"""
DISCLAIMER: 

This software is provided solely for educational and research purposes. 
It is not intended to provide investment advice, and no investment recommendations are made herein. 
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software. 
Always consult a professional financial advisor before making any investment decisions.
"""

import os
import time
import math
from datetime import datetime, timedelta, timezone

import requests

try:
    # Official Polygon Python client
    from polygon import RESTClient as PolygonRESTClient  # type: ignore
except Exception:  # pragma: no cover
    PolygonRESTClient = None  # type: ignore
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---- Configure your Polygon API key ----
# Recommended: export POLYGON_API_KEY in your environment.
# Note: Read from the standard env var name, not a literal key.
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
POLYGON_API_BASE = "https://api.polygon.io"

# Optional debug mode: set DEBUG=1 to print extra logs
DEBUG = bool(os.environ.get("DEBUG"))


def _debug_log(*args):
    if DEBUG:
        print("[DEBUG]", *args)

# Initialize REST client if library is available
POLYGON_CLIENT = None
if PolygonRESTClient and POLYGON_API_KEY:
    try:
        POLYGON_CLIENT = PolygonRESTClient(POLYGON_API_KEY)
    except Exception:
        POLYGON_CLIENT = None


def _polygon_get(path, params=None):
    if not POLYGON_API_KEY:
        raise ValueError(
            "Missing POLYGON_API_KEY. Set env var POLYGON_API_KEY to your Polygon API key."
        )
    params = dict(params or {})
    # Polygon supports apiKey as query param
    params.setdefault("apiKey", POLYGON_API_KEY)
    url = path if path.startswith("http") else f"{POLYGON_API_BASE}{path}"
    safe_params = {**params}
    if "apiKey" in safe_params:
        safe_params["apiKey"] = "***"
    _debug_log("GET", url, safe_params)
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        body = None
        try:
            body = e.response.json()
        except Exception:
            body = (e.response.text or "").strip() if e.response is not None else ""
        raise RuntimeError(f"Polygon HTTP {status} for {url}: {body}") from e
    except requests.RequestException as e:
        raise RuntimeError(f"Network error calling {url}: {e}") from e


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[: i + 1]]
            break
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data["High"] / price_data["Open"]).apply(np.log)
    log_lo = (price_data["Low"] / price_data["Open"]).apply(np.log)
    log_co = (price_data["Close"] / price_data["Open"]).apply(np.log)
    log_oc = (price_data["Open"] / price_data["Close"].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    log_cc = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (
        1.0 / (window - 1.0)
    )
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (
        1.0 / (window - 1.0)
    )
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(
        trading_periods
    )
    if return_last_only:
        # Return the last non-NaN value if available; otherwise NaN
        last = result.dropna()
        return last.iloc[-1] if not last.empty else float("nan")
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    # Prepare arrays and drop non-finite values
    days_arr = np.asarray(days, dtype=float)
    ivs_arr = np.asarray(ivs, dtype=float)
    mask = np.isfinite(days_arr) & np.isfinite(ivs_arr)
    days_arr = days_arr[mask]
    ivs_arr = ivs_arr[mask]

    # Deduplicate days by averaging IVs for the same day (prevents zero division in interp1d)
    if days_arr.size == 0:
        def term_spline(_):
            return float("nan")
        return term_spline

    unique = {}
    for d, v in zip(days_arr, ivs_arr):
        di = int(round(d))
        if di in unique:
            unique[di].append(float(v))
        else:
            unique[di] = [float(v)]

    uniq_days = np.array(sorted(unique.keys()), dtype=float)
    uniq_ivs = np.array([float(np.mean(unique[d])) for d in uniq_days], dtype=float)

    # If only one unique day, return a constant function
    if uniq_days.size == 1:
        const_iv = float(uniq_ivs[0])
        def term_spline(_):
            return const_iv
        return term_spline

    # Build a linear interpolator on strictly increasing x
    spline = interp1d(uniq_days, uniq_ivs, kind="linear", fill_value="extrapolate")

    def term_spline(dte):
        if dte < uniq_days[0]:
            return float(uniq_ivs[0])
        elif dte > uniq_days[-1]:
            return float(uniq_ivs[-1])
        else:
            return float(spline(dte))

    return term_spline


def get_current_price(symbol):
    # Prefer official client if available
    if POLYGON_CLIENT is not None:
        # Try last trade
        try:
            if hasattr(POLYGON_CLIENT, "get_last_trade"):
                lt = POLYGON_CLIENT.get_last_trade(symbol)
                price = getattr(lt, "price", None) or getattr(lt, "p", None)
                if price is not None:
                    return float(price)
        except Exception:
            pass
        # Fallback to previous close via client
        try:
            # handle potential method name differences
            if hasattr(POLYGON_CLIENT, "get_previous_close"):
                pc = POLYGON_CLIENT.get_previous_close(symbol, adjusted=True)
            elif hasattr(POLYGON_CLIENT, "get_previous_close_agg"):
                pc = POLYGON_CLIENT.get_previous_close_agg(symbol, adjusted=True)
            else:
                pc = None
            if pc is not None:
                results = getattr(pc, "results", None) or getattr(pc, "results", [])
                if results:
                    c = getattr(results[0], "c", None) or results[0].get("c")
                    if c is not None:
                        return float(c)
        except Exception:
            pass

    # Try snapshot endpoint for most recent trade price via HTTP
    try:
        data = _polygon_get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")
        # Structure: { ticker: {..., lastTrade: { p: price } } }
        last_trade = data.get("ticker", {}).get("lastTrade", {})
        price = last_trade.get("p")
        if price is not None:
            return float(price)
    except Exception:
        pass

    # Fallback to previous close aggregate if snapshot not available
    try:
        data = _polygon_get(
            f"/v2/aggs/ticker/{symbol}/prev", params={"adjusted": "true"}
        )
        results = data.get("results", [])
        if results:
            return float(results[0].get("c"))
    except Exception:
        pass

    raise ValueError("Unable to retrieve current price from Polygon")


def get_price_history(symbol, period_days=90):
    # Use timezone-aware UTC datetime to avoid deprecated utcnow()
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=period_days)

    # Prefer official client if available
    if POLYGON_CLIENT is not None:
        try:
            # method name in client is typically get_aggs
            if hasattr(POLYGON_CLIENT, "get_aggs"):
                aggs = POLYGON_CLIENT.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=from_date.strftime("%Y-%m-%d"),
                    to=to_date.strftime("%Y-%m-%d"),
                    adjusted=True,
                    sort="asc",
                    limit=50000,
                )
                results = getattr(aggs, "results", None) or []
                if results:
                    df = pd.DataFrame(
                        {
                            "Open": [
                                getattr(r, "o", None) or r.get("o") for r in results
                            ],
                            "High": [
                                getattr(r, "h", None) or r.get("h") for r in results
                            ],
                            "Low": [
                                getattr(r, "l", None) or r.get("l") for r in results
                            ],
                            "Close": [
                                getattr(r, "c", None) or r.get("c") for r in results
                            ],
                            "Volume": [
                                getattr(r, "v", None) or r.get("v") for r in results
                            ],
                        },
                        index=pd.to_datetime(
                            [getattr(r, "t", None) or r.get("t") for r in results],
                            unit="ms",
                            utc=True,
                        ).tz_convert(None),
                    )
                    return df
        except Exception:
            pass

    # HTTP fallback
    path = (
        f"/v2/aggs/ticker/{symbol}/range/1/day/{from_date:%Y-%m-%d}/{to_date:%Y-%m-%d}"
    )
    data = _polygon_get(path, params={"adjusted": "true", "sort": "asc", "limit": 5000})
    if data.get("resultsCount", 0) == 0 or not data.get("results"):
        raise Exception("Error fetching price history from Polygon")
    results = data["results"]
    df = pd.DataFrame(
        {
            "Open": [row.get("o") for row in results],
            "High": [row.get("h") for row in results],
            "Low": [row.get("l") for row in results],
            "Close": [row.get("c") for row in results],
            "Volume": [row.get("v") for row in results],
        },
        index=pd.to_datetime(
            [row.get("t") for row in results], unit="ms", utc=True
        ).tz_convert(None),
    )
    return df


def get_option_expirations(symbol):
    # Prefer official client if available
    expirations = set()
    if POLYGON_CLIENT is not None and hasattr(POLYGON_CLIENT, "list_options_contracts"):
        try:
            for contract in POLYGON_CLIENT.list_options_contracts(
                underlying_ticker=symbol,
                expired=False,
                limit=1000,
                sort="expiration_date",
                order="asc",
            ):
                exp = getattr(contract, "expiration_date", None) or contract.get(
                    "expiration_date"
                )
                if exp:
                    expirations.add(str(exp))
            if expirations:
                return sorted(expirations)
        except Exception:
            pass

    # HTTP fallback
    url = f"{POLYGON_API_BASE}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": symbol,
        "expired": "false",
        "limit": 1000,
        "sort": "expiration_date",
        "order": "asc",
        "apiKey": POLYGON_API_KEY,
    }
    while True:
        data = _polygon_get(url, params)
        results = data.get("results", [])
        for item in results:
            exp = item.get("expiration_date")
            if exp:
                expirations.add(exp)
        next_url = data.get("next_url")
        if not next_url:
            break
        time.sleep(0.05)
        url = next_url
        params = {}
    return sorted(expirations)


def get_option_chain(symbol, expiration, underlying_price=None):
    # Prefer official client if available
    if POLYGON_CLIENT is not None and hasattr(POLYGON_CLIENT, "list_options_snapshots"):
        for lim in (250, 150, 100, 50):
            try:
                calls, puts = [], []
                it = POLYGON_CLIENT.list_options_snapshots(
                    underlying_asset=symbol,
                    expiration_date=expiration,
                    limit=lim,
                    order="asc",
                    sort="strike_price",
                )
                for snap in it:
                    details = getattr(snap, "details", None) or {}
                    last_quote = getattr(snap, "last_quote", None) or {}
                    contract_type = (
                        getattr(details, "contract_type", None)
                        if hasattr(details, "contract_type")
                        else (
                            details.get("contract_type")
                            if isinstance(details, dict)
                            else None
                        )
                    )
                    strike = (
                        getattr(details, "strike_price", None)
                        if hasattr(details, "strike_price")
                        else (
                            details.get("strike_price")
                            if isinstance(details, dict)
                            else None
                        )
                    )
                    iv = (
                        getattr(snap, "implied_volatility", None)
                        if hasattr(snap, "implied_volatility")
                        else None
                    )
                    bid = (
                        getattr(last_quote, "bid", None)
                        if hasattr(last_quote, "bid")
                        else last_quote.get("bid") if isinstance(last_quote, dict) else None
                    )
                    ask = (
                        getattr(last_quote, "ask", None)
                        if hasattr(last_quote, "ask")
                        else last_quote.get("ask") if isinstance(last_quote, dict) else None
                    )
                    mark = None
                    if isinstance(last_quote, dict):
                        mark = (
                            last_quote.get("mid")
                            or last_quote.get("midpoint")
                            or last_quote.get("mark")
                            or last_quote.get("mark_price")
                        )
                    # Optional last trade price as fallback
                    last_trade = getattr(snap, "last_trade", None)
                    last_price = None
                    if last_trade is not None:
                        if hasattr(last_trade, "price"):
                            last_price = getattr(last_trade, "price", None)
                        elif isinstance(last_trade, dict):
                            last_price = last_trade.get("price") or last_trade.get("p")
                    row = {
                        "strike": float(strike) if strike is not None else None,
                        "impliedVolatility": float(iv) if iv is not None else None,
                        "bid": float(bid) if bid is not None else None,
                        "ask": float(ask) if ask is not None else None,
                        "mark": float(mark) if mark is not None else None,
                        "last": float(last_price) if last_price is not None else None,
                    }
                    if contract_type and str(contract_type).lower() == "call":
                        calls.append(row)
                    elif contract_type and str(contract_type).lower() == "put":
                        puts.append(row)
                return {"calls": calls, "puts": puts}
            except Exception as e:
                _debug_log(f"list_options_snapshots failed with limit={lim}: {e}")
                continue

    # HTTP fallback
    data = None
    last_err = None
    # Build optional strike filters around the underlying to reduce result size
    strike_params = {}
    if underlying_price is not None:
        # +/- 30% window around spot
        lower = max(0.01, float(underlying_price) * 0.7)
        upper = float(underlying_price) * 1.3
        strike_params = {
            "strike_price.gte": round(lower, 2),
            "strike_price.lte": round(upper, 2),
        }
    # Try with conservative limits first and expand if needed
    for lim in (25, 10, 5, None):
        try:
            params = {
                "expiration_date": expiration,
                "order": "asc",
                "sort": "strike_price",
                **strike_params,
            }
            if lim is not None:
                params["limit"] = lim
            data = _polygon_get(
                f"/v3/snapshot/options/{symbol}",
                params=params,
            )
            break
        except Exception as e:
            _debug_log(f"HTTP snapshot retry failed (limit={lim}, filters={strike_params}): {e}")
            last_err = e
            data = None
            continue
    if data is None and last_err:
        raise last_err
    results = data.get("results", [])
    calls, puts = [], []
    for item in results:
        details = item.get("details", {})
        last_quote = item.get("last_quote", {}) or item.get("lastQuote", {})
        # Contract type normalization across possible keys
        contract_type = (
            details.get("contract_type")
            or details.get("contractType")
            or item.get("contract_type")
            or item.get("contractType")
            or details.get("option_type")
            or details.get("optionType")
            or item.get("option_type")
            or item.get("optionType")
            or details.get("type")
            or item.get("type")
            or details.get("put_call")
            or item.get("put_call")
        )
        strike = details.get("strike_price") or details.get("strikePrice")
        iv = (
            item.get("implied_volatility")
            or item.get("iv")
            or (item.get("greeks", {}) or {}).get("implied_volatility")
        )
        bid = (
            last_quote.get("bid")
            or last_quote.get("bid_price")
            or last_quote.get("pBid")
        )
        ask = (
            last_quote.get("ask")
            or last_quote.get("ask_price")
            or last_quote.get("pAsk")
        )
        mark = (
            last_quote.get("mid")
            or last_quote.get("midpoint")
            or last_quote.get("mark")
            or last_quote.get("mark_price")
        )
        last_trade = item.get("last_trade") or item.get("lastTrade") or {}
        last_price = (
            last_trade.get("price")
            or last_trade.get("p")
            or item.get("last_price")
            or item.get("lastPrice")
        )
        row = {
            "strike": float(strike) if strike is not None else None,
            "impliedVolatility": float(iv) if iv is not None else None,
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "mark": float(mark) if mark is not None else None,
            "last": float(last_price) if last_price is not None else None,
        }
        if contract_type and contract_type.lower() == "call":
            calls.append(row)
        elif contract_type and contract_type.lower() == "put":
            puts.append(row)
    return {"calls": calls, "puts": puts}


def compute_recommendation(symbol):
    # Normalize symbol formatting early
    symbol = (symbol or "").strip().upper()
    try:
        symbol = symbol.strip().upper()
        if not symbol:
            return "No stock symbol provided."

        # Get current price first to enable strike filtering
        try:
            underlying_price = get_current_price(symbol)
            if underlying_price is None:
                raise ValueError("No market price found.")
        except Exception:
            return "Error: Unable to retrieve underlying stock price."

        # Get option expirations
        try:
            exp_dates = get_option_expirations(symbol)
            _debug_log("Found expirations count:", len(exp_dates))
            if not exp_dates:
                return f"Error: No options found for '{symbol}' (empty expirations)."
        except Exception as e:
            return f"Error fetching options for '{symbol}': {e}"

        try:
            exp_dates = filter_dates(exp_dates)
        except Exception:
            return "Error: Not enough option data."

        # Get option chains for each expiration using strike filters around spot
        options_chains = {}
        for exp_date in exp_dates:
            options_chains[exp_date] = get_option_chain(symbol, exp_date, underlying_price)

        # Helpers to compute robust mid prices and ATM IV/straddle
        def _mid_from_quotes(bid, ask, last=None, mark=None):
            vals = [v for v in (bid, ask) if v is not None and np.isfinite(v) and v > 0]
            if len(vals) == 2:
                return (vals[0] + vals[1]) / 2.0
            # Prefer mark then last if available
            if mark is not None and np.isfinite(mark) and mark > 0:
                return float(mark)
            if last is not None and np.isfinite(last) and last > 0:
                return float(last)
            if len(vals) == 1:
                return vals[0]
            return None

        def _nearest_mid(side, target):
            if not side:
                return None
            strikes = np.array([s.get("strike") for s in side], dtype=float)
            # Filter to finite strikes
            finite = np.isfinite(strikes)
            if not finite.any():
                return None
            idxs = np.where(finite)[0]
            order = idxs[np.argsort(np.abs(strikes[idxs] - target))]
            for j in order:
                bid = side[j].get("bid")
                ask = side[j].get("ask")
                last = side[j].get("last")
                mark = side[j].get("mark")
                mid = _mid_from_quotes(bid, ask, last, mark)
                if mid is not None and mid > 0:
                    return float(mid)
            return None

        # Find ATM IV and straddle for each expiration
        atm_iv = {}
        straddle = None
        for exp_date, chain in options_chains.items():
            calls = chain.get("calls", [])
            puts = chain.get("puts", [])
            if not calls or not puts:
                continue

            # Find ATM call and put
            call_strikes = np.array([c["strike"] for c in calls], dtype=float)
            put_strikes = np.array([p["strike"] for p in puts], dtype=float)
            call_idx = np.nanargmin(np.abs(call_strikes - underlying_price)) if call_strikes.size else None
            put_idx = np.nanargmin(np.abs(put_strikes - underlying_price)) if put_strikes.size else None
            call_iv = calls[call_idx].get("impliedVolatility") if call_idx is not None else None
            put_iv = puts[put_idx].get("impliedVolatility") if put_idx is not None else None

            if call_iv is None or put_iv is None:
                continue

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            # Capture first available straddle (near-term) from nearest quotes
            if straddle is None:
                call_mid = _nearest_mid(calls, underlying_price)
                put_mid = _nearest_mid(puts, underlying_price)
                if call_mid is not None and put_mid is not None:
                    straddle = call_mid + put_mid

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.today().date()
        dtes = []
        ivs = []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)

        term_spline = build_term_structure(dtes, ivs)
        # Guard against zero denominator if dtes[0] equals 45
        denom = (45 - dtes[0])
        ts_slope_0_45 = (
            (term_spline(45) - term_spline(dtes[0])) / denom if denom != 0 else 0.0
        )

        price_history = get_price_history(symbol)
        rv30 = yang_zhang(price_history)
        iv30 = term_spline(30)
        iv30_rv30 = (iv30 / rv30) if np.isfinite(rv30) and rv30 > 0 else float("nan")
        # Use a rolling window sized to available history; require at least 1 observation
        window = int(min(30, max(1, len(price_history))))
        avg_volume = (
            price_history["Volume"].rolling(window=window, min_periods=1).mean().iloc[-1]
        )
        # Compute expected move as percent of underlying using nearest valid straddle; fallback to IV-based estimate
        expected_move = None
        if straddle is not None and underlying_price and underlying_price > 0:
            expected_move = f"{round((straddle / underlying_price) * 100.0, 2)}%"
        else:
            # Fallback: use shortest-dated ATM IV to estimate EM ~ IV * sqrt(T)
            if dtes and ivs:
                # Identify the smallest positive DTE with an IV
                pairs = [(d, v) for d, v in zip(dtes, ivs) if d is not None and d > 0 and np.isfinite(v)]
                if pairs:
                    pairs.sort(key=lambda x: x[0])
                    dte, iv_atm = pairs[0]
                    T = max(0.0, float(dte)) / 365.0
                    em_pct = iv_atm * math.sqrt(T) * 100.0
                    expected_move = f"{round(em_pct, 2)}%"

        return {
            "avg_volume": avg_volume >= 1500000,
            "iv30_rv30": iv30_rv30 >= 1.25,
            "ts_slope_0_45": ts_slope_0_45 <= -0.00406,
            "expected_move": expected_move,
        }
    except Exception as e:
        return f"Error occurred processing: {e}"


def print_recommendation(result):
    if isinstance(result, str):
        print(result)
        return
    avg_volume_bool = result["avg_volume"]
    iv30_rv30_bool = result["iv30_rv30"]
    ts_slope_bool = result["ts_slope_0_45"]
    expected_move = result["expected_move"]
    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        title = "Recommended"
    elif ts_slope_bool and (
        (avg_volume_bool and not iv30_rv30_bool)
        or (iv30_rv30_bool and not avg_volume_bool)
    ):
        title = "Consider"
    else:
        title = "Avoid"
    print(f"\nResult: {title}")
    print(f"avg_volume:    {'PASS' if avg_volume_bool else 'FAIL'}")
    print(f"iv30_rv30:     {'PASS' if iv30_rv30_bool else 'FAIL'}")
    print(f"ts_slope_0_45: {'PASS' if ts_slope_bool else 'FAIL'}")
    print(f"Expected Move: {expected_move}")


def main():
    print("Earnings Position Checker (CLI, Polygon)")
    print("----------------------------------------")
    while True:
        stock = input("Enter Stock Symbol (or 'exit' to quit): ").strip()
        if stock.lower() == "exit":
            print("Exiting.")
            break
        if not stock:
            print("Please enter a valid stock symbol.")
            continue
        print("Processing, please wait...")
        result = compute_recommendation(stock)
        print_recommendation(result)
        print("-" * 40)


if __name__ == "__main__":
    main()
