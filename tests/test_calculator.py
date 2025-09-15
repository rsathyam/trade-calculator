import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import calculator as calc


def _mk_history(n=5, base=100.0):
    idx = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=n, freq="D", tz="UTC").tz_convert(None)
    prices = np.linspace(base * 0.98, base * 1.02, n)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": np.linspace(1_000_000, 2_000_000, n),
        },
        index=idx,
    )
    return df


def test_expected_move_populates_with_partial_quotes(monkeypatch):
    symbol = "TEST"
    today = datetime.today().date()
    exp1 = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    exp2 = (today + timedelta(days=46)).strftime("%Y-%m-%d")

    # Underlying spot
    monkeypatch.setattr(calc, "get_current_price", lambda s: 100.0)

    # Two expirations, including a >=45d date so filter_dates passes
    monkeypatch.setattr(calc, "get_option_expirations", lambda s: [exp1, exp2])

    # Option chain near-the-money, with partial quotes (only bid present)
    def fake_chain(sym, expiration, underlying_price=None):
        strikes = [95.0, 100.0, 105.0]
        calls = [
            {"strike": k, "impliedVolatility": 0.5, "bid": 5.0 if k == 100.0 else None, "ask": None}
            for k in strikes
        ]
        puts = [
            {"strike": k, "impliedVolatility": 0.7, "bid": 4.0 if k == 100.0 else None, "ask": None}
            for k in strikes
        ]
        return {"calls": calls, "puts": puts}

    monkeypatch.setattr(calc, "get_option_chain", fake_chain)

    # Provide some price history; exact values don't matter for Expected Move
    monkeypatch.setattr(calc, "get_price_history", lambda s: _mk_history())

    result = calc.compute_recommendation(symbol)
    assert isinstance(result, dict)
    # Straddle is 5 + 4 = 9; expected move is 9% of 100
    assert result.get("expected_move") == "9.0%"


def test_expected_move_uses_last_trade_when_no_quotes(monkeypatch):
    symbol = "TEST2"
    today = datetime.today().date()
    exp1 = (today + timedelta(days=10)).strftime("%Y-%m-%d")
    exp2 = (today + timedelta(days=50)).strftime("%Y-%m-%d")

    monkeypatch.setattr(calc, "get_current_price", lambda s: 200.0)
    monkeypatch.setattr(calc, "get_option_expirations", lambda s: [exp1, exp2])

    def fake_chain(sym, expiration, underlying_price=None):
        strikes = [190.0, 200.0, 210.0]
        calls = [
            {"strike": k, "impliedVolatility": 0.4, "bid": None, "ask": None, "last": 6.0 if k == 200.0 else None}
            for k in strikes
        ]
        puts = [
            {"strike": k, "impliedVolatility": 0.4, "bid": None, "ask": None, "last": 5.0 if k == 200.0 else None}
            for k in strikes
        ]
        return {"calls": calls, "puts": puts}

    monkeypatch.setattr(calc, "get_option_chain", fake_chain)
    monkeypatch.setattr(calc, "get_price_history", lambda s: _mk_history())

    result = calc.compute_recommendation(symbol)
    assert isinstance(result, dict)
    # Straddle from last trades: 6 + 5 = 11; 11 / 200 = 5.5%
    assert result.get("expected_move") == "5.5%"


def test_term_structure_deduplicates_days():
    ts = calc.build_term_structure([30, 30, 45], [0.5, 0.7, 0.6])
    assert abs(ts(30) - 0.6) < 1e-9
    # Interpolation remains stable between 30 and 45
    mid = ts(37.5)
    assert abs(mid - 0.6) < 1e-9


def test_yang_zhang_handles_insufficient_data():
    # Single row leads to insufficient rolling windows; function should return NaN
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1_000_000],
        },
        index=pd.to_datetime([datetime.utcnow()]),
    )
    val = calc.yang_zhang(df)
    assert isinstance(val, float) and math.isnan(val)


def test_expected_move_falls_back_to_iv(monkeypatch):
    symbol = "TEST3"
    today = datetime.today().date()
    exp1 = (today + timedelta(days=20)).strftime("%Y-%m-%d")
    exp2 = (today + timedelta(days=60)).strftime("%Y-%m-%d")

    monkeypatch.setattr(calc, "get_current_price", lambda s: 150.0)
    monkeypatch.setattr(calc, "get_option_expirations", lambda s: [exp1, exp2])

    # No quotes and no last, but IV present
    def fake_chain(sym, expiration, underlying_price=None):
        strikes = [140.0, 150.0, 160.0]
        calls = [
            {"strike": k, "impliedVolatility": 0.4, "bid": None, "ask": None, "last": None, "mark": None}
            for k in strikes
        ]
        puts = [
            {"strike": k, "impliedVolatility": 0.4, "bid": None, "ask": None, "last": None, "mark": None}
            for k in strikes
        ]
        return {"calls": calls, "puts": puts}

    monkeypatch.setattr(calc, "get_option_chain", fake_chain)
    monkeypatch.setattr(calc, "get_price_history", lambda s: _mk_history())

    result = calc.compute_recommendation(symbol)
    assert isinstance(result, dict)
    # Fallback uses IV * sqrt(T); with 20 days and IV=0.4 => 0.4*sqrt(20/365)=~0.118
    # So about 11.8%
    assert result.get("expected_move") == f"{round(0.4 * math.sqrt(20/365) * 100.0, 2)}%"
