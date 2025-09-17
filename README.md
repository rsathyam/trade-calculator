# Trade Calculator

CLI tool that analyzes earnings positioning using options data and historical volatility.

Now uses Polygon.io instead of Finnhub for:
- Current price (snapshot/prev close fallback)
- Daily OHLCV aggregates (last 90 days)
- Option expirations and option chain snapshots

Setup
- Create and export your Polygon API key: `export POLYGON_API_KEY=your_key_here`
- Install dependencies: `pip install -r requirements.txt`

Run
- `python calculator.py`
  - Interactive prompts support two modes:
    - Single symbol: enter a ticker like `AAPL` to get a recommendation.
    - Auto-scan: type `auto` to scan all tickers with earnings in the next 5 business days (Polygon), or `auto N` to set the window to N business days.

Notes
- The app queries Polygon v2/v3 endpoints. A valid API key is required.
- Option IV, bid, ask are derived from Polygon snapshot fields where available.
- Auto-scan uses Polygon `/v3/reference/earnings` filtered by `announcement_date` within the next N business days (weekdays only; holidays not accounted for).
 - If your Polygon plan does not include earnings endpoints or they return 404, you can provide upcoming earnings tickers via:
   - `EARNINGS_TICKERS`: comma-separated list, e.g., `export EARNINGS_TICKERS=AAPL,MSFT,NVDA`
   - `EARNINGS_FILE`: path to a file containing tickers (CSV or one per line; `#` comments allowed), e.g., `export EARNINGS_FILE=/path/to/upcoming_earnings.txt`
