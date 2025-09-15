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

Notes
- The app queries Polygon v2/v3 endpoints. A valid API key is required.
- Option IV, bid, ask are derived from Polygon snapshot fields where available.
