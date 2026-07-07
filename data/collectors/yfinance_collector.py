"""
Kairos Engine — Yahoo Finance collector (FREE, no signup)

pip install yfinance — that's it. No API key, no account.
~15 min delayed on Indian markets. 2m interval gives 60 days history.
"""

import numpy as np
from data.models.candle import Candle
from data.collectors.base import BaseCollector


SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "SBIN": "SBIN.NS",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "SPY": "SPY",
}

PERIOD_MAP = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "1d": "max",
}


class YFinanceCollector(BaseCollector):
    def __init__(self):
        self.yf = None
        self._connected = False

    @property
    def name(self) -> str:
        return "Yahoo Finance (free, ~15min delayed)"

    def connect(self) -> bool:
        try:
            import yfinance

            self.yf = yfinance
            self._connected = True
            print("  [yfinance] Ready — no login needed")
            print("  [yfinance] Note: Indian market data is ~15 min delayed")
            return True
        except ImportError:
            print("  [yfinance] Not installed. Run: pip install yfinance")
            return False

    def get_candles(self, symbol="NIFTY", interval="2m", count=200):
        if not self._connected or not self.yf:
            return []

        ticker_symbol = SYMBOL_MAP.get(symbol.upper(), f"{symbol}.NS")
        period = PERIOD_MAP.get(interval, "60d")

        try:
            ticker = self.yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"  [yfinance] No data for {ticker_symbol}")
                return []

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            candles = []
            for ts, row in df.tail(count).iterrows():
                if np.isnan(row["Close"]):
                    continue
                candles.append(
                    Candle(
                        timestamp=ts.to_pydatetime(),
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row.get("Volume", 0)),
                        symbol=symbol,
                    )
                )

            if candles:
                print(
                    f"  [yfinance] {ticker_symbol}: {len(candles)} candles, "
                    f"latest={candles[-1].timestamp.strftime('%Y-%m-%d %H:%M')} "
                    f"@ {candles[-1].close:,.2f}"
                )
            return candles

        except Exception as e:
            print(f"  [yfinance] Error: {e}")
            return []

    def get_ltp(self, symbol="NIFTY"):
        if not self._connected or not self.yf:
            return 0.0
        ticker_symbol = SYMBOL_MAP.get(symbol.upper(), f"{symbol}.NS")
        try:
            ticker = self.yf.Ticker(ticker_symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass
        return 0.0

    def disconnect(self):
        self._connected = False
