"""
Kairos Engine — CSV Data Loader

Load real NIFTY (or any) OHLCV candle data from CSV files.
Supports common formats from TradingView, Zerodha Kite, and generic exports.

Expected columns (case-insensitive, flexible naming):
  timestamp/date/datetime, open, high, low, close, volume

Usage:
    loader = CSVLoader()
    candles = loader.load("path/to/nifty_1min.csv")
    closes, highs, lows, opens, volumes = loader.to_arrays(candles)
"""

import csv
from datetime import datetime
from pathlib import Path
import numpy as np

from data.models.candle import Candle
from engine.core.types import FloatArray


# Common column name mappings
COLUMN_MAP = {
    "timestamp": ["timestamp", "date", "datetime", "time", "ts"],
    "open": ["open", "o", "open_price"],
    "high": ["high", "h", "high_price"],
    "low": ["low", "l", "low_price"],
    "close": ["close", "c", "close_price", "ltp"],
    "volume": ["volume", "vol", "v", "qty", "quantity"],
}

DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d",
]


class CSVLoader:
    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol

    def load(self, path: str) -> list[Candle]:
        """Load candles from a CSV file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        with open(p, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no headers")

            col_map = self._map_columns(reader.fieldnames)
            candles = []

            for row in reader:
                try:
                    ts = self._parse_timestamp(row[col_map["timestamp"]])
                    candle = Candle(
                        timestamp=ts,
                        open=float(row[col_map["open"]]),
                        high=float(row[col_map["high"]]),
                        low=float(row[col_map["low"]]),
                        close=float(row[col_map["close"]]),
                        volume=float(row.get(col_map.get("volume", ""), "0") or "0"),
                        symbol=self.symbol,
                    )
                    candles.append(candle)
                except (ValueError, KeyError):
                    continue

        candles.sort(key=lambda c: c.timestamp)
        return candles

    def to_arrays(
        self, candles: list[Candle]
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Convert candle list to numpy arrays (closes, highs, lows, opens, volumes)."""
        closes = np.array([c.close for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        opens = np.array([c.open for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)
        return closes, highs, lows, opens, volumes

    def _map_columns(self, headers: list[str]) -> dict[str, str]:
        """Map CSV headers to standard names."""
        lower_headers = {h.lower().strip(): h for h in headers}
        result = {}

        for std_name, aliases in COLUMN_MAP.items():
            for alias in aliases:
                if alias in lower_headers:
                    result[std_name] = lower_headers[alias]
                    break

        required = ["timestamp", "open", "high", "low", "close"]
        missing = [r for r in required if r not in result]
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Found: {list(lower_headers.keys())}"
            )

        return result

    def _parse_timestamp(self, value: str) -> datetime:
        """Try multiple date formats."""
        value = value.strip()
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse timestamp: {value}")
