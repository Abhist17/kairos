"""
Kairos Engine — Multi-Asset Configuration

Supports: NIFTY, BANKNIFTY, SENSEX, Gold, BTC, ETH, US stocks.
Each asset has its own strike step, ticker, and scaling.
"""

from dataclasses import dataclass


@dataclass
class AssetConfig:
    symbol: str  # internal name
    yfinance_ticker: str  # yfinance symbol
    tv_ticker: str  # TradingView symbol
    strike_step: float  # option strike gap (or price rounding for non-options)
    asset_type: str  # "index_option", "commodity", "crypto", "stock"
    currency: str  # INR, USD
    lot_size: int  # contract lot size
    exchange: str  # NSE, MCX, COMEX, CRYPTO


ASSETS = {
    # Indian Index Options
    "NIFTY": AssetConfig(
        "NIFTY", "^NSEI", "NSE:NIFTY", 50, "index_option", "INR", 25, "NSE"
    ),
    "BANKNIFTY": AssetConfig(
        "BANKNIFTY", "^NSEBANK", "NSE:BANKNIFTY", 100, "index_option", "INR", 15, "NSE"
    ),
    "SENSEX": AssetConfig(
        "SENSEX", "^BSESN", "BSE:SENSEX", 100, "index_option", "INR", 10, "BSE"
    ),
    "FINNIFTY": AssetConfig(
        "FINNIFTY",
        "NIFTY_FIN_SERVICE.NS",
        "NSE:FINNIFTY",
        50,
        "index_option",
        "INR",
        25,
        "NSE",
    ),
    # Gold
    "GOLD": AssetConfig(
        "GOLD", "GC=F", "COMEX:GC1!", 10, "commodity", "USD", 1, "COMEX"
    ),
    "GOLDM": AssetConfig(
        "GOLDM", "GC=F", "MCX:GOLD1!", 100, "commodity", "INR", 1, "MCX"
    ),
    "SILVER": AssetConfig(
        "SILVER", "SI=F", "COMEX:SI1!", 0.5, "commodity", "USD", 1, "COMEX"
    ),
    # Crypto
    "BTC": AssetConfig(
        "BTC", "BTC-USD", "BINANCE:BTCUSDT", 500, "crypto", "USD", 1, "CRYPTO"
    ),
    "ETH": AssetConfig(
        "ETH", "ETH-USD", "BINANCE:ETHUSDT", 50, "crypto", "USD", 1, "CRYPTO"
    ),
    "SOL": AssetConfig(
        "SOL", "SOL-USD", "BINANCE:SOLUSDT", 5, "crypto", "USD", 1, "CRYPTO"
    ),
    # US Stocks
    "AAPL": AssetConfig(
        "AAPL", "AAPL", "NASDAQ:AAPL", 2.5, "stock", "USD", 100, "NASDAQ"
    ),
    "TSLA": AssetConfig(
        "TSLA", "TSLA", "NASDAQ:TSLA", 5, "stock", "USD", 100, "NASDAQ"
    ),
    "SPY": AssetConfig("SPY", "SPY", "AMEX:SPY", 1, "stock", "USD", 100, "AMEX"),
    "QQQ": AssetConfig("QQQ", "QQQ", "NASDAQ:QQQ", 1, "stock", "USD", 100, "NASDAQ"),
    "NVDA": AssetConfig(
        "NVDA", "NVDA", "NASDAQ:NVDA", 5, "stock", "USD", 100, "NASDAQ"
    ),
    # Indian Stocks
    "RELIANCE": AssetConfig(
        "RELIANCE", "RELIANCE.NS", "NSE:RELIANCE", 20, "stock", "INR", 250, "NSE"
    ),
    "HDFCBANK": AssetConfig(
        "HDFCBANK", "HDFCBANK.NS", "NSE:HDFCBANK", 20, "stock", "INR", 550, "NSE"
    ),
    "TCS": AssetConfig("TCS", "TCS.NS", "NSE:TCS", 50, "stock", "INR", 175, "NSE"),
    "SBIN": AssetConfig("SBIN", "SBIN.NS", "NSE:SBIN", 5, "stock", "INR", 1500, "NSE"),
}


def get_asset(symbol: str) -> AssetConfig:
    key = symbol.upper().replace(" ", "")
    if key in ASSETS:
        return ASSETS[key]
    # Default: treat as Indian stock
    return AssetConfig(
        symbol=symbol,
        yfinance_ticker=f"{symbol}.NS",
        tv_ticker=f"NSE:{symbol}",
        strike_step=10,
        asset_type="stock",
        currency="INR",
        lot_size=1,
        exchange="NSE",
    )
