"""
Kairos Engine — NSE Direct Live Data (FREE, real-time, no signup)

NSE blocks simple requests. We mimic a real browser session:
  1. Hit nseindia.com with full browser headers
  2. Follow redirects, accept cookies
  3. Then call API endpoints with those cookies
"""

import time
import requests
import numpy as np
from datetime import datetime
from data.models.candle import Candle
from data.collectors.base import BaseCollector


NSE_BASE = "https://www.nseindia.com"

INDEX_ID_MAP = {
    "NIFTY": "NIFTY 50",
    "NIFTY 50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK",
    "NIFTY BANK": "NIFTY BANK",
    "FINNIFTY": "NIFTY FIN SERVICE",
}

INDEX_SYMBOL_MAP = INDEX_ID_MAP.copy()


class NSELiveCollector(BaseCollector):
    def __init__(self):
        self.session = requests.Session()
        self._connected = False
        self._last_cookie_time = 0
        self._cookie_ttl = 90

        # Full browser-like headers — this is what makes NSE accept us
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
                "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Linux"',
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
            }
        )

    @property
    def name(self) -> str:
        return "NSE India Direct (free, real-time)"

    def connect(self) -> bool:
        return self._refresh_cookies()

    def _refresh_cookies(self) -> bool:
        now = time.time()
        if self._connected and (now - self._last_cookie_time) < self._cookie_ttl:
            return True

        try:
            # Step 1: Hit homepage like a browser
            self.session.headers["Referer"] = "https://www.google.com/"
            self.session.headers["Sec-Fetch-Site"] = "cross-site"
            self.session.headers["Sec-Fetch-Mode"] = "navigate"
            self.session.headers["Sec-Fetch-Dest"] = "document"
            self.session.headers["Accept"] = (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            )

            r = self.session.get(NSE_BASE, timeout=15, allow_redirects=True)

            if r.status_code != 200:
                # Try the option chain page instead — sometimes works better
                r = self.session.get(
                    f"{NSE_BASE}/option-chain", timeout=15, allow_redirects=True
                )

            if r.status_code == 200:
                # Reset headers for API calls
                self.session.headers["Referer"] = "https://www.nseindia.com/"
                self.session.headers["Sec-Fetch-Site"] = "same-origin"
                self.session.headers["Sec-Fetch-Mode"] = "cors"
                self.session.headers["Sec-Fetch-Dest"] = "empty"
                self.session.headers["Accept"] = "application/json, text/plain, */*"

                self._connected = True
                self._last_cookie_time = now

                cookies = list(self.session.cookies.keys())
                print(f"  [NSE] Connected — got {len(cookies)} cookies")
                return True

            print(f"  [NSE] Homepage returned {r.status_code}")
            return False

        except requests.exceptions.ConnectionError:
            print("  [NSE] Connection error — check internet")
            return False
        except Exception as e:
            print(f"  [NSE] Cookie refresh failed: {e}")
            return False

    def _get(self, url, retries=3):
        for attempt in range(retries):
            self._refresh_cookies()
            try:
                time.sleep(0.5)  # be polite to NSE
                r = self.session.get(url, timeout=15)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code in (401, 403):
                    print(
                        f"  [NSE] {r.status_code} — refreshing cookies (attempt {attempt + 1})"
                    )
                    self._last_cookie_time = 0
                    self._connected = False
                    time.sleep(2)
                else:
                    print(f"  [NSE] {r.status_code} on {url.split('?')[0]}")
            except requests.exceptions.JSONDecodeError:
                print(f"  [NSE] Non-JSON response (attempt {attempt + 1})")
                self._last_cookie_time = 0
                self._connected = False
                time.sleep(2)
            except Exception as e:
                print(f"  [NSE] Request error: {e}")
                time.sleep(2)
        return None

    def get_candles(self, symbol="NIFTY", interval="1m", count=200):
        index_name = INDEX_ID_MAP.get(symbol.upper(), "NIFTY 50")
        encoded = index_name.replace(" ", "%20")
        url = f"{NSE_BASE}/api/chart-databyindex?index={encoded}&indices=true"

        data = self._get(url)
        if not data or "gpiData" not in data:
            print("  [NSE] Chart API failed, trying quote fallback...")
            return self._candles_from_quote(symbol, count)

        gpi = data["gpiData"]
        ticks = []

        for point in gpi:
            try:
                ts_ms = int(point[0])
                price = float(point[1])
                ts = datetime.fromtimestamp(ts_ms / 1000)
                ticks.append((ts, price))
            except (IndexError, ValueError, TypeError):
                continue

        if not ticks:
            return self._candles_from_quote(symbol, count)

        # Group ticks into 1-min OHLC bars
        bars = {}
        for ts, price in ticks:
            key = ts.replace(second=0, microsecond=0)
            if key not in bars:
                bars[key] = []
            bars[key].append(price)

        candles = []
        for ts in sorted(bars.keys()):
            prices = bars[ts]
            candles.append(
                Candle(
                    timestamp=ts,
                    open=prices[0],
                    high=max(prices),
                    low=min(prices),
                    close=prices[-1],
                    volume=0.0,
                    symbol=symbol,
                )
            )

        candles = candles[-count:]

        if candles:
            print(
                f"  [NSE] {symbol}: {len(candles)} candles, "
                f"latest={candles[-1].timestamp.strftime('%H:%M:%S')} "
                f"@ {candles[-1].close:,.2f}"
            )
        return candles

    def _candles_from_quote(self, symbol, count):
        """Fallback: build history from repeated LTP snapshots."""
        price = self.get_ltp(symbol)
        if price <= 0:
            print(f"  [NSE] Could not get LTP for {symbol}")
            return []

        print(f"  [NSE] Using LTP snapshot: {price:,.2f}")
        now = datetime.now()
        # Return a single candle — will accumulate over polls
        return [
            Candle(
                timestamp=now,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0.0,
                symbol=symbol,
            )
        ]

    def get_ltp(self, symbol="NIFTY"):
        index_name = INDEX_SYMBOL_MAP.get(symbol.upper(), "NIFTY 50")
        encoded = index_name.replace(" ", "%20")
        url = f"{NSE_BASE}/api/equity-stockIndices?index={encoded}"

        data = self._get(url)
        if not data or "data" not in data:
            return 0.0

        for item in data["data"]:
            if item.get("index") == index_name or item.get("symbol") == index_name:
                return float(item.get("lastPrice", item.get("last", 0)))

        if data["data"]:
            return float(data["data"][0].get("lastPrice", 0))

        return 0.0

    def get_option_chain(self, symbol="NIFTY"):
        url = f"{NSE_BASE}/api/option-chain-indices?symbol={symbol.upper()}"
        data = self._get(url)

        if not data or "records" not in data or "data" not in data["records"]:
            return {}

        chain = {}
        for rec in data["records"]["data"]:
            strike = float(rec["strikePrice"])
            entry = {
                "ce_oi": 0,
                "pe_oi": 0,
                "ce_iv": 0.0,
                "pe_iv": 0.0,
                "ce_ltp": 0.0,
                "pe_ltp": 0.0,
            }

            if "CE" in rec:
                ce = rec["CE"]
                entry["ce_oi"] = int(ce.get("openInterest", 0))
                entry["ce_iv"] = float(ce.get("impliedVolatility", 0))
                entry["ce_ltp"] = float(ce.get("lastPrice", 0))

            if "PE" in rec:
                pe = rec["PE"]
                entry["pe_oi"] = int(pe.get("openInterest", 0))
                entry["pe_iv"] = float(pe.get("impliedVolatility", 0))
                entry["pe_ltp"] = float(pe.get("lastPrice", 0))

            chain[strike] = entry

        if chain:
            print(f"  [NSE] Option chain: {len(chain)} strikes with real OI")

        return chain

    def get_oi_arrays(self, symbol="NIFTY"):
        chain = self.get_option_chain(symbol)
        if not chain:
            return None, None, None

        sorted_strikes = sorted(chain.keys())
        strikes = np.array(sorted_strikes, dtype=np.float64)
        call_oi = np.array(
            [chain[s]["ce_oi"] for s in sorted_strikes], dtype=np.float64
        )
        put_oi = np.array([chain[s]["pe_oi"] for s in sorted_strikes], dtype=np.float64)

        return strikes, call_oi, put_oi

    def disconnect(self):
        self._connected = False
        self.session.close()
