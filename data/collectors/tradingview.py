import os
import json
import requests
from datetime import datetime

from engine.pipeline.signal_generator import TradeSignal
from data.models.market_state import MarketState
from data.collectors.assets import AssetConfig, get_asset


class TradingViewAlert:
    def __init__(
        self,
        webhook_url: str | None = None,
        log_dir: str = "data/storage/tv_signals",
    ):
        self.webhook_url = webhook_url or os.getenv("TV_WEBHOOK_URL", "")
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.enabled = bool(self.webhook_url)
        if self.enabled:
            print(f"  [TradingView] Webhook: {self.webhook_url[:40]}...")
        else:
            print("  [TradingView] No webhook URL. Signals logged to files only.")
            print("  [TradingView] Set TV_WEBHOOK_URL for live alerts.")

    def send_signal(
        self,
        signal: TradeSignal,
        state: MarketState,
        asset: AssetConfig | None = None,
    ) -> bool:
        if asset is None:
            asset = get_asset(signal.symbol)

        # Build TradingView-compatible alert payload
        payload = {
            "timestamp": datetime.now().isoformat(),
            "ticker": asset.tv_ticker,
            "exchange": asset.exchange,
            "action": "sell" if signal.option_type == "PE" else "buy",
            "contracts": 1,
            "price": signal.spot,
            "strike": signal.strike,
            "option_type": signal.option_type,
            "premium": signal.estimated_premium,
            "stoploss": signal.stoploss_premium,
            "target": signal.target_premium,
            "risk_reward": signal.risk_reward,
            "confidence": signal.confidence,
            "survival_minutes": signal.survival_minutes,
            "regime": state.regime.value,
            "bias": state.thesis.primary_bias.value,
            "thesis_separation": state.thesis.separation,
            "reason": signal.reason,
            # TradingView strategy fields
            "strategy": {
                "position_size": asset.lot_size,
                "order_action": "buy",
                "order_contracts": 1,
                "order_price": signal.estimated_premium,
                "order_id": f"kairos_{datetime.now().strftime('%H%M%S')}",
            },
            # For display
            "message": (
                f"KAIROS: {signal.action} {signal.symbol} "
                f"{signal.strike:.0f} {signal.option_type} "
                f"@ {asset.currency}{signal.estimated_premium:.2f} "
                f"| SL:{signal.stoploss_premium:.2f} "
                f"T:{signal.target_premium:.2f} "
                f"| {signal.confidence:.0%} conf"
            ),
        }

        # Always log to file
        self._log_signal(payload)

        # Send webhook if configured
        if self.enabled:
            return self._send_webhook(payload)

        return True

    def _send_webhook(self, payload: dict) -> bool:
        try:
            r = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            if r.status_code == 200:
                print("  [TradingView] Alert sent ✓")
                return True
            else:
                print(f"  [TradingView] Webhook {r.status_code}")
                return False
        except Exception as e:
            print(f"  [TradingView] Send failed: {e}")
            return False

    def _log_signal(self, payload: dict):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"signal_{ts}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def generate_pine_script(self, symbol: str = "NIFTY") -> str:
        """
        Generate a Pine Script indicator that displays Kairos signals
        on a TradingView chart. Copy-paste into TV Pine Editor.
        """
        asset = get_asset(symbol)

        script = f"""
//@version=5
indicator("Kairos Engine Signals", overlay=true)

// ============================================
// KAIROS ENGINE — TradingView Signal Display
// ============================================
// This indicator reads signals from Kairos Engine
// and displays them as buy/sell markers on your chart.
//
// HOW TO USE:
// 1. Run Kairos: python live.py --broker yfinance --symbol {symbol}
// 2. When a signal fires, manually add an alert on TV
//    at the signal price with the signal details.
// 3. Or use this indicator to mark levels visually.
//
// ASSET: {asset.tv_ticker}
// STRIKE STEP: {asset.strike_step}
// ============================================

// Inputs — update these when Kairos fires a signal
signal_active = input.bool(false, "Signal Active?")
signal_price = input.float(0, "Signal Spot Price")
signal_strike = input.float(0, "Strike Price")
signal_sl = input.float(0, "Stoploss Premium")
signal_target = input.float(0, "Target Premium")
signal_type = input.string("CE", "Option Type", options=["CE", "PE"])
signal_time = input.int(0, "Signal Bar Index")

// Plot signal levels
plot_color = signal_type == "CE" ? color.green : color.red

// Entry level
hline_price = signal_active ? signal_price : na
plot(hline_price, "Entry Spot", color=plot_color, linewidth=2, style=plot.style_circles)

// Strike reference
hline_strike = signal_active ? signal_strike : na
plot(hline_strike, "Strike", color=color.blue, linewidth=1, style=plot.style_cross)

// Buy/Sell markers
plotshape(signal_active and bar_index == signal_time,
    style=signal_type == "CE" ? shape.triangleup : shape.triangledown,
    location=location.belowbar,
    color=plot_color,
    size=size.large,
    text=signal_type == "CE" ? "BUY CE" : "BUY PE")

// Background on signal
bgcolor(signal_active and bar_index >= signal_time and bar_index <= signal_time + 15 ?
    color.new(plot_color, 90) : na)

// Info table
var table info = table.new(position.top_right, 2, 6)
if signal_active
    table.cell(info, 0, 0, "Kairos Signal", bgcolor=plot_color, text_color=color.white)
    table.cell(info, 1, 0, signal_type + " " + str.tostring(signal_strike), bgcolor=plot_color, text_color=color.white)
    table.cell(info, 0, 1, "Spot")
    table.cell(info, 1, 1, str.tostring(signal_price, "#.##"))
    table.cell(info, 0, 2, "SL Premium")
    table.cell(info, 1, 2, str.tostring(signal_sl, "#.##"))
    table.cell(info, 0, 3, "Target Premium")
    table.cell(info, 1, 3, str.tostring(signal_target, "#.##"))
"""
        return script


class TVSignalLogger:
    """
    Logs all signals to a single JSON file that can be
    read by external tools, bots, or TradingView scripts.
    """

    def __init__(self, path: str = "data/storage/tv_signals/latest.json"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.signals: list[dict] = []

    def add(self, signal: TradeSignal, state: MarketState):
        asset = get_asset(signal.symbol)
        self.signals.append(
            {
                "time": datetime.now().isoformat(),
                "symbol": signal.symbol,
                "tv_ticker": asset.tv_ticker,
                "type": signal.option_type,
                "strike": signal.strike,
                "spot": signal.spot,
                "premium": signal.estimated_premium,
                "sl": signal.stoploss_premium,
                "target": signal.target_premium,
                "rr": signal.risk_reward,
                "confidence": signal.confidence,
                "regime": state.regime.value,
                "bias": state.thesis.primary_bias.value,
                "separation": state.thesis.separation,
            }
        )
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump({"signals": self.signals}, f, indent=2, default=str)

    def get_latest(self) -> dict | None:
        if self.signals:
            return self.signals[-1]
        return None
