"""
Kairos Engine — Signal Generator V3

All 9 filters applied AFTER state machine (covers both Path A and B):
1. Opening range exclusion (45 min)
2. 10-minute cooldown
3. Regime whitelist
4. MTF alignment
5. Thesis separation >= 0.20
6. IV overexpanded block
7. Momentum confirmation (2 candles)
8. Move feasibility >= 1.0
9. Confidence >= 60%
"""

import numpy as np
from datetime import datetime

from engine.pipeline.market_pipeline import MarketPipeline
from engine.options.strike_selector import StrikeSelector, TradeSignal
from engine.core.enums import EntryWindow, MarketBias, IVState
from engine.core.config import TradeFilterConfig, RiskConfig
from engine.core.types import FloatArray
from engine.features.multi_timeframe import MultiTimeframe
from engine.features.volatility import atr
from data.models.market_state import MarketState


# Fix 4: ATR multiplier by regime
ATR_MULTIPLIER = {
    "TREND_EXPANSION": 2.0,
    "COMPRESSION": 3.0,
    "TREND_EXHAUSTION": 1.5,
    "MEAN_REVERSION": 1.0,
    "CHAOTIC": 1.5,
    "UNKNOWN": 1.5,
}


class SignalGenerator:
    def __init__(
        self,
        strike_step: float = 50.0,
        filters: TradeFilterConfig | None = None,
        risk: RiskConfig | None = None,
    ):
        self.pipeline = MarketPipeline()
        self.selector = StrikeSelector(strike_step=strike_step)
        self.filters = filters or TradeFilterConfig()
        self.risk = risk or RiskConfig()
        self.mtf = MultiTimeframe(resample_factor=self.filters.mtf_resample_factor)
        self._last_signal_time: datetime | None = None
        self._rejection_reason: str = ""
        self._open_positions: int = 0

    @property
    def min_candles(self) -> int:
        return self.pipeline.min_candles

    @property
    def last_rejection(self) -> str:
        return self._rejection_reason

    def process(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        opens: FloatArray,
        volumes: FloatArray,
        iv_series: FloatArray | None = None,
        symbol: str = "NIFTY",
        timestamp: datetime | None = None,
        dte_minutes: float = 375.0,
    ) -> tuple[MarketState, TradeSignal | None]:

        state = self.pipeline.process(
            closes, highs, lows, opens, volumes,
            iv_series, symbol, timestamp,
        )

        signal = None
        self._rejection_reason = ""

        if state.entry_window == EntryWindow.OPEN:
            signal = self._apply_filters(
                state, closes, highs, lows,
                timestamp or datetime.now(), dte_minutes, symbol,
            )

        return state, signal

    def _apply_filters(
        self, state, closes, highs, lows, now, dte_minutes, symbol,
    ) -> TradeSignal | None:
        f = self.filters

        # --- F1: Opening range (45 min) ---
        market_open = now.replace(hour=f.market_open_hour, minute=f.market_open_minute, second=0)
        mins_open = (now - market_open).total_seconds() / 60
        if 0 < mins_open < f.opening_range_skip_minutes:
            self._rejection_reason = f"Opening range ({mins_open:.0f}m < {f.opening_range_skip_minutes}m)"
            return None

        # --- F2: Cooldown ---
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds() / 60
            if elapsed < f.signal_cooldown_minutes:
                self._rejection_reason = f"Cooldown ({elapsed:.1f}m < {f.signal_cooldown_minutes}m)"
                return None

        # --- F3: Regime whitelist ---
        if state.regime not in f.allowed_regimes:
            self._rejection_reason = f"Regime {state.regime.value} not allowed"
            return None

        # --- F4: MTF alignment ---
        if f.require_mtf_alignment:
            mtf = self.mtf.analyze(closes, highs, lows, state.regime, state.thesis.primary_bias)
            if not mtf.aligned:
                self._rejection_reason = (
                    f"MTF: {mtf.primary_bias.value} vs HTF {mtf.higher_bias.value}"
                )
                return None

        # --- F5: Thesis separation ---
        if state.thesis.separation < f.min_thesis_separation:
            self._rejection_reason = f"Separation {state.thesis.separation:.3f} < {f.min_thesis_separation}"
            return None

        # --- F6: IV block ---
        if f.block_overexpanded_iv and state.iv.state == IVState.OVEREXPANDED:
            self._rejection_reason = "IV OVEREXPANDED"
            return None

        # --- F7: Momentum confirmation ---
        bias = state.thesis.primary_bias
        if bias == MarketBias.NEUTRAL:
            self._rejection_reason = "Neutral bias"
            return None

        nc = f.confirmation_candles
        if len(closes) < nc + 1:
            self._rejection_reason = "Not enough candles"
            return None

        recent = np.diff(closes[-(nc + 1):])
        if bias == MarketBias.BULLISH:
            confirmed = all(m > 0 for m in recent)
        else:
            confirmed = all(m < 0 for m in recent)

        if not confirmed:
            self._rejection_reason = f"Momentum not confirmed ({nc} candles)"
            return None

        # --- F10: Max positions ---
        if self._open_positions >= self.risk.max_positions:
            self._rejection_reason = f"Max positions ({self.risk.max_positions})"
            return None

        # --- Generate strike ---
        # Fix 4: ATR-based expected move
        true_ranges = atr(highs, lows, closes)
        if len(true_ranges) >= 14:
            current_atr = float(np.mean(true_ranges[-14:]))
            multiplier = ATR_MULTIPLIER.get(state.regime.value, 1.5)
            expected_move = current_atr * multiplier
        else:
            expected_move = 30.0

        iv_val = state.iv.current_iv if state.iv.current_iv > 0 else 0.15

        signal = self.selector.select(
            spot=state.last_price,
            bias=bias,
            expected_move=expected_move,
            iv=iv_val,
            dte_minutes=dte_minutes,
            regime_confidence=state.regime_metrics.regime_confidence,
            thesis_separation=state.thesis.separation,
            symbol=symbol,
        )

        if not signal:
            self._rejection_reason = "No suitable strike"
            return None

        # --- F8: Feasibility floor ---
        if signal.all_candidates:
            if signal.all_candidates[0].move_feasibility < f.min_move_feasibility:
                self._rejection_reason = (
                    f"Feasibility {signal.all_candidates[0].move_feasibility:.2f} < {f.min_move_feasibility}"
                )
                return None

        # --- F9: Confidence ---
        if signal.confidence < f.min_signal_confidence:
            self._rejection_reason = f"Confidence {signal.confidence:.0%} < {f.min_signal_confidence:.0%}"
            return None

        # --- Fix 6: Position sizing ---
        max_premium = self.risk.account_size * self.risk.max_premium_pct
        if signal.estimated_premium > max_premium:
            self._rejection_reason = f"Premium ₹{signal.estimated_premium:.0f} > max ₹{max_premium:.0f}"
            return None

        self._last_signal_time = now
        return signal


def format_signal(signal: TradeSignal) -> str:
    if signal is None:
        return ""

    GRN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"
    B = "\033[1m"; R = "\033[0m"; MAG = "\033[95m"

    side_color = GRN if signal.option_type == "CE" else RED
    conf_color = GRN if signal.confidence > 0.6 else (YEL if signal.confidence > 0.4 else RED)

    lines = [
        "",
        f"  {B}{side_color}╔══════════════════════════════════════════════════════════════╗{R}",
        f"  {B}{side_color}║  ▶ {signal.action}  {signal.symbol} {signal.strike:.0f} {signal.option_type:<4s}"
        f"                                     ║{R}",
        f"  {B}{side_color}╠══════════════════════════════════════════════════════════════╣{R}",
        f"  {B}{side_color}║{R}"
        f"  Spot: {signal.spot:,.2f}"
        f"  │  Premium: {B}₹{signal.estimated_premium:.2f}{R}"
        f"{'':>{24 - len(f'{signal.estimated_premium:.2f}')}}"
        f"{side_color}║{R}",
        f"  {B}{side_color}║{R}"
        f"  {RED}SL: ₹{signal.stoploss_premium:.2f}{R}"
        f"  │  {GRN}Target: ₹{signal.target_premium:.2f}{R}"
        f"  │  R:R 1:{signal.risk_reward:.1f}"
        f"{'':>{10 - len(f'{signal.risk_reward:.1f}')}}"
        f"{side_color}║{R}",
        f"  {B}{side_color}║{R}"
        f"  Survival: {signal.survival_minutes:.0f} min"
        f"  │  Confidence: {conf_color}{signal.confidence:.0%}{R}"
        f"{'':>{22 - len(f'{signal.confidence:.0%}')}}"
        f"{side_color}║{R}",
        f"  {B}{side_color}║{R}"
        f"  {signal.reason}"
        f"{'':>{60 - len(signal.reason)}}"
        f"{side_color}║{R}",
        f"  {B}{side_color}╚══════════════════════════════════════════════════════════════╝{R}",
        f"  {MAG}Alternatives:{R}",
    ]

    for c in signal.all_candidates[:3]:
        flag = " ◀" if c.strike == signal.strike else ""
        lines.append(
            f"    {c.strike:.0f} {c.option_type} {c.moneyness:<5s}"
            f"  ₹{c.estimated_premium:>7.2f}"
            f"  δ={c.estimated_delta:.3f}"
            f"  feas={c.move_feasibility:.2f}"
            f"  γ/θ={c.gamma_theta_ratio:.1f}"
            f"  surv={c.theta_survival_minutes:.0f}m"
            f"  score={c.score:.3f}{flag}"
        )

    lines.append("")
    return "\n".join(lines)