"""
Kairos Engine — Trade Signal Generator

Sits on top of the pipeline. When ENTRY_WINDOW_OPEN fires,
this picks the optimal strike and generates an actionable signal.

Output example:
    ▶ BUY  NIFTY 24500 CE  @ ₹148
      SL: ₹104  │  Target: ₹236  │  R:R 1:2.0
      Survival: 14 min  │  Confidence: 0.72
"""

import numpy as np
from datetime import datetime

from engine.pipeline.market_pipeline import MarketPipeline
from engine.options.strike_selector import StrikeSelector, TradeSignal
from engine.core.enums import EntryWindow
from engine.core.types import FloatArray
from data.models.market_state import MarketState


class SignalGenerator:
    def __init__(self, strike_step: float = 50.0):
        self.pipeline = MarketPipeline()
        self.selector = StrikeSelector(strike_step=strike_step)
        self._last_signal_time: datetime | None = None
        self._cooldown_seconds: int = 120  # no repeat signal within 2 min

    @property
    def min_candles(self) -> int:
        return self.pipeline.min_candles

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
        """
        Run full pipeline and generate trade signal if entry window opens.
        Returns (MarketState, TradeSignal or None).
        """
        state = self.pipeline.process(
            closes,
            highs,
            lows,
            opens,
            volumes,
            iv_series,
            symbol,
            timestamp,
        )

        signal = None

        if state.entry_window == EntryWindow.OPEN:
            # Cooldown check
            now = timestamp or datetime.now()
            if self._last_signal_time:
                elapsed = (now - self._last_signal_time).total_seconds()
                if elapsed < self._cooldown_seconds:
                    return state, None

            # Calculate expected move from recent price action
            if len(closes) > 20:
                recent_std = float(np.std(np.diff(closes[-20:])))
                expected_move = recent_std * 3.0  # 3-sigma move
            else:
                expected_move = 30.0

            # Get IV
            iv = state.iv.current_iv if state.iv.current_iv > 0 else 0.15

            signal = self.selector.select(
                spot=state.last_price,
                bias=state.thesis.primary_bias,
                expected_move=expected_move,
                iv=iv,
                dte_minutes=dte_minutes,
                regime_confidence=state.regime_metrics.regime_confidence,
                thesis_separation=state.thesis.separation,
                symbol=symbol,
            )

            if signal:
                self._last_signal_time = now

        return state, signal


def format_signal(signal: TradeSignal) -> str:
    """Format a trade signal for terminal display."""
    if signal is None:
        return ""

    GRN = "\033[92m"
    RED = "\033[91m"
    YEL = "\033[93m"
    B = "\033[1m"
    R = "\033[0m"
    MAG = "\033[95m"

    side_color = GRN if signal.option_type == "CE" else RED
    conf_color = (
        GRN if signal.confidence > 0.6 else (YEL if signal.confidence > 0.4 else RED)
    )

    lines = []
    lines.append("")
    lines.append(
        f"  {B}{side_color}╔══════════════════════════════════════════════════════════════╗{R}"
    )
    lines.append(
        f"  {B}{side_color}║  ▶ {signal.action}  {signal.symbol} {signal.strike:.0f} {signal.option_type:<4s}"
        f"                                     ║{R}"
    )
    lines.append(
        f"  {B}{side_color}╠══════════════════════════════════════════════════════════════╣{R}"
    )
    lines.append(
        f"  {B}{side_color}║{R}"
        f"  Spot: {signal.spot:,.2f}"
        f"  │  Premium: {B}₹{signal.estimated_premium:.2f}{R}"
        f"{'':>{24 - len(f'{signal.estimated_premium:.2f}')}}"
        f"{side_color}║{R}"
    )
    lines.append(
        f"  {B}{side_color}║{R}"
        f"  {RED}SL: ₹{signal.stoploss_premium:.2f}{R}"
        f"  │  {GRN}Target: ₹{signal.target_premium:.2f}{R}"
        f"  │  R:R 1:{signal.risk_reward:.1f}"
        f"{'':>{10 - len(f'{signal.risk_reward:.1f}')}}"
        f"{side_color}║{R}"
    )
    lines.append(
        f"  {B}{side_color}║{R}"
        f"  Survival: {signal.survival_minutes:.0f} min"
        f"  │  Confidence: {conf_color}{signal.confidence:.0%}{R}"
        f"{'':>{22 - len(f'{signal.confidence:.0%}')}}"
        f"{side_color}║{R}"
    )
    lines.append(
        f"  {B}{side_color}║{R}"
        f"  {signal.reason}"
        f"{'':>{60 - len(signal.reason)}}"
        f"{side_color}║{R}"
    )
    lines.append(
        f"  {B}{side_color}╚══════════════════════════════════════════════════════════════╝{R}"
    )

    # Show top 3 candidates
    lines.append(f"  {MAG}Alternative strikes:{R}")
    for c in signal.all_candidates[:3]:
        flag = " ◀ SELECTED" if c.strike == signal.strike else ""
        lines.append(
            f"    {c.strike:.0f} {c.option_type} {c.moneyness:<5s}"
            f"  ₹{c.estimated_premium:>7.2f}"
            f"  δ={c.estimated_delta:.3f}"
            f"  feas={c.move_feasibility:.2f}"
            f"  γ/θ={c.gamma_theta_ratio:.1f}"
            f"  surv={c.theta_survival_minutes:.0f}m"
            f"  score={c.score:.3f}"
            f"{flag}"
        )

    lines.append("")
    return "\n".join(lines)
