"""
Kairos Engine — Trade Signal Generator V2

Filters applied based on backtest failures:
1. Opening range exclusion (first 30 min)
2. 10-minute cooldown between signals
3. Regime filter (only TREND_EXPANSION + COMPRESSION)
4. Momentum confirmation (2 candles in thesis direction)
5. Higher thesis separation (0.20 minimum)
6. IV overexpanded block
7. Move feasibility floor (>= 1.0)
"""

import numpy as np
from datetime import datetime

from engine.pipeline.market_pipeline import MarketPipeline
from engine.options.strike_selector import StrikeSelector, TradeSignal
from engine.core.enums import EntryWindow, MarketBias, IVState
from engine.core.config import TradeFilterConfig
from engine.features.multi_timeframe import MultiTimeframe
from engine.core.types import FloatArray
from data.models.market_state import MarketState


class SignalGenerator:
    def __init__(
        self,
        strike_step: float = 50.0,
        filters: TradeFilterConfig | None = None,
    ):
        self.pipeline = MarketPipeline()
        self.selector = StrikeSelector(strike_step=strike_step)
        self.filters = filters or TradeFilterConfig()
        self.mtf = MultiTimeframe(resample_factor=7)
        self._last_signal_time: datetime | None = None
        self._rejection_reason: str = ""

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
        self._rejection_reason = ""

        if state.entry_window == EntryWindow.OPEN:
            signal = self._try_generate_signal(
                state, closes, timestamp or datetime.now(), dte_minutes, symbol
            )

        return state, signal

    def _try_generate_signal(
        self,
        state: MarketState,
        closes: FloatArray,
        now: datetime,
        dte_minutes: float,
        symbol: str,
    ) -> TradeSignal | None:
        f = self.filters

        # --- Filter 1: Opening range ---
        market_open = now.replace(
            hour=f.market_open_hour, minute=f.market_open_minute, second=0
        )
        minutes_since_open = (now - market_open).total_seconds() / 60

        if 0 < minutes_since_open < f.opening_range_skip_minutes:
            self._rejection_reason = (
                f"Opening range: {minutes_since_open:.0f}m since open, "
                f"need {f.opening_range_skip_minutes}m"
            )
            return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = (
            self.mtf.analyze(
                closes, highs, lows, state.regime, state.thesis.primary_bias
            )
            if len(closes) > 0
            else None
        )
        if mtf_result and not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        # --- Filter 2: Signal cooldown ---
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds() / 60
            if elapsed < f.signal_cooldown_minutes:
                self._rejection_reason = (
                    f"Cooldown: {elapsed:.1f}m since last signal, "
                    f"need {f.signal_cooldown_minutes}m"
                )
                return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = (
            self.mtf.analyze(
                closes, highs, lows, state.regime, state.thesis.primary_bias
            )
            if len(closes) > 0
            else None
        )
        if mtf_result and not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        # --- Filter 3: Regime filter ---
        if state.regime not in f.allowed_regimes:
            self._rejection_reason = (
                f"Regime {state.regime.value} not in allowed: "
                f"{[r.value for r in f.allowed_regimes]}"
            )
            return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = self.mtf.analyze(
            closes, highs, lows, state.regime, state.thesis.primary_bias
        )
        if not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        # --- Filter 4: Thesis separation ---
        if state.thesis.separation < f.min_thesis_separation:
            self._rejection_reason = (
                f"Thesis separation {state.thesis.separation:.3f} < "
                f"{f.min_thesis_separation}"
            )
            return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = (
            self.mtf.analyze(
                closes, highs, lows, state.regime, state.thesis.primary_bias
            )
            if len(closes) > 0
            else None
        )
        if mtf_result and not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        # --- Filter 5: IV overexpanded block ---
        if f.block_overexpanded_iv and state.iv.state == IVState.OVEREXPANDED:
            self._rejection_reason = "IV is OVEREXPANDED — options too expensive"
            return None

        # --- Filter 6: Momentum confirmation ---
        n_confirm = f.confirmation_candles
        if len(closes) < n_confirm + 1:
            self._rejection_reason = "Not enough candles for confirmation"
            return None

        # Check last N candles moved in thesis direction
        recent_moves = np.diff(closes[-(n_confirm + 1) :])
        bias = state.thesis.primary_bias

        if bias == MarketBias.BULLISH:
            confirmed = all(m > 0 for m in recent_moves)
        elif bias == MarketBias.BEARISH:
            confirmed = all(m < 0 for m in recent_moves)
        else:
            self._rejection_reason = "Neutral bias — no direction to confirm"
            return None

        if not confirmed:
            self._rejection_reason = (
                f"Momentum not confirmed: last {n_confirm} candles "
                f"not all in {bias.value} direction"
            )
            return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = (
            self.mtf.analyze(
                closes, highs, lows, state.regime, state.thesis.primary_bias
            )
            if len(closes) > 0
            else None
        )
        if mtf_result and not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        # --- All filters passed, generate signal ---
        if len(closes) > 20:
            recent_std = float(np.std(np.diff(closes[-20:])))
            expected_move = recent_std * 3.0
        else:
            expected_move = 30.0

        iv = state.iv.current_iv if state.iv.current_iv > 0 else 0.15

        signal = self.selector.select(
            spot=state.last_price,
            bias=bias,
            expected_move=expected_move,
            iv=iv,
            dte_minutes=dte_minutes,
            regime_confidence=state.regime_metrics.regime_confidence,
            thesis_separation=state.thesis.separation,
            symbol=symbol,
        )

        # --- Filter 7: Move feasibility floor ---
        if signal and signal.all_candidates:
            best = signal.all_candidates[0]
            if best.move_feasibility < f.min_move_feasibility:
                self._rejection_reason = (
                    f"Feasibility {best.move_feasibility:.2f} < "
                    f"{f.min_move_feasibility}"
                )
                return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = (
            self.mtf.analyze(
                closes, highs, lows, state.regime, state.thesis.primary_bias
            )
            if len(closes) > 0
            else None
        )
        if mtf_result and not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        # --- Filter 8: Minimum confidence ---
        if signal and signal.confidence < f.min_signal_confidence:
            self._rejection_reason = (
                f"Confidence {signal.confidence:.0%} < {f.min_signal_confidence:.0%}"
            )
            return None
        # --- Filter 3b: Multi-timeframe alignment ---
        mtf_result = (
            self.mtf.analyze(
                closes, highs, lows, state.regime, state.thesis.primary_bias
            )
            if len(closes) > 0
            else None
        )
        if mtf_result and not mtf_result.aligned:
            self._rejection_reason = (
                f"MTF conflict: primary={mtf_result.primary_bias.value} "
                f"but higher TF={mtf_result.higher_bias.value} "
                f"({mtf_result.higher_regime.value})"
            )
            return None

        if signal:
            self._last_signal_time = now

        return signal


def format_signal(signal: TradeSignal) -> str:
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
