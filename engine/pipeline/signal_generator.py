"""
Kairos Engine — Signal Generator V6 (FINAL)

Problem diagnosed: filters kept killing valid signals.
NIFTY dropped 362 pts on Jul 8 and engine said nothing.

Root cause: regime_age filter + high ER threshold + long cooldown
= by the time all filters pass, the move is over.

Fix: SCORE-BASED approach instead of chain of binary kill-switches.
Each factor CONTRIBUTES to a score. High enough total = signal.
No single factor can kill a signal alone (except opening range).

Score components (0-100):
  - Regime quality (0-25): TREND_EXPANSION=25, EXHAUSTION=15, COMPRESSION=10
  - Efficiency ratio (0-20): scaled by ER value
  - Thesis separation (0-20): scaled by separation
  - Momentum (0-15): confirmed candles in direction
  - Time quality (0-10): 1-3 PM bonus
  - Option efficiency (0-10): contract is mathematically sound

Signal threshold: 50/100 = fire
This means a TREND_EXPANSION(25) + good ER(15) + decent separation(12)
= 52 → signal, even without perfect momentum or prime time.
"""

import numpy as np
from datetime import datetime

from engine.pipeline.market_pipeline import MarketPipeline
from engine.options.strike_selector import StrikeSelector, TradeSignal
from engine.core.enums import EntryWindow, MarketBias, MarketRegime
from engine.core.types import FloatArray
from engine.features.volatility import atr
from data.models.market_state import MarketState


ATR_MULTIPLIER = {
    "TREND_EXPANSION": 2.0,
    "COMPRESSION": 3.0,
    "TREND_EXHAUSTION": 1.5,
    "MEAN_REVERSION": 1.0,
    "CHAOTIC": 1.5,
    "UNKNOWN": 1.5,
}


class SignalGenerator:
    def __init__(self, strike_step: float = 50.0):
        self.pipeline = MarketPipeline()
        self.selector = StrikeSelector(strike_step=strike_step, min_survival=5.0)
        self._last_signal_time: datetime | None = None
        self._rejection_reason: str = ""
        self._signal_count_today: int = 0
        self._current_date: str = ""
        self._daily_pnl: float = 0.0

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
        now = timestamp or datetime.now()

        # Reset daily counter
        today = now.strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._signal_count_today = 0
            self._daily_pnl = 0.0
            self._last_signal_time = None

        if state.entry_window == EntryWindow.OPEN:
            signal = self._score_and_decide(
                state,
                closes,
                highs,
                lows,
                now,
                dte_minutes,
                symbol,
            )

        return state, signal

    def _score_and_decide(
        self,
        state,
        closes,
        highs,
        lows,
        now,
        dte_minutes,
        symbol,
    ) -> TradeSignal | None:

        # === HARD BLOCKS (only things that absolutely must block) ===

        # Opening range: first 30 min is chaos
        market_open = now.replace(hour=9, minute=15, second=0)
        mins_open = (now - market_open).total_seconds() / 60
        if 0 < mins_open < 30:
            self._rejection_reason = f"Opening range ({mins_open:.0f}m)"
            return None

        # Cooldown: 20 min between signals
        if self._last_signal_time:
            elapsed = (now - self._last_signal_time).total_seconds() / 60
            if elapsed < 20:
                self._rejection_reason = f"Cooldown ({elapsed:.0f}m)"
                return None

        # Max 3 signals per day
        if self._signal_count_today >= 3:
            self._rejection_reason = "Max 3 signals today"
            return None

        # Daily loss limit
        if self._daily_pnl <= -1200:
            self._rejection_reason = "Daily loss limit ₹1200"
            return None

        # Must have a direction
        bias = state.thesis.primary_bias
        if bias == MarketBias.NEUTRAL:
            self._rejection_reason = "Neutral bias"
            return None

        # === SCORE-BASED EVALUATION ===
        score = 0.0
        score_details = []

        # 1. Regime quality (0-25)
        regime_scores = {
            MarketRegime.TREND_EXPANSION: 25,
            MarketRegime.TREND_EXHAUSTION: 18,
            MarketRegime.COMPRESSION: 12,
            MarketRegime.MEAN_REVERSION: 5,
            MarketRegime.CHAOTIC: 3,
            MarketRegime.UNKNOWN: 0,
        }
        regime_pts = regime_scores.get(state.regime, 0)
        score += regime_pts
        score_details.append(f"regime={regime_pts}")

        # 2. Efficiency ratio (0-20)
        er = state.regime_metrics.efficiency_ratio
        er_pts = min(20, er * 25)  # ER=0.80 → 20 pts, ER=0.40 → 10 pts
        score += er_pts
        score_details.append(f"ER={er_pts:.0f}")

        # 3. Thesis separation (0-20)
        sep = state.thesis.separation
        sep_pts = min(20, sep * 60)  # sep=0.33 → 20 pts, sep=0.15 → 9 pts
        score += sep_pts
        score_details.append(f"sep={sep_pts:.0f}")

        # 4. Momentum confirmation (0-15)
        mom_pts = 0
        if len(closes) >= 3:
            recent = np.diff(closes[-3:])
            if bias == MarketBias.BULLISH and all(m > 0 for m in recent):
                mom_pts = 15
            elif bias == MarketBias.BEARISH and all(m < 0 for m in recent):
                mom_pts = 15
            elif bias == MarketBias.BULLISH and recent[-1] > 0:
                mom_pts = 8  # at least last candle confirms
            elif bias == MarketBias.BEARISH and recent[-1] < 0:
                mom_pts = 8
        score += mom_pts
        score_details.append(f"mom={mom_pts}")

        # 5. Time quality (0-10)
        hour = now.hour
        time_pts = 0
        if 13 <= hour < 15:  # prime window 1-3 PM
            time_pts = 10
        elif 10 <= hour < 13:  # decent window
            time_pts = 6
        elif hour == 9 and now.minute >= 45:  # after opening range
            time_pts = 4
        elif hour == 15:  # closing
            time_pts = 3
        score += time_pts
        score_details.append(f"time={time_pts}")

        # 6. Option efficiency (0-10)
        opt_pts = 10 if state.option.is_efficient else 3
        score += opt_pts
        score_details.append(f"opt={opt_pts}")

        # === THRESHOLD CHECK ===
        threshold = 50
        self._rejection_reason = (
            f"Score {score:.0f}/{threshold} ({', '.join(score_details)})"
        )

        if score < threshold:
            return None

        # === PASSED — Generate strike ===
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

        self._last_signal_time = now
        self._signal_count_today += 1
        self._rejection_reason = ""
        return signal


def format_signal(signal: TradeSignal) -> str:
    if signal is None:
        return ""

    GRN = "\033[92m"
    RED = "\033[91m"
    YEL = "\033[93m"
    B = "\033[1m"
    R = "\033[0m"

    side_color = GRN if signal.option_type == "CE" else RED
    conf_color = GRN if signal.confidence > 0.6 else YEL

    lines = [
        "",
        f"  {B}{side_color}╔══════════════════════════════════════════════════════════╗{R}",
        f"  {B}{side_color}║  ▶ {signal.action}  {signal.symbol} {signal.strike:.0f} {signal.option_type}"
        f"{'':>{44 - len(f'{signal.strike:.0f}')}}║{R}",
        f"  {B}{side_color}╠══════════════════════════════════════════════════════════╣{R}",
        f"  {B}{side_color}║{R}  Spot: {signal.spot:,.2f}  │  Premium: {B}₹{signal.estimated_premium:.2f}{R}"
        f"{'':>{20 - len(f'{signal.estimated_premium:.2f}')}}{side_color}║{R}",
        f"  {B}{side_color}║{R}  {RED}SL: ₹{signal.stoploss_premium:.2f}{R}"
        f"  │  {GRN}Target: ₹{signal.target_premium:.2f}{R}"
        f"  │  R:R 1:{signal.risk_reward:.1f}"
        f"{'':>{6 - len(f'{signal.risk_reward:.1f}')}}{side_color}║{R}",
        f"  {B}{side_color}║{R}  Survival: {signal.survival_minutes:.0f}m"
        f"  │  Conf: {conf_color}{signal.confidence:.0%}{R}"
        f"{'':>{24 - len(f'{signal.confidence:.0%}')}}{side_color}║{R}",
        f"  {B}{side_color}╚══════════════════════════════════════════════════════════╝{R}",
    ]

    for c in signal.all_candidates[:3]:
        flag = " ◀" if c.strike == signal.strike else ""
        lines.append(
            f"    {c.strike:.0f} {c.option_type} {c.moneyness:<5s}"
            f"  ₹{c.estimated_premium:>7.2f}"
            f"  feas={c.move_feasibility:.2f}"
            f"  score={c.score:.3f}{flag}"
        )
    lines.append("")
    return "\n".join(lines)
