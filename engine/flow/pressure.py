"""
Kairos Engine — Pressure / Order Flow Analyzer

Without real tick-by-tick order flow, we estimate pressure from
price action patterns that IMPLY order flow:

1. Level Tests: price repeatedly hitting a level = orders absorbing there
2. Pullback Shrinkage: each pullback from the level is smaller = buyers
   are more aggressive, willing to pay higher each time
3. Test Frequency: tests getting closer together = urgency increasing
4. Aggression Ratio: ratio of bullish vs bearish candle bodies near the level
5. Liquidity Depletion: range narrowing near the level = orders being
   consumed, breakout imminent

These are ESTIMATES. Real order flow (L2, trade tape) would be better.
"""

import numpy as np
from engine.core.types import FloatArray
from engine.flow.models import FlowResult


class PressureAnalyzer:
    def __init__(
        self,
        test_threshold_pct: float = 0.001,  # within 0.1% of level = a test
        lookback: int = 30,
    ):
        self.test_threshold_pct = test_threshold_pct
        self.lookback = lookback

    def analyze(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        opens: FloatArray,
        target_level: float,
    ) -> FlowResult:
        n = len(closes)
        if n < self.lookback or target_level <= 0:
            return self._default()

        window_c = closes[-self.lookback :]
        window_h = highs[-self.lookback :]
        window_l = lows[-self.lookback :]
        window_o = opens[-self.lookback :]

        threshold = target_level * self.test_threshold_pct

        # --- Level tests ---
        tests = []
        for i in range(len(window_c)):
            if (
                abs(window_h[i] - target_level) <= threshold
                or abs(window_l[i] - target_level) <= threshold
            ):
                tests.append(i)
        n_tests = len(tests)
        test_freq = n_tests / self.lookback

        # --- Pullback shrinkage ---
        pullback_shrinking = False
        if n_tests >= 3:
            pullbacks = []
            for t in tests:
                dist = abs(window_c[t] - target_level)
                pullbacks.append(dist)
            if len(pullbacks) >= 3:
                first_half = np.mean(pullbacks[: len(pullbacks) // 2])
                second_half = np.mean(pullbacks[len(pullbacks) // 2 :])
                pullback_shrinking = second_half < first_half * 0.8

        # --- Aggression ratio ---
        bull_body = 0.0
        bear_body = 0.0
        for i in range(len(window_c)):
            body = window_c[i] - window_o[i]
            if body > 0:
                bull_body += body
            else:
                bear_body += abs(body)
        total_body = bull_body + bear_body
        aggression = 0.5
        if total_body > 0:
            aggression = bull_body / total_body

        # --- Pressure asymmetry ---
        above_tests = sum(1 for t in tests if window_c[t] > target_level)
        below_tests = n_tests - above_tests
        asymmetry = 0.0
        if n_tests > 0:
            asymmetry = abs(above_tests - below_tests) / n_tests

        # --- Liquidity thinning ---
        if len(window_c) >= 10:
            recent_range = float(np.mean(window_h[-5:] - window_l[-5:]))
            older_range = float(np.mean(window_h[:5] - window_l[:5]))
            liq_thin = recent_range < older_range * 0.6
        else:
            liq_thin = False

        # --- Composite score ---
        test_score = min(n_tests / 5.0, 1.0)
        freq_score = min(test_freq / 0.3, 1.0)
        shrink_score = 1.0 if pullback_shrinking else 0.0
        asym_score = asymmetry
        liq_score = 1.0 if liq_thin else 0.0

        pressure = (
            0.25 * test_score
            + 0.20 * freq_score
            + 0.20 * shrink_score
            + 0.15 * asym_score
            + 0.20 * liq_score
        )

        return FlowResult(
            pressure_score=round(min(pressure, 1.0), 4),
            level_tests=n_tests,
            test_frequency=round(test_freq, 4),
            pullback_shrinking=pullback_shrinking,
            aggression_ratio=round(aggression, 4),
            pressure_asymmetry=round(asymmetry, 4),
            liquidity_thinning=liq_thin,
        )

    def _default(self) -> FlowResult:
        return FlowResult(
            pressure_score=0.0,
            level_tests=0,
            test_frequency=0.0,
            pullback_shrinking=False,
            aggression_ratio=0.5,
            pressure_asymmetry=0.0,
            liquidity_thinning=False,
        )
