"""
Kairos Engine — Multi-Timeframe Confirmation V2

User watches 1m, 5m, 15m on TradingView.
We resample 2m candles to approximate 5m and 15m,
then check if all timeframes agree on direction.

For a signal to pass:
  - Primary (2m): must show the setup
  - 5m equivalent: must agree on bias (or be neutral)
  - 15m equivalent: must agree on bias (or be neutral)

If 2m says BEARISH but 15m says BULLISH → CONFLICT → no signal
"""

import numpy as np
from engine.core.types import FloatArray
from engine.core.enums import MarketBias, MarketRegime
from engine.regime.classifier import RegimeClassifier
from dataclasses import dataclass


@dataclass
class MTFResult:
    primary_regime: MarketRegime
    primary_bias: MarketBias
    mid_regime: MarketRegime  # ~5m
    mid_bias: MarketBias
    higher_regime: MarketRegime  # ~15m
    higher_bias: MarketBias
    aligned: bool
    conflict_reason: str = ""


class MultiTimeframe:
    def __init__(self, resample_factor: int = 7):
        self.mid_factor = 3  # 2m * 3 = ~6m (approximates 5m)
        self.high_factor = resample_factor  # 2m * 7 = ~14m (approximates 15m)
        self.mid_clf = RegimeClassifier()
        self.high_clf = RegimeClassifier()

    def analyze(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        primary_regime: MarketRegime,
        primary_bias: MarketBias,
    ) -> MTFResult:
        n = len(closes)

        # Default: aligned if not enough data
        default = MTFResult(
            primary_regime=primary_regime,
            primary_bias=primary_bias,
            mid_regime=MarketRegime.UNKNOWN,
            mid_bias=MarketBias.NEUTRAL,
            higher_regime=MarketRegime.UNKNOWN,
            higher_bias=MarketBias.NEUTRAL,
            aligned=True,
        )

        # Need enough data for both timeframes
        min_needed = self.high_factor * 25
        if n < min_needed:
            return default

        # Analyze mid timeframe (~5m)
        mid_c, mid_h, mid_l = self._resample(closes, highs, lows, self.mid_factor)
        if len(mid_c) >= 20:
            mid_result = self.mid_clf.classify(mid_c, mid_h, mid_l)
            mid_regime = mid_result.regime
            mid_bias = mid_result.bias
        else:
            mid_regime = MarketRegime.UNKNOWN
            mid_bias = MarketBias.NEUTRAL

        # Analyze higher timeframe (~15m)
        high_c, high_h, high_l = self._resample(closes, highs, lows, self.high_factor)
        if len(high_c) >= 20:
            high_result = self.high_clf.classify(high_c, high_h, high_l)
            higher_regime = high_result.regime
            higher_bias = high_result.bias
        else:
            higher_regime = MarketRegime.UNKNOWN
            higher_bias = MarketBias.NEUTRAL

        # Check alignment
        aligned = True
        conflict = ""

        # Primary vs higher TF: must not contradict
        if primary_bias != MarketBias.NEUTRAL and higher_bias != MarketBias.NEUTRAL:
            if primary_bias != higher_bias:
                aligned = False
                conflict = f"2m={primary_bias.value} vs 15m={higher_bias.value}"

        # Primary vs mid TF: must not contradict
        if primary_bias != MarketBias.NEUTRAL and mid_bias != MarketBias.NEUTRAL:
            if primary_bias != mid_bias:
                aligned = False
                conflict = f"2m={primary_bias.value} vs 5m={mid_bias.value}"

        return MTFResult(
            primary_regime=primary_regime,
            primary_bias=primary_bias,
            mid_regime=mid_regime,
            mid_bias=mid_bias,
            higher_regime=higher_regime,
            higher_bias=higher_bias,
            aligned=aligned,
            conflict_reason=conflict,
        )

    def _resample(
        self, closes: FloatArray, highs: FloatArray, lows: FloatArray, factor: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        n = len(closes)
        n_bars = n // factor

        if n_bars < 2:
            return closes, highs, lows

        trim = n_bars * factor
        c = closes[-trim:]
        h = highs[-trim:]
        lo = lows[-trim:]

        htf_closes = np.array([c[(i + 1) * factor - 1] for i in range(n_bars)])
        htf_highs = np.array(
            [np.max(h[i * factor : (i + 1) * factor]) for i in range(n_bars)]
        )
        htf_lows = np.array(
            [np.min(lo[i * factor : (i + 1) * factor]) for i in range(n_bars)]
        )

        return htf_closes, htf_highs, htf_lows
