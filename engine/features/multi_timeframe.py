"""
Kairos Engine — Multi-Timeframe Confirmation

Problem: 2-min chart shows TREND_EXPANSION but 15-min chart
is actually in MEAN_REVERSION. The 2-min trend is just noise
within a larger range.

Solution: Resample candles to higher timeframe, run regime
classifier on both, require agreement.

If 2m says TREND_EXPANSION + BULLISH
and 15m also says BULLISH (any trending regime)
→ CONFIRMED

If they disagree → CONFLICTING → don't signal
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
    higher_regime: MarketRegime
    higher_bias: MarketBias
    aligned: bool  # both timeframes agree on direction
    higher_tf_label: str  # e.g. "15m"


class MultiTimeframe:
    def __init__(self, resample_factor: int = 7):
        """
        resample_factor: how many primary candles make one higher TF candle.
        For 2m primary → factor 7 ≈ ~15 minute higher TF.
        """
        self.resample_factor = resample_factor
        self.higher_clf = RegimeClassifier()

    def analyze(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        primary_regime: MarketRegime,
        primary_bias: MarketBias,
    ) -> MTFResult:
        factor = self.resample_factor
        n = len(closes)

        if n < factor * 20:  # need enough for higher TF lookback
            return MTFResult(
                primary_regime=primary_regime,
                primary_bias=primary_bias,
                higher_regime=MarketRegime.UNKNOWN,
                higher_bias=MarketBias.NEUTRAL,
                aligned=True,  # don't block if insufficient data
                higher_tf_label=f"{factor}x",
            )

        # Resample to higher timeframe
        htf_closes, htf_highs, htf_lows = self._resample(closes, highs, lows, factor)

        if len(htf_closes) < 20:
            return MTFResult(
                primary_regime=primary_regime,
                primary_bias=primary_bias,
                higher_regime=MarketRegime.UNKNOWN,
                higher_bias=MarketBias.NEUTRAL,
                aligned=True,
                higher_tf_label=f"{factor}x",
            )

        # Classify higher TF
        result = self.higher_clf.classify(htf_closes, htf_highs, htf_lows)

        # Check alignment: biases must agree or higher be neutral
        aligned = (
            result.bias == primary_bias
            or result.bias == MarketBias.NEUTRAL
            or primary_bias == MarketBias.NEUTRAL
        )

        return MTFResult(
            primary_regime=primary_regime,
            primary_bias=primary_bias,
            higher_regime=result.regime,
            higher_bias=result.bias,
            aligned=aligned,
            higher_tf_label=f"{factor}x",
        )

    def _resample(
        self, closes: FloatArray, highs: FloatArray, lows: FloatArray, factor: int
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Resample OHLC to higher timeframe by grouping candles."""
        n = len(closes)
        n_bars = n // factor

        if n_bars < 2:
            return closes, highs, lows

        # Trim to exact multiple
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
