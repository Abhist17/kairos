"""
Kairos Engine — Compression Detector

Detects when the market is coiling: range shrinking, volatility decaying,
energy building for an expansion move.

WHY THIS MATTERS FOR OPTIONS:
Compression is the BEST regime for option entry because:
1. IV tends to be low → options are cheap
2. Gamma acceleration is maximized when price breaks out of compression
3. Theta cost is low relative to the potential move
4. The move, when it comes, tends to be fast and directional

Compression Velocity: how FAST is volatility contracting?
Fast compression → breakout likely sooner.
Slow compression → could grind sideways longer.

Compression Half-Life: at the current decay rate, how many candles
until the range halves again? This estimates urgency.
Short half-life → compression is accelerating → watch closely.
Long half-life → compression is stalling → patience needed.
"""

import numpy as np

from engine.core.logger import get_logger
from engine.core.types import FloatArray
from engine.compression.models import CompressionResult
from engine.features.volatility import (
    atr,
    atr_contraction_ratio,
    bollinger_bandwidth_percentile,
)
from engine.features.returns import log_returns

logger = get_logger("compression.detector")


class CompressionDetector:
    def __init__(
        self,
        atr_threshold: float = 0.70,
        range_threshold: float = 0.70,
        rv_decay_threshold: float = 0.70,
        bbw_percentile_threshold: float = 25.0,
        score_threshold: float = 0.55,
        lookback: int = 20,
        long_lookback: int = 50,
    ):
        self.atr_threshold = atr_threshold
        self.range_threshold = range_threshold
        self.rv_decay_threshold = rv_decay_threshold
        self.bbw_percentile_threshold = bbw_percentile_threshold
        self.score_threshold = score_threshold
        self.lookback = lookback
        self.long_lookback = long_lookback

        self._candles_compressed: int = 0
        self._pre_compression_vol: float = 0.0
        self._was_compressed: bool = False

    def detect(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
    ) -> CompressionResult:
        n = len(closes)
        if n < self.long_lookback + self.lookback:
            return self._default_result()

        # --- ATR contraction ---
        atr_ratio = atr_contraction_ratio(
            highs[-self.long_lookback :],
            lows[-self.long_lookback :],
            closes[-self.long_lookback :],
            short_window=5,
            long_window=self.lookback,
        )

        # --- Range contraction ---
        range_ratio = self._range_contraction(highs, lows)

        # --- Realized vol decay ---
        rv_decay = self._rv_decay(closes)

        # --- Bollinger bandwidth percentile ---
        bbw_pct = bollinger_bandwidth_percentile(
            closes, bb_window=self.lookback, percentile_lookback=self.long_lookback
        )

        # --- Compression velocity ---
        comp_velocity = self._compression_velocity(highs, lows, closes)

        # --- Compression half-life ---
        half_life = self._compression_half_life(comp_velocity, atr_ratio)

        # --- Composite score ---
        atr_score = max(0.0, (self.atr_threshold - atr_ratio) / self.atr_threshold)
        range_score = max(
            0.0, (self.range_threshold - range_ratio) / self.range_threshold
        )
        rv_score = max(
            0.0, (self.rv_decay_threshold - rv_decay) / self.rv_decay_threshold
        )
        bbw_score = max(
            0.0,
            (self.bbw_percentile_threshold - bbw_pct) / self.bbw_percentile_threshold,
        )

        compression_score = (
            0.30 * atr_score + 0.25 * range_score + 0.25 * rv_score + 0.20 * bbw_score
        )
        compression_score = min(1.0, compression_score)

        is_compressed = compression_score >= self.score_threshold

        # --- Track duration ---
        if is_compressed:
            if not self._was_compressed:
                rets = log_returns(closes)
                self._pre_compression_vol = (
                    float(np.std(rets[-self.long_lookback :], ddof=1))
                    if len(rets) >= self.long_lookback
                    else 0.0
                )
            self._candles_compressed += 1
        else:
            self._candles_compressed = 0

        self._was_compressed = is_compressed

        return CompressionResult(
            is_compressed=is_compressed,
            compression_score=round(compression_score, 4),
            atr_contraction=round(atr_ratio, 4),
            range_contraction=round(range_ratio, 4),
            rv_decay=round(rv_decay, 4),
            bbw_percentile=round(bbw_pct, 2),
            compression_velocity=round(comp_velocity, 6),
            compression_half_life=round(half_life, 2),
            candles_compressed=self._candles_compressed,
            pre_compression_vol=round(self._pre_compression_vol, 6),
        )

    def _range_contraction(self, highs: FloatArray, lows: FloatArray) -> float:
        """Short-term candle range / long-term candle range median."""
        ranges = highs - lows
        if len(ranges) < self.long_lookback:
            return 1.0

        short_range = float(np.mean(ranges[-5:]))
        long_median = float(np.median(ranges[-self.long_lookback :]))

        if long_median == 0:
            return 1.0

        return short_range / long_median

    def _rv_decay(self, closes: FloatArray) -> float:
        """Current short-window realized vol / longer-window median realized vol."""
        rets = log_returns(closes)
        if len(rets) < self.long_lookback:
            return 1.0

        short_vol = float(np.std(rets[-self.lookback :], ddof=1))

        # Rolling vol over long lookback
        rolling_vols = np.array(
            [
                np.std(rets[i : i + self.lookback], ddof=1)
                for i in range(len(rets) - self.lookback + 1)
            ]
        )

        if len(rolling_vols) < 2:
            return 1.0

        median_vol = float(np.median(rolling_vols[-self.long_lookback :]))

        if median_vol == 0:
            return 1.0

        return short_vol / median_vol

    def _compression_velocity(
        self, highs: FloatArray, lows: FloatArray, closes: FloatArray
    ) -> float:
        """
        Rate of change of ATR over recent candles.
        Negative = ATR shrinking = compressing.
        Units: ATR change per candle, normalized by price.
        """
        true_ranges = atr(highs, lows, closes)
        if len(true_ranges) < self.lookback:
            return 0.0

        recent_tr = true_ranges[-self.lookback :]
        mean_price = float(np.mean(closes[-self.lookback :]))

        if mean_price == 0:
            return 0.0

        # Linear regression slope of true ranges
        x = np.arange(len(recent_tr), dtype=np.float64)
        x_mean = np.mean(x)

        numerator = float(np.sum((x - x_mean) * (recent_tr - np.mean(recent_tr))))
        denominator = float(np.sum((x - x_mean) ** 2))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope / mean_price  # normalized

    def _compression_half_life(
        self, velocity: float, current_atr_ratio: float
    ) -> float:
        """
        Estimated candles until ATR ratio halves from current level.
        Only meaningful when velocity is negative (compressing).
        """
        if velocity >= 0:
            return float("inf")  # not compressing

        # velocity is normalized ATR change per candle
        # half_life = current_level / (2 * |decay_rate|)
        if abs(velocity) < 1e-10:
            return float("inf")

        half_life = current_atr_ratio / (2.0 * abs(velocity))
        return max(1.0, half_life)

    def _default_result(self) -> CompressionResult:
        return CompressionResult(
            is_compressed=False,
            compression_score=0.0,
            atr_contraction=1.0,
            range_contraction=1.0,
            rv_decay=1.0,
            bbw_percentile=50.0,
            compression_velocity=0.0,
            compression_half_life=float("inf"),
            candles_compressed=0,
            pre_compression_vol=0.0,
        )
