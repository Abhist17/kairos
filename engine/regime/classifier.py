"""
Kairos Engine — Regime Classifier V1

Rule-based regime classification using quantitative features.
Each regime has a multi-factor score (0-1). Highest wins.
Below min_confidence -> UNKNOWN.
"""

import numpy as np

from engine.core.config import RegimeConfig
from engine.core.enums import MarketBias, MarketRegime
from engine.core.logger import get_logger
from engine.core.types import FloatArray
from engine.features.returns import (
    directional_persistence,
    log_returns,
    mean_log_return,
)
from engine.features.trend import (
    directional_entropy,
    kaufman_efficiency_ratio,
    normalized_slope,
)
from engine.features.volatility import (
    atr_contraction_ratio,
    realized_volatility,
    volatility_percentile,
)
from engine.regime.models import RegimeResult

logger = get_logger("regime.classifier")


class RegimeClassifier:
    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self._prev_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_age: int = 0

    def classify(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
    ) -> RegimeResult:
        cfg = self.config
        n = len(closes)

        if n < cfg.lookback:
            return self._unknown_result()

        window_closes = closes[-cfg.lookback :]
        window_highs = highs[-cfg.lookback :]
        window_lows = lows[-cfg.lookback :]

        rets = log_returns(window_closes)

        er = kaufman_efficiency_ratio(window_closes)
        persistence = directional_persistence(rets)
        slope = normalized_slope(window_closes)
        entropy = directional_entropy(rets)
        rv = realized_volatility(rets)
        vol_pct = volatility_percentile(
            log_returns(closes),
            current_window=cfg.lookback,
            history_window=cfg.vol_percentile_window,
        )
        atr_ratio = atr_contraction_ratio(window_highs, window_lows, window_closes)
        ret_mean = mean_log_return(rets)

        scores: dict[MarketRegime, float] = {
            MarketRegime.TREND_EXPANSION: self._score_trend_expansion(
                er, persistence, entropy, slope
            ),
            MarketRegime.TREND_EXHAUSTION: self._score_trend_exhaustion(
                er, persistence, vol_pct, entropy
            ),
            MarketRegime.COMPRESSION: self._score_compression(
                atr_ratio, er, entropy, vol_pct
            ),
            MarketRegime.MEAN_REVERSION: self._score_mean_reversion(
                er, persistence, entropy, atr_ratio
            ),
            MarketRegime.CHAOTIC: self._score_chaotic(
                entropy, er, persistence, vol_pct
            ),
        }

        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_regime]

        if best_score < cfg.min_confidence:
            best_regime = MarketRegime.UNKNOWN
            best_score = 0.0

        bias = self._compute_bias(ret_mean, slope, persistence)

        transition_from = (
            self._prev_regime if best_regime != self._prev_regime else None
        )

        if best_regime == self._prev_regime:
            self._regime_age += 1
        else:
            self._regime_age = 1

        self._prev_regime = best_regime

        return RegimeResult(
            regime=best_regime,
            confidence=round(best_score, 4),
            bias=bias,
            efficiency_ratio=round(er, 4),
            directional_persistence=round(persistence, 4),
            normalized_slope=round(slope, 6),
            directional_entropy=round(entropy, 4),
            realized_volatility=round(rv, 4),
            volatility_percentile=round(vol_pct, 2),
            atr_contraction=round(atr_ratio, 4),
            log_return_mean=round(ret_mean, 8),
            regime_age=self._regime_age,
            transition_from=transition_from,
        )

    def _score_trend_expansion(self, er, persistence, entropy, slope):
        cfg = self.config
        er_score = _sigmoid_score(er, center=cfg.efficiency_ratio_trend, steepness=12)
        persist_score = _sigmoid_score(
            persistence, center=cfg.persistence_trend, steepness=12
        )
        entropy_score = _sigmoid_score(
            1.0 - entropy, center=1.0 - cfg.entropy_trending, steepness=10
        )
        slope_score = min(abs(slope) / 0.001, 1.0)
        return (
            0.35 * er_score
            + 0.30 * persist_score
            + 0.20 * entropy_score
            + 0.15 * slope_score
        )

    def _score_trend_exhaustion(self, er, persistence, vol_pct, entropy):
        cfg = self.config
        er_moderate = 1.0 - abs(er - 0.40) / 0.40
        er_moderate = max(0.0, er_moderate)
        persist_moderate = _sigmoid_score(persistence, center=0.55, steepness=8)
        vol_high = _sigmoid_score(
            vol_pct, center=cfg.vol_percentile_exhaustion, steepness=0.08
        )
        entropy_rising = _sigmoid_score(entropy, center=0.80, steepness=8)
        return (
            0.25 * er_moderate
            + 0.20 * persist_moderate
            + 0.30 * vol_high
            + 0.25 * entropy_rising
        )

    def _score_compression(self, atr_ratio, er, entropy, vol_pct):
        cfg = self.config
        atr_score = _sigmoid_score(
            1.0 - atr_ratio,
            center=1.0 - cfg.atr_contraction_threshold,
            steepness=10,
        )
        er_low = _sigmoid_score(1.0 - er, center=0.55, steepness=8)
        vol_low = _sigmoid_score(
            100.0 - vol_pct,
            center=100.0 - cfg.bbw_percentile_compressed,
            steepness=0.06,
        )
        entropy_moderate = 1.0 - abs(entropy - 0.80) / 0.30
        entropy_moderate = max(0.0, min(1.0, entropy_moderate))
        return (
            0.40 * atr_score + 0.20 * er_low + 0.25 * vol_low + 0.15 * entropy_moderate
        )

    def _score_mean_reversion(self, er, persistence, entropy, atr_ratio):
        er_low = _sigmoid_score(1.0 - er, center=0.65, steepness=8)
        persist_low = _sigmoid_score(1.0 - persistence, center=0.50, steepness=8)
        entropy_high = _sigmoid_score(entropy, center=0.85, steepness=8)
        atr_normal = 1.0 - abs(atr_ratio - 1.0) / 0.5
        atr_normal = max(0.0, min(1.0, atr_normal))
        return (
            0.30 * er_low + 0.30 * persist_low + 0.25 * entropy_high + 0.15 * atr_normal
        )

    def _score_chaotic(self, entropy, er, persistence, vol_pct):
        entropy_high = _sigmoid_score(entropy, center=0.85, steepness=12)
        er_low = _sigmoid_score(1.0 - er, center=0.70, steepness=10)
        persist_low = _sigmoid_score(1.0 - persistence, center=0.50, steepness=8)
        vol_high = _sigmoid_score(vol_pct, center=70.0, steepness=0.08)
        return (
            0.25 * entropy_high + 0.20 * er_low + 0.20 * persist_low + 0.35 * vol_high
        )

    def _compute_bias(self, ret_mean, slope, persistence):
        if ret_mean > 0 and slope > 0 and persistence > 0.55:
            return MarketBias.BULLISH
        elif ret_mean < 0 and slope < 0 and persistence > 0.55:
            return MarketBias.BEARISH
        return MarketBias.NEUTRAL

    def _unknown_result(self):
        return RegimeResult(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            bias=MarketBias.NEUTRAL,
            efficiency_ratio=0.0,
            directional_persistence=0.5,
            normalized_slope=0.0,
            directional_entropy=1.0,
            realized_volatility=0.0,
            volatility_percentile=50.0,
            atr_contraction=1.0,
            log_return_mean=0.0,
        )


def _sigmoid_score(value: float, center: float, steepness: float) -> float:
    return float(1.0 / (1.0 + np.exp(-steepness * (value - center))))
