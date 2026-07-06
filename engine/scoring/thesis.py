"""
Kairos Engine — Thesis Scoring

Builds a directional thesis from all engine outputs and measures
how SEPARATED the primary thesis is from the counter thesis.

A high bullish score alone is NOT enough. If bearish score is
nearly as high, the setup is ambiguous and not tradeable.

Thesis Separation = Primary Score - Counter Score
High separation = clear, tradeable thesis.
Low separation = conflicting signals, stay out.
"""

from engine.core.enums import MarketBias, MarketRegime
from engine.scoring.models import ThesisResult


class ThesisEngine:
    def __init__(self, min_separation: float = 0.15):
        self.min_separation = min_separation

    def score(
        self,
        regime: MarketRegime,
        bias: MarketBias,
        regime_confidence: float,
        compression_score: float,
        is_compressed: bool,
        structure_score: float,
        nearest_zone_dist_pct: float,
        pressure_score: float,
        aggression_ratio: float,
        iv_percentile: float,
        option_efficient: bool,
    ) -> ThesisResult:

        bull_points = 0.0
        bear_points = 0.0
        total_weight = 0.0

        # --- Bias from regime engine (weight: 0.25) ---
        w = 0.25
        total_weight += w
        if bias == MarketBias.BULLISH:
            bull_points += w * regime_confidence
        elif bias == MarketBias.BEARISH:
            bear_points += w * regime_confidence
        else:
            bull_points += w * 0.3
            bear_points += w * 0.3

        # --- Aggression ratio (weight: 0.20) ---
        w = 0.20
        total_weight += w
        bull_points += w * aggression_ratio
        bear_points += w * (1.0 - aggression_ratio)

        # --- Pressure (weight: 0.20) ---
        w = 0.20
        total_weight += w
        if bias == MarketBias.BULLISH:
            bull_points += w * pressure_score
        elif bias == MarketBias.BEARISH:
            bear_points += w * pressure_score
        else:
            bull_points += w * pressure_score * 0.5
            bear_points += w * pressure_score * 0.5

        # --- Structure proximity (weight: 0.15) ---
        w = 0.15
        total_weight += w
        prox_score = max(0, 1.0 - nearest_zone_dist_pct / 0.5)
        bull_points += w * prox_score * (1.0 if bias != MarketBias.BEARISH else 0.3)
        bear_points += w * prox_score * (1.0 if bias != MarketBias.BULLISH else 0.3)

        # --- Compression bonus (weight: 0.10) ---
        w = 0.10
        total_weight += w
        if is_compressed:
            if bias == MarketBias.BULLISH:
                bull_points += w * compression_score
            elif bias == MarketBias.BEARISH:
                bear_points += w * compression_score
            else:
                bull_points += w * compression_score * 0.5
                bear_points += w * compression_score * 0.5

        # --- IV favorability (weight: 0.10) ---
        w = 0.10
        total_weight += w
        iv_score = max(0, 1.0 - iv_percentile / 100.0)
        if bias == MarketBias.BULLISH:
            bull_points += w * iv_score
        elif bias == MarketBias.BEARISH:
            bear_points += w * iv_score

        # Normalize
        if total_weight > 0:
            bull_points /= total_weight
            bear_points /= total_weight

        # Determine primary
        if bull_points >= bear_points:
            primary_bias = MarketBias.BULLISH
            primary = bull_points
            counter = bear_points
        else:
            primary_bias = MarketBias.BEARISH
            primary = bear_points
            counter = bull_points

        separation = primary - counter
        valid = separation >= self.min_separation

        return ThesisResult(
            primary_bias=primary_bias,
            primary_score=round(primary, 4),
            counter_score=round(counter, 4),
            separation=round(separation, 4),
            thesis_valid=valid,
        )
