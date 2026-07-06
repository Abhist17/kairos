"""Kairos Engine — Regime models."""

from dataclasses import dataclass

from engine.core.enums import MarketBias, MarketRegime


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    bias: MarketBias

    efficiency_ratio: float
    directional_persistence: float
    normalized_slope: float
    directional_entropy: float
    realized_volatility: float
    volatility_percentile: float
    atr_contraction: float
    log_return_mean: float

    regime_age: int = 0
    transition_from: MarketRegime | None = None
