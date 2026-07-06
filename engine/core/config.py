"""
Kairos Engine — Configuration

All tunable thresholds in one place.
Every threshold has a comment explaining WHY that value.
These are starting points — Setup Memory will eventually calibrate them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegimeConfig:
    lookback: int = 20
    vol_percentile_window: int = 100

    efficiency_ratio_trend: float = 0.55
    efficiency_ratio_chop: float = 0.25

    persistence_trend: float = 0.65
    persistence_chop: float = 0.45

    atr_contraction_threshold: float = 0.70
    bbw_percentile_compressed: float = 20.0

    vol_percentile_exhaustion: float = 75.0
    efficiency_exhaustion_ceiling: float = 0.50

    entropy_chaotic: float = 0.95
    entropy_trending: float = 0.75

    min_confidence: float = 0.40


@dataclass(frozen=True)
class EngineConfig:
    regime: RegimeConfig = field(default_factory=RegimeConfig)
