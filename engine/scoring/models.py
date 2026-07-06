"""Kairos Engine — Thesis scoring models."""

from dataclasses import dataclass
from engine.core.enums import MarketBias


@dataclass
class ThesisResult:
    primary_bias: MarketBias
    primary_score: float  # 0-1
    counter_score: float  # 0-1
    separation: float  # primary - counter (high = clear thesis)
    thesis_valid: bool  # separation above minimum threshold
