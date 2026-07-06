"""Kairos Engine — Flow/Pressure models."""

from dataclasses import dataclass


@dataclass
class FlowResult:
    pressure_score: float  # 0-1 composite
    level_tests: int  # how many times price tested nearest zone
    test_frequency: float  # tests per N candles
    pullback_shrinking: bool  # are pullbacks getting smaller?
    aggression_ratio: float  # buying aggression vs selling
    pressure_asymmetry: float  # 0=symmetric, 1=fully one-sided
    liquidity_thinning: bool  # range shrinking near level = liquidity depleting
