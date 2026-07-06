"""Kairos Engine — IV State models."""

from dataclasses import dataclass
from engine.core.enums import IVState


@dataclass
class IVResult:
    state: IVState
    current_iv: float
    iv_percentile: float  # 0-100
    iv_velocity: float  # rate of IV change per candle
    iv_mean: float
    iv_std: float
