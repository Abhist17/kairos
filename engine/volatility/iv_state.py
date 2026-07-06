"""
Kairos Engine — IV State Classifier

Classifies implied volatility regime from a time series of IV readings.
In live trading, IV comes from the option chain. Here we estimate it
from realized vol as a proxy until real data is connected.

WHY THIS MATTERS:
- COMPRESSED IV → options are cheap → best time to buy
- OVEREXPANDED IV → options are expensive → worst time to buy
- COLLAPSING IV → you lose even if direction is right (IV crush)
"""

import numpy as np
from engine.core.enums import IVState
from engine.core.types import FloatArray
from engine.volatility.models import IVResult


class IVClassifier:
    def __init__(
        self,
        compressed_pct: float = 20.0,
        overexpanded_pct: float = 85.0,
        velocity_expanding: float = 0.02,
        velocity_collapsing: float = -0.02,
        lookback: int = 20,
        history: int = 100,
    ):
        self.compressed_pct = compressed_pct
        self.overexpanded_pct = overexpanded_pct
        self.velocity_expanding = velocity_expanding
        self.velocity_collapsing = velocity_collapsing
        self.lookback = lookback
        self.history = history

    def classify(self, iv_series: FloatArray) -> IVResult:
        if len(iv_series) < self.lookback + 10:
            return IVResult(
                state=IVState.NORMAL,
                current_iv=0.0,
                iv_percentile=50.0,
                iv_velocity=0.0,
                iv_mean=0.0,
                iv_std=0.0,
            )

        current = float(iv_series[-1])
        hist = (
            iv_series[-self.history :] if len(iv_series) >= self.history else iv_series
        )
        iv_mean = float(np.mean(hist))
        iv_std = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0

        pct = float(np.sum(hist <= current) / len(hist) * 100)

        # Velocity: slope of recent IV
        recent = iv_series[-self.lookback :]
        x = np.arange(len(recent), dtype=np.float64)
        x_m = np.mean(x)
        denom = float(np.sum((x - x_m) ** 2))
        velocity = 0.0
        if denom > 0 and iv_mean > 0:
            velocity = (
                float(np.sum((x - x_m) * (recent - np.mean(recent))) / denom) / iv_mean
            )

        # Classify
        if velocity < self.velocity_collapsing and pct > 40:
            state = IVState.COLLAPSING
        elif pct >= self.overexpanded_pct:
            state = IVState.OVEREXPANDED
        elif velocity > self.velocity_expanding and pct > 50:
            state = IVState.EXPANDING
        elif pct <= self.compressed_pct:
            state = IVState.COMPRESSED
        else:
            state = IVState.NORMAL

        return IVResult(
            state=state,
            current_iv=round(current, 4),
            iv_percentile=round(pct, 2),
            iv_velocity=round(velocity, 6),
            iv_mean=round(iv_mean, 4),
            iv_std=round(iv_std, 4),
        )
