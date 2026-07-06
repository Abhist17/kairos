"""
Kairos Engine — Return-based features.

Log returns are used instead of simple returns because:
1. They are additive across time.
2. They are approximately normally distributed for small moves.
3. They are symmetric: +1% and -1% have equal magnitude.
"""

import numpy as np

from engine.core.types import FloatArray


def log_returns(closes: FloatArray) -> FloatArray:
    """ln(close[t] / close[t-1]). Returns array of length len(closes) - 1."""
    if len(closes) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(np.log(closes))


def directional_persistence(returns: FloatArray) -> float:
    """
    Fraction of returns in the dominant direction.
    Direction-agnostic: max(frac_positive, frac_negative).
    """
    if len(returns) == 0:
        return 0.5

    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.5

    n_positive = np.sum(nonzero > 0)
    n_total = len(nonzero)
    frac_positive = n_positive / n_total

    return float(max(frac_positive, 1.0 - frac_positive))


def mean_log_return(returns: FloatArray) -> float:
    """Mean of log returns. Sign = bias, magnitude = strength."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))
