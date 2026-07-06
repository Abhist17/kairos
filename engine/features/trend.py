"""
Kairos Engine — Trend-quality features.

Measures HOW efficiently price is moving, not just whether it moved.
"""

import numpy as np

from engine.core.types import FloatArray


def kaufman_efficiency_ratio(closes: FloatArray) -> float:
    """
    ER = |net displacement| / total path length
    ER ~ 1.0 = straight line trend. ER ~ 0.0 = chop.
    """
    if len(closes) < 2:
        return 0.0

    net_displacement = abs(closes[-1] - closes[0])
    total_path = float(np.sum(np.abs(np.diff(closes))))

    if total_path == 0:
        return 0.0

    return float(net_displacement / total_path)


def normalized_slope(closes: FloatArray) -> float:
    """
    Slope of least-squares line through closes, normalized by mean price.
    Positive = up, negative = down. Magnitude = steepness relative to price.
    """
    n = len(closes)
    if n < 2:
        return 0.0

    mean_price = np.mean(closes)
    if mean_price == 0:
        return 0.0

    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    y_mean = mean_price

    numerator = np.sum((x - x_mean) * (closes - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    raw_slope = numerator / denominator
    return float(raw_slope / mean_price)


def directional_entropy(returns: FloatArray, n_bins: int = 10) -> float:
    """
    Shannon entropy of the return distribution, normalized to [0, 1].
    High (->1) = random/chaotic. Low (->0) = directional.
    """
    if len(returns) < 2:
        return 1.0

    if np.std(returns) < 1e-15:
        return 1.0

    counts, _ = np.histogram(returns, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_bins)

    if max_entropy == 0:
        return 1.0

    return float(entropy / max_entropy)
