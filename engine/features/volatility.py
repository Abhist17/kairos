"""
Kairos Engine — Volatility features.

Volatility is the single most important variable for options pricing.
"""

import numpy as np

from engine.core.types import FloatArray


def realized_volatility(returns: FloatArray) -> float:
    """Annualized realized vol from log returns (1-min NIFTY)."""
    if len(returns) < 2:
        return 0.0
    annualization = np.sqrt(93_750)
    return float(np.std(returns, ddof=1) * annualization)


def volatility_percentile(
    returns: FloatArray,
    current_window: int = 20,
    history_window: int = 100,
) -> float:
    """
    Where does current realized vol sit relative to recent history?
    Returns 0-100.
    """
    if len(returns) < current_window + history_window:
        return 50.0

    rolling_vols = np.array(
        [
            np.std(returns[i : i + current_window], ddof=1)
            for i in range(len(returns) - current_window + 1)
        ]
    )

    if len(rolling_vols) < 2:
        return 50.0

    current_vol = rolling_vols[-1]
    history = rolling_vols[-history_window:]

    if np.max(history) < 1e-15:
        return 0.0

    percentile = float(np.sum(history <= current_vol) / len(history) * 100)
    return percentile


def atr(highs: FloatArray, lows: FloatArray, closes: FloatArray) -> FloatArray:
    """True Range per candle. Returns array of length len(closes) - 1."""
    if len(closes) < 2:
        return np.array([], dtype=np.float64)

    high_low = highs[1:] - lows[1:]
    high_prev_close = np.abs(highs[1:] - closes[:-1])
    low_prev_close = np.abs(lows[1:] - closes[:-1])

    true_ranges = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    return true_ranges


def atr_contraction_ratio(
    highs: FloatArray,
    lows: FloatArray,
    closes: FloatArray,
    short_window: int = 5,
    long_window: int = 20,
) -> float:
    """
    Short-term ATR / long-term ATR median.
    <0.7 = contracting (compression). ~1.0 = normal. >1.3 = expanding.
    """
    true_ranges = atr(highs, lows, closes)

    if len(true_ranges) < long_window:
        return 1.0

    recent_atr = float(np.mean(true_ranges[-short_window:]))
    median_atr = float(np.median(true_ranges[-long_window:]))

    if median_atr == 0:
        return 1.0

    return recent_atr / median_atr


def bollinger_bandwidth(
    closes: FloatArray,
    window: int = 20,
    num_std: float = 2.0,
) -> float:
    """Bollinger Bandwidth: (upper - lower) / middle * 100."""
    if len(closes) < window:
        return 0.0

    recent = closes[-window:]
    middle = float(np.mean(recent))
    std = float(np.std(recent, ddof=1))

    if middle == 0:
        return 0.0

    upper = middle + num_std * std
    lower = middle - num_std * std

    return (upper - lower) / middle * 100


def bollinger_bandwidth_percentile(
    closes: FloatArray,
    bb_window: int = 20,
    percentile_lookback: int = 100,
) -> float:
    """Percentile rank of current BB width. Low (<20) = compression."""
    total_needed = bb_window + percentile_lookback
    if len(closes) < total_needed:
        return 50.0

    bandwidths = []
    for i in range(percentile_lookback):
        end = len(closes) - percentile_lookback + i + 1
        start = end - bb_window
        segment = closes[start:end]
        bw = bollinger_bandwidth(segment, window=bb_window)
        bandwidths.append(bw)

    bandwidths = np.array(bandwidths)
    current_bw = bandwidths[-1]

    percentile = float(np.sum(bandwidths <= current_bw) / len(bandwidths) * 100)
    return percentile
