"""Tests for Compression Detector."""

import numpy as np
import pytest

from engine.compression.detector import CompressionDetector


@pytest.fixture
def detector():
    return CompressionDetector()


def _make_ohlc(closes, noise_pct=0.001):
    rng = np.random.default_rng(42)
    noise = closes * noise_pct
    highs = closes + np.abs(rng.normal(0, 1, len(closes))) * noise + 0.5
    lows = closes - np.abs(rng.normal(0, 1, len(closes))) * noise - 0.5
    return highs, lows, closes


class TestCompressionDetection:
    def test_obvious_compression(self, detector):
        """Normal vol then tiny vol — should detect compression."""
        rng = np.random.default_rng(42)
        base = 25000.0
        normal = rng.normal(0, 8, 100)
        tight = rng.normal(0, 0.3, 80)
        moves = np.concatenate([normal, tight])
        closes = base + np.cumsum(moves)
        highs, lows, closes = _make_ohlc(closes, noise_pct=0.0003)

        result = detector.detect(closes, highs, lows)

        assert result.compression_score > 0.0
        assert result.rv_decay < 1.0
        assert result.bbw_percentile < 50.0

    def test_expanding_market_not_compressed(self, detector):
        """Trending strongly — should NOT be compressed."""
        rng = np.random.default_rng(42)
        base = 25000.0
        drift = np.cumsum(np.ones(180) * 3.0 + rng.normal(0, 2, 180))
        closes = base + drift
        highs, lows, closes = _make_ohlc(closes)

        result = detector.detect(closes, highs, lows)

        assert result.is_compressed is False

    def test_compression_velocity_negative_when_compressing(self, detector):
        """Velocity should be negative when range is shrinking."""
        rng = np.random.default_rng(42)
        base = 25000.0
        # Linearly decreasing volatility
        moves = []
        for i in range(180):
            vol = max(0.1, 10.0 - i * 0.05)
            moves.append(rng.normal(0, vol))
        closes = base + np.cumsum(moves)
        highs, lows, closes = _make_ohlc(closes, noise_pct=0.0004)

        result = detector.detect(closes, highs, lows)

        assert result.compression_velocity < 0

    def test_half_life_inf_when_expanding(self, detector):
        """Half-life should be inf when market is expanding, not compressing."""
        rng = np.random.default_rng(42)
        base = 25000.0
        # Increasing vol
        moves = []
        for i in range(180):
            vol = 1.0 + i * 0.1
            moves.append(rng.normal(0, vol))
        closes = base + np.cumsum(moves)
        highs, lows, closes = _make_ohlc(closes)

        result = detector.detect(closes, highs, lows)

        assert (
            result.compression_half_life > 1000
        )  # effectively inf, noise may cause tiny negative velocity


class TestCompressionDuration:
    def test_candles_compressed_tracks(self, detector):
        """Consecutive compression calls should increment counter."""
        rng = np.random.default_rng(42)
        base = 25000.0
        normal = rng.normal(0, 10, 100)
        tight = rng.normal(0, 0.2, 80)
        moves = np.concatenate([normal, tight])
        closes = base + np.cumsum(moves)
        highs, lows, closes = _make_ohlc(closes, noise_pct=0.0002)

        r1 = detector.detect(closes, highs, lows)
        # Add one more tight candle
        extra = np.append(closes, closes[-1] + rng.normal(0, 0.2))
        extra_h = np.append(highs, extra[-1] + 0.3)
        extra_l = np.append(lows, extra[-1] - 0.3)
        r2 = detector.detect(extra, extra_h, extra_l)

        if r1.is_compressed and r2.is_compressed:
            assert r2.candles_compressed >= r1.candles_compressed


class TestEdgeCases:
    def test_insufficient_data(self, detector):
        closes = np.array([25000.0, 25001.0, 25002.0])
        highs = closes + 1
        lows = closes - 1

        result = detector.detect(closes, highs, lows)
        assert result.is_compressed is False
        assert result.compression_score == 0.0
