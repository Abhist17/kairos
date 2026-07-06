"""Tests for Regime Classifier V1."""

import numpy as np
import pytest

from engine.core.enums import MarketBias, MarketRegime
from engine.regime.classifier import RegimeClassifier


@pytest.fixture
def classifier():
    return RegimeClassifier()


def _make_candles(closes, noise_pct=0.001):
    rng = np.random.default_rng(42)
    noise = closes * noise_pct
    highs = closes + np.abs(rng.normal(0, 1, len(closes))) * noise
    lows = closes - np.abs(rng.normal(0, 1, len(closes))) * noise
    return highs, lows, closes


class TestTrendExpansion:
    def test_strong_uptrend(self, classifier):
        rng = np.random.default_rng(42)
        base = 25000.0
        drift = np.cumsum(np.ones(150) * 2.0 + rng.normal(0, 0.3, 150))
        closes = base + drift
        highs, lows, closes = _make_candles(closes)

        result = classifier.classify(closes, highs, lows)

        assert result.regime == MarketRegime.TREND_EXPANSION
        assert result.confidence > 0.5
        assert result.bias == MarketBias.BULLISH
        assert result.efficiency_ratio > 0.4

    def test_strong_downtrend(self, classifier):
        rng = np.random.default_rng(42)
        base = 25000.0
        drift = np.cumsum(np.ones(150) * -2.0 + rng.normal(0, 0.3, 150))
        closes = base + drift
        highs, lows, closes = _make_candles(closes)

        result = classifier.classify(closes, highs, lows)

        assert result.regime == MarketRegime.TREND_EXPANSION
        assert result.bias == MarketBias.BEARISH


class TestCompression:
    def test_narrowing_range(self, classifier):
        rng = np.random.default_rng(42)
        base = 25000.0

        normal = rng.normal(0, 5, 100)
        compressed = rng.normal(0, 0.5, 50)

        all_moves = np.concatenate([normal, compressed])
        closes = base + np.cumsum(all_moves)
        highs, lows, closes = _make_candles(closes, noise_pct=0.0005)

        result = classifier.classify(closes, highs, lows)

        assert result.regime in (
            MarketRegime.COMPRESSION,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.UNKNOWN,
        )


class TestChaotic:
    def test_high_vol_random(self, classifier):
        rng = np.random.default_rng(42)
        base = 25000.0

        mild = rng.normal(0, 2, 100)
        violent = rng.normal(0, 30, 50)

        all_moves = np.concatenate([mild, violent])
        closes = base + np.cumsum(all_moves)
        highs, lows, closes = _make_candles(closes, noise_pct=0.002)

        result = classifier.classify(closes, highs, lows)

        assert result.regime in (
            MarketRegime.CHAOTIC,
            MarketRegime.TREND_EXHAUSTION,
        )
        assert result.volatility_percentile > 60


class TestTransitions:
    def test_regime_age_increments(self, classifier):
        rng = np.random.default_rng(42)
        closes = 25000.0 + np.cumsum(np.ones(150) * 2.0 + rng.normal(0, 0.3, 150))
        highs, lows, closes = _make_candles(closes)

        r1 = classifier.classify(closes, highs, lows)
        r2 = classifier.classify(closes, highs, lows)

        if r1.regime == r2.regime:
            assert r2.regime_age == 2
            assert r2.transition_from is None

    def test_transition_recorded(self, classifier):
        rng = np.random.default_rng(42)

        trend_closes = 25000.0 + np.cumsum(np.ones(150) * 2.0 + rng.normal(0, 0.3, 150))
        h, lo, c = _make_candles(trend_closes)
        r1 = classifier.classify(c, h, lo)

        flat_closes = np.full(150, trend_closes[-1]) + np.cumsum(
            rng.normal(0, 0.2, 150)
        )
        h2, l2, c2 = _make_candles(flat_closes, noise_pct=0.0003)
        r2 = classifier.classify(c2, h2, l2)

        if r2.regime != r1.regime:
            assert r2.transition_from == r1.regime
            assert r2.regime_age == 1


class TestEdgeCases:
    def test_insufficient_data(self, classifier):
        closes = np.array([25000.0, 25001.0, 25002.0])
        highs = closes + 1
        lows = closes - 1

        result = classifier.classify(closes, highs, lows)
        assert result.regime == MarketRegime.UNKNOWN
        assert result.confidence == 0.0

    def test_flat_market(self, classifier):
        closes = np.full(150, 25000.0)
        highs = closes + 0.5
        lows = closes - 0.5

        result = classifier.classify(closes, highs, lows)
        assert result.regime in (
            MarketRegime.COMPRESSION,
            MarketRegime.UNKNOWN,
            MarketRegime.MEAN_REVERSION,
        )
