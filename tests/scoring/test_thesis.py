"""Tests for Thesis Engine."""

import pytest
from engine.core.enums import MarketBias, MarketRegime
from engine.scoring.thesis import ThesisEngine


@pytest.fixture
def engine():
    return ThesisEngine()


class TestThesis:
    def test_strong_bullish(self, engine):
        r = engine.score(
            regime=MarketRegime.TREND_EXPANSION,
            bias=MarketBias.BULLISH,
            regime_confidence=0.8,
            compression_score=0.5,
            is_compressed=True,
            structure_score=0.7,
            nearest_zone_dist_pct=0.1,
            pressure_score=0.6,
            aggression_ratio=0.8,
            iv_percentile=20.0,
            option_efficient=True,
        )
        assert r.primary_bias == MarketBias.BULLISH
        assert r.separation > 0.15
        assert r.thesis_valid

    def test_conflicting_signals(self, engine):
        r = engine.score(
            regime=MarketRegime.CHAOTIC,
            bias=MarketBias.NEUTRAL,
            regime_confidence=0.4,
            compression_score=0.0,
            is_compressed=False,
            structure_score=0.3,
            nearest_zone_dist_pct=0.5,
            pressure_score=0.1,
            aggression_ratio=0.5,
            iv_percentile=50.0,
            option_efficient=False,
        )
        assert r.separation < 0.15

    def test_separation_bounded(self, engine):
        r = engine.score(
            regime=MarketRegime.TREND_EXPANSION,
            bias=MarketBias.BEARISH,
            regime_confidence=0.9,
            compression_score=0.8,
            is_compressed=True,
            structure_score=0.9,
            nearest_zone_dist_pct=0.05,
            pressure_score=0.9,
            aggression_ratio=0.1,
            iv_percentile=10.0,
            option_efficient=True,
        )
        assert 0 <= r.primary_score <= 1
        assert 0 <= r.counter_score <= 1
