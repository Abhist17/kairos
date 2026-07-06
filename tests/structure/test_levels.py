"""Tests for Structure Analyzer."""

import numpy as np
import pytest

from engine.structure.levels import StructureAnalyzer


@pytest.fixture
def analyzer():
    return StructureAnalyzer()


def _make_session(base=25000.0, n=200, seed=42):
    rng = np.random.default_rng(seed)
    moves = rng.normal(0, 3, n)
    closes = base + np.cumsum(moves)
    highs = closes + np.abs(rng.normal(0, 1, n)) * 2 + 0.5
    lows = closes - np.abs(rng.normal(0, 1, n)) * 2 - 0.5
    volumes = rng.integers(5000, 50000, n).astype(float)
    return closes, highs, lows, volumes


class TestLevelDetection:
    def test_detects_pdh_pdl(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        labels = [lv.label for lv in result.levels]
        assert "PDH" in labels
        assert "PDL" in labels

    def test_detects_vwap(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        labels = [lv.label for lv in result.levels]
        assert "VWAP" in labels

    def test_detects_opening_range(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        labels = [lv.label for lv in result.levels]
        assert "OR_HIGH" in labels
        assert "OR_LOW" in labels

    def test_detects_swing_points(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        labels = [lv.label for lv in result.levels]
        assert "SWING_H" in labels or "SWING_L" in labels


class TestMagneticZones:
    def test_zones_created(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        assert len(result.zones) > 0

    def test_confluence_counts(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        for zone in result.zones:
            assert zone.confluence == len(zone.levels)
            assert zone.confluence >= 1

    def test_zones_sorted_by_strength(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        strengths = [z.total_strength for z in result.zones]
        assert strengths == sorted(strengths, reverse=True)


class TestLocation:
    def test_nearest_zone_found(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        assert result.nearest_zone is not None
        assert result.nearest_zone_distance >= 0

    def test_above_below_count(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        total = (
            result.above_zones + result.below_zones + (1 if result.inside_zone else 0)
        )
        assert total <= len(result.zones)

    def test_structure_score_bounded(self, analyzer):
        closes, highs, lows, volumes = _make_session()
        result = analyzer.analyze(closes, highs, lows, volumes)

        assert 0.0 <= result.structure_score <= 1.0


class TestEdgeCases:
    def test_insufficient_data(self, analyzer):
        closes = np.array([25000.0] * 5)
        highs = closes + 1
        lows = closes - 1
        volumes = np.ones(5) * 1000

        result = analyzer.analyze(closes, highs, lows, volumes)
        assert len(result.levels) == 0
        assert len(result.zones) == 0
