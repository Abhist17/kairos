"""Tests for OI Gravity and Gamma Map."""

import numpy as np
from engine.options.oi_gravity import OIGravityTracker
from engine.options.gamma_map import GammaGravityMap


class TestOIGravity:
    def test_synthetic_analysis(self):
        tracker = OIGravityTracker()
        result = tracker.analyze(25000.0)
        assert result.call_gravity > 0
        assert result.put_gravity > 0
        assert result.max_pain > 0
        assert result.pcr_overall > 0
        assert len(result.strike_data) > 0

    def test_velocity_tracks(self):
        tracker = OIGravityTracker()
        tracker.analyze(25000.0)
        r2 = tracker.analyze(25050.0)
        assert r2.gravity_velocity != 0

    def test_custom_oi(self):
        tracker = OIGravityTracker()
        strikes = np.array([24800, 24900, 25000, 25100, 25200], dtype=float)
        call_oi = np.array([1000, 5000, 50000, 20000, 3000], dtype=float)
        put_oi = np.array([3000, 20000, 50000, 5000, 1000], dtype=float)
        result = tracker.analyze(25000.0, call_oi, put_oi, strikes)
        assert abs(result.max_pain - 25000) <= 100


class TestGammaMap:
    def test_estimate_runs(self):
        gm = GammaGravityMap()
        result = gm.estimate(25000.0)
        assert len(result.gamma_concentration) > 0
        assert result.max_gamma_strike > 0
        assert "ESTIMATED" in result.regime_note

    def test_gamma_values_positive(self):
        gm = GammaGravityMap()
        result = gm.estimate(25000.0)
        for sg in result.gamma_concentration:
            assert sg.estimated_gamma >= 0
