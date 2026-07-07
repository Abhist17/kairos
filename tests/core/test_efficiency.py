"""Tests for Option Efficiency Engine."""

import pytest
from engine.options.efficiency import OptionEfficiencyEngine


@pytest.fixture
def engine():
    return OptionEfficiencyEngine()


class TestOptionEfficiency:
    def test_efficient_setup(self, engine):
        r = engine.evaluate(
            spot=25000,
            strike=25000,
            delta=0.50,
            gamma=0.05,
            theta=-1.0,
            vega=5.0,
            expected_move=50.0,
            required_move=30.0,
            dte_minutes=300,
        )
        assert r.is_efficient
        assert r.move_feasibility > 1.0
        assert r.delta_acceleration > 0

    def test_inefficient_far_otm(self, engine):
        r = engine.evaluate(
            spot=25000,
            strike=25500,
            delta=0.05,
            gamma=0.001,
            theta=-0.5,
            vega=1.0,
            expected_move=20.0,
            required_move=500.0,
            dte_minutes=100,
        )
        assert not r.is_efficient
        assert r.move_feasibility < 1.0

    def test_theta_survival_finite(self, engine):
        r = engine.evaluate(
            spot=25000,
            strike=25000,
            delta=0.50,
            gamma=0.05,
            theta=-5.0,
            vega=5.0,
            expected_move=50.0,
            required_move=30.0,
            dte_minutes=300,
        )
        assert r.theta_survival_minutes < 9999

    def test_zero_theta_infinite_survival(self, engine):
        r = engine.evaluate(
            spot=25000,
            strike=25000,
            delta=0.50,
            gamma=0.05,
            theta=0.0,
            vega=5.0,
            expected_move=50.0,
            required_move=30.0,
            dte_minutes=300,
        )
        assert r.theta_survival_minutes >= 9999
