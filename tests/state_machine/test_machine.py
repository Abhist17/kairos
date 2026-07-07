"""Tests for Entry State Machine."""

import pytest
from engine.core.enums import TradeState, EntryWindow, MarketRegime
from engine.state_machine.machine import EntryStateMachine


@pytest.fixture
def sm():
    return EntryStateMachine()


class TestStateMachine:
    def test_all_gates_open(self, sm):
        r = sm.evaluate(
            regime=MarketRegime.TREND_EXPANSION,
            structure_score=0.7,
            nearest_zone_dist_pct=0.1,
            compression_score=0.5,
            is_compressed=True,
            pressure_score=0.5,
            option_efficient=True,
            theta_survival=60.0,
            thesis_valid=True,
            thesis_separation=0.3,
        )
        assert r.state == TradeState.ENTRY_WINDOW_OPEN
        assert r.entry_window == EntryWindow.OPEN
        assert r.estimated_window_seconds > 0
        assert all(g.ready for g in r.gates)

    def test_no_structure(self, sm):
        r = sm.evaluate(
            regime=MarketRegime.TREND_EXPANSION,
            structure_score=0.1,
            nearest_zone_dist_pct=0.5,
            compression_score=0.5,
            is_compressed=True,
            pressure_score=0.5,
            option_efficient=True,
            theta_survival=60.0,
            thesis_valid=True,
            thesis_separation=0.3,
        )
        assert r.state == TradeState.NO_SETUP
        assert r.entry_window == EntryWindow.CLOSED

    def test_gates_sequential(self, sm):
        """If structure fails, nothing downstream should be ready."""
        r = sm.evaluate(
            regime=MarketRegime.CHAOTIC,
            structure_score=0.1,
            nearest_zone_dist_pct=0.8,
            compression_score=0.8,
            is_compressed=True,
            pressure_score=0.8,
            option_efficient=True,
            theta_survival=60.0,
            thesis_valid=True,
            thesis_separation=0.5,
        )
        assert r.gates[0].ready is False
        for g in r.gates[1:]:
            assert g.ready is False

    def test_partial_gates(self, sm):
        r = sm.evaluate(
            regime=MarketRegime.COMPRESSION,
            structure_score=0.5,
            nearest_zone_dist_pct=0.1,
            compression_score=0.5,
            is_compressed=True,
            pressure_score=0.5,
            option_efficient=False,
            theta_survival=60.0,
            thesis_valid=True,
            thesis_separation=0.3,
        )
        assert r.state == TradeState.PRESSURE_BUILDING
        assert r.entry_window == EntryWindow.CLOSED

    def test_window_seconds_scales(self, sm):
        r = sm.evaluate(
            regime=MarketRegime.COMPRESSION,
            structure_score=0.7,
            nearest_zone_dist_pct=0.1,
            compression_score=0.9,
            is_compressed=True,
            pressure_score=0.5,
            option_efficient=True,
            theta_survival=30.0,
            thesis_valid=True,
            thesis_separation=0.3,
        )
        assert r.estimated_window_seconds >= 45
