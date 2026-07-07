"""Tests for Backtest Engine."""

import numpy as np
import pytest
from backtest.engine import BacktestEngine


@pytest.fixture
def bt_engine(tmp_path):
    db = str(tmp_path / "test_bt.db")
    engine = BacktestEngine(db_path=db, step=5, record_all=True)
    yield engine
    engine.close()


def _synth(n=250, seed=42):
    rng = np.random.default_rng(seed)
    base = 25000.0
    c = base + np.cumsum(rng.normal(2, 1.5, n))
    h = c + np.abs(rng.normal(0, 1, n)) * 2 + 0.5
    lo = c - np.abs(rng.normal(0, 1, n)) * 2 - 0.5
    o = np.roll(c, 1)
    o[0] = base
    v = rng.integers(5000, 50000, n).astype(float)
    return c, h, lo, o, v


class TestBacktest:
    def test_runs_and_records(self, bt_engine):
        c, h, lo, o, v = _synth()
        report = bt_engine.run(c, h, lo, o, v)
        assert report["candles_processed"] > 0
        assert report["setups_recorded"] > 0
        assert report["outcomes_filled"] > 0

    def test_forward_outcomes_filled(self, bt_engine):
        c, h, lo, o, v = _synth()
        report = bt_engine.run(c, h, lo, o, v)
        stats = report["db_stats"]
        assert stats["outcomes_filled"] > 0
