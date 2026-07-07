"""Tests for Setup Memory."""

import numpy as np
import pytest
from engine.pipeline.market_pipeline import MarketPipeline
from data.storage.setup_memory import SetupMemory


@pytest.fixture
def memory(tmp_path):
    db = str(tmp_path / "test_setups.db")
    mem = SetupMemory(db_path=db)
    yield mem
    mem.close()


@pytest.fixture
def sample_state():
    pipe = MarketPipeline()
    rng = np.random.default_rng(42)
    c = 25000.0 + np.cumsum(rng.normal(0, 3, 200))
    h = c + 2
    lo = c - 2
    o = np.roll(c, 1)
    o[0] = 25000
    v = rng.integers(5000, 50000, 200).astype(float)
    return pipe.process(c, h, lo, o, v)


class TestSetupMemory:
    def test_record_and_stats(self, memory, sample_state):
        rid = memory.record_setup(sample_state)
        assert rid > 0
        stats = memory.get_stats()
        assert stats["total_setups"] == 1

    def test_fill_forward(self, memory, sample_state):
        rid = memory.record_setup(sample_state)
        memory.fill_forward_outcomes(rid, 10.0, 25.0, 40.0, 80.0, 100.0)
        stats = memory.get_stats()
        assert stats["outcomes_filled"] == 1

    def test_multiple_setups(self, memory, sample_state):
        for _ in range(5):
            memory.record_setup(sample_state)
        stats = memory.get_stats()
        assert stats["total_setups"] == 5
