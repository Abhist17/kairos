"""Tests for the full MarketPipeline."""

import numpy as np
import pytest
from engine.pipeline.market_pipeline import MarketPipeline
from engine.core.enums import MarketRegime, TradeState


@pytest.fixture
def pipeline():
    return MarketPipeline()


def _synth(n=200, drift=0.0, noise=3.0, seed=42):
    rng = np.random.default_rng(seed)
    base = 25000.0
    closes = base + np.cumsum(np.ones(n) * drift + rng.normal(0, noise, n))
    highs = closes + np.abs(rng.normal(0, 1, n)) * 2 + 0.5
    lows = closes - np.abs(rng.normal(0, 1, n)) * 2 - 0.5
    opens = np.roll(closes, 1)
    opens[0] = base
    volumes = rng.integers(5000, 50000, n).astype(float)
    return closes, highs, lows, opens, volumes


class TestPipelineBasic:
    def test_returns_market_state(self, pipeline):
        c, h, lo, o, v = _synth()
        state = pipeline.process(c, h, lo, o, v)
        assert state.symbol == "NIFTY"
        assert state.last_price > 0
        assert state.regime in MarketRegime
        assert state.trade_state in TradeState

    def test_trending_data(self, pipeline):
        c, h, lo, o, v = _synth(n=200, drift=3.0, noise=0.5)
        state = pipeline.process(c, h, lo, o, v)
        assert state.regime == MarketRegime.TREND_EXPANSION

    def test_all_metrics_populated(self, pipeline):
        c, h, lo, o, v = _synth()
        state = pipeline.process(c, h, lo, o, v)
        assert state.regime_metrics.efficiency_ratio >= 0
        assert state.compression.compression_score >= 0
        assert state.structure.structure_score >= 0
        assert 0 <= state.flow.aggression_ratio <= 1
        assert state.thesis.primary_score >= 0
        assert len(state.state_machine.gates) == 6

    def test_pipeline_min_candles(self, pipeline):
        assert pipeline.min_candles == 120
