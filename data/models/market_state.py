"""
Kairos Engine — MarketState

The single object that flows through the pipeline.
Each engine reads what it needs and writes ONLY to its own namespace.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from engine.core.enums import (
    EntryWindow,
    MarketBias,
    MarketRegime,
    TradeState,
)


class RegimeMetrics(BaseModel):
    efficiency_ratio: float = 0.0
    directional_persistence: float = 0.0
    normalized_slope: float = 0.0
    directional_entropy: float = 0.0
    realized_volatility: float = 0.0
    volatility_percentile: float = 0.0
    atr_contraction: float = 0.0
    log_return_mean: float = 0.0
    regime_confidence: float = 0.0
    regime_age: int = 0
    transition_from: MarketRegime | None = None


class CompressionMetrics(BaseModel):
    is_compressed: bool = False
    compression_score: float = 0.0
    atr_contraction: float = 1.0
    range_contraction: float = 1.0
    rv_decay: float = 1.0
    bbw_percentile: float = 50.0
    compression_velocity: float = 0.0
    compression_half_life: float = float("inf")
    candles_compressed: int = 0
    pre_compression_vol: float = 0.0


class MarketState(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    symbol: str = "NIFTY"
    timestamp: datetime = Field(default_factory=datetime.now)

    last_price: float = 0.0
    candle_open: float = 0.0
    candle_high: float = 0.0
    candle_low: float = 0.0
    candle_close: float = 0.0
    candle_volume: float = 0.0

    regime: MarketRegime = MarketRegime.UNKNOWN
    bias: MarketBias = MarketBias.NEUTRAL
    regime_metrics: RegimeMetrics = Field(default_factory=RegimeMetrics)

    compression: CompressionMetrics = Field(default_factory=CompressionMetrics)

    structure: dict[str, Any] = Field(default_factory=dict)
    option_efficiency: dict[str, Any] = Field(default_factory=dict)
    flow: dict[str, Any] = Field(default_factory=dict)
    thesis: dict[str, Any] = Field(default_factory=dict)

    trade_state: TradeState = TradeState.NO_SETUP
    entry_window: EntryWindow = EntryWindow.CLOSED
