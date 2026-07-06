"""Kairos Engine — MarketState (complete)."""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
from engine.core.enums import (
    EntryWindow,
    IVState,
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


class StructureMetrics(BaseModel):
    structure_score: float = 0.0
    nearest_zone_distance: float = 0.0
    nearest_zone_distance_pct: float = 0.0
    nearest_zone_center: float = 0.0
    nearest_zone_confluence: int = 0
    above_zones: int = 0
    below_zones: int = 0
    inside_zone: bool = False
    total_levels: int = 0
    total_zones: int = 0


class IVMetrics(BaseModel):
    state: IVState = IVState.NORMAL
    current_iv: float = 0.0
    iv_percentile: float = 50.0
    iv_velocity: float = 0.0


class OptionMetrics(BaseModel):
    is_efficient: bool = False
    delta_acceleration: float = 0.0
    gamma_theta_ratio: float = 0.0
    theta_survival_minutes: float = 9999.0
    move_feasibility: float = 0.0
    strike: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0


class FlowMetrics(BaseModel):
    pressure_score: float = 0.0
    level_tests: int = 0
    test_frequency: float = 0.0
    pullback_shrinking: bool = False
    aggression_ratio: float = 0.5
    pressure_asymmetry: float = 0.0
    liquidity_thinning: bool = False


class ThesisMetrics(BaseModel):
    primary_bias: MarketBias = MarketBias.NEUTRAL
    primary_score: float = 0.0
    counter_score: float = 0.0
    separation: float = 0.0
    thesis_valid: bool = False


class GateInfo(BaseModel):
    name: str = ""
    ready: bool = False
    reason: str = ""


class StateMachineMetrics(BaseModel):
    gates: list[GateInfo] = Field(default_factory=list)
    estimated_window_seconds: int = 0
    thesis_survival_minutes: float = 0.0


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
    structure: StructureMetrics = Field(default_factory=StructureMetrics)
    iv: IVMetrics = Field(default_factory=IVMetrics)
    option: OptionMetrics = Field(default_factory=OptionMetrics)
    flow: FlowMetrics = Field(default_factory=FlowMetrics)
    thesis: ThesisMetrics = Field(default_factory=ThesisMetrics)
    state_machine: StateMachineMetrics = Field(default_factory=StateMachineMetrics)

    trade_state: TradeState = TradeState.NO_SETUP
    entry_window: EntryWindow = EntryWindow.CLOSED
