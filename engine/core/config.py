"""Kairos Engine — Config tuned for aggressive scalp entries."""

from dataclasses import dataclass, field
from engine.core.enums import MarketRegime


@dataclass(frozen=True)
class RegimeConfig:
    lookback: int = 20
    vol_percentile_window: int = 100
    efficiency_ratio_trend: float = 0.55
    efficiency_ratio_chop: float = 0.25
    persistence_trend: float = 0.65
    persistence_chop: float = 0.45
    atr_contraction_threshold: float = 0.70
    bbw_percentile_compressed: float = 20.0
    vol_percentile_exhaustion: float = 75.0
    efficiency_exhaustion_ceiling: float = 0.50
    entropy_chaotic: float = 0.95
    entropy_trending: float = 0.75
    min_confidence: float = 0.40


@dataclass(frozen=True)
class TradeFilterConfig:
    opening_range_skip_minutes: int = 30
    market_open_hour: int = 9
    market_open_minute: int = 15
    signal_cooldown_minutes: int = 30

    # Accept TREND_EXPANSION + TREND_EXHAUSTION
    # Exhaustion still has direction — scalpers profit from it
    allowed_regimes: tuple = (
        MarketRegime.TREND_EXPANSION,
        MarketRegime.TREND_EXHAUSTION,
        MarketRegime.COMPRESSION,
    )

    confirmation_candles: int = 2
    min_thesis_separation: float = 0.20
    min_signal_confidence: float = 0.60  # lowered from 0.60

    # Smart IV: only block in non-trending
    block_iv_in_non_trend: bool = True


@dataclass(frozen=True)
class RiskConfig:
    account_size: float = 15000.0
    max_risk_pct: float = 0.03
    max_positions: int = 1
    sl_pct: float = 0.30
    target_multiplier: float = 2.0
    max_premium_pct: float = 0.50


@dataclass(frozen=True)
class DisciplineConfig:
    max_trades_per_day: int = 2
    hard_max_trades: int = 3
    daily_loss_limit: float = 1200.0
    prime_start_hour: int = 13
    prime_start_minute: int = 0
    prime_end_hour: int = 15
    prime_end_minute: int = 0
    opening_avoid_minutes: int = 30
    closing_avoid_minutes: int = 10
    consecutive_loss_warning: int = 2
    detect_revenge_trades: bool = True


@dataclass(frozen=True)
class EngineConfig:
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trade_filter: TradeFilterConfig = field(default_factory=TradeFilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    discipline: DisciplineConfig = field(default_factory=DisciplineConfig)
