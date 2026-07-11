"""Kairos Engine — Configuration tuned for scalping."""

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
    # Opening range: 20 min (9:15-9:35)
    # First 15 min is pure chaos, 15-20 min is transition
    # After 9:35 scalp setups start forming
    opening_range_skip_minutes: int = 30
    market_open_hour: int = 9
    market_open_minute: int = 15

    # 10 min cooldown between signals (prevents duplicate signals)
    signal_cooldown_minutes: int = 10

    # Only signal in trending or compressed regimes
    allowed_regimes: tuple = (
        MarketRegime.TREND_EXPANSION,
        MarketRegime.COMPRESSION,
    )

    # MTF alignment
    require_mtf_alignment: bool = True
    mtf_resample_factor: int = 7

    # 2 candle momentum confirmation
    confirmation_candles: int = 2

    # Thesis separation
    min_thesis_separation: float = 0.20

    # Minimum confidence
    min_signal_confidence: float = 0.60

    # IV filter: SMART — not a blanket block
    # Only block overexpanded IV when regime is NOT trending
    # During TREND_EXPANSION, high IV is expected and the move compensates
    block_iv_in_non_trend: bool = True

    # Move feasibility floor
    min_move_feasibility: float = 0.8  # lowered for scalps (smaller targets)

    # Regime age for Path B
    min_regime_age_trend: int = 3


@dataclass(frozen=True)
class RiskConfig:
    """Tuned for ₹10-20K scalp account."""

    account_size: float = 15000.0  # ₹15K midpoint
    max_risk_pct: float = 0.03  # 3% per trade = ₹450 max risk
    max_positions: int = 1  # one trade at a time for small account
    sl_pct: float = 0.30  # 30% of premium as SL
    target_multiplier: float = 2.0  # 1:2 R:R
    max_premium_pct: float = 0.50  # up to 50% of account on one trade
    # ₹15K × 50% = ₹7500 max premium


@dataclass(frozen=True)
class EngineConfig:
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trade_filter: TradeFilterConfig = field(default_factory=TradeFilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
