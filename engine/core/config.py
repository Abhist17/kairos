"""Kairos Engine — Configuration. All thresholds with rationale."""

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
    # Fix 2: Opening range extended to 45 min (9:15-10:00)
    # Backtest showed 50% loss rate in first 30 min, still lossy at 30-45
    opening_range_skip_minutes: int = 45
    market_open_hour: int = 9
    market_open_minute: int = 15

    # Fix 1: Cooldown applied to ALL paths (was only Path A)
    signal_cooldown_minutes: int = 10

    # Regime whitelist for signal generation
    allowed_regimes: tuple = (
        MarketRegime.TREND_EXPANSION,
        MarketRegime.COMPRESSION,
    )

    # Fix 7: Multi-timeframe must agree
    require_mtf_alignment: bool = True
    mtf_resample_factor: int = 7  # 2m * 7 = ~15 min higher TF

    # Momentum confirmation candles
    confirmation_candles: int = 2

    # Fix 1: Minimum thesis separation raised
    min_thesis_separation: float = 0.20

    # Minimum signal confidence
    min_signal_confidence: float = 0.60

    # Block overexpanded IV (options too expensive to buy)
    block_overexpanded_iv: bool = True

    # Fix 4: Move feasibility floor
    min_move_feasibility: float = 1.0

    # Fix 3: Minimum regime age for Path B (trend must be established)
    # Prevents signaling on a "trend" that just started 2 candles ago
    min_regime_age_trend: int = 5  # ~10 min on 2m candles


@dataclass(frozen=True)
class RiskConfig:
    """Fix 6: Position sizing and risk management."""
    # Account size in INR
    account_size: float = 100000.0

    # Max risk per trade as % of account
    max_risk_pct: float = 0.02  # 2% = ₹2000 on 1L account

    # Max simultaneous open positions
    max_positions: int = 2

    # Stoploss as % of premium
    sl_pct: float = 0.30  # 30% of premium

    # Target as multiple of risk
    target_multiplier: float = 2.0  # 1:2 R:R

    # Max premium as % of account (don't blow on one trade)
    max_premium_pct: float = 0.05  # 5% of account per trade


@dataclass(frozen=True)
class EngineConfig:
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trade_filter: TradeFilterConfig = field(default_factory=TradeFilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)