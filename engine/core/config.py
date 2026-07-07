"""
Kairos Engine — Configuration
All tunable thresholds with rationale.
"""

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
    """
    Filters learned from backtest analysis.
    Each one exists because of a specific failure mode.
    """

    # --- Opening range filter ---
    # First 15 min (9:15-9:30) are chaotic. Signals here had 0% win rate.
    # Skip the first N minutes of the session.
    opening_range_skip_minutes: int = 30

    # Market open hour (IST 24h). Used to detect opening range.
    market_open_hour: int = 9
    market_open_minute: int = 15

    # --- Signal cooldown ---
    # Backtest showed 3 signals in 4 minutes = same setup duplicated.
    # Wait at least N minutes between signals.
    signal_cooldown_minutes: int = 10

    # --- Regime filter ---
    # Mean reversion had 0% accuracy. Only signal in these regimes.
    allowed_regimes: tuple = (
        MarketRegime.TREND_EXPANSION,
        MarketRegime.COMPRESSION,
    )

    # --- Momentum confirmation ---
    # Engine entered before the move started. Require N consecutive
    # candles moving in the thesis direction AFTER gates open.
    confirmation_candles: int = 2

    # --- Minimum thesis separation ---
    # Backtest showed low-separation signals were wrong.
    # Raise from 0.10 to 0.20.
    min_thesis_separation: float = 0.20

    # --- Minimum confidence ---
    # Only signal above this confidence level.
    min_signal_confidence: float = 0.60

    # --- Volatility filter ---
    # Don't signal when IV is overexpanded (options too expensive to buy).
    block_overexpanded_iv: bool = True

    # --- Move feasibility floor ---
    # Backtest: signals with feasibility < 1.0 were mostly losses.
    min_move_feasibility: float = 1.0


@dataclass(frozen=True)
class EngineConfig:
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    trade_filter: TradeFilterConfig = field(default_factory=TradeFilterConfig)