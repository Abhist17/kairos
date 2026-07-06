"""
Kairos Engine — Core Enumerations

Every discrete label the engine produces lives here.
No magic strings anywhere else in the codebase.
"""

from enum import Enum


class MarketRegime(str, Enum):
    TREND_EXPANSION = "TREND_EXPANSION"
    TREND_EXHAUSTION = "TREND_EXHAUSTION"
    COMPRESSION = "COMPRESSION"
    MEAN_REVERSION = "MEAN_REVERSION"
    CHAOTIC = "CHAOTIC"
    UNKNOWN = "UNKNOWN"


class MarketBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class IVState(str, Enum):
    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    EXPANDING = "EXPANDING"
    OVEREXPANDED = "OVEREXPANDED"
    COLLAPSING = "COLLAPSING"


class TradeState(str, Enum):
    NO_SETUP = "NO_SETUP"
    STRUCTURAL_INTEREST = "STRUCTURAL_INTEREST"
    COMPRESSION = "COMPRESSION"
    PRESSURE_BUILDING = "PRESSURE_BUILDING"
    OPTION_EFFICIENCY = "OPTION_EFFICIENCY"
    FLOW_CONFIRMATION = "FLOW_CONFIRMATION"
    ENTRY_WINDOW_OPEN = "ENTRY_WINDOW_OPEN"


class EntryWindow(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    EXPIRED = "EXPIRED"
