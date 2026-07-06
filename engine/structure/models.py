"""Kairos Engine — Structure models."""

from dataclasses import dataclass, field


@dataclass
class PriceLevel:
    """A single significant price level."""

    price: float
    label: str  # e.g. "PDH", "PDL", "VWAP", "OR_HIGH", "VOL_CLUSTER"
    strength: float = 1.0  # how many sources agree on this zone
    touches: int = 0  # how many times price tested this level


@dataclass
class MagneticZone:
    """
    Cluster of nearby price levels that form a magnetic zone.
    When multiple levels (PDH, VWAP, volume node) converge within
    a tight band, that zone becomes much more significant than
    any single level alone.
    """

    center: float  # weighted average of levels in the zone
    width: float  # price range of the zone
    levels: list[PriceLevel] = field(default_factory=list)
    confluence: int = 0  # number of levels in the zone
    total_strength: float = 0.0


@dataclass
class StructureResult:
    """Complete structure analysis output."""

    # All detected raw levels
    levels: list[PriceLevel] = field(default_factory=list)

    # Clustered magnetic zones
    zones: list[MagneticZone] = field(default_factory=list)

    # Location metrics
    nearest_zone_distance: float = 0.0  # points to nearest zone
    nearest_zone_distance_pct: float = 0.0  # as % of price
    nearest_zone: MagneticZone | None = None

    # Position relative to structure
    above_zones: int = 0  # how many zones below current price
    below_zones: int = 0  # how many zones above current price
    inside_zone: bool = False  # price is within a magnetic zone

    # Structure quality
    structure_score: float = 0.0  # 0-1: how well-defined the structure is
