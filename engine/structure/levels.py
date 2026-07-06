"""
Kairos Engine — Price Level Detection & Magnetic Zone Clustering

Detects significant price levels from OHLC data:
- Previous Day High / Low (PDH/PDL)
- Opening Range High / Low
- Session VWAP
- Volume-clustered price nodes
- Swing highs / lows

Then clusters nearby levels into Magnetic Zones.

WHY MAGNETIC ZONES:
A single PDH at 25100 is interesting.
PDH at 25100 + VWAP at 25095 + volume node at 25105 = magnetic zone
at 25100 ± 5 with confluence 3. Price is drawn to this zone and
will react strongly when it gets there.

For options: knowing you're 50 points from a 3-confluence zone
tells you the expected move has a likely pause/reversal point,
which directly affects move feasibility and theta survival.
"""

import numpy as np

from engine.core.types import FloatArray
from engine.structure.models import MagneticZone, PriceLevel, StructureResult


class StructureAnalyzer:
    def __init__(
        self,
        cluster_threshold_pct: float = 0.0015,  # levels within 0.15% are clustered
        opening_range_candles: int = 15,  # first 15 candles = opening range
        swing_lookback: int = 5,  # candles each side for swing detection
        volume_node_bins: int = 50,  # histogram bins for volume profile
        min_volume_node_pct: float = 0.70,  # top 30% volume bins become nodes
    ):
        self.cluster_threshold_pct = cluster_threshold_pct
        self.opening_range_candles = opening_range_candles
        self.swing_lookback = swing_lookback
        self.volume_node_bins = volume_node_bins
        self.min_volume_node_pct = min_volume_node_pct

    def analyze(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        volumes: FloatArray,
        current_price: float | None = None,
    ) -> StructureResult:
        n = len(closes)
        if n < self.opening_range_candles + 10:
            return StructureResult()

        if current_price is None:
            current_price = float(closes[-1])

        # --- Detect all raw levels ---
        levels: list[PriceLevel] = []

        levels.extend(self._previous_day_levels(highs, lows, closes))
        levels.extend(self._opening_range(highs, lows))
        levels.extend(self._vwap(closes, volumes))
        levels.extend(self._volume_nodes(closes, volumes))
        levels.extend(self._swing_points(highs, lows))

        if not levels:
            return StructureResult()

        # --- Cluster into magnetic zones ---
        zones = self._cluster_levels(levels, current_price)

        # --- Location analysis ---
        nearest_zone = None
        nearest_dist = float("inf")
        nearest_dist_pct = 0.0
        above = 0
        below = 0
        inside = False

        for zone in zones:
            dist = abs(current_price - zone.center)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_zone = zone

            half_w = zone.width / 2
            if current_price > zone.center + half_w:
                above += 1
            elif current_price < zone.center - half_w:
                below += 1
            else:
                inside = True

        if current_price > 0:
            nearest_dist_pct = nearest_dist / current_price * 100

        # --- Structure score ---
        # More zones + higher confluence = better defined structure
        max_confluence = max((z.confluence for z in zones), default=0)
        zone_count_score = min(len(zones) / 5.0, 1.0)
        confluence_score = min(max_confluence / 4.0, 1.0)
        structure_score = 0.5 * zone_count_score + 0.5 * confluence_score

        return StructureResult(
            levels=levels,
            zones=zones,
            nearest_zone_distance=round(nearest_dist, 2),
            nearest_zone_distance_pct=round(nearest_dist_pct, 4),
            nearest_zone=nearest_zone,
            above_zones=above,
            below_zones=below,
            inside_zone=inside,
            structure_score=round(structure_score, 4),
        )

    def _previous_day_levels(
        self, highs: FloatArray, lows: FloatArray, closes: FloatArray
    ) -> list[PriceLevel]:
        """
        PDH/PDL from the first half of data (simulating 'previous session').
        In live trading this would come from actual previous day data.
        """
        n = len(highs)
        split = n // 2
        if split < 10:
            return []

        prev_highs = highs[:split]
        prev_lows = lows[:split]

        pdh = float(np.max(prev_highs))
        pdl = float(np.min(prev_lows))
        pdc = float(closes[split - 1])

        return [
            PriceLevel(price=pdh, label="PDH", strength=1.5),
            PriceLevel(price=pdl, label="PDL", strength=1.5),
            PriceLevel(price=pdc, label="PDC", strength=1.0),
        ]

    def _opening_range(self, highs: FloatArray, lows: FloatArray) -> list[PriceLevel]:
        """Opening range from first N candles."""
        or_highs = highs[: self.opening_range_candles]
        or_lows = lows[: self.opening_range_candles]

        or_high = float(np.max(or_highs))
        or_low = float(np.min(or_lows))

        return [
            PriceLevel(price=or_high, label="OR_HIGH", strength=1.2),
            PriceLevel(price=or_low, label="OR_LOW", strength=1.2),
        ]

    def _vwap(self, closes: FloatArray, volumes: FloatArray) -> list[PriceLevel]:
        """Volume-weighted average price."""
        if len(volumes) == 0 or np.sum(volumes) == 0:
            return []

        vwap = float(np.sum(closes * volumes) / np.sum(volumes))
        return [PriceLevel(price=vwap, label="VWAP", strength=2.0)]

    def _volume_nodes(
        self, closes: FloatArray, volumes: FloatArray
    ) -> list[PriceLevel]:
        """High-volume price nodes from volume profile."""
        if len(closes) < 20:
            return []

        price_min = float(np.min(closes))
        price_max = float(np.max(closes))

        if price_max - price_min < 1.0:
            return []

        bin_edges = np.linspace(price_min, price_max, self.volume_node_bins + 1)
        bin_indices = np.digitize(closes, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.volume_node_bins - 1)

        vol_profile = np.zeros(self.volume_node_bins)
        for i, bi in enumerate(bin_indices):
            vol_profile[bi] += volumes[i]

        # Find high-volume nodes (top percentile)
        threshold = np.percentile(vol_profile, self.min_volume_node_pct * 100)
        levels = []

        for i in range(self.volume_node_bins):
            if vol_profile[i] >= threshold:
                center = (bin_edges[i] + bin_edges[i + 1]) / 2
                levels.append(
                    PriceLevel(
                        price=round(float(center), 2),
                        label="VOL_NODE",
                        strength=1.0,
                    )
                )

        return levels

    def _swing_points(self, highs: FloatArray, lows: FloatArray) -> list[PriceLevel]:
        """Detect swing highs and swing lows."""
        lb = self.swing_lookback
        n = len(highs)
        levels = []

        if n < 2 * lb + 1:
            return levels

        for i in range(lb, n - lb):
            # Swing high: higher than all neighbors
            if highs[i] == np.max(highs[i - lb : i + lb + 1]):
                levels.append(
                    PriceLevel(
                        price=float(highs[i]),
                        label="SWING_H",
                        strength=0.8,
                    )
                )
            # Swing low
            if lows[i] == np.min(lows[i - lb : i + lb + 1]):
                levels.append(
                    PriceLevel(
                        price=float(lows[i]),
                        label="SWING_L",
                        strength=0.8,
                    )
                )

        return levels

    def _cluster_levels(
        self, levels: list[PriceLevel], reference_price: float
    ) -> list[MagneticZone]:
        """
        Cluster nearby levels into magnetic zones using simple
        distance-based agglomerative clustering.
        """
        if not levels:
            return []

        threshold = reference_price * self.cluster_threshold_pct
        sorted_levels = sorted(levels, key=lambda lv: lv.price)

        zones: list[MagneticZone] = []
        current_group: list[PriceLevel] = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            if sorted_levels[i].price - current_group[-1].price <= threshold:
                current_group.append(sorted_levels[i])
            else:
                zones.append(self._make_zone(current_group))
                current_group = [sorted_levels[i]]

        zones.append(self._make_zone(current_group))

        # Sort by total strength descending
        zones.sort(key=lambda z: z.total_strength, reverse=True)
        return zones

    def _make_zone(self, levels: list[PriceLevel]) -> MagneticZone:
        """Create a MagneticZone from a group of clustered levels."""
        prices = [lv.price for lv in levels]
        strengths = [lv.strength for lv in levels]
        total_strength = sum(strengths)

        # Strength-weighted center
        center = sum(p * s for p, s in zip(prices, strengths)) / total_strength

        return MagneticZone(
            center=round(center, 2),
            width=round(max(prices) - min(prices), 2) if len(prices) > 1 else 0.0,
            levels=levels,
            confluence=len(levels),
            total_strength=round(total_strength, 2),
        )
