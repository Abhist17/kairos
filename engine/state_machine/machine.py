"""
Kairos Engine — Entry State Machine V2

Two entry paths:

PATH A — BREAKOUT (original):
  Structure → Compression → Pressure → Option_Eff → Thesis → Entry
  For: price near a zone, vol compressed, pressure building

PATH B — TREND FOLLOWING (new):
  Strong Trend → Momentum → Thesis Strong → Option_Eff → Entry
  For: market already moving hard, catch the wave
  Bypasses structure/compression/pressure gates

Today's SENSEX crash (Jul 8) was a textbook Path B:
  TREND_EXPANSION + BEARISH + separation 0.30+ for 30 minutes
  but 0/6 gates because no zone nearby. Path B fixes this.
"""

from engine.core.enums import TradeState, EntryWindow, MarketRegime, MarketBias
from engine.state_machine.models import StateMachineResult, GateStatus


class EntryStateMachine:
    def __init__(
        self,
        # Path A thresholds
        structure_min: float = 0.3,
        structure_max_dist_pct: float = 0.3,
        compression_min: float = 0.2,
        pressure_min: float = 0.2,
        thesis_min_separation: float = 0.1,
        favorable_regimes: tuple = (
            MarketRegime.COMPRESSION,
            MarketRegime.TREND_EXPANSION,
        ),
        # Path B thresholds
        trend_min_confidence: float = 0.60,
        trend_min_separation: float = 0.25,
        trend_min_er: float = 0.70,
        trend_regimes: tuple = (MarketRegime.TREND_EXPANSION,),
    ):
        self.structure_min = structure_min
        self.structure_max_dist_pct = structure_max_dist_pct
        self.compression_min = compression_min
        self.pressure_min = pressure_min
        self.thesis_min_separation = thesis_min_separation
        self.favorable_regimes = favorable_regimes

        self.trend_min_confidence = trend_min_confidence
        self.trend_min_separation = trend_min_separation
        self.trend_min_er = trend_min_er
        self.trend_regimes = trend_regimes

    def evaluate(
        self,
        regime: MarketRegime,
        bias: MarketBias = MarketBias.NEUTRAL,
        regime_confidence: float = 0.0,
        efficiency_ratio: float = 0.0,
        structure_score: float = 0.0,
        nearest_zone_dist_pct: float = 0.0,
        compression_score: float = 0.0,
        is_compressed: bool = False,
        pressure_score: float = 0.0,
        option_efficient: bool = False,
        theta_survival: float = 0.0,
        thesis_valid: bool = False,
        thesis_separation: float = 0.0,
    ) -> StateMachineResult:

        # Try Path B first (trend following) — catches fast moves
        path_b = self._evaluate_path_b(
            regime, bias, regime_confidence, efficiency_ratio,
            option_efficient, theta_survival, thesis_valid,
            thesis_separation,
        )

        if path_b.entry_window == EntryWindow.OPEN:
            return path_b

        # Fall back to Path A (breakout)
        return self._evaluate_path_a(
            regime, structure_score, nearest_zone_dist_pct,
            compression_score, is_compressed, pressure_score,
            option_efficient, theta_survival, thesis_valid,
            thesis_separation,
        )

    def _evaluate_path_b(
        self,
        regime: MarketRegime,
        bias: MarketBias,
        regime_confidence: float,
        efficiency_ratio: float,
        option_efficient: bool,
        theta_survival: float,
        thesis_valid: bool,
        thesis_separation: float,
    ) -> StateMachineResult:
        """
        PATH B — Trend Following

        Gate 1: STRONG_TREND — regime is TREND_EXPANSION with high confidence + ER
        Gate 2: DIRECTION — bias is not NEUTRAL
        Gate 3: THESIS_STRONG — separation above trend threshold
        Gate 4: OPTION_EFF — option is mathematically efficient
        Gate 5: ENTRY — all above pass
        """
        gates: list[GateStatus] = []
        current_state = TradeState.NO_SETUP

        # Gate 1: Strong trend
        trend_ok = (
            regime in self.trend_regimes
            and regime_confidence >= self.trend_min_confidence
            and efficiency_ratio >= self.trend_min_er
        )
        reason = ""
        if not trend_ok:
            if regime not in self.trend_regimes:
                reason = f"regime {regime.value} not trending"
            elif regime_confidence < self.trend_min_confidence:
                reason = f"confidence {regime_confidence:.2f} < {self.trend_min_confidence}"
            else:
                reason = f"ER {efficiency_ratio:.3f} < {self.trend_min_er}"
        gates.append(GateStatus("STRONG_TREND", trend_ok, reason))

        if trend_ok:
            current_state = TradeState.STRUCTURAL_INTEREST

        # Gate 2: Clear direction
        dir_ok = bias != MarketBias.NEUTRAL
        reason = "neutral bias" if not dir_ok else ""
        gates.append(GateStatus("DIRECTION", dir_ok and trend_ok, reason))

        if trend_ok and dir_ok:
            current_state = TradeState.COMPRESSION

        # Gate 3: Strong thesis
        thesis_ok = (
            thesis_valid
            and thesis_separation >= self.trend_min_separation
        )
        reason = ""
        if not thesis_ok:
            reason = f"separation {thesis_separation:.3f} < {self.trend_min_separation}"
        gates.append(GateStatus("THESIS_STRONG", thesis_ok and dir_ok and trend_ok, reason))

        if trend_ok and dir_ok and thesis_ok:
            current_state = TradeState.PRESSURE_BUILDING

        # Gate 4: Option efficiency
        opt_ok = option_efficient
        reason = "" if opt_ok else "option not efficient"
        gates.append(GateStatus("OPTION_EFF", opt_ok and thesis_ok and dir_ok and trend_ok, reason))

        if trend_ok and dir_ok and thesis_ok and opt_ok:
            current_state = TradeState.OPTION_EFFICIENCY

        # Gate 5: Entry
        all_ready = trend_ok and dir_ok and thesis_ok and opt_ok
        entry_window = EntryWindow.OPEN if all_ready else EntryWindow.CLOSED

        if all_ready:
            current_state = TradeState.ENTRY_WINDOW_OPEN

        # Pad gates to match 6-gate display
        gates.append(GateStatus("FLOW_CONF", all_ready, "" if all_ready else "path B: N/A"))
        gates.append(GateStatus("ENTRY", all_ready, "" if all_ready else "waiting"))

        window_secs = int(45 + regime_confidence * 75) if all_ready else 0

        return StateMachineResult(
            state=current_state,
            entry_window=entry_window,
            gates=gates,
            estimated_window_seconds=window_secs,
            thesis_survival_minutes=round(min(theta_survival, 9999), 1),
        )

    def _evaluate_path_a(
        self,
        regime: MarketRegime,
        structure_score: float,
        nearest_zone_dist_pct: float,
        compression_score: float,
        is_compressed: bool,
        pressure_score: float,
        option_efficient: bool,
        theta_survival: float,
        thesis_valid: bool,
        thesis_separation: float,
    ) -> StateMachineResult:
        """PATH A — Breakout (original logic)."""
        gates: list[GateStatus] = []
        current_state = TradeState.NO_SETUP

        # Gate 1: STRUCTURE
        struct_ok = (
            structure_score >= self.structure_min
            and nearest_zone_dist_pct <= self.structure_max_dist_pct
        )
        reason = ""
        if not struct_ok:
            if structure_score < self.structure_min:
                reason = f"struct {structure_score:.2f} < {self.structure_min}"
            else:
                reason = f"dist {nearest_zone_dist_pct:.3f}% > {self.structure_max_dist_pct}%"
        gates.append(GateStatus("STRUCTURE", struct_ok, reason))

        if struct_ok:
            current_state = TradeState.STRUCTURAL_INTEREST

        # Gate 2: COMPRESSION / REGIME
        comp_ok = (
            is_compressed
            or compression_score >= self.compression_min
            or regime in self.favorable_regimes
        )
        reason = ""
        if not comp_ok:
            reason = f"no compression and regime={regime.value}"
        gates.append(GateStatus("COMPRESSION", comp_ok and struct_ok, reason))

        if struct_ok and comp_ok:
            current_state = TradeState.COMPRESSION

        # Gate 3: PRESSURE
        press_ok = pressure_score >= self.pressure_min
        reason = ""
        if not press_ok:
            reason = f"pressure {pressure_score:.2f} < {self.pressure_min}"
        gates.append(GateStatus("PRESSURE", press_ok and comp_ok and struct_ok, reason))

        if struct_ok and comp_ok and press_ok:
            current_state = TradeState.PRESSURE_BUILDING

        # Gate 4: OPTION EFFICIENCY
        opt_ok = option_efficient
        reason = "" if opt_ok else "option not efficient"
        all_prior = struct_ok and comp_ok and press_ok
        gates.append(GateStatus("OPTION_EFF", opt_ok and all_prior, reason))

        if all_prior and opt_ok:
            current_state = TradeState.OPTION_EFFICIENCY

        # Gate 5: THESIS
        flow_ok = thesis_valid and thesis_separation >= self.thesis_min_separation
        reason = ""
        if not flow_ok:
            reason = f"sep {thesis_separation:.3f} or invalid"
        all_prior2 = all_prior and opt_ok
        gates.append(GateStatus("FLOW_CONF", flow_ok and all_prior2, reason))

        if all_prior2 and flow_ok:
            current_state = TradeState.FLOW_CONFIRMATION

        # Gate 6: ENTRY
        all_ready = all_prior2 and flow_ok
        entry_window = EntryWindow.OPEN if all_ready else EntryWindow.CLOSED

        if all_ready:
            current_state = TradeState.ENTRY_WINDOW_OPEN

        gates.append(GateStatus("ENTRY", all_ready, "" if all_ready else "waiting"))

        window_secs = int(45 + compression_score * 75) if all_ready else 0

        return StateMachineResult(
            state=current_state,
            entry_window=entry_window,
            gates=gates,
            estimated_window_seconds=window_secs,
            thesis_survival_minutes=round(min(theta_survival, 9999), 1),
        )