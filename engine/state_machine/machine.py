"""
Kairos Engine — Entry State Machine

NOT a score threshold. A STATE MACHINE.

Each gate must be READY before the next gate is even evaluated.
ALL gates READY = ENTRY_WINDOW_OPEN.

Gates in order:
1. STRUCTURE  — price near a meaningful zone with defined structure
2. COMPRESSION — market compressed OR in favorable regime
3. PRESSURE   — order flow pressure building toward thesis direction
4. OPTION_EFF — option contract is mathematically efficient to enter
5. FLOW_CONF  — thesis is valid with sufficient separation from counter
6. ENTRY      — all above ready → window opens for estimated duration
"""

from engine.core.enums import TradeState, EntryWindow, MarketRegime
from engine.state_machine.models import StateMachineResult, GateStatus


class EntryStateMachine:
    def __init__(
        self,
        structure_min: float = 0.3,
        structure_max_dist_pct: float = 0.3,
        compression_min: float = 0.2,
        pressure_min: float = 0.2,
        thesis_min_separation: float = 0.1,
        favorable_regimes: tuple = (
            MarketRegime.COMPRESSION,
            MarketRegime.TREND_EXPANSION,
        ),
    ):
        self.structure_min = structure_min
        self.structure_max_dist_pct = structure_max_dist_pct
        self.compression_min = compression_min
        self.pressure_min = pressure_min
        self.thesis_min_separation = thesis_min_separation
        self.favorable_regimes = favorable_regimes

    def evaluate(
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
                reason = f"structure_score {structure_score:.2f} < {self.structure_min}"
            else:
                reason = f"zone_dist {nearest_zone_dist_pct:.3f}% > {self.structure_max_dist_pct}%"
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
            reason = (
                f"no compression ({compression_score:.2f}) and regime={regime.value}"
            )
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
        reason = "" if opt_ok else "option contract not efficient"
        gates.append(
            GateStatus(
                "OPTION_EFF", opt_ok and press_ok and comp_ok and struct_ok, reason
            )
        )

        if struct_ok and comp_ok and press_ok and opt_ok:
            current_state = TradeState.OPTION_EFFICIENCY

        # Gate 5: FLOW / THESIS CONFIRMATION
        flow_ok = thesis_valid and thesis_separation >= self.thesis_min_separation
        reason = ""
        if not flow_ok:
            reason = f"thesis_sep {thesis_separation:.3f} or invalid"
        gates.append(
            GateStatus(
                "FLOW_CONF",
                flow_ok and opt_ok and press_ok and comp_ok and struct_ok,
                reason,
            )
        )

        if struct_ok and comp_ok and press_ok and opt_ok and flow_ok:
            current_state = TradeState.FLOW_CONFIRMATION

        # Gate 6: ENTRY WINDOW
        all_ready = struct_ok and comp_ok and press_ok and opt_ok and flow_ok
        entry_window = EntryWindow.OPEN if all_ready else EntryWindow.CLOSED

        if all_ready:
            current_state = TradeState.ENTRY_WINDOW_OPEN

        gates.append(
            GateStatus(
                "ENTRY", all_ready, "" if all_ready else "waiting for prior gates"
            )
        )

        # Estimate window duration (45-120s based on compression energy)
        window_secs = 0
        if all_ready:
            window_secs = int(45 + compression_score * 75)

        return StateMachineResult(
            state=current_state,
            entry_window=entry_window,
            gates=gates,
            estimated_window_seconds=window_secs,
            thesis_survival_minutes=round(min(theta_survival, 9999), 1),
        )
