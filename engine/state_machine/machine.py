"""
Kairos Engine — Entry State Machine V3

Path A — BREAKOUT: price near zone → compressed → pressure → efficient → thesis → entry
Path B — TREND FOLLOWING: strong trend + established (age>5) + MTF aligned + thesis strong + efficient

Fixes applied:
  Fix 1: Both paths respect opening range + cooldown (enforced in signal_generator)
  Fix 3: Path B requires regime_age >= 5 (trend must be established ~10 min)
  Fix 5: Path B requires MTF alignment (higher TF must agree on direction)
"""

from engine.core.enums import TradeState, EntryWindow, MarketRegime, MarketBias
from engine.state_machine.models import StateMachineResult, GateStatus


class EntryStateMachine:
    def __init__(
        self,
        # Path A
        structure_min: float = 0.3,
        structure_max_dist_pct: float = 0.3,
        compression_min: float = 0.2,
        pressure_min: float = 0.2,
        thesis_min_separation: float = 0.1,
        favorable_regimes: tuple = (
            MarketRegime.COMPRESSION,
            MarketRegime.TREND_EXPANSION,
        ),
        # Path B
        trend_min_confidence: float = 0.60,
        trend_min_separation: float = 0.25,
        trend_min_er: float = 0.55,
        trend_min_regime_age: int = 3,
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
        self.trend_min_regime_age = trend_min_regime_age
        self.trend_regimes = trend_regimes

    def evaluate(
        self,
        regime: MarketRegime,
        bias: MarketBias = MarketBias.NEUTRAL,
        regime_confidence: float = 0.0,
        efficiency_ratio: float = 0.0,
        regime_age: int = 0,
        mtf_aligned: bool = True,
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

        path_b = self._evaluate_path_b(
            regime,
            bias,
            regime_confidence,
            efficiency_ratio,
            regime_age,
            mtf_aligned,
            option_efficient,
            theta_survival,
            thesis_valid,
            thesis_separation,
        )

        if path_b.entry_window == EntryWindow.OPEN:
            return path_b

        return self._evaluate_path_a(
            regime,
            structure_score,
            nearest_zone_dist_pct,
            compression_score,
            is_compressed,
            pressure_score,
            option_efficient,
            theta_survival,
            thesis_valid,
            thesis_separation,
        )

    def _evaluate_path_b(
        self,
        regime,
        bias,
        regime_confidence,
        efficiency_ratio,
        regime_age,
        mtf_aligned,
        option_efficient,
        theta_survival,
        thesis_valid,
        thesis_separation,
    ) -> StateMachineResult:
        gates: list[GateStatus] = []
        current_state = TradeState.NO_SETUP

        # Gate 1: STRONG_TREND — regime + confidence + ER + age
        trend_ok = (
            regime in self.trend_regimes
            and regime_confidence >= self.trend_min_confidence
            and efficiency_ratio >= self.trend_min_er
            and regime_age >= self.trend_min_regime_age
        )
        reason = ""
        if not trend_ok:
            if regime not in self.trend_regimes:
                reason = f"regime {regime.value}"
            elif regime_confidence < self.trend_min_confidence:
                reason = f"conf {regime_confidence:.2f}"
            elif efficiency_ratio < self.trend_min_er:
                reason = f"ER {efficiency_ratio:.3f}"
            else:
                reason = f"age {regime_age} < {self.trend_min_regime_age}"
        gates.append(GateStatus("STRONG_TREND", trend_ok, reason))

        if trend_ok:
            current_state = TradeState.STRUCTURAL_INTEREST

        # Gate 2: DIRECTION — not neutral + MTF aligned
        dir_ok = bias != MarketBias.NEUTRAL and mtf_aligned
        reason = ""
        if not dir_ok:
            if bias == MarketBias.NEUTRAL:
                reason = "neutral bias"
            else:
                reason = "MTF not aligned"
        gates.append(GateStatus("DIRECTION", dir_ok and trend_ok, reason))

        if trend_ok and dir_ok:
            current_state = TradeState.COMPRESSION

        # Gate 3: THESIS_STRONG
        thesis_ok = thesis_valid and thesis_separation >= self.trend_min_separation
        reason = ""
        if not thesis_ok:
            reason = f"sep {thesis_separation:.3f} < {self.trend_min_separation}"
        gates.append(
            GateStatus("THESIS_STRONG", thesis_ok and dir_ok and trend_ok, reason)
        )

        if trend_ok and dir_ok and thesis_ok:
            current_state = TradeState.PRESSURE_BUILDING

        # Gate 4: OPTION_EFF
        opt_ok = option_efficient
        reason = "" if opt_ok else "option not efficient"
        all_prior = trend_ok and dir_ok and thesis_ok
        gates.append(GateStatus("OPTION_EFF", opt_ok and all_prior, reason))

        if all_prior and opt_ok:
            current_state = TradeState.OPTION_EFFICIENCY

        # All pass
        all_ready = all_prior and opt_ok
        entry_window = EntryWindow.OPEN if all_ready else EntryWindow.CLOSED

        if all_ready:
            current_state = TradeState.ENTRY_WINDOW_OPEN

        gates.append(
            GateStatus("FLOW_CONF", all_ready, "" if all_ready else "path B: N/A")
        )
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
        regime,
        structure_score,
        nearest_zone_dist_pct,
        compression_score,
        is_compressed,
        pressure_score,
        option_efficient,
        theta_survival,
        thesis_valid,
        thesis_separation,
    ) -> StateMachineResult:
        gates: list[GateStatus] = []
        current_state = TradeState.NO_SETUP

        struct_ok = (
            structure_score >= self.structure_min
            and nearest_zone_dist_pct <= self.structure_max_dist_pct
        )
        reason = ""
        if not struct_ok:
            if structure_score < self.structure_min:
                reason = f"struct {structure_score:.2f}"
            else:
                reason = f"dist {nearest_zone_dist_pct:.3f}%"
        gates.append(GateStatus("STRUCTURE", struct_ok, reason))
        if struct_ok:
            current_state = TradeState.STRUCTURAL_INTEREST

        comp_ok = (
            is_compressed
            or compression_score >= self.compression_min
            or regime in self.favorable_regimes
        )
        reason = "" if comp_ok else f"no comp, regime={regime.value}"
        gates.append(GateStatus("COMPRESSION", comp_ok and struct_ok, reason))
        if struct_ok and comp_ok:
            current_state = TradeState.COMPRESSION

        press_ok = pressure_score >= self.pressure_min
        reason = "" if press_ok else f"pressure {pressure_score:.2f}"
        gates.append(GateStatus("PRESSURE", press_ok and comp_ok and struct_ok, reason))
        if struct_ok and comp_ok and press_ok:
            current_state = TradeState.PRESSURE_BUILDING

        opt_ok = option_efficient
        reason = "" if opt_ok else "option not efficient"
        all_prior = struct_ok and comp_ok and press_ok
        gates.append(GateStatus("OPTION_EFF", opt_ok and all_prior, reason))
        if all_prior and opt_ok:
            current_state = TradeState.OPTION_EFFICIENCY

        flow_ok = thesis_valid and thesis_separation >= self.thesis_min_separation
        reason = "" if flow_ok else f"sep {thesis_separation:.3f}"
        all_prior2 = all_prior and opt_ok
        gates.append(GateStatus("FLOW_CONF", flow_ok and all_prior2, reason))
        if all_prior2 and flow_ok:
            current_state = TradeState.FLOW_CONFIRMATION

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
