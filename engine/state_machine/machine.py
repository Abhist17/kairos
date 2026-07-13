"""
Kairos Engine — State Machine V4

Path A — BREAKOUT: near zone, compressed, pressure building (unchanged)
Path B — MOMENTUM CATCH: big candles moving one direction = GO

Path B redesigned for scalper who catches fast moves early:
  - Accepts TREND_EXPANSION AND TREND_EXHAUSTION (crash = exhaustion but still tradeable)
  - No regime age requirement (catch it at the START)
  - Lower ER threshold (0.40 — even moderate trends count)
  - Key: last 3 candles bigger than average AND same direction
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
        # Path B — aggressive momentum catch
        trend_min_confidence: float = 0.50,
        trend_min_separation: float = 0.20,
        trend_min_er: float = 0.50,
        trend_regimes: tuple = (
            MarketRegime.TREND_EXPANSION,
            MarketRegime.TREND_EXHAUSTION,
        ),
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

        path_b = self._path_b(
            regime,
            bias,
            regime_confidence,
            efficiency_ratio,
            option_efficient,
            theta_survival,
            thesis_valid,
            thesis_separation,
        )
        if path_b.entry_window == EntryWindow.OPEN:
            return path_b

        return self._path_a(
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

    def _path_b(
        self,
        regime,
        bias,
        confidence,
        er,
        option_efficient,
        theta_survival,
        thesis_valid,
        separation,
    ) -> StateMachineResult:
        """
        Path B — Momentum Catch

        Gate 1: MOMENTUM — trending regime + ER shows direction
        Gate 2: DIRECTION — clear bias (not neutral)
        Gate 3: THESIS — separation confirms it's not conflicting
        Gate 4: OPTION_EFF — the contract makes mathematical sense
        """
        gates: list[GateStatus] = []
        state = TradeState.NO_SETUP

        # Gate 1: Momentum — is the market moving directionally?
        mom_ok = regime in self.trend_regimes and er >= self.trend_min_er
        reason = ""
        if not mom_ok:
            if regime not in self.trend_regimes:
                reason = f"regime {regime.value}"
            else:
                reason = f"ER {er:.3f} < {self.trend_min_er}"
        gates.append(GateStatus("MOMENTUM", mom_ok, reason))
        if mom_ok:
            state = TradeState.STRUCTURAL_INTEREST

        # Gate 2: Direction — is bias clear?
        dir_ok = bias != MarketBias.NEUTRAL
        reason = "neutral" if not dir_ok else ""
        gates.append(GateStatus("DIRECTION", dir_ok and mom_ok, reason))
        if mom_ok and dir_ok:
            state = TradeState.COMPRESSION

        # Gate 3: Thesis — is the thesis separated enough?
        thesis_ok = thesis_valid and separation >= self.trend_min_separation
        reason = f"sep {separation:.3f}" if not thesis_ok else ""
        gates.append(GateStatus("THESIS", thesis_ok and dir_ok and mom_ok, reason))
        if mom_ok and dir_ok and thesis_ok:
            state = TradeState.PRESSURE_BUILDING

        # Gate 4: Option efficiency
        opt_ok = option_efficient
        reason = "" if opt_ok else "not efficient"
        all_ok = mom_ok and dir_ok and thesis_ok
        gates.append(GateStatus("OPTION_EFF", opt_ok and all_ok, reason))
        if all_ok and opt_ok:
            state = TradeState.OPTION_EFFICIENCY

        # Entry
        entry = all_ok and opt_ok
        if entry:
            state = TradeState.ENTRY_WINDOW_OPEN

        gates.append(GateStatus("FLOW", entry, ""))
        gates.append(GateStatus("ENTRY", entry, ""))

        return StateMachineResult(
            state=state,
            entry_window=EntryWindow.OPEN if entry else EntryWindow.CLOSED,
            gates=gates,
            estimated_window_seconds=int(45 + confidence * 75) if entry else 0,
            thesis_survival_minutes=round(min(theta_survival, 9999), 1),
        )

    def _path_a(
        self,
        regime,
        structure_score,
        zone_dist,
        compression_score,
        is_compressed,
        pressure_score,
        option_efficient,
        theta_survival,
        thesis_valid,
        separation,
    ) -> StateMachineResult:
        """Path A — Breakout (original)."""
        gates: list[GateStatus] = []
        state = TradeState.NO_SETUP

        s_ok = (
            structure_score >= self.structure_min
            and zone_dist <= self.structure_max_dist_pct
        )
        gates.append(
            GateStatus(
                "STRUCTURE",
                s_ok,
                f"s={structure_score:.2f} d={zone_dist:.3f}" if not s_ok else "",
            )
        )
        if s_ok:
            state = TradeState.STRUCTURAL_INTEREST

        c_ok = (
            is_compressed
            or compression_score >= self.compression_min
            or regime in self.favorable_regimes
        )
        gates.append(GateStatus("COMPRESSION", c_ok and s_ok, ""))
        if s_ok and c_ok:
            state = TradeState.COMPRESSION

        p_ok = pressure_score >= self.pressure_min
        gates.append(GateStatus("PRESSURE", p_ok and c_ok and s_ok, ""))
        if s_ok and c_ok and p_ok:
            state = TradeState.PRESSURE_BUILDING

        o_ok = option_efficient
        a3 = s_ok and c_ok and p_ok
        gates.append(GateStatus("OPTION_EFF", o_ok and a3, ""))
        if a3 and o_ok:
            state = TradeState.OPTION_EFFICIENCY

        f_ok = thesis_valid and separation >= self.thesis_min_separation
        a4 = a3 and o_ok
        gates.append(GateStatus("FLOW_CONF", f_ok and a4, ""))
        if a4 and f_ok:
            state = TradeState.FLOW_CONFIRMATION

        entry = a4 and f_ok
        if entry:
            state = TradeState.ENTRY_WINDOW_OPEN
        gates.append(GateStatus("ENTRY", entry, ""))

        return StateMachineResult(
            state=state,
            entry_window=EntryWindow.OPEN if entry else EntryWindow.CLOSED,
            gates=gates,
            estimated_window_seconds=int(45 + compression_score * 75) if entry else 0,
            thesis_survival_minutes=round(min(theta_survival, 9999), 1),
        )
