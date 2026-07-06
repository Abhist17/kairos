"""
Kairos Engine — Full Pipeline Demo
Regime → Compression → Structure → IV → Options → Flow → Thesis → State Machine
"""

import numpy as np
from datetime import datetime, timedelta

from engine.core.enums import MarketRegime, TradeState, EntryWindow
from engine.core.config import RegimeConfig
from engine.regime.classifier import RegimeClassifier
from engine.compression.detector import CompressionDetector
from engine.structure.levels import StructureAnalyzer
from engine.volatility.iv_state import IVClassifier
from engine.options.efficiency import OptionEfficiencyEngine
from engine.flow.pressure import PressureAnalyzer
from engine.scoring.thesis import ThesisEngine
from engine.state_machine.machine import EntryStateMachine
from data.models.market_state import (
    MarketState,
    RegimeMetrics,
    CompressionMetrics,
    StructureMetrics,
    IVMetrics,
    OptionMetrics,
    FlowMetrics,
    ThesisMetrics,
    StateMachineMetrics,
    GateInfo,
)


def generate_synthetic_nifty(n_candles=600, seed=42):
    rng = np.random.default_rng(seed)
    base = 25000.0
    prices = [base]
    phases = [
        (100, 0.0, 0.8, "COMPRESSION"),
        (100, 3.0, 1.2, "TREND_UP"),
        (80, 1.0, 8.0, "EXHAUSTION"),
        (100, 0.0, 4.0, "MEAN_REV"),
        (120, -2.5, 1.5, "TREND_DN"),
        (100, 0.0, 15.0, "CHAOTIC"),
    ]
    labels = []
    for n, drift, noise, label in phases:
        for _ in range(n):
            prices.append(prices[-1] + drift + rng.normal(0, noise))
            labels.append(label)
    prices = np.array(prices[1:])
    n = len(prices)
    highs = prices + np.abs(rng.normal(0, 1, n)) * prices * 0.0003 + 0.5
    lows = prices - np.abs(rng.normal(0, 1, n)) * prices * 0.0003 - 0.5
    opens = np.roll(prices, 1)
    opens[0] = base
    volumes = rng.integers(5000, 50000, n).astype(float)
    # Synthetic IV series: base 15% + noise correlated with vol changes
    iv_base = 0.15
    iv_series = iv_base + np.cumsum(rng.normal(0, 0.001, n))
    iv_series = np.clip(iv_series, 0.05, 0.60)
    return dict(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=prices,
        volumes=volumes,
        labels=labels,
        iv_series=iv_series,
    )


COLORS = {
    MarketRegime.TREND_EXPANSION: "\033[92m",
    MarketRegime.TREND_EXHAUSTION: "\033[93m",
    MarketRegime.COMPRESSION: "\033[96m",
    MarketRegime.MEAN_REVERSION: "\033[95m",
    MarketRegime.CHAOTIC: "\033[91m",
    MarketRegime.UNKNOWN: "\033[90m",
}
R = "\033[0m"
B = "\033[1m"
G = "\033[92m"
RED = "\033[91m"
Y = "\033[93m"
C = "\033[96m"

STATE_COLOR = {
    TradeState.NO_SETUP: "\033[90m",
    TradeState.STRUCTURAL_INTEREST: "\033[37m",
    TradeState.COMPRESSION: C,
    TradeState.PRESSURE_BUILDING: Y,
    TradeState.OPTION_EFFICIENCY: "\033[95m",
    TradeState.FLOW_CONFIRMATION: "\033[94m",
    TradeState.ENTRY_WINDOW_OPEN: G,
}


def print_state(s: MarketState, idx: int, label: str):
    rc = COLORS.get(s.regime, R)
    sc = STATE_COLOR.get(s.trade_state, R)

    gates_str = ""
    for g in s.state_machine.gates[:6]:
        gates_str += f"{G}●{R}" if g.ready else f"{RED}○{R}"

    entry = ""
    if s.entry_window == EntryWindow.OPEN:
        entry = f" {B}{G}▶ ENTRY WINDOW OPEN ~{s.state_machine.estimated_window_seconds}s survival={s.state_machine.thesis_survival_minutes:.0f}m{R}"

    print(
        f"  {idx:>4d} | {s.last_price:>9.1f} | "
        f"{rc}{s.regime.value:<16s}{R} "
        f"{s.bias.value:<7s} | "
        f"iv={s.iv.state.value:<12s} "
        f"comp={s.compression.compression_score:.2f} "
        f"struct={s.structure.structure_score:.2f} "
        f"press={s.flow.pressure_score:.2f} "
        f"sep={s.thesis.separation:+.2f} | "
        f"[{gates_str}] "
        f"{sc}{s.trade_state.value:<20s}{R}"
        f"{entry}"
        f"  [{label}]"
    )


def main():
    print(f"\n{'=' * 140}")
    print(f"  {B}KAIROS ENGINE — Full Pipeline{R}")
    print(
        "  Regime → Compression → Structure → IV → Options → Flow → Thesis → State Machine"
    )
    print(f"{'=' * 140}\n")

    data = generate_synthetic_nifty()
    cfg = RegimeConfig()
    regime_clf = RegimeClassifier()
    comp_det = CompressionDetector()
    struct_ana = StructureAnalyzer()
    iv_clf = IVClassifier()
    opt_eng = OptionEfficiencyEngine()
    press_ana = PressureAnalyzer()
    thesis_eng = ThesisEngine()
    sm = EntryStateMachine()

    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    opens = data["opens"]
    volumes = data["volumes"]
    labels = data["labels"]
    iv_series = data["iv_series"]

    min_c = cfg.vol_percentile_window + cfg.lookback
    step = 5

    for i in range(min_c, len(closes), step):
        c, h, lo, o, v = closes[:i], highs[:i], lows[:i], opens[:i], volumes[:i]
        price = float(c[-1])

        # 1. Regime
        reg = regime_clf.classify(c, h, lo)

        # 2. Compression
        comp = comp_det.detect(c, h, lo)

        # 3. Structure
        struct = struct_ana.analyze(c, h, lo, v, price)

        # 4. IV State
        iv = iv_clf.classify(iv_series[:i])

        # 5. Options Efficiency (synthetic greeks)
        strike = round(price / 50) * 50  # nearest 50-strike
        dte_min = max(60, 375 - (i % 375))
        # Synthetic greeks scaled by moneyness
        moneyness = abs(price - strike) / price
        syn_delta = max(0.1, 0.5 - moneyness * 5)
        syn_gamma = max(0.001, 0.05 - moneyness)
        syn_theta = -(0.5 + iv.current_iv * 2)
        syn_vega = max(0.1, 5.0 - moneyness * 20)
        exp_move = float(np.std(np.diff(c[-20:]))) * 3 if len(c) > 21 else 10.0
        req_move = abs(price - strike) + 5.0

        opt = opt_eng.evaluate(
            spot=price,
            strike=strike,
            delta=syn_delta,
            gamma=syn_gamma,
            theta=syn_theta,
            vega=syn_vega,
            expected_move=exp_move,
            required_move=req_move,
            dte_minutes=dte_min,
        )

        # 6. Flow/Pressure
        target = struct.nearest_zone.center if struct.nearest_zone else price
        flow = press_ana.analyze(c, h, lo, o, target)

        # 7. Thesis
        thesis = thesis_eng.score(
            regime=reg.regime,
            bias=reg.bias,
            regime_confidence=reg.confidence,
            compression_score=comp.compression_score,
            is_compressed=comp.is_compressed,
            structure_score=struct.structure_score,
            nearest_zone_dist_pct=struct.nearest_zone_distance_pct,
            pressure_score=flow.pressure_score,
            aggression_ratio=flow.aggression_ratio,
            iv_percentile=iv.iv_percentile,
            option_efficient=opt.is_efficient,
        )

        # 8. State Machine
        sm_result = sm.evaluate(
            regime=reg.regime,
            structure_score=struct.structure_score,
            nearest_zone_dist_pct=struct.nearest_zone_distance_pct,
            compression_score=comp.compression_score,
            is_compressed=comp.is_compressed,
            pressure_score=flow.pressure_score,
            option_efficient=opt.is_efficient,
            theta_survival=opt.theta_survival_minutes,
            thesis_valid=thesis.thesis_valid,
            thesis_separation=thesis.separation,
        )

        # Build state
        state = MarketState(
            symbol="NIFTY",
            timestamp=datetime(2025, 1, 6, 9, 15) + timedelta(minutes=i),
            last_price=price,
            candle_close=price,
            candle_high=float(h[-1]),
            candle_low=float(lo[-1]),
            candle_open=float(o[-1]),
            candle_volume=float(v[-1]),
            regime=reg.regime,
            bias=reg.bias,
            regime_metrics=RegimeMetrics(
                efficiency_ratio=reg.efficiency_ratio,
                directional_persistence=reg.directional_persistence,
                normalized_slope=reg.normalized_slope,
                directional_entropy=reg.directional_entropy,
                realized_volatility=reg.realized_volatility,
                volatility_percentile=reg.volatility_percentile,
                atr_contraction=reg.atr_contraction,
                log_return_mean=reg.log_return_mean,
                regime_confidence=reg.confidence,
                regime_age=reg.regime_age,
                transition_from=reg.transition_from,
            ),
            compression=CompressionMetrics(
                is_compressed=comp.is_compressed,
                compression_score=comp.compression_score,
                atr_contraction=comp.atr_contraction,
                range_contraction=comp.range_contraction,
                rv_decay=comp.rv_decay,
                bbw_percentile=comp.bbw_percentile,
                compression_velocity=comp.compression_velocity,
                compression_half_life=comp.compression_half_life,
                candles_compressed=comp.candles_compressed,
                pre_compression_vol=comp.pre_compression_vol,
            ),
            structure=StructureMetrics(
                structure_score=struct.structure_score,
                nearest_zone_distance=struct.nearest_zone_distance,
                nearest_zone_distance_pct=struct.nearest_zone_distance_pct,
                nearest_zone_center=struct.nearest_zone.center
                if struct.nearest_zone
                else 0,
                nearest_zone_confluence=struct.nearest_zone.confluence
                if struct.nearest_zone
                else 0,
                above_zones=struct.above_zones,
                below_zones=struct.below_zones,
                inside_zone=struct.inside_zone,
                total_levels=len(struct.levels),
                total_zones=len(struct.zones),
            ),
            iv=IVMetrics(
                state=iv.state,
                current_iv=iv.current_iv,
                iv_percentile=iv.iv_percentile,
                iv_velocity=iv.iv_velocity,
            ),
            option=OptionMetrics(
                is_efficient=opt.is_efficient,
                delta_acceleration=opt.delta_acceleration,
                gamma_theta_ratio=opt.gamma_theta_ratio,
                theta_survival_minutes=opt.theta_survival_minutes,
                move_feasibility=opt.move_feasibility,
                strike=opt.strike,
                delta=opt.delta,
                gamma=opt.gamma,
                theta=opt.theta,
            ),
            flow=FlowMetrics(
                pressure_score=flow.pressure_score,
                level_tests=flow.level_tests,
                test_frequency=flow.test_frequency,
                pullback_shrinking=flow.pullback_shrinking,
                aggression_ratio=flow.aggression_ratio,
                pressure_asymmetry=flow.pressure_asymmetry,
                liquidity_thinning=flow.liquidity_thinning,
            ),
            thesis=ThesisMetrics(
                primary_bias=thesis.primary_bias,
                primary_score=thesis.primary_score,
                counter_score=thesis.counter_score,
                separation=thesis.separation,
                thesis_valid=thesis.thesis_valid,
            ),
            state_machine=StateMachineMetrics(
                gates=[
                    GateInfo(name=g.name, ready=g.ready, reason=g.reason)
                    for g in sm_result.gates
                ],
                estimated_window_seconds=sm_result.estimated_window_seconds,
                thesis_survival_minutes=sm_result.thesis_survival_minutes,
            ),
            trade_state=sm_result.state,
            entry_window=sm_result.entry_window,
        )

        print_state(state, i, labels[i - 1])

    print(f"\n{'=' * 140}")
    print(f"  {B}Full Pipeline Complete{R} — 8 engines, state machine with 6 gates")
    print(f"{'=' * 140}\n")


if __name__ == "__main__":
    main()
