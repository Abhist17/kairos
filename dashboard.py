"""
Kairos Engine — Real-Time Terminal Dashboard

Runs the pipeline continuously over candle data and renders
a full-screen terminal display with all engine states.

Usage:
    python dashboard.py                          # synthetic demo
    python dashboard.py --csv path/to/candles.csv # real data
"""

import argparse
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta

from engine.pipeline.market_pipeline import MarketPipeline
from engine.options.oi_gravity import OIGravityTracker
from engine.options.gamma_map import GammaGravityMap
from engine.core.enums import MarketRegime, TradeState, EntryWindow
from data.models.market_state import MarketState


# --- Colors ---
R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
RED = "\033[91m"; GRN = "\033[92m"; YEL = "\033[93m"
BLU = "\033[94m"; MAG = "\033[95m"; CYN = "\033[96m"; WHT = "\033[97m"; GRY = "\033[90m"

REGIME_CLR = {
    MarketRegime.TREND_EXPANSION: GRN, MarketRegime.TREND_EXHAUSTION: YEL,
    MarketRegime.COMPRESSION: CYN, MarketRegime.MEAN_REVERSION: MAG,
    MarketRegime.CHAOTIC: RED, MarketRegime.UNKNOWN: GRY,
}


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def bar(value: float, max_val: float = 1.0, width: int = 20, color: str = GRN) -> str:
    filled = int(value / max_val * width) if max_val > 0 else 0
    filled = max(0, min(width, filled))
    return f"{color}{'█' * filled}{GRY}{'░' * (width - filled)}{R}"


def gate_display(gates) -> str:
    parts = []
    for g in gates:
        if g.ready:
            parts.append(f"{GRN}● {g.name}{R}")
        else:
            parts.append(f"{RED}○ {g.name}{R}")
    return "  ".join(parts)


def render(state: MarketState, oi_result, gamma_result, idx: int, total: int):
    clear()
    rc = REGIME_CLR.get(state.regime, R)
    rm = state.regime_metrics
    cm = state.compression
    sm = state.state_machine

    print(f"{B}{CYN}╔══════════════════════════════════════════════════════════════════════════════════╗{R}")
    print(f"{B}{CYN}║  KAIROS ENGINE — Real-Time Options Entry Intelligence{R}{' ' * 27}{CYN}║{R}")
    print(f"{B}{CYN}╚══════════════════════════════════════════════════════════════════════════════════╝{R}")

    # Price & Regime
    print(f"\n  {B}{state.symbol}{R}  {B}{WHT}{state.last_price:,.2f}{R}  "
          f"│  Regime: {rc}{B}{state.regime.value}{R}  "
          f"│  Bias: {B}{state.bias.value}{R}  "
          f"│  Conf: {rm.regime_confidence:.2f}  "
          f"│  Age: {rm.regime_age}")

    # Feature bars
    print(f"\n  {DIM}── Features ──{R}")
    print(f"  Efficiency   {bar(rm.efficiency_ratio, 1.0, 25, GRN)} {rm.efficiency_ratio:.3f}")
    print(f"  Persistence  {bar(rm.directional_persistence, 1.0, 25, GRN)} {rm.directional_persistence:.3f}")
    print(f"  Entropy      {bar(rm.directional_entropy, 1.0, 25, YEL)} {rm.directional_entropy:.3f}")
    print(f"  Vol Pctile   {bar(rm.volatility_percentile, 100, 25, RED)} {rm.volatility_percentile:.1f}%")

    # Compression
    print(f"\n  {DIM}── Compression ──{R}")
    comp_flag = f"{CYN}{B}■ COMPRESSED{R}" if cm.is_compressed else f"{GRY}□ not compressed{R}"
    hl = f"{cm.compression_half_life:.0f}" if cm.compression_half_life < 99999 else "∞"
    print(f"  Score: {bar(cm.compression_score, 1.0, 20, CYN)} {cm.compression_score:.3f}  "
          f"Velocity: {cm.compression_velocity:+.5f}  Half-life: {hl}  {comp_flag}")

    # Structure
    st = state.structure
    print(f"\n  {DIM}── Structure ──{R}")
    print(f"  Score: {bar(st.structure_score, 1.0, 20, BLU)} {st.structure_score:.2f}  "
          f"Zones: {st.total_zones}  Levels: {st.total_levels}  "
          f"Nearest: {st.nearest_zone_center:.0f} ({st.nearest_zone_distance_pct:.3f}%)  "
          f"{'INSIDE' if st.inside_zone else ''}")

    # IV State
    iv = state.iv
    iv_clr = GRN if iv.state.value == "COMPRESSED" else (RED if "OVER" in iv.state.value else WHT)
    print(f"\n  {DIM}── IV State ──{R}")
    print(f"  State: {iv_clr}{B}{iv.state.value}{R}  "
          f"IV: {iv.current_iv:.4f}  Pctile: {iv.iv_percentile:.1f}%  "
          f"Velocity: {iv.iv_velocity:+.5f}")

    # OI Gravity
    print(f"\n  {DIM}── OI Gravity (ESTIMATED) ──{R}")
    print(f"  Call Gravity: {oi_result.call_gravity:.0f}  "
          f"Put Gravity: {oi_result.put_gravity:.0f}  "
          f"Max Pain: {oi_result.max_pain:.0f}  "
          f"PCR: {oi_result.pcr_overall:.2f}  "
          f"Velocity: {oi_result.gravity_velocity:+.1f}")

    # Gamma Map
    print(f"\n  {DIM}── Gamma Map (ESTIMATED) ──{R}")
    gclr = GRN if gamma_result.total_gamma_exposure > 0 else RED
    print(f"  {gclr}{gamma_result.regime_note}{R}")
    print(f"  GEX: {gamma_result.total_gamma_exposure:+,.0f}  "
          f"Flip: {gamma_result.gamma_flip_strike:.0f}  "
          f"Max Gamma: {gamma_result.max_gamma_strike:.0f}")

    # Option Efficiency
    op = state.option
    eff_clr = GRN if op.is_efficient else RED
    print(f"\n  {DIM}── Option Efficiency ──{R}")
    print(f"  {eff_clr}{B}{'✓ EFFICIENT' if op.is_efficient else '✗ NOT EFFICIENT'}{R}  "
          f"Δ-accel: {op.delta_acceleration:.3f}  "
          f"γ/θ: {op.gamma_theta_ratio:.2f}  "
          f"θ-survival: {op.theta_survival_minutes:.0f}m  "
          f"Feasibility: {op.move_feasibility:.2f}")

    # Flow
    fl = state.flow
    print(f"\n  {DIM}── Pressure / Flow ──{R}")
    print(f"  Pressure: {bar(fl.pressure_score, 1.0, 20, YEL)} {fl.pressure_score:.3f}  "
          f"Tests: {fl.level_tests}  "
          f"Aggression: {fl.aggression_ratio:.2f}  "
          f"{'↓ SHRINKING' if fl.pullback_shrinking else ''}  "
          f"{'⚡ LIQ THIN' if fl.liquidity_thinning else ''}")

    # Thesis
    th = state.thesis
    sep_clr = GRN if th.separation > 0.15 else (YEL if th.separation > 0.05 else RED)
    print(f"\n  {DIM}── Thesis ──{R}")
    print(f"  Primary: {B}{th.primary_bias.value}{R} ({th.primary_score:.3f})  "
          f"Counter: ({th.counter_score:.3f})  "
          f"Separation: {sep_clr}{B}{th.separation:+.3f}{R}  "
          f"{'✓ VALID' if th.thesis_valid else '✗ INVALID'}")

    # State Machine — THE BIG ONE
    print(f"\n  {DIM}── State Machine ──{R}")
    print(f"  {gate_display(sm.gates)}")

    if state.entry_window == EntryWindow.OPEN:
        print(f"\n  {B}{GRN}╔══════════════════════════════════════════════════════════════╗{R}")
        print(f"  {B}{GRN}║  ▶ ENTRY WINDOW OPEN                                        ║{R}")
        print(f"  {B}{GRN}║  Estimated Window: ~{sm.estimated_window_seconds}s"
              f"{'':>{42 - len(str(sm.estimated_window_seconds))}}║{R}")
        print(f"  {B}{GRN}║  Thesis Survival: {sm.thesis_survival_minutes:.0f} minutes"
              f"{'':>{40 - len(str(int(sm.thesis_survival_minutes)))}}║{R}")
        print(f"  {B}{GRN}╚══════════════════════════════════════════════════════════════╝{R}")
    else:
        sc = REGIME_CLR.get(state.regime, GRY)
        print(f"\n  State: {sc}{B}{state.trade_state.value}{R}  │  Entry: {RED}CLOSED{R}")

    print(f"\n  {GRY}Candle {idx}/{total}  │  {state.timestamp.strftime('%H:%M:%S')}{R}")


def run_synthetic():
    """Demo with synthetic data."""
    from main import generate_synthetic_nifty

    data = generate_synthetic_nifty(n_candles=600)
    pipeline = MarketPipeline()
    oi = OIGravityTracker()
    gm = GammaGravityMap()

    closes, highs, lows = data["closes"], data["highs"], data["lows"]
    opens, volumes = data["opens"], data["volumes"]
    iv_series = data["iv_series"]

    min_c = pipeline.min_candles
    total = len(closes)

    for i in range(min_c, total, 3):
        c, h, lo, o, v = closes[:i], highs[:i], lows[:i], opens[:i], volumes[:i]
        price = float(c[-1])

        state = pipeline.process(c, h, lo, o, v, iv_series[:i],
                                 timestamp=datetime(2025, 1, 6, 9, 15) + timedelta(minutes=i))

        oi_result = oi.analyze(price)
        gm_result = gm.estimate(price, iv=state.iv.current_iv or 0.15)

        render(state, oi_result, gm_result, i, total)
        time.sleep(0.15)


def run_csv(path: str):
    """Run from CSV file."""
    from data.collectors.csv_loader import CSVLoader

    loader = CSVLoader()
    candles = loader.load(path)
    closes, highs, lows, opens, volumes = loader.to_arrays(candles)

    pipeline = MarketPipeline()
    oi = OIGravityTracker()
    gm = GammaGravityMap()

    min_c = pipeline.min_candles
    total = len(closes)

    if total < min_c:
        print(f"Need at least {min_c} candles, got {total}")
        return

    for i in range(min_c, total, 3):
        c, h, lo, o, v = closes[:i], highs[:i], lows[:i], opens[:i], volumes[:i]
        price = float(c[-1])

        state = pipeline.process(c, h, lo, o, v,
                                 timestamp=candles[i - 1].timestamp,
                                 symbol=candles[0].symbol)

        oi_result = oi.analyze(price)
        gm_result = gm.estimate(price, iv=state.iv.current_iv or 0.15)

        render(state, oi_result, gm_result, i, total)
        time.sleep(0.08)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos Dashboard")
    parser.add_argument("--csv", type=str, help="Path to CSV candle file")
    args = parser.parse_args()

    if args.csv:
        run_csv(args.csv)
    else:
        run_synthetic()