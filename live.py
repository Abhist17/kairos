"""
Kairos Engine — Live Market Runner with Trade Signals

python live.py --demo                         # synthetic demo
python live.py --broker yfinance              # FREE, ~15min delayed
python live.py --broker yfinance --symbol BANKNIFTY
"""

import argparse
import time
import sys
import os
from datetime import datetime, timedelta

from engine.pipeline.signal_generator import SignalGenerator, format_signal
from engine.options.oi_gravity import OIGravityTracker
from engine.options.gamma_map import GammaGravityMap
from data.collectors.csv_loader import CSVLoader
from data.storage.setup_memory import SetupMemory
from engine.core.enums import EntryWindow
from dashboard import render

B = "\033[1m"
R = "\033[0m"
GRN = "\033[92m"
RED = "\033[91m"
YEL = "\033[93m"
CYN = "\033[96m"


def get_collector(broker):
    if broker == "nse":
        from data.collectors.nse_live import NSELiveCollector

        return NSELiveCollector()
    elif broker == "yfinance":
        from data.collectors.yfinance_collector import YFinanceCollector

        return YFinanceCollector()
    elif broker == "shoonya":
        from data.collectors.shoonya import ShoonyaCollector

        return ShoonyaCollector()
    elif broker == "angel":
        from data.collectors.angel import AngelCollector

        return AngelCollector()
    elif broker == "zerodha":
        from data.collectors.zerodha import ZerodhaCollector

        return ZerodhaCollector()
    return None


def run_live_broker(collector, symbol, interval, poll_seconds):
    sig_gen = SignalGenerator()
    oi_tracker = OIGravityTracker()
    gamma_map = GammaGravityMap()
    memory = SetupMemory(db_path="data/storage/live_setups.db")

    min_candles = sig_gen.min_candles
    entry_count = 0
    total_ticks = 0
    signals_log = []

    print(f"\n{B}{CYN}Kairos Engine — Live Mode with Trade Signals{R}")
    print(f"  Broker:   {collector.name}")
    print(f"  Symbol:   {symbol}")
    print(f"  Interval: {interval}")
    print(f"  Poll:     every {poll_seconds}s")
    print(f"  Min candles: {min_candles}")
    print("  Ctrl+C to stop\n")

    if not collector.connect():
        print(f"{RED}Failed to connect.{R}")
        return

    use_real_oi = hasattr(collector, "get_oi_arrays")

    try:
        while True:
            candles = collector.get_candles(
                symbol, interval, count=max(200, min_candles + 20)
            )

            if not candles or len(candles) < min_candles:
                n = len(candles) if candles else 0
                print(f"  Waiting... got {n}/{min_candles} candles")
                time.sleep(poll_seconds)
                continue

            loader = CSVLoader(symbol)
            closes, highs, lows, opens, volumes = loader.to_arrays(candles)
            price = float(closes[-1])
            total_ticks += 1

            # Run pipeline + signal generator
            state, signal = sig_gen.process(
                closes,
                highs,
                lows,
                opens,
                volumes,
                symbol=symbol,
                timestamp=candles[-1].timestamp,
            )

            # OI
            if use_real_oi:
                strikes, call_oi, put_oi = collector.get_oi_arrays(symbol)
                if strikes is not None:
                    oi_result = oi_tracker.analyze(price, call_oi, put_oi, strikes)
                    gm_result = gamma_map.estimate(
                        price,
                        strikes=strikes,
                        call_oi=call_oi,
                        put_oi=put_oi,
                        iv=state.iv.current_iv or 0.15,
                    )
                else:
                    oi_result = oi_tracker.analyze(price)
                    gm_result = gamma_map.estimate(
                        price, iv=state.iv.current_iv or 0.15
                    )
            else:
                oi_result = oi_tracker.analyze(price)
                gm_result = gamma_map.estimate(price, iv=state.iv.current_iv or 0.15)

            # Render dashboard
            render(state, oi_result, gm_result, total_ticks, len(candles))

            # Record setup
            memory.record_setup(state)

            # Show signal if generated
            if signal:
                entry_count += 1
                signals_log.append((datetime.now(), signal))
                print(format_signal(signal))
            elif state.entry_window == EntryWindow.OPEN:
                entry_count += 1
                print(f"\n  {YEL}Entry window open but no suitable strike found{R}\n")

            oi_tag = " (real OI)" if use_real_oi else " (est OI)"
            stats = memory.get_stats()
            print(
                f"\n  {B}Live:{R} ticks={total_ticks}  "
                f"setups={stats['total_setups']}  signals={len(signals_log)}"
                f"{oi_tag}  │  next in {poll_seconds}s  │  Ctrl+C to stop"
            )

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print(f"\n\n{YEL}Stopping...{R}")
        if signals_log:
            print(f"\n  {B}Signals generated this session:{R}")
            for ts, sig in signals_log:
                print(
                    f"    {ts.strftime('%H:%M:%S')}  {sig.action} {sig.symbol} "
                    f"{sig.strike:.0f} {sig.option_type}  "
                    f"₹{sig.estimated_premium:.2f}  "
                    f"conf={sig.confidence:.0%}"
                )
        stats = memory.get_stats()
        print(
            f"\n  Recorded {stats['total_setups']} setups, {len(signals_log)} trade signals"
        )
    finally:
        collector.disconnect()
        memory.close()


def run_live_csv(csv_path, poll_seconds, symbol):
    sig_gen = SignalGenerator()
    oi_tracker = OIGravityTracker()
    gamma_map = GammaGravityMap()
    memory = SetupMemory(db_path="data/storage/live_setups.db")
    loader = CSVLoader(symbol)

    min_candles = sig_gen.min_candles
    last_count = 0
    total_ticks = 0
    signals_log = []

    print(f"\n{B}{CYN}Kairos Engine — CSV Polling with Signals{R}")
    print(f"  File: {csv_path}  │  Poll: {poll_seconds}s  │  Ctrl+C to stop\n")

    try:
        while True:
            if not os.path.exists(csv_path):
                print(f"  Waiting for file: {csv_path}")
                time.sleep(poll_seconds)
                continue

            candles = loader.load(csv_path)
            if len(candles) <= last_count:
                time.sleep(poll_seconds)
                continue

            last_count = len(candles)
            if len(candles) < min_candles:
                print(f"  Have {len(candles)}/{min_candles} candles...")
                time.sleep(poll_seconds)
                continue

            closes, highs, lows, opens, volumes = loader.to_arrays(candles)
            total_ticks += 1

            state, signal = sig_gen.process(
                closes,
                highs,
                lows,
                opens,
                volumes,
                symbol=symbol,
                timestamp=candles[-1].timestamp,
            )

            oi_result = oi_tracker.analyze(float(closes[-1]))
            gm_result = gamma_map.estimate(
                float(closes[-1]), iv=state.iv.current_iv or 0.15
            )
            render(state, oi_result, gm_result, total_ticks, len(candles))
            memory.record_setup(state)

            if signal:
                signals_log.append((datetime.now(), signal))
                print(format_signal(signal))

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print(f"\n{YEL}Stopped.{R}")
    finally:
        memory.close()


def run_demo():
    from main import generate_synthetic_nifty

    sig_gen = SignalGenerator()
    oi_tracker = OIGravityTracker()
    gamma_map = GammaGravityMap()
    memory = SetupMemory(db_path="data/storage/demo_setups.db")

    data = generate_synthetic_nifty(n_candles=600)
    closes, highs, lows = data["closes"], data["highs"], data["lows"]
    opens, volumes, iv_series = data["opens"], data["volumes"], data["iv_series"]

    min_c = sig_gen.min_candles
    signals_log = []

    print(f"\n{B}{CYN}Kairos Engine — Demo with Trade Signals{R}")
    print("  Ctrl+C to stop\n")
    time.sleep(2)

    try:
        for i in range(min_c, len(closes)):
            c, h, lo, o, v = closes[:i], highs[:i], lows[:i], opens[:i], volumes[:i]
            price = float(c[-1])

            state, signal = sig_gen.process(
                c,
                h,
                lo,
                o,
                v,
                iv_series[:i],
                symbol="NIFTY",
                timestamp=datetime(2025, 1, 6, 9, 15) + timedelta(minutes=i),
            )

            oi_result = oi_tracker.analyze(price)
            gm_result = gamma_map.estimate(price, iv=state.iv.current_iv or 0.15)
            render(state, oi_result, gm_result, i, len(closes))
            memory.record_setup(state)

            if signal:
                signals_log.append((state.timestamp, signal))
                print(format_signal(signal))
                time.sleep(5)  # pause on signals
            else:
                time.sleep(0.25)

    except KeyboardInterrupt:
        pass

    if signals_log:
        print(f"\n  {B}All signals this session:{R}")
        for ts, sig in signals_log:
            print(
                f"    {ts.strftime('%H:%M:%S')}  {sig.action} NIFTY "
                f"{sig.strike:.0f} {sig.option_type}  "
                f"₹{sig.estimated_premium:.2f}  conf={sig.confidence:.0%}  "
                f"SL=₹{sig.stoploss_premium:.2f}  T=₹{sig.target_premium:.2f}"
            )

    stats = memory.get_stats()
    print(f"\n{B}Done.{R} Setups: {stats['total_setups']}  Signals: {len(signals_log)}")
    memory.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos Engine — Live")
    parser.add_argument(
        "--broker",
        type=str,
        default="demo",
        choices=["nse", "yfinance", "shoonya", "angel", "zerodha", "csv", "demo"],
    )
    parser.add_argument("--symbol", type=str, default="NIFTY")
    parser.add_argument("--interval", type=str, default="2m")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--csv", type=str)

    args = parser.parse_args()

    if args.broker == "demo":
        run_demo()
    elif args.broker == "csv":
        if not args.csv:
            print("Provide --csv path")
            sys.exit(1)
        run_live_csv(args.csv, args.poll, args.symbol)
    else:
        collector = get_collector(args.broker)
        if collector:
            run_live_broker(collector, args.symbol, args.interval, args.poll)
