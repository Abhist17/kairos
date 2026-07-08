"""
Kairos Engine — Live Market Runner (Multi-Asset + TradingView Alerts)

python live.py --broker demo
python live.py --broker yfinance --symbol NIFTY
python live.py --broker yfinance --symbol BTC --interval 2m
python live.py --broker yfinance --symbol GOLD --interval 5m
python live.py --broker yfinance --symbol AAPL --interval 2m
python live.py --broker yfinance --symbol SENSEX --tv-alerts
"""

import argparse
import time
import sys
import os
import numpy as np
from datetime import datetime, timedelta

from engine.pipeline.signal_generator import SignalGenerator, format_signal
from engine.options.oi_gravity import OIGravityTracker
from engine.options.gamma_map import GammaGravityMap
from data.collectors.csv_loader import CSVLoader
from data.collectors.assets import get_asset
from data.storage.setup_memory import SetupMemory
from engine.core.enums import EntryWindow
from dashboard import render

B = "\033[1m"; R = "\033[0m"
GRN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; CYN = "\033[96m"


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


def run_live_broker(collector, symbol, interval, poll_seconds, tv_alerts=False, telegram_alerts=False):
    asset = get_asset(symbol)
    sig_gen = SignalGenerator(strike_step=asset.strike_step)
    oi_tracker = OIGravityTracker(strike_step=asset.strike_step)
    gamma_map = GammaGravityMap(strike_step=asset.strike_step)
    memory = SetupMemory(db_path="data/storage/live_setups.db")

    # Optional integrations
    tv = None
    tg = None

    if tv_alerts:
        from data.collectors.tradingview import TradingViewAlert
        tv = TradingViewAlert()

    if telegram_alerts:
        from data.collectors.telegram import TelegramAlert
        tg = TelegramAlert()
        tg.send_startup(symbol, collector.name)

    min_candles = sig_gen.min_candles
    entry_count = 0
    total_ticks = 0
    signals_log = []

    print(f"\n{B}{CYN}Kairos Engine — Live Mode{R}")
    print(f"  Broker:   {collector.name}")
    print(f"  Asset:    {symbol} ({asset.asset_type})")
    print(f"  TV:       {asset.tv_ticker}")
    print(f"  Strike:   {asset.strike_step} step")
    print(f"  Currency: {asset.currency}")
    print(f"  Interval: {interval}  │  Poll: {poll_seconds}s")
    print(f"  Alerts:   {'TV ' if tv else ''}{'Telegram ' if tg else ''}{'terminal only' if not tv and not tg else ''}")
    print(f"  Ctrl+C to stop\n")

    # Override yfinance ticker for the collector
    if hasattr(collector, 'yf') or hasattr(collector, '_connected'):
        from data.collectors.yfinance_collector import SYMBOL_MAP
        SYMBOL_MAP[symbol.upper()] = asset.yfinance_ticker

    if not collector.connect():
        print(f"{RED}Failed to connect.{R}")
        return

    use_real_oi = hasattr(collector, "get_oi_arrays")

    try:
        while True:
            candles = collector.get_candles(symbol, interval, count=max(200, min_candles + 20))

            if not candles or len(candles) < min_candles:
                n = len(candles) if candles else 0
                print(f"  Waiting... {n}/{min_candles} candles")
                time.sleep(poll_seconds)
                continue

            loader = CSVLoader(symbol)
            closes, highs, lows, opens, volumes = loader.to_arrays(candles)
            price = float(closes[-1])
            total_ticks += 1

            state, signal = sig_gen.process(
                closes, highs, lows, opens, volumes,
                symbol=symbol, timestamp=candles[-1].timestamp,
            )

            if use_real_oi:
                strikes, call_oi, put_oi = collector.get_oi_arrays(symbol)
                if strikes is not None:
                    oi_result = oi_tracker.analyze(price, call_oi, put_oi, strikes)
                    gm_result = gamma_map.estimate(
                        price, strikes=strikes, call_oi=call_oi, put_oi=put_oi,
                        iv=state.iv.current_iv or 0.15,
                    )
                else:
                    oi_result = oi_tracker.analyze(price)
                    gm_result = gamma_map.estimate(price, iv=state.iv.current_iv or 0.15)
            else:
                oi_result = oi_tracker.analyze(price)
                gm_result = gamma_map.estimate(price, iv=state.iv.current_iv or 0.15)

            render(state, oi_result, gm_result, total_ticks, len(candles))
            memory.record_setup(state)

            if signal:
                entry_count += 1
                signals_log.append((datetime.now(), signal))
                print(format_signal(signal))

                # Send to TradingView
                if tv:
                    tv.send_signal(signal, state, asset)

                # Send to Telegram
                if tg:
                    tg.send_signal(signal, state)

            elif sig_gen.last_rejection and state.entry_window == EntryWindow.OPEN:
                print(f"  {YEL}Filtered: {sig_gen.last_rejection}{R}")

            oi_tag = " (real OI)" if use_real_oi else ""
            stats = memory.get_stats()
            print(f"\n  {B}Live:{R} ticks={total_ticks}  "
                  f"setups={stats['total_setups']}  signals={len(signals_log)}"
                  f"{oi_tag}  │  {symbol}@{asset.tv_ticker}  │  Ctrl+C to stop")

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        print(f"\n\n{YEL}Stopping...{R}")
        if signals_log:
            print(f"\n  {B}Signals this session:{R}")
            for ts, sig in signals_log:
                print(f"    {ts.strftime('%H:%M:%S')}  {sig.action} {sig.symbol} "
                      f"{sig.strike:.0f} {sig.option_type}  "
                      f"{asset.currency}{sig.estimated_premium:.2f}  "
                      f"conf={sig.confidence:.0%}")
        stats = memory.get_stats()
        print(f"\n  Recorded {stats['total_setups']} setups, {len(signals_log)} signals")
    finally:
        collector.disconnect()
        memory.close()


def run_live_csv(csv_path, poll_seconds, symbol):
    asset = get_asset(symbol)
    sig_gen = SignalGenerator(strike_step=asset.strike_step)
    oi_tracker = OIGravityTracker(strike_step=asset.strike_step)
    gamma_map = GammaGravityMap(strike_step=asset.strike_step)
    memory = SetupMemory(db_path="data/storage/live_setups.db")
    loader = CSVLoader(symbol)

    min_candles = sig_gen.min_candles
    last_count = 0
    total_ticks = 0

    print(f"\n{B}{CYN}Kairos — CSV Mode ({symbol}){R}")
    print(f"  File: {csv_path}  │  Poll: {poll_seconds}s  │  Ctrl+C to stop\n")

    try:
        while True:
            if not os.path.exists(csv_path):
                print(f"  Waiting for file...")
                time.sleep(poll_seconds)
                continue

            candles = loader.load(csv_path)
            if len(candles) <= last_count:
                time.sleep(poll_seconds)
                continue

            last_count = len(candles)
            if len(candles) < min_candles:
                time.sleep(poll_seconds)
                continue

            closes, highs, lows, opens, volumes = loader.to_arrays(candles)
            total_ticks += 1
            state, signal = sig_gen.process(
                closes, highs, lows, opens, volumes,
                symbol=symbol, timestamp=candles[-1].timestamp,
            )
            oi_result = oi_tracker.analyze(float(closes[-1]))
            gm_result = gamma_map.estimate(float(closes[-1]))
            render(state, oi_result, gm_result, total_ticks, len(candles))
            memory.record_setup(state)

            if signal:
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

    print(f"\n{B}{CYN}Kairos — Demo{R}\n")
    time.sleep(1)

    try:
        for i in range(min_c, len(closes)):
            c, h, lo, o, v = closes[:i], highs[:i], lows[:i], opens[:i], volumes[:i]
            price = float(c[-1])
            state, signal = sig_gen.process(
                c, h, lo, o, v, iv_series[:i],
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
                time.sleep(3)
            else:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass

    if signals_log:
        print(f"\n  {B}Signals:{R}")
        for ts, sig in signals_log:
            print(f"    {ts.strftime('%H:%M:%S')}  {sig.action} NIFTY "
                  f"{sig.strike:.0f} {sig.option_type}  conf={sig.confidence:.0%}")

    stats = memory.get_stats()
    print(f"\n{B}Done.{R} Setups: {stats['total_setups']}  Signals: {len(signals_log)}")
    memory.close()


def generate_pine(symbol):
    from data.collectors.tradingview import TradingViewAlert
    tv = TradingViewAlert()
    script = tv.generate_pine_script(symbol)
    path = f"data/storage/tv_signals/kairos_{symbol.lower()}.pine"
    with open(path, "w") as f:
        f.write(script)
    print(f"\n{GRN}Pine Script saved: {path}{R}")
    print(f"Copy-paste into TradingView Pine Editor to display signals on chart.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairos Engine — Live")
    parser.add_argument(
        "--broker", type=str, default="demo",
        choices=["nse", "yfinance", "shoonya", "angel", "zerodha", "csv", "demo"],
    )
    parser.add_argument("--symbol", type=str, default="NIFTY")
    parser.add_argument("--interval", type=str, default="2m")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--tv-alerts", action="store_true", help="Send TradingView webhook alerts")
    parser.add_argument("--telegram", action="store_true", help="Send Telegram alerts")
    parser.add_argument("--pine", action="store_true", help="Generate Pine Script and exit")

    args = parser.parse_args()

    if args.pine:
        generate_pine(args.symbol)
        sys.exit(0)

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
            run_live_broker(
                collector, args.symbol, args.interval, args.poll,
                tv_alerts=args.tv_alerts, telegram_alerts=args.telegram,
            )