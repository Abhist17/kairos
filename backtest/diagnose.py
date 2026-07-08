"""
Kairos Engine — Diagnostic Backtest

Shows EVERY entry window the engine opened, and exactly which
filter blocked or passed each one. This tells us what to fix.

Usage:
    python -m backtest.diagnose --symbol NIFTY --interval 1m
    python -m backtest.diagnose --symbol NIFTY --interval 2m
"""

import argparse
import numpy as np

from engine.pipeline.market_pipeline import MarketPipeline
from engine.core.enums import EntryWindow, MarketBias, MarketRegime, IVState
from engine.core.config import TradeFilterConfig
from engine.features.multi_timeframe import MultiTimeframe

B = "\033[1m"
R = "\033[0m"
GRN = "\033[92m"
RED = "\033[91m"
YEL = "\033[93m"
CYN = "\033[96m"
MAG = "\033[95m"
GRY = "\033[90m"

SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}
PERIOD_MAP = {"1m": "7d", "2m": "60d", "5m": "60d"}


def download(symbol, interval):
    import yfinance as yf

    ticker_sym = SYMBOL_MAP.get(symbol.upper(), f"{symbol}.NS")
    period = PERIOD_MAP.get(interval, "60d")
    print(f"  Downloading {ticker_sym} ({interval}) last {period}...")
    ticker = yf.Ticker(ticker_sym)
    df = ticker.history(period=period, interval=interval)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Close"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--interval", default="2m")
    parser.add_argument(
        "--date", default=None, help="Filter to specific date YYYY-MM-DD"
    )
    args = parser.parse_args()

    df = download(args.symbol, args.interval)
    if df.empty:
        print(f"{RED}No data{R}")
        return

    closes = df["Close"].values.astype(np.float64)
    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)
    opens = df["Open"].values.astype(np.float64)
    volumes = df["Volume"].values.astype(np.float64)
    timestamps = [ts.to_pydatetime() for ts in df.index]

    pipeline = MarketPipeline()
    mtf = MultiTimeframe(resample_factor=7)
    f = TradeFilterConfig()
    min_c = pipeline.min_candles

    # Filter to specific date if requested
    if args.date:
        target_date = args.date
    else:
        target_date = timestamps[-1].strftime("%Y-%m-%d")

    print(f"\n{B}{CYN}{'=' * 90}{R}")
    print(
        f"{B}{CYN}  KAIROS DIAGNOSTIC — {args.symbol} ({args.interval}) — {target_date}{R}"
    )
    print(f"{B}{CYN}{'=' * 90}{R}")
    print(f"  Total candles: {len(closes)}")
    print("  Analyzing every candle, showing ALL entry windows + filter verdicts\n")

    entry_windows = 0
    signal_would_pass = 0
    last_signal_time = None

    print(
        f"  {'Time':<8s} {'Price':>10s} {'Regime':<18s} {'Bias':<8s} "
        f"{'Gates':>5s} {'Sep':>6s} {'IV':<12s} {'Filters':<40s} {'Fwd5':>7s}"
    )
    print(f"  {'-' * 120}")

    for i in range(min_c, len(closes)):
        ts = timestamps[i - 1]
        ts_date = ts.strftime("%Y-%m-%d")

        # Only show requested date
        if ts_date != target_date:
            continue

        c, h, lo, o, v = closes[:i], highs[:i], lows[:i], opens[:i], volumes[:i]
        price = float(c[-1])

        state = pipeline.process(c, h, lo, o, v, symbol=args.symbol, timestamp=ts)

        gates_ready = sum(1 for g in state.state_machine.gates if g.ready)

        # Show every state where at least 3 gates are ready, or entry window open
        if gates_ready < 3 and state.entry_window != EntryWindow.OPEN:
            continue

        # Calculate forward move
        fwd5 = 0.0
        if i + 5 < len(closes):
            fwd5 = float(closes[i + 5]) - price

        # Run every filter manually to show which blocks
        filters_log = []
        would_pass = True

        # F1: Opening range
        market_open = ts.replace(hour=9, minute=15, second=0)
        mins_open = (ts - market_open).total_seconds() / 60
        if 0 < mins_open < f.opening_range_skip_minutes:
            filters_log.append(f"{RED}OR({mins_open:.0f}m){R}")
            would_pass = False
        else:
            filters_log.append(f"{GRN}OR✓{R}")

        # F2: Cooldown
        if last_signal_time:
            elapsed = (ts - last_signal_time).total_seconds() / 60
            if elapsed < f.signal_cooldown_minutes:
                filters_log.append(f"{RED}CD({elapsed:.0f}m){R}")
                would_pass = False
            else:
                filters_log.append(f"{GRN}CD✓{R}")
        else:
            filters_log.append(f"{GRN}CD✓{R}")

        # F3: Regime
        if state.regime in f.allowed_regimes:
            filters_log.append(f"{GRN}RG✓{R}")
        else:
            filters_log.append(f"{RED}RG({state.regime.value[:6]}){R}")
            would_pass = False

        # F4: MTF
        mtf_r = mtf.analyze(c, h, lo, state.regime, state.thesis.primary_bias)
        if mtf_r.aligned:
            filters_log.append(f"{GRN}MTF✓{R}")
        else:
            filters_log.append(f"{RED}MTF({mtf_r.higher_bias.value[:4]}){R}")
            would_pass = False

        # F5: Separation
        if state.thesis.separation >= f.min_thesis_separation:
            filters_log.append(f"{GRN}SEP✓{R}")
        else:
            filters_log.append(f"{RED}SEP({state.thesis.separation:.2f}){R}")
            would_pass = False

        # F6: IV
        if state.iv.state == IVState.OVEREXPANDED and f.block_overexpanded_iv:
            filters_log.append(f"{RED}IV!{R}")
            would_pass = False
        else:
            filters_log.append(f"{GRN}IV✓{R}")

        # F7: Momentum
        bias = state.thesis.primary_bias
        if bias != MarketBias.NEUTRAL and len(c) > f.confirmation_candles + 1:
            recent = np.diff(c[-(f.confirmation_candles + 1) :])
            if bias == MarketBias.BULLISH:
                mom_ok = all(m > 0 for m in recent)
            else:
                mom_ok = all(m < 0 for m in recent)
            if mom_ok:
                filters_log.append(f"{GRN}MOM✓{R}")
            else:
                filters_log.append(f"{RED}MOM✗{R}")
                would_pass = False
        else:
            filters_log.append(f"{RED}MOM(N){R}")
            would_pass = False

        # Mark entry windows
        is_entry = state.entry_window == EntryWindow.OPEN
        if is_entry:
            entry_windows += 1

        if would_pass and is_entry:
            signal_would_pass += 1
            last_signal_time = ts

        # Color code the row
        regime_color = (
            GRN
            if state.regime == MarketRegime.TREND_EXPANSION
            else (CYN if state.regime == MarketRegime.COMPRESSION else YEL)
        )
        fwd_color = (
            GRN
            if (
                (bias == MarketBias.BULLISH and fwd5 > 0)
                or (bias == MarketBias.BEARISH and fwd5 < 0)
            )
            else RED
        )

        entry_marker = f"{GRN}★{R}" if is_entry else " "
        pass_marker = (
            f"{GRN}▶{R}"
            if (would_pass and is_entry)
            else (f"{YEL}○{R}" if is_entry else " ")
        )

        filters_str = " ".join(filters_log)

        print(
            f" {entry_marker}{pass_marker}"
            f"{ts.strftime('%H:%M'):<7s} "
            f"{price:>10,.2f} "
            f"{regime_color}{state.regime.value:<18s}{R} "
            f"{bias.value:<8s} "
            f"{gates_ready:>1d}/6   "
            f"{state.thesis.separation:>+.3f} "
            f"{state.iv.state.value:<12s} "
            f"{filters_str}  "
            f"{fwd_color}{fwd5:>+7.1f}{R}"
        )

    print(f"\n  {B}Summary for {target_date}:{R}")
    print(f"    Entry windows opened:      {entry_windows}")
    print(f"    Would pass all filters:    {signal_would_pass}")
    print(f"    Blocked by filters:        {entry_windows - signal_would_pass}")

    # Show which filters block the most
    print(f"\n  {B}Filter Impact:{R}")
    print(f"    If a filter shows {RED}RED{R} frequently on rows with good fwd5,")
    print("    that filter is too aggressive and should be relaxed.")
    print(
        f"    If it shows {RED}RED{R} on bad fwd5, the filter is working correctly.\n"
    )

    print(f"{CYN}{'=' * 90}{R}\n")


if __name__ == "__main__":
    main()
