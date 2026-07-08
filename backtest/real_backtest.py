"""
Kairos Engine — Real Data Backtest

Downloads actual NIFTY history via yfinance, runs every candle
through the full pipeline + signal generator, tracks what
would have happened if you followed every signal.

No cherry-picking. Every signal recorded with forward outcomes.

Usage:
    python -m backtest.real_backtest
    python -m backtest.real_backtest --symbol BANKNIFTY --days 30
    python -m backtest.real_backtest --interval 5m --days 60
"""

import argparse
import os
import sqlite3
import numpy as np
from pathlib import Path
from engine.core.enums import EntryWindow

from engine.pipeline.signal_generator import SignalGenerator


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
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "SBIN": "SBIN.NS",
}

PERIOD_MAP = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
}


class RealBacktest:
    def __init__(self, db_path: str = "data/storage/real_backtest.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.db_path = db_path

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                candle_idx INTEGER,
                spot REAL,
                strike REAL,
                option_type TEXT,
                bias TEXT,
                premium REAL,
                stoploss REAL,
                target REAL,
                confidence REAL,
                feasibility REAL,
                gamma_theta REAL,
                survival_minutes REAL,
                regime TEXT,
                thesis_separation REAL,
                fwd_1 REAL, fwd_2 REAL, fwd_3 REAL,
                fwd_5 REAL, fwd_10 REAL, fwd_15 REAL, fwd_30 REAL,
                max_favorable REAL,
                max_adverse REAL,
                direction_correct INTEGER,
                hit_target INTEGER,
                hit_stoploss INTEGER,
                result TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS all_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candle_idx INTEGER,
                timestamp TEXT,
                price REAL,
                regime TEXT,
                bias TEXT,
                trade_state TEXT,
                entry_window TEXT,
                regime_confidence REAL,
                compression_score REAL,
                structure_score REAL,
                pressure_score REAL,
                thesis_separation REAL,
                iv_state TEXT,
                gates_ready INTEGER
            )
        """)
        self.conn.commit()

    def run(
        self,
        symbol: str = "NIFTY",
        interval: str = "2m",
        days: int = 30,
    ) -> dict:
        # Download data
        print(f"\n{B}{CYN}{'=' * 80}{R}")
        print(f"{B}{CYN}  KAIROS — Real Data Backtest{R}")
        print(f"{B}{CYN}{'=' * 80}{R}")

        closes, highs, lows, opens, volumes, timestamps = self._download(
            symbol, interval, days
        )

        if closes is None or len(closes) < 130:
            print(
                f"{RED}Not enough data. Got {len(closes) if closes is not None else 0} candles.{R}"
            )
            return {}

        print(f"  Candles: {len(closes)}")
        print(f"  Range:   {timestamps[0]} → {timestamps[-1]}")
        print(f"  Price:   {closes[0]:,.2f} → {closes[-1]:,.2f}")

        # Run pipeline
        sig_gen = SignalGenerator()
        min_c = sig_gen.min_candles
        total = len(closes)
        signals = []
        total_states = 0

        print(f"\n  Running pipeline on {total - min_c} candles...\n")

        progress_step = max(1, (total - min_c) // 40)

        for i in range(min_c, total):
            c = closes[:i]
            h = highs[:i]
            lo = lows[:i]
            o = opens[:i]
            v = volumes[:i]

            state, signal = sig_gen.process(
                c,
                h,
                lo,
                o,
                v,
                symbol=symbol,
                timestamp=timestamps[i - 1],
            )

            # Count gates ready
            gates_ready = sum(1 for g in state.state_machine.gates if g.ready)

            # Store state
            self.conn.execute(
                """INSERT INTO all_states
                   (candle_idx, timestamp, price, regime, bias, trade_state,
                    entry_window, regime_confidence, compression_score,
                    structure_score, pressure_score, thesis_separation,
                    iv_state, gates_ready)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    i,
                    str(timestamps[i - 1]),
                    float(c[-1]),
                    state.regime.value,
                    state.bias.value,
                    state.trade_state.value,
                    state.entry_window.value,
                    state.regime_metrics.regime_confidence,
                    state.compression.compression_score,
                    state.structure.structure_score,
                    state.flow.pressure_score,
                    state.thesis.separation,
                    state.iv.state.value,
                    gates_ready,
                ),
            )
            total_states += 1

            # Process signal
            if signal:
                fwd = self._calc_forward(closes, i, signal.option_type)
                max_fav, max_adv = self._calc_extremes(
                    closes, i, signal.option_type, lookforward=30
                )

                # Did direction play out?
                fwd_5 = fwd.get(5, 0)
                if signal.option_type == "CE":
                    direction_ok = 1 if fwd_5 > 0 else 0
                else:
                    direction_ok = 1 if fwd_5 < 0 else 0

                # Would SL or target have been hit?
                # Approximate: premium moves ~delta * spot_move
                delta = (
                    signal.all_candidates[0].estimated_delta
                    if signal.all_candidates
                    else 0.5
                )
                sl_dist = signal.estimated_premium - signal.stoploss_premium
                tgt_dist = signal.target_premium - signal.estimated_premium

                sl_spot_needed = sl_dist / delta if delta > 0 else 999
                tgt_spot_needed = tgt_dist / delta if delta > 0 else 999

                hit_tgt = 1 if abs(max_fav) >= tgt_spot_needed else 0
                hit_sl = 1 if abs(max_adv) >= sl_spot_needed else 0

                # Result classification
                if hit_tgt and not hit_sl:
                    result = "WIN_TARGET"
                elif hit_tgt and hit_sl:
                    # Check which came first (simplified: use 5-candle check)
                    result = "MIXED"
                elif hit_sl:
                    result = "LOSS_SL"
                elif direction_ok:
                    result = "WIN_PARTIAL"
                else:
                    result = "LOSS_DIRECTION"

                self.conn.execute(
                    """INSERT INTO signals
                       (timestamp, candle_idx, spot, strike, option_type, bias,
                        premium, stoploss, target, confidence, feasibility,
                        gamma_theta, survival_minutes, regime, thesis_separation,
                        fwd_1, fwd_2, fwd_3, fwd_5, fwd_10, fwd_15, fwd_30,
                        max_favorable, max_adverse, direction_correct,
                        hit_target, hit_stoploss, result)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(timestamps[i - 1]),
                        i,
                        float(c[-1]),
                        signal.strike,
                        signal.option_type,
                        signal.all_candidates[0].moneyness
                        if signal.all_candidates
                        else "",
                        signal.estimated_premium,
                        signal.stoploss_premium,
                        signal.target_premium,
                        signal.confidence,
                        signal.all_candidates[0].move_feasibility
                        if signal.all_candidates
                        else 0,
                        signal.all_candidates[0].gamma_theta_ratio
                        if signal.all_candidates
                        else 0,
                        signal.survival_minutes,
                        state.regime.value,
                        state.thesis.separation,
                        fwd.get(1, 0),
                        fwd.get(2, 0),
                        fwd.get(3, 0),
                        fwd.get(5, 0),
                        fwd.get(10, 0),
                        fwd.get(15, 0),
                        fwd.get(30, 0),
                        max_fav,
                        max_adv,
                        direction_ok,
                        hit_tgt,
                        hit_sl,
                        result,
                    ),
                )

                # Log rejection if entry window open but signal filtered
                if state.entry_window == EntryWindow.OPEN and not signal:
                    rejection = sig_gen.last_rejection
                    if rejection and (i - min_c) % 50 == 0:
                        print(f"  {GRY}  filtered: {rejection}{R}")

                signals.append(
                    {
                        "idx": i,
                        "ts": timestamps[i - 1],
                        "spot": float(c[-1]),
                        "strike": signal.strike,
                        "type": signal.option_type,
                        "conf": signal.confidence,
                        "fwd_5": fwd.get(5, 0),
                        "result": result,
                        "premium": signal.estimated_premium,
                    }
                )

                marker = GRN if direction_ok else RED
                print(
                    f"  {marker}▶ Signal #{len(signals):>3d}{R}  "
                    f"{timestamps[i - 1].strftime('%Y-%m-%d %H:%M')}  "
                    f"{signal.option_type} {signal.strike:.0f}  "
                    f"spot={float(c[-1]):,.2f}  "
                    f"conf={signal.confidence:.0%}  "
                    f"fwd5={fwd.get(5, 0):+.1f}  "
                    f"{marker}{result}{R}"
                )

            # Progress bar
            if (i - min_c) % progress_step == 0:
                pct = (i - min_c) / (total - min_c) * 100
                done = int(pct / 2.5)
                print(
                    f"\r  [{GRN}{'█' * done}{GRY}{'░' * (40 - done)}{R}] {pct:.0f}%  "
                    f"candle {i}/{total}  signals={len(signals)}",
                    end="",
                    flush=True,
                )

        self.conn.commit()
        print(f"\r  [{'█' * 40}] 100%  Done.{' ' * 30}")

        return self._print_report(signals, total_states, symbol, interval, timestamps)

    def _download(self, symbol, interval, days):
        try:
            import yfinance as yf
        except ImportError:
            print(f"{RED}yfinance not installed. Run: pip install yfinance{R}")
            return None, None, None, None, None, None

        ticker_sym = SYMBOL_MAP.get(symbol.upper(), f"{symbol}.NS")
        period = PERIOD_MAP.get(interval, "60d")

        print(f"\n  Downloading {ticker_sym} ({interval}) last {period}...")

        try:
            ticker = yf.Ticker(ticker_sym)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"{RED}No data returned for {ticker_sym}{R}")
                return None, None, None, None, None, None

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df.dropna(subset=["Close"])

            closes = df["Close"].values.astype(np.float64)
            highs = df["High"].values.astype(np.float64)
            lows = df["Low"].values.astype(np.float64)
            opens = df["Open"].values.astype(np.float64)
            volumes = df["Volume"].values.astype(np.float64)
            timestamps = [ts.to_pydatetime() for ts in df.index]

            print(f"  Downloaded {len(closes)} candles")
            return closes, highs, lows, opens, volumes, timestamps

        except Exception as e:
            print(f"{RED}Download failed: {e}{R}")
            return None, None, None, None, None, None

    def _calc_forward(self, closes, idx, option_type, offsets=(1, 2, 3, 5, 10, 15, 30)):
        base = float(closes[idx - 1])
        result = {}
        for off in offsets:
            fwd_idx = idx + off
            if fwd_idx < len(closes):
                move = float(closes[fwd_idx]) - base
                result[off] = round(move, 2)
            else:
                result[off] = 0.0
        return result

    def _calc_extremes(self, closes, idx, option_type, lookforward=30):
        base = float(closes[idx - 1])
        end = min(idx + lookforward, len(closes))
        if end <= idx:
            return 0.0, 0.0

        future = closes[idx:end]
        moves = future - base

        if option_type == "CE":
            max_favorable = float(np.max(moves))
            max_adverse = float(np.min(moves))
        else:
            max_favorable = float(-np.min(moves))  # down is favorable for PE
            max_adverse = float(-np.max(moves))  # up is adverse for PE

        return round(max_favorable, 2), round(abs(max_adverse), 2)

    def _print_report(self, signals, total_states, symbol, interval, timestamps):
        print(f"\n{B}{CYN}{'=' * 80}{R}")
        print(f"{B}{CYN}  BACKTEST RESULTS — {symbol} ({interval}){R}")
        print(
            f"{B}{CYN}  {timestamps[0].strftime('%Y-%m-%d')} → {timestamps[-1].strftime('%Y-%m-%d')}{R}"
        )
        print(f"{B}{CYN}{'=' * 80}{R}")

        n = len(signals)
        print(f"\n  Total candles analyzed: {total_states}")
        print(f"  Total signals:         {n}")

        if n == 0:
            print(f"\n  {YEL}No signals generated. Engine stayed out of market.{R}")
            print("  This could mean: data too short, or no setups met all 6 gates.\n")
            return {"signals": 0}

        # Load from DB for detailed analysis
        rows = self.conn.execute("SELECT * FROM signals ORDER BY candle_idx").fetchall()

        # Direction accuracy
        correct = sum(1 for r in rows if r["direction_correct"])
        accuracy = correct / n * 100

        # Result breakdown
        results = {}
        for r in rows:
            res = r["result"]
            results[res] = results.get(res, 0) + 1

        # Forward moves
        fwd_moves = {
            "1": [r["fwd_1"] for r in rows],
            "2": [r["fwd_2"] for r in rows],
            "3": [r["fwd_3"] for r in rows],
            "5": [r["fwd_5"] for r in rows],
            "10": [r["fwd_10"] for r in rows],
            "15": [r["fwd_15"] for r in rows],
            "30": [r["fwd_30"] for r in rows],
        }

        # Directional forward (positive = correct direction)
        dir_fwd = []
        for r in rows:
            if r["option_type"] == "CE":
                dir_fwd.append(r["fwd_5"])
            else:
                dir_fwd.append(-r["fwd_5"])

        # CE vs PE breakdown
        ce_signals = [r for r in rows if r["option_type"] == "CE"]
        pe_signals = [r for r in rows if r["option_type"] == "PE"]

        ce_correct = sum(1 for r in ce_signals if r["direction_correct"])
        pe_correct = sum(1 for r in pe_signals if r["direction_correct"])

        # Print
        acc_color = GRN if accuracy > 55 else (RED if accuracy < 45 else YEL)

        print(f"\n  {B}Direction Accuracy:{R}")
        print(f"    Overall:  {acc_color}{B}{accuracy:.1f}%{R}  ({correct}/{n})")

        if ce_signals:
            ce_acc = ce_correct / len(ce_signals) * 100
            ce_color = GRN if ce_acc > 55 else RED
            print(
                f"    CE (Buy): {ce_color}{ce_acc:.1f}%{R}  ({ce_correct}/{len(ce_signals)})"
            )

        if pe_signals:
            pe_acc = pe_correct / len(pe_signals) * 100
            pe_color = GRN if pe_acc > 55 else RED
            print(
                f"    PE (Buy): {pe_color}{pe_acc:.1f}%{R}  ({pe_correct}/{len(pe_signals)})"
            )

        print(f"\n  {B}Result Breakdown:{R}")
        for res, count in sorted(results.items(), key=lambda x: -x[1]):
            pct = count / n * 100
            color = GRN if "WIN" in res else (RED if "LOSS" in res else YEL)
            bar = "█" * int(pct / 3)
            print(
                f"    {color}{res:<18s}{R} {count:>4d}  ({pct:5.1f}%)  {color}{bar}{R}"
            )

        print(f"\n  {B}Average Spot Move After Signal (points):{R}")
        print(
            f"    {'Offset':<8s} {'Avg Move':>10s} {'Correct Dir':>12s} {'Abs Move':>10s}"
        )
        print(f"    {'-' * 44}")
        for label, moves in fwd_moves.items():
            if not moves:
                continue
            avg = np.mean(moves)
            avg_abs = np.mean(np.abs(moves))
            # Directional avg
            dir_m = []
            for i, r in enumerate(rows):
                m = moves[i]
                if r["option_type"] == "PE":
                    m = -m
                dir_m.append(m)
            avg_dir = np.mean(dir_m) if dir_m else 0
            dir_color = GRN if avg_dir > 0 else RED

            print(
                f"    {label + ' candle':<8s} {avg:>+10.2f} {dir_color}{avg_dir:>+12.2f}{R} {avg_abs:>10.2f}"
            )

        # Max favorable / adverse
        avg_fav = np.mean([r["max_favorable"] for r in rows])
        avg_adv = np.mean([r["max_adverse"] for r in rows])
        print(f"\n  {B}Max Move Within 30 Candles After Signal:{R}")
        print(
            f"    Avg max favorable (in signal direction): {GRN}{avg_fav:+.2f}{R} pts"
        )
        print(
            f"    Avg max adverse (against signal):        {RED}{avg_adv:+.2f}{R} pts"
        )
        print(
            f"    Favorable/Adverse ratio:                 {avg_fav / avg_adv:.2f}x"
            if avg_adv > 0
            else ""
        )

        # Confidence vs accuracy
        high_conf = [r for r in rows if r["confidence"] >= 0.6]
        low_conf = [r for r in rows if r["confidence"] < 0.6]

        if high_conf and low_conf:
            hc_acc = (
                sum(1 for r in high_conf if r["direction_correct"])
                / len(high_conf)
                * 100
            )
            lc_acc = (
                sum(1 for r in low_conf if r["direction_correct"]) / len(low_conf) * 100
            )
            print(f"\n  {B}Confidence Calibration:{R}")
            hc_color = GRN if hc_acc > 55 else RED
            lc_color = GRN if lc_acc > 55 else RED
            print(
                f"    High confidence (≥60%): {hc_color}{hc_acc:.1f}%{R} accurate  (n={len(high_conf)})"
            )
            print(
                f"    Low confidence  (<60%): {lc_color}{lc_acc:.1f}%{R} accurate  (n={len(low_conf)})"
            )

        # By regime
        regime_groups = {}
        for r in rows:
            reg = r["regime"]
            if reg not in regime_groups:
                regime_groups[reg] = {"total": 0, "correct": 0}
            regime_groups[reg]["total"] += 1
            if r["direction_correct"]:
                regime_groups[reg]["correct"] += 1

        if len(regime_groups) > 1:
            print(f"\n  {B}Accuracy by Regime:{R}")
            for reg, data in sorted(
                regime_groups.items(), key=lambda x: -x[1]["total"]
            ):
                acc = data["correct"] / data["total"] * 100
                color = GRN if acc > 55 else RED
                print(f"    {reg:<22s} {color}{acc:5.1f}%{R}  (n={data['total']})")

        # Signal list
        print(f"\n  {B}All Signals:{R}")
        print(
            f"    {'#':>3s}  {'Date':>16s}  {'Type':<4s} {'Strike':>7s}  "
            f"{'Spot':>10s}  {'Conf':>5s}  {'Fwd5':>7s}  {'MaxFav':>7s}  {'Result':<16s}"
        )
        print(f"    {'-' * 90}")

        for i, r in enumerate(rows):
            color = GRN if r["direction_correct"] else RED
            print(
                f"    {i + 1:>3d}  {r['timestamp'][:16]:>16s}  "
                f"{r['option_type']:<4s} {r['strike']:>7.0f}  "
                f"{r['spot']:>10.2f}  {r['confidence']:>5.0%}  "
                f"{color}{r['fwd_5']:>+7.1f}{R}  "
                f"{r['max_favorable']:>+7.1f}  "
                f"{color}{r['result']:<16s}{R}"
            )

        print(f"\n{CYN}{'=' * 80}{R}")
        print(f"  Database saved: {self.db_path}")
        print("  Rerun anytime: python -m backtest.real_backtest")
        print(f"{CYN}{'=' * 80}{R}\n")

        self.conn.close()

        return {
            "signals": n,
            "accuracy": accuracy,
            "results": results,
            "avg_fwd_5_directional": float(np.mean(dir_fwd)) if dir_fwd else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Kairos — Real Data Backtest")
    parser.add_argument("--symbol", type=str, default="NIFTY")
    parser.add_argument(
        "--interval",
        type=str,
        default="2m",
        help="1m (7 days), 2m (60 days), 5m (60 days)",
    )
    parser.add_argument("--days", type=int, default=60)

    args = parser.parse_args()

    bt = RealBacktest()
    bt.run(symbol=args.symbol, interval=args.interval, days=args.days)


if __name__ == "__main__":
    main()
