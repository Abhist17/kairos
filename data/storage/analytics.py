"""
Kairos Engine — Setup Memory Analytics

Analyzes stored setups to answer:
- Do ENTRY_WINDOW_OPEN states actually lead to profitable moves?
- Which trade_state has the best forward outcomes?
- What's the average move at 1/3/5/10/15 min for each state?
- Win rate by regime, by gate progression level

This is where the engine proves or disproves itself.
No claiming profitability — just data.
"""

import sqlite3
from pathlib import Path


class SetupAnalytics:
    def __init__(self, db_path: str = "data/storage/backtest.db"):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"No database at {db_path}. Run backtest first.")
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def summary(self) -> dict:
        """Overall stats."""
        row = self.conn.execute(
            "SELECT COUNT(*) as total, SUM(fwd_filled) as filled FROM setups"
        ).fetchone()
        return {"total": row["total"], "filled": row["filled"] or 0}

    def outcomes_by_state(self) -> list[dict]:
        """Average forward moves grouped by trade_state."""
        rows = self.conn.execute("""
            SELECT
                trade_state,
                COUNT(*) as n,
                ROUND(AVG(fwd_1m), 2) as avg_1m,
                ROUND(AVG(fwd_3m), 2) as avg_3m,
                ROUND(AVG(fwd_5m), 2) as avg_5m,
                ROUND(AVG(fwd_10m), 2) as avg_10m,
                ROUND(AVG(fwd_15m), 2) as avg_15m,
                ROUND(AVG(ABS(fwd_5m)), 2) as avg_abs_5m
            FROM setups
            WHERE fwd_filled = 1
            GROUP BY trade_state
            ORDER BY avg_abs_5m DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def outcomes_by_regime(self) -> list[dict]:
        """Average forward moves grouped by regime."""
        rows = self.conn.execute("""
            SELECT
                regime,
                COUNT(*) as n,
                ROUND(AVG(fwd_1m), 2) as avg_1m,
                ROUND(AVG(fwd_5m), 2) as avg_5m,
                ROUND(AVG(fwd_15m), 2) as avg_15m
            FROM setups
            WHERE fwd_filled = 1
            GROUP BY regime
            ORDER BY n DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def entry_window_performance(self) -> dict:
        """How do ENTRY_WINDOW_OPEN setups perform vs others?"""
        entry = self.conn.execute("""
            SELECT
                COUNT(*) as n,
                ROUND(AVG(fwd_1m), 2) as avg_1m,
                ROUND(AVG(fwd_3m), 2) as avg_3m,
                ROUND(AVG(fwd_5m), 2) as avg_5m,
                ROUND(AVG(fwd_10m), 2) as avg_10m,
                ROUND(AVG(fwd_15m), 2) as avg_15m,
                ROUND(AVG(ABS(fwd_5m)), 2) as avg_move_5m
            FROM setups
            WHERE entry_window = 'OPEN' AND fwd_filled = 1
        """).fetchone()

        non_entry = self.conn.execute("""
            SELECT
                COUNT(*) as n,
                ROUND(AVG(fwd_1m), 2) as avg_1m,
                ROUND(AVG(fwd_5m), 2) as avg_5m,
                ROUND(AVG(fwd_15m), 2) as avg_15m,
                ROUND(AVG(ABS(fwd_5m)), 2) as avg_move_5m
            FROM setups
            WHERE entry_window != 'OPEN' AND fwd_filled = 1
        """).fetchone()

        return {"entry_open": dict(entry), "non_entry": dict(non_entry)}

    def directional_accuracy(self) -> dict:
        """For entry windows with a bias, did price move in that direction?"""
        rows = self.conn.execute("""
            SELECT
                bias,
                COUNT(*) as n,
                SUM(CASE WHEN bias = 'BULLISH' AND fwd_5m > 0 THEN 1
                         WHEN bias = 'BEARISH' AND fwd_5m < 0 THEN 1
                         ELSE 0 END) as correct,
                ROUND(AVG(fwd_5m), 2) as avg_5m
            FROM setups
            WHERE entry_window = 'OPEN' AND fwd_filled = 1
                AND bias != 'NEUTRAL'
            GROUP BY bias
        """).fetchall()

        result = {}
        for r in rows:
            d = dict(r)
            d["accuracy_pct"] = (
                round(d["correct"] / d["n"] * 100, 1) if d["n"] > 0 else 0
            )
            result[d["bias"]] = d
        return result

    def print_report(self):
        """Print a full analytics report to terminal."""
        G = "\033[92m"
        Y = "\033[93m"
        R = "\033[91m"
        B = "\033[1m"
        RST = "\033[0m"

        print(f"\n{'=' * 80}")
        print(f"  {B}KAIROS — SETUP MEMORY ANALYTICS{RST}")
        print(f"{'=' * 80}\n")

        s = self.summary()
        print(f"  Total setups: {s['total']}  |  Outcomes filled: {s['filled']}\n")

        # By state
        print(f"  {B}Forward Outcomes by Trade State:{RST}")
        print(
            f"  {'State':<25s} {'N':>5s} {'1m':>8s} {'3m':>8s} {'5m':>8s} {'10m':>8s} {'15m':>8s}"
        )
        print(f"  {'-' * 73}")
        for r in self.outcomes_by_state():
            color = G if r["trade_state"] == "ENTRY_WINDOW_OPEN" else RST
            print(
                f"  {color}{r['trade_state']:<25s}{RST} "
                f"{r['n']:>5d} "
                f"{r['avg_1m']:>8.2f} {r['avg_3m']:>8.2f} "
                f"{r['avg_5m']:>8.2f} {r['avg_10m']:>8.2f} "
                f"{r['avg_15m']:>8.2f}"
            )

        # By regime
        print(f"\n  {B}Forward Outcomes by Regime:{RST}")
        print(f"  {'Regime':<22s} {'N':>5s} {'1m':>8s} {'5m':>8s} {'15m':>8s}")
        print(f"  {'-' * 50}")
        for r in self.outcomes_by_regime():
            print(
                f"  {r['regime']:<22s} {r['n']:>5d} "
                f"{r['avg_1m']:>8.2f} {r['avg_5m']:>8.2f} {r['avg_15m']:>8.2f}"
            )

        # Entry vs non-entry
        perf = self.entry_window_performance()
        print(f"\n  {B}Entry Window vs Non-Entry:{RST}")
        e = perf["entry_open"]
        ne = perf["non_entry"]
        print(
            f"  ENTRY_OPEN   (n={e['n']:>4d}): 1m={e['avg_1m']:>7.2f}  5m={e['avg_5m']:>7.2f}  15m={e['avg_15m']:>7.2f}  |move|={e['avg_move_5m']:>6.2f}"
        )
        print(
            f"  NON-ENTRY    (n={ne['n']:>4d}): 1m={ne['avg_1m']:>7.2f}  5m={ne['avg_5m']:>7.2f}  15m={ne['avg_15m']:>7.2f}  |move|={ne['avg_move_5m']:>6.2f}"
        )

        # Directional accuracy
        acc = self.directional_accuracy()
        if acc:
            print(f"\n  {B}Directional Accuracy (Entry Windows only):{RST}")
            for bias, d in acc.items():
                color = (
                    G
                    if d["accuracy_pct"] > 55
                    else (R if d["accuracy_pct"] < 45 else Y)
                )
                print(
                    f"  {bias:<10s} n={d['n']:>3d}  correct={d['correct']:>3d}  {color}accuracy={d['accuracy_pct']:.1f}%{RST}  avg_5m={d['avg_5m']:>7.2f}"
                )

        print(f"\n{'=' * 80}\n")

    def close(self):
        self.conn.close()
