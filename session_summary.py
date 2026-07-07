"""
Kairos Engine — Session Summary

Run at end of trading day to see:
  - All signals generated
  - Trades taken (if logged)
  - Win/loss stats
  - Regime distribution
  - Best/worst setups

Usage: python session_summary.py
       python session_summary.py --date 2025-01-06
"""

import argparse
from datetime import datetime

from data.storage.setup_memory import SetupMemory
from data.storage.trade_tracker import TradeTracker

B = "\033[1m"; R = "\033[0m"
GRN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; CYN = "\033[96m"; MAG = "\033[95m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--setup-db", type=str, default="data/storage/live_setups.db")
    parser.add_argument("--trade-db", type=str, default="data/storage/trades.db")
    args = parser.parse_args()

    date = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"\n{B}{CYN}{'='*70}{R}")
    print(f"{B}{CYN}  KAIROS — SESSION SUMMARY  │  {date}{R}")
    print(f"{B}{CYN}{'='*70}{R}")

    # --- Setup Memory Stats ---
    try:
        memory = SetupMemory(db_path=args.setup_db)
        stats = memory.get_stats()

        print(f"\n  {B}Setup Memory:{R}")
        print(f"    Total setups recorded: {stats['total_setups']}")
        print(f"    Entry windows:         {stats['entry_windows']}")

        if stats.get("by_state"):
            print(f"\n    {B}By State:{R}")
            for state, count in sorted(stats["by_state"].items(), key=lambda x: -x[1]):
                pct = count / stats["total_setups"] * 100
                print(f"      {state:<25s} {count:>5d}  ({pct:.1f}%)")

        memory.close()
    except FileNotFoundError:
        print(f"\n  {YEL}No setup database found at {args.setup_db}{R}")
        print(f"  Run live.py first to generate data")

    # --- Trade Stats ---
    try:
        tracker = TradeTracker(db_path=args.trade_db)
        trades = tracker.get_session_trades(date)
        t_stats = tracker.get_stats()

        print(f"\n  {B}Trade Tracker:{R}")

        if not trades:
            print(f"    No trades logged for {date}")
            print(f"    Use the interactive trade logger to record trades")
        else:
            print(f"    Trades today: {len(trades)}")
            print(f"")
            print(f"    {'Time':<10s} {'Symbol':<8s} {'Strike':>8s} {'Type':<4s} "
                  f"{'Entry':>8s} {'Exit':>8s} {'P&L':>8s} {'Status':<8s}")
            print(f"    {'-'*66}")

            for t in trades:
                pnl_str = f"₹{t['pnl']:.2f}" if t["pnl"] is not None else "—"
                pnl_color = GRN if (t["pnl"] or 0) > 0 else (RED if (t["pnl"] or 0) < 0 else R)
                exit_str = f"₹{t['exit_price']:.2f}" if t["exit_price"] else "—"
                ts = t["entry_time"][:8] if t["entry_time"] else "—"

                print(f"    {ts:<10s} {t['symbol']:<8s} {t['strike']:>8.0f} {t['option_type']:<4s} "
                      f"₹{t['entry_price']:>7.2f} {exit_str:>8s} "
                      f"{pnl_color}{pnl_str:>8s}{R} {t['status']:<8s}")

        if t_stats["closed"] > 0:
            print(f"")
            wr_color = GRN if t_stats["win_rate"] > 55 else (RED if t_stats["win_rate"] < 45 else YEL)
            pnl_color = GRN if t_stats["total_pnl"] > 0 else RED
            print(f"    {B}All-time stats:{R}")
            print(f"      Closed: {t_stats['closed']}  "
                  f"W: {GRN}{t_stats['wins']}{R}  L: {RED}{t_stats['losses']}{R}  "
                  f"WR: {wr_color}{t_stats['win_rate']:.1f}%{R}  "
                  f"P&L: {pnl_color}{B}₹{t_stats['total_pnl']:.2f}{R}")

        tracker.close()
    except Exception:
        print(f"\n  {YEL}No trade database found. Trades not logged yet.{R}")

    print(f"\n{CYN}{'='*70}{R}\n")


if __name__ == "__main__":
    main()