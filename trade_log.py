"""
Kairos Engine — Interactive Trade Logger

Quick CLI to log trades you actually take based on signals.

Usage:
    python trade_log.py entry NIFTY 24500 CE 148.50 104 237
    python trade_log.py exit 1 180.00 target
    python trade_log.py exit 1 95.00 stoploss
    python trade_log.py open                    # show open trades
    python trade_log.py stats                   # show all stats
"""

import sys
from data.storage.trade_tracker import TradeTracker

B = "\033[1m"; R = "\033[0m"
GRN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; CYN = "\033[96m"


def main():
    if len(sys.argv) < 2:
        print(f"""
{B}Kairos Trade Logger{R}

  {CYN}Log an entry:{R}
    python trade_log.py entry NIFTY 24500 CE 148.50 104 237

  {CYN}Log an exit:{R}
    python trade_log.py exit <trade_id> <exit_price> <reason>
    python trade_log.py exit 1 180.00 target
    python trade_log.py exit 1 95.00 stoploss

  {CYN}View open trades:{R}
    python trade_log.py open

  {CYN}View stats:{R}
    python trade_log.py stats
""")
        return

    tracker = TradeTracker()
    cmd = sys.argv[1].lower()

    if cmd == "entry":
        if len(sys.argv) < 8:
            print(f"  {RED}Usage: trade_log.py entry SYMBOL STRIKE TYPE ENTRY SL TARGET{R}")
            print(f"  Example: trade_log.py entry NIFTY 24500 CE 148.50 104 237")
            tracker.close()
            return

        symbol = sys.argv[2]
        strike = float(sys.argv[3])
        opt_type = sys.argv[4].upper()
        entry = float(sys.argv[5])
        sl = float(sys.argv[6])
        target = float(sys.argv[7])
        notes = " ".join(sys.argv[8:]) if len(sys.argv) > 8 else ""

        tid = tracker.log_entry(
            symbol=symbol, strike=strike, option_type=opt_type,
            entry_price=entry, stoploss=sl, target=target, notes=notes,
        )

        rr = (target - entry) / (entry - sl) if entry > sl else 0
        print(f"\n  {GRN}✓ Trade #{tid} logged{R}")
        print(f"    BUY {symbol} {strike:.0f} {opt_type} @ ₹{entry:.2f}")
        print(f"    SL: ₹{sl:.2f}  │  Target: ₹{target:.2f}  │  R:R 1:{rr:.1f}\n")

    elif cmd == "exit":
        if len(sys.argv) < 4:
            print(f"  {RED}Usage: trade_log.py exit <trade_id> <exit_price> [reason]{R}")
            tracker.close()
            return

        tid = int(sys.argv[2])
        exit_price = float(sys.argv[3])
        reason = sys.argv[4] if len(sys.argv) > 4 else "manual"

        tracker.log_exit(tid, exit_price, reason)

        # Show result
        trade = tracker.conn.execute(
            "SELECT * FROM trades WHERE id = ?", (tid,)
        ).fetchone()

        if trade:
            pnl = trade["pnl"]
            pnl_color = GRN if pnl > 0 else RED
            emoji = "✅" if pnl > 0 else "❌"
            print(f"\n  {emoji} Trade #{tid} closed")
            print(f"    {trade['symbol']} {trade['strike']:.0f} {trade['option_type']}")
            print(f"    Entry: ₹{trade['entry_price']:.2f}  →  Exit: ₹{exit_price:.2f}")
            print(f"    P&L: {pnl_color}{B}₹{pnl:.2f}{R}  ({reason})\n")

    elif cmd == "open":
        trades = tracker.get_open_trades()
        if not trades:
            print(f"\n  {YEL}No open trades{R}\n")
        else:
            print(f"\n  {B}Open Trades:{R}")
            for t in trades:
                print(f"    #{t['id']}  {t['symbol']} {t['strike']:.0f} {t['option_type']}  "
                      f"@ ₹{t['entry_price']:.2f}  "
                      f"SL: ₹{t['stoploss']:.2f}  T: ₹{t['target']:.2f}  "
                      f"({t['entry_time'][:16]})")
            print()

    elif cmd == "stats":
        tracker.print_summary()

    else:
        print(f"  {RED}Unknown command: {cmd}{R}")
        print(f"  Commands: entry, exit, open, stats")

    tracker.close()


if __name__ == "__main__":
    main()