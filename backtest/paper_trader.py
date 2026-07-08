"""
Kairos Engine — Paper Trading Auto-Validator

Runs live, generates signals, then AUTOMATICALLY tracks what
happens after each signal without you placing any trade.

After each signal:
  - Records entry price
  - Watches if SL or Target gets hit
  - Records time to hit
  - Logs max favorable / max adverse move
  - Calculates what P&L would have been

Run this during market hours for a few days BEFORE risking real money.

Usage:
    python -m backtest.paper_trader --broker yfinance --symbol NIFTY
"""

import argparse
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

from engine.pipeline.signal_generator import SignalGenerator, format_signal, TradeSignal
from engine.options.oi_gravity import OIGravityTracker
from engine.options.gamma_map import GammaGravityMap
from data.collectors.csv_loader import CSVLoader
from dashboard import render

B = "\033[1m"
R = "\033[0m"
GRN = "\033[92m"
RED = "\033[91m"
YEL = "\033[93m"
CYN = "\033[96m"
MAG = "\033[95m"


@dataclass
class PaperTrade:
    signal_time: datetime
    symbol: str
    strike: float
    option_type: str  # CE or PE
    entry_spot: float
    estimated_premium: float
    stoploss_premium: float
    target_premium: float
    confidence: float
    delta: float
    status: str = "OPEN"  # OPEN, WIN_TARGET, LOSS_SL, EXPIRED
    exit_spot: float = 0.0
    exit_time: datetime | None = None
    max_favorable_spot: float = 0.0
    max_adverse_spot: float = 0.0
    estimated_pnl: float = 0.0
    candles_held: int = 0
    survival_limit: float = 30.0  # minutes


@dataclass
class PaperSession:
    trades: list[PaperTrade] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def open_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades if t.status == "OPEN"]

    @property
    def closed_trades(self) -> list[PaperTrade]:
        return [t for t in self.trades if t.status != "OPEN"]

    def add(self, signal: TradeSignal, spot: float, ts: datetime, delta: float = 0.5):
        self.trades.append(
            PaperTrade(
                signal_time=ts,
                symbol=signal.symbol,
                strike=signal.strike,
                option_type=signal.option_type,
                entry_spot=spot,
                estimated_premium=signal.estimated_premium,
                stoploss_premium=signal.stoploss_premium,
                target_premium=signal.target_premium,
                confidence=signal.confidence,
                delta=delta,
                survival_limit=signal.survival_minutes,
            )
        )

    def update(self, current_spot: float, current_time: datetime):
        """Check all open trades against current price."""
        for trade in self.open_trades:
            trade.candles_held += 1

            # Track extremes
            if trade.option_type == "CE":
                favorable = current_spot - trade.entry_spot
                adverse = trade.entry_spot - current_spot
            else:
                favorable = trade.entry_spot - current_spot
                adverse = current_spot - trade.entry_spot

            trade.max_favorable_spot = max(trade.max_favorable_spot, favorable)
            trade.max_adverse_spot = max(trade.max_adverse_spot, max(0, adverse))

            # Estimate premium change from spot move
            spot_move = current_spot - trade.entry_spot
            if trade.option_type == "PE":
                spot_move = -spot_move

            premium_change = spot_move * trade.delta
            current_premium = trade.estimated_premium + premium_change

            # Check target
            if current_premium >= trade.target_premium:
                trade.status = "WIN_TARGET"
                trade.exit_spot = current_spot
                trade.exit_time = current_time
                trade.estimated_pnl = trade.target_premium - trade.estimated_premium
                continue

            # Check stoploss
            if current_premium <= trade.stoploss_premium:
                trade.status = "LOSS_SL"
                trade.exit_spot = current_spot
                trade.exit_time = current_time
                trade.estimated_pnl = trade.stoploss_premium - trade.estimated_premium
                continue

            # Check time expiry
            elapsed = (current_time - trade.signal_time).total_seconds() / 60
            if elapsed > trade.survival_limit:
                trade.status = "EXPIRED"
                trade.exit_spot = current_spot
                trade.exit_time = current_time
                trade.estimated_pnl = premium_change
                continue

    def print_status(self):
        open_t = self.open_trades
        closed = self.closed_trades

        if not self.trades:
            return

        wins = sum(1 for t in closed if "WIN" in t.status)
        losses = sum(1 for t in closed if "LOSS" in t.status)
        expired = sum(1 for t in closed if t.status == "EXPIRED")
        total_pnl = sum(t.estimated_pnl for t in closed)

        print(f"\n  {B}Paper Trading:{R}")
        print(
            f"    Open: {len(open_t)}  │  "
            f"Closed: {len(closed)}  │  "
            f"{GRN}W:{wins}{R}  {RED}L:{losses}{R}  Exp:{expired}  │  "
            f"Est P&L: {GRN if total_pnl > 0 else RED}₹{total_pnl:+.2f}{R}"
        )

        for t in open_t:
            elapsed = (datetime.now() - t.signal_time).total_seconds() / 60
            color = GRN if t.max_favorable_spot > 0 else YEL
            print(
                f"    {color}◎ {t.symbol} {t.strike:.0f} {t.option_type}  "
                f"entry={t.entry_spot:,.2f}  "
                f"maxfav={t.max_favorable_spot:+.1f}  "
                f"held={t.candles_held}  "
                f"elapsed={elapsed:.0f}m{R}"
            )

        for t in closed[-3:]:  # last 3 closed
            color = GRN if "WIN" in t.status else RED
            print(
                f"    {color}{'✓' if 'WIN' in t.status else '✗'} {t.symbol} "
                f"{t.strike:.0f} {t.option_type}  "
                f"pnl=₹{t.estimated_pnl:+.2f}  "
                f"{t.status}  held={t.candles_held}{R}"
            )

    def print_final_report(self):
        closed = self.closed_trades

        print(f"\n{B}{CYN}{'=' * 70}{R}")
        print(f"{B}{CYN}  PAPER TRADING REPORT{R}")
        print(f"{CYN}{'=' * 70}{R}")

        if not closed:
            print("  No closed trades.\n")
            return

        wins = [t for t in closed if "WIN" in t.status]
        losses = [t for t in closed if "LOSS" in t.status]
        expired = [t for t in closed if t.status == "EXPIRED"]
        total_pnl = sum(t.estimated_pnl for t in closed)
        win_rate = len(wins) / len(closed) * 100

        wr_color = GRN if win_rate > 55 else (RED if win_rate < 45 else YEL)
        pnl_color = GRN if total_pnl > 0 else RED

        print(f"\n  Total signals: {len(self.trades)}")
        print(f"  Closed:        {len(closed)}")
        print(f"  Still open:    {len(self.open_trades)}")
        print(f"\n  {GRN}Wins:    {len(wins)}{R}")
        print(f"  {RED}Losses:  {len(losses)}{R}")
        print(f"  Expired: {len(expired)}")
        print(f"  Win Rate: {wr_color}{B}{win_rate:.1f}%{R}")
        print(f"\n  Total Est P&L: {pnl_color}{B}₹{total_pnl:+.2f}{R}")

        if wins:
            avg_win = sum(t.estimated_pnl for t in wins) / len(wins)
            print(f"  Avg Win:       {GRN}₹{avg_win:+.2f}{R}")
        if losses:
            avg_loss = sum(t.estimated_pnl for t in losses) / len(losses)
            print(f"  Avg Loss:      {RED}₹{avg_loss:+.2f}{R}")

        avg_favorable = np.mean([t.max_favorable_spot for t in closed])
        avg_adverse = np.mean([t.max_adverse_spot for t in closed])
        print(f"\n  Avg Max Favorable: {GRN}+{avg_favorable:.1f}{R} pts")
        print(f"  Avg Max Adverse:   {RED}-{avg_adverse:.1f}{R} pts")

        print(f"\n  {B}All Trades:{R}")
        print(
            f"    {'Time':<18s} {'Type':<8s} {'Spot':>10s} {'P&L':>8s} "
            f"{'Held':>5s} {'MaxFav':>7s} {'Status':<14s}"
        )
        print(f"    {'-' * 72}")

        for t in closed:
            color = GRN if "WIN" in t.status else RED
            print(
                f"    {t.signal_time.strftime('%Y-%m-%d %H:%M'):<18s} "
                f"{t.option_type + ' ' + str(int(t.strike)):<8s} "
                f"{t.entry_spot:>10,.2f} "
                f"{color}₹{t.estimated_pnl:>+7.2f}{R} "
                f"{t.candles_held:>5d} "
                f"{t.max_favorable_spot:>+7.1f} "
                f"{color}{t.status:<14s}{R}"
            )

        print(f"\n{CYN}{'=' * 70}{R}\n")


def get_collector(broker):
    if broker == "yfinance":
        from data.collectors.yfinance_collector import YFinanceCollector

        return YFinanceCollector()
    elif broker == "nse":
        from data.collectors.nse_live import NSELiveCollector

        return NSELiveCollector()
    return None


def main():
    parser = argparse.ArgumentParser(description="Kairos Paper Trader")
    parser.add_argument(
        "--broker", type=str, default="yfinance", choices=["yfinance", "nse"]
    )
    parser.add_argument("--symbol", type=str, default="NIFTY")
    parser.add_argument("--interval", type=str, default="2m")
    parser.add_argument("--poll", type=int, default=60)

    args = parser.parse_args()

    sig_gen = SignalGenerator()
    oi_tracker = OIGravityTracker()
    gamma_map = GammaGravityMap()
    session = PaperSession()
    collector = get_collector(args.broker)

    if not collector:
        print(f"{RED}Unknown broker{R}")
        return

    print(f"\n{B}{CYN}Kairos — Paper Trading Mode{R}")
    print(f"  Broker: {collector.name}")
    print(f"  Symbol: {args.symbol}")
    print("  NO real money. Auto-tracking signal outcomes.\n")

    if not collector.connect():
        print(f"{RED}Connection failed{R}")
        return

    min_c = sig_gen.min_candles
    total_ticks = 0

    try:
        while True:
            candles = collector.get_candles(
                args.symbol, args.interval, count=max(200, min_c + 20)
            )

            if not candles or len(candles) < min_c:
                n = len(candles) if candles else 0
                print(f"  Waiting... {n}/{min_c} candles")
                time.sleep(args.poll)
                continue

            loader = CSVLoader(args.symbol)
            closes, highs, lows, opens, volumes = loader.to_arrays(candles)
            price = float(closes[-1])
            ts = candles[-1].timestamp
            total_ticks += 1

            state, signal = sig_gen.process(
                closes,
                highs,
                lows,
                opens,
                volumes,
                symbol=args.symbol,
                timestamp=ts,
            )

            # Update open paper trades
            session.update(price, ts)

            oi_result = oi_tracker.analyze(price)
            gm_result = gamma_map.estimate(price, iv=state.iv.current_iv or 0.15)
            render(state, oi_result, gm_result, total_ticks, len(candles))

            # New signal → open paper trade
            if signal:
                delta = (
                    signal.all_candidates[0].estimated_delta
                    if signal.all_candidates
                    else 0.5
                )
                session.add(signal, price, ts, delta)
                print(format_signal(signal))
                print(f"  {MAG}▸ Paper trade opened — auto-tracking outcome{R}")
            elif sig_gen.last_rejection:
                print(f"  {YEL}Filtered: {sig_gen.last_rejection}{R}")

            session.print_status()

            print(f"\n  tick={total_ticks}  │  next in {args.poll}s  │  Ctrl+C to stop")
            time.sleep(args.poll)

    except KeyboardInterrupt:
        pass

    session.print_final_report()
    collector.disconnect()


if __name__ == "__main__":
    main()
