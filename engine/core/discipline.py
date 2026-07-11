"""
Kairos Engine — Discipline Guard

Built from Abhist's actual trading weaknesses:
  1. FOMO entries without setup → engine only signals when ALL gates pass
  2. Revenge trading after losses → daily trade limit + loss limit
  3. No consistent SL → engine calculates SL, shows countdown timer
  4. Averaging down → engine blocks signals on same direction after loss
  5. Trading outside best hours → highlights 1-3 PM sweet spot

This module doesn't BLOCK — it WARNS loudly. The trader decides.
"""

from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    time: datetime
    symbol: str
    strike: float
    option_type: str
    premium: float
    result: str = "OPEN"  # OPEN, WIN, LOSS
    pnl: float = 0.0


@dataclass
class DisciplineState:
    # Warnings (shown to trader)
    warnings: list[str] = field(default_factory=list)
    can_trade: bool = True
    reason_blocked: str = ""

    # Stats
    trades_today: int = 0
    daily_pnl: float = 0.0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_losses: int = 0

    # Time assessment
    in_prime_window: bool = False
    time_quality: str = ""  # "PRIME", "OK", "AVOID"


class DisciplineGuard:
    def __init__(
        self,
        max_trades_per_day: int = 2,
        hard_max_trades: int = 3,
        daily_loss_limit: float = 1200.0,
        account_size: float = 15000.0,
        prime_start_hour: int = 13,  # 1 PM
        prime_start_minute: int = 0,
        prime_end_hour: int = 15,  # 3 PM
        prime_end_minute: int = 0,
        opening_avoid_minutes: int = 30,
        closing_avoid_minutes: int = 10,  # last 10 min before 3:30
    ):
        self.max_trades = max_trades_per_day
        self.hard_max = hard_max_trades
        self.daily_loss_limit = daily_loss_limit
        self.account_size = account_size
        self.prime_start = (prime_start_hour, prime_start_minute)
        self.prime_end = (prime_end_hour, prime_end_minute)
        self.opening_avoid = opening_avoid_minutes
        self.closing_avoid = closing_avoid_minutes

        self._trades: list[TradeRecord] = []
        self._current_date: str = ""
        self._last_loss_direction: str = ""  # prevents revenge same direction

    def check(self, now: datetime | None = None) -> DisciplineState:
        """Check discipline state before any signal processing."""
        now = now or datetime.now()
        today = now.strftime("%Y-%m-%d")

        # Reset on new day
        if today != self._current_date:
            self._trades = []
            self._current_date = today
            self._last_loss_direction = ""

        state = DisciplineState()
        today_trades = [t for t in self._trades if t.time.strftime("%Y-%m-%d") == today]

        state.trades_today = len(today_trades)
        state.daily_pnl = sum(t.pnl for t in today_trades if t.result != "OPEN")
        state.wins_today = sum(1 for t in today_trades if t.result == "WIN")
        state.losses_today = sum(1 for t in today_trades if t.result == "LOSS")

        # Consecutive losses
        recent_results = [t.result for t in today_trades if t.result != "OPEN"]
        state.consecutive_losses = 0
        for r in reversed(recent_results):
            if r == "LOSS":
                state.consecutive_losses += 1
            else:
                break

        # --- Time quality ---
        market_open = now.replace(hour=9, minute=15, second=0)
        market_close = now.replace(hour=15, minute=30, second=0)
        mins_since_open = (now - market_open).total_seconds() / 60
        mins_to_close = (market_close - now).total_seconds() / 60

        prime_start = now.replace(hour=self.prime_start[0], minute=self.prime_start[1])
        prime_end = now.replace(hour=self.prime_end[0], minute=self.prime_end[1])

        if prime_start <= now <= prime_end:
            state.in_prime_window = True
            state.time_quality = "PRIME"
        elif mins_since_open < self.opening_avoid:
            state.time_quality = "AVOID"
            state.warnings.append(
                f"⚠ Opening range ({mins_since_open:.0f}m since open). "
                f"Wait until 9:45+ for cleaner setups."
            )
        elif mins_to_close < self.closing_avoid:
            state.time_quality = "AVOID"
            state.warnings.append("⚠ Market closing soon. New positions risky.")
        else:
            state.time_quality = "OK"

        # --- Trade count ---
        if state.trades_today >= self.hard_max:
            state.warnings.append(
                f"🛑 HARD LIMIT: {state.trades_today}/{self.hard_max} trades today. "
                f"You said max 3. Stop trading. Go for a walk."
            )
            state.can_trade = False
            state.reason_blocked = "Hard trade limit reached"
        elif state.trades_today >= self.max_trades:
            state.warnings.append(
                f"⚠ Soft limit: {state.trades_today}/{self.max_trades} trades. "
                f"Consider stopping. One more allowed but is it worth it?"
            )

        # --- Daily loss limit ---
        if state.daily_pnl <= -self.daily_loss_limit:
            state.warnings.append(
                f"🛑 DAILY LOSS LIMIT HIT: ₹{abs(state.daily_pnl):,.0f} lost today. "
                f"Your limit is ₹{self.daily_loss_limit:,.0f}. "
                f"Stop. Trading more will NOT recover losses."
            )
            state.can_trade = False
            state.reason_blocked = "Daily loss limit hit"
        elif state.daily_pnl <= -self.daily_loss_limit * 0.7:
            remaining = self.daily_loss_limit + state.daily_pnl
            state.warnings.append(
                f"⚠ Approaching loss limit: ₹{abs(state.daily_pnl):,.0f} lost. "
                f"Only ₹{remaining:,.0f} left before limit."
            )

        # --- Consecutive losses ---
        if state.consecutive_losses >= 2:
            state.warnings.append(
                f"⚠ {state.consecutive_losses} losses in a row. "
                f"Take a 15-minute break. Don't revenge trade."
            )

        # --- Prime window encouragement ---
        if state.in_prime_window and state.can_trade and state.trades_today == 0:
            state.warnings.append(
                "✅ Prime window (1-3 PM). Your best historical window. "
                "Wait for a clean signal."
            )

        return state

    def record_trade(
        self,
        time: datetime,
        symbol: str,
        strike: float,
        option_type: str,
        premium: float,
    ) -> int:
        """Record a new trade entry. Returns trade index."""
        trade = TradeRecord(
            time=time,
            symbol=symbol,
            strike=strike,
            option_type=option_type,
            premium=premium,
        )
        self._trades.append(trade)
        return len(self._trades) - 1

    def close_trade(self, index: int, pnl: float):
        """Close a trade with P&L."""
        if 0 <= index < len(self._trades):
            trade = self._trades[index]
            trade.pnl = pnl
            trade.result = "WIN" if pnl > 0 else "LOSS"

            if trade.result == "LOSS":
                self._last_loss_direction = trade.option_type

    def check_revenge(self, option_type: str) -> str | None:
        """Check if this trade direction is revenge trading."""
        if self._last_loss_direction == option_type:
            return (
                f"⚠ REVENGE ALERT: Your last loss was also {option_type}. "
                f"Same direction after a loss = emotional, not analytical. "
                f"Are you sure this is a fresh setup?"
            )
        return None

    def format_status(self, state: DisciplineState) -> str:
        """Format discipline status for terminal display."""
        GRN = "\033[92m"
        RED = "\033[91m"
        YEL = "\033[93m"
        B = "\033[1m"
        R = "\033[0m"

        lines = []

        # Time quality bar
        if state.time_quality == "PRIME":
            lines.append(f"  {GRN}{B}⏰ PRIME WINDOW (1-3 PM){R}")
        elif state.time_quality == "AVOID":
            lines.append(f"  {RED}⏰ AVOID ({state.time_quality}){R}")
        else:
            lines.append(f"  {YEL}⏰ {state.time_quality}{R}")

        # Daily stats
        pnl_color = GRN if state.daily_pnl >= 0 else RED
        lines.append(
            f"  📊 Trades: {state.trades_today}/{self.max_trades}  "
            f"W:{GRN}{state.wins_today}{R} L:{RED}{state.losses_today}{R}  "
            f"P&L: {pnl_color}₹{state.daily_pnl:+,.0f}{R}"
        )

        # Warnings
        for w in state.warnings:
            if "🛑" in w:
                lines.append(f"  {RED}{B}{w}{R}")
            elif "⚠" in w:
                lines.append(f"  {YEL}{w}{R}")
            elif "✅" in w:
                lines.append(f"  {GRN}{w}{R}")

        return "\n".join(lines)
