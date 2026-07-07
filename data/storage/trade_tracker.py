"""
Kairos Engine — Trade Tracker

Log actual trades you take based on signals, then compare
signal prediction vs what actually happened.

This closes the loop: Engine → Signal → You Trade → Track Outcome → Improve Engine
"""

import sqlite3
from datetime import datetime
from pathlib import Path


class TradeTracker:
    def __init__(self, db_path: str = "data/storage/trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                strike REAL,
                option_type TEXT,
                action TEXT DEFAULT 'BUY',
                entry_price REAL,
                stoploss REAL,
                target REAL,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL,
                signal_confidence REAL,
                signal_feasibility REAL,
                signal_survival REAL,
                regime TEXT,
                bias TEXT,
                thesis_separation REAL,
                status TEXT DEFAULT 'OPEN',
                entry_time TEXT,
                exit_time TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def log_entry(
        self,
        symbol: str,
        strike: float,
        option_type: str,
        entry_price: float,
        stoploss: float,
        target: float,
        confidence: float = 0.0,
        feasibility: float = 0.0,
        survival: float = 0.0,
        regime: str = "",
        bias: str = "",
        separation: float = 0.0,
        notes: str = "",
    ) -> int:
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            """
            INSERT INTO trades (
                timestamp, symbol, strike, option_type, action,
                entry_price, stoploss, target,
                signal_confidence, signal_feasibility, signal_survival,
                regime, bias, thesis_separation, status, entry_time, notes
            ) VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """,
            (now, symbol, strike, option_type, entry_price, stoploss, target,
             confidence, feasibility, survival, regime, bias, separation, now, notes),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_exit(
        self, trade_id: int, exit_price: float, reason: str = "manual"
    ):
        trade = self.conn.execute(
            "SELECT entry_price FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()

        if not trade:
            return

        pnl = exit_price - trade["entry_price"]
        now = datetime.now().isoformat()

        self.conn.execute(
            """
            UPDATE trades SET
                exit_price = ?, exit_reason = ?, pnl = ?,
                status = 'CLOSED', exit_time = ?
            WHERE id = ?
            """,
            (exit_price, reason, round(pnl, 2), now, trade_id),
        )
        self.conn.commit()

    def get_open_trades(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_trades(self, date: str | None = None) -> list[dict]:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        rows = self.conn.execute(
            "SELECT * FROM trades WHERE timestamp LIKE ? ORDER BY created_at",
            (f"{date}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        closed = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()[0]

        if closed == 0:
            return {"total": total, "closed": 0, "wins": 0, "losses": 0,
                    "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

        wins = self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND pnl > 0"
        ).fetchone()[0]

        row = self.conn.execute(
            "SELECT AVG(pnl), SUM(pnl) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()

        return {
            "total": total,
            "closed": closed,
            "wins": wins,
            "losses": closed - wins,
            "win_rate": round(wins / closed * 100, 1),
            "avg_pnl": round(row[0] or 0, 2),
            "total_pnl": round(row[1] or 0, 2),
        }

    def print_summary(self):
        B = "\033[1m"; R = "\033[0m"
        GRN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"

        stats = self.get_stats()
        wr_color = GRN if stats["win_rate"] > 55 else (RED if stats["win_rate"] < 45 else YEL)
        pnl_color = GRN if stats["total_pnl"] > 0 else RED

        print(f"\n{'='*60}")
        print(f"  {B}TRADE TRACKER SUMMARY{R}")
        print(f"{'='*60}")
        print(f"  Total trades: {stats['total']}")
        print(f"  Closed:       {stats['closed']}")
        print(f"  Wins:         {GRN}{stats['wins']}{R}")
        print(f"  Losses:       {RED}{stats['losses']}{R}")
        print(f"  Win rate:     {wr_color}{stats['win_rate']:.1f}%{R}")
        print(f"  Avg P&L:      {pnl_color}₹{stats['avg_pnl']:.2f}{R}")
        print(f"  Total P&L:    {pnl_color}{B}₹{stats['total_pnl']:.2f}{R}")
        print(f"{'='*60}\n")

    def close(self):
        self.conn.close()