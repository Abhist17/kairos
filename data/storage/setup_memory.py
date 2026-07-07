"""
Kairos Engine — Setup Memory

Stores every detected setup (including rejected ones) with the
complete MarketState snapshot, then records forward price outcomes
at 1, 3, 5, 10, and 15 minutes.

This creates our own labelled market-state dataset.
Only AFTER collecting enough data should ML be considered.

Storage is SQLite for simplicity — no external dependencies.
"""

import sqlite3
from pathlib import Path

from data.models.market_state import MarketState


class SetupMemory:
    def __init__(self, db_path: str = "data/storage/setups.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
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
                iv_percentile REAL,
                option_efficient INTEGER,
                theta_survival REAL,
                move_feasibility REAL,
                full_state_json TEXT,
                fwd_1m REAL,
                fwd_3m REAL,
                fwd_5m REAL,
                fwd_10m REAL,
                fwd_15m REAL,
                fwd_filled INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def record_setup(self, state: MarketState) -> int:
        """Store a setup snapshot. Returns the row ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO setups (
                timestamp, symbol, price, regime, bias, trade_state,
                entry_window, regime_confidence, compression_score,
                structure_score, pressure_score, thesis_separation,
                iv_state, iv_percentile, option_efficient, theta_survival,
                move_feasibility, full_state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.timestamp.isoformat(),
                state.symbol,
                state.last_price,
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
                state.iv.iv_percentile,
                1 if state.option.is_efficient else 0,
                state.option.theta_survival_minutes,
                state.option.move_feasibility,
                state.model_dump_json(),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore

    def fill_forward_outcomes(
        self,
        setup_id: int,
        fwd_1m: float,
        fwd_3m: float,
        fwd_5m: float,
        fwd_10m: float,
        fwd_15m: float,
    ):
        """Fill forward price outcomes after the setup was recorded."""
        self.conn.execute(
            """
            UPDATE setups SET
                fwd_1m = ?, fwd_3m = ?, fwd_5m = ?,
                fwd_10m = ?, fwd_15m = ?, fwd_filled = 1
            WHERE id = ?
            """,
            (fwd_1m, fwd_3m, fwd_5m, fwd_10m, fwd_15m, setup_id),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        """Quick summary of stored setups."""
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(fwd_filled) FROM setups"
        ).fetchone()
        total = row[0] or 0
        filled = row[1] or 0

        by_state = {}
        for r in self.conn.execute(
            "SELECT trade_state, COUNT(*) FROM setups GROUP BY trade_state"
        ):
            by_state[r[0]] = r[1]

        entry_count = self.conn.execute(
            "SELECT COUNT(*) FROM setups WHERE entry_window = 'OPEN'"
        ).fetchone()[0]

        return {
            "total_setups": total,
            "outcomes_filled": filled,
            "entry_windows": entry_count,
            "by_state": by_state,
        }

    def close(self):
        self.conn.close()
