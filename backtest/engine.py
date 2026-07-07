"""
Kairos Engine — Backtest Engine

Runs the full pipeline over historical OHLCV data candle-by-candle,
records every setup into Setup Memory, and fills forward outcomes
at 1, 3, 5, 10, 15 minute marks.

This is HOW we build the labelled dataset. No cherry-picking.
Every single state gets recorded with what actually happened next.

Usage:
    engine = BacktestEngine()
    report = engine.run(closes, highs, lows, opens, volumes)
    engine.print_report(report)
"""

from datetime import datetime, timedelta

from engine.pipeline.market_pipeline import MarketPipeline
from data.storage.setup_memory import SetupMemory
from engine.core.enums import TradeState, EntryWindow
from engine.core.types import FloatArray


class BacktestEngine:
    def __init__(
        self,
        db_path: str = "data/storage/backtest.db",
        step: int = 1,
        record_all: bool = True,
        record_only_entries: bool = False,
    ):
        self.pipeline = MarketPipeline()
        self.memory = SetupMemory(db_path=db_path)
        self.step = step
        self.record_all = record_all
        self.record_only_entries = record_only_entries

    def run(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        opens: FloatArray,
        volumes: FloatArray,
        iv_series: FloatArray | None = None,
        symbol: str = "NIFTY",
        start_time: datetime | None = None,
    ) -> dict:
        n = len(closes)
        min_c = self.pipeline.min_candles
        start_time = start_time or datetime(2025, 1, 6, 9, 15)

        total = 0
        entries = 0
        recorded_ids: list[tuple[int, int]] = []  # (setup_id, candle_index)

        for i in range(min_c, n, self.step):
            ts = start_time + timedelta(minutes=i)

            state = self.pipeline.process(
                closes[:i],
                highs[:i],
                lows[:i],
                opens[:i],
                volumes[:i],
                iv_series[:i] if iv_series is not None else None,
                symbol=symbol,
                timestamp=ts,
            )

            total += 1
            is_entry = state.entry_window == EntryWindow.OPEN

            if is_entry:
                entries += 1

            should_record = (
                self.record_all
                or (self.record_only_entries and is_entry)
                or state.trade_state != TradeState.NO_SETUP
            )

            if should_record:
                rid = self.memory.record_setup(state)
                recorded_ids.append((rid, i))

        # Fill forward outcomes
        filled = 0
        forward_offsets = [1, 3, 5, 10, 15]

        for rid, idx in recorded_ids:
            fwd = []
            base_price = float(closes[idx - 1])

            for offset in forward_offsets:
                fwd_idx = idx + offset
                if fwd_idx < n:
                    fwd_price = float(closes[fwd_idx])
                    fwd.append(fwd_price - base_price)
                else:
                    fwd.append(0.0)

            self.memory.fill_forward_outcomes(
                rid, fwd[0], fwd[1], fwd[2], fwd[3], fwd[4]
            )
            filled += 1

        stats = self.memory.get_stats()

        return {
            "candles_processed": total,
            "setups_recorded": len(recorded_ids),
            "entry_windows": entries,
            "outcomes_filled": filled,
            "db_stats": stats,
        }

    def close(self):
        self.memory.close()
