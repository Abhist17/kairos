"""
Kairos Engine — Run Backtest

Generates synthetic data, runs the full pipeline over it,
stores all setups with forward outcomes, then prints analytics.

Usage: python -m backtest.run_backtest
"""

import numpy as np
import os
from backtest.engine import BacktestEngine
from data.storage.analytics import SetupAnalytics


def generate_long_session(n=2000, seed=42):
    """Longer synthetic session with multiple regime cycles."""
    rng = np.random.default_rng(seed)
    base = 25000.0
    prices = [base]

    phases = [
        (150, 0.0, 0.6, "COMPRESS"),
        (200, 3.5, 1.0, "TREND_UP"),
        (100, 0.5, 9.0, "EXHAUST"),
        (150, 0.0, 3.5, "MEAN_REV"),
        (200, -3.0, 1.2, "TREND_DN"),
        (100, 0.0, 12.0, "CHAOTIC"),
        (120, 0.0, 0.4, "COMPRESS2"),
        (180, 2.8, 1.5, "TREND_UP2"),
        (100, -0.5, 7.0, "EXHAUST2"),
        (200, 0.0, 4.0, "MEAN_REV2"),
        (200, -2.0, 2.0, "TREND_DN2"),
        (200, 0.0, 10.0, "CHAOTIC2"),
    ]

    labels = []
    for nc, drift, noise, label in phases:
        for _ in range(nc):
            prices.append(prices[-1] + drift + rng.normal(0, noise))
            labels.append(label)

    prices = np.array(prices[1:])
    n = len(prices)
    highs = prices + np.abs(rng.normal(0, 1, n)) * prices * 0.0003 + 0.5
    lows = prices - np.abs(rng.normal(0, 1, n)) * prices * 0.0003 - 0.5
    opens = np.roll(prices, 1)
    opens[0] = base
    volumes = rng.integers(5000, 50000, n).astype(float)

    return prices, highs, lows, opens, volumes, labels


def main():
    B = "\033[1m"
    R = "\033[0m"

    db_path = "data/storage/backtest.db"

    # Remove old db
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"\n{B}Running backtest...{R}")

    closes, highs, lows, opens, volumes, labels = generate_long_session()
    print(f"  Candles: {len(closes)}")

    bt = BacktestEngine(db_path=db_path, step=1, record_all=True)
    report = bt.run(closes, highs, lows, opens, volumes)
    bt.close()

    print(f"  Processed: {report['candles_processed']}")
    print(f"  Recorded:  {report['setups_recorded']}")
    print(f"  Entries:   {report['entry_windows']}")
    print(f"  Filled:    {report['outcomes_filled']}")

    # Analytics
    analytics = SetupAnalytics(db_path=db_path)
    analytics.print_report()
    analytics.close()


if __name__ == "__main__":
    main()
