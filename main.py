"""
Kairos Engine — Main Entry Point
Regime Engine V1 + Compression Engine V1
"""

import numpy as np
from datetime import datetime, timedelta

from engine.core.enums import MarketRegime
from engine.core.config import RegimeConfig
from engine.regime.classifier import RegimeClassifier
from engine.compression.detector import CompressionDetector
from data.models.market_state import MarketState, RegimeMetrics, CompressionMetrics


def generate_synthetic_nifty(n_candles: int = 600, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    base_price = 25000.0
    prices = [base_price]

    phases = [
        (100, 0.0, 0.8, "COMPRESSION"),
        (100, 3.0, 1.2, "TREND_UP"),
        (80, 1.0, 8.0, "EXHAUSTION"),
        (100, 0.0, 4.0, "MEAN_REVERSION"),
        (120, -2.5, 1.5, "TREND_DOWN"),
        (100, 0.0, 15.0, "CHAOTIC"),
    ]

    phase_labels = []

    for n, drift, noise, label in phases:
        for _ in range(n):
            move = drift + rng.normal(0, noise)
            prices.append(prices[-1] + move)
            phase_labels.append(label)

    prices = np.array(prices[1:])
    n = len(prices)

    noise_factor = 0.0003
    highs = prices + np.abs(rng.normal(0, 1, n)) * prices * noise_factor + 0.5
    lows = prices - np.abs(rng.normal(0, 1, n)) * prices * noise_factor - 0.5
    opens = np.roll(prices, 1)
    opens[0] = base_price
    volumes = rng.integers(5000, 50000, n).astype(float)

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": prices,
        "volumes": volumes,
        "phase_labels": phase_labels,
    }


REGIME_COLORS = {
    MarketRegime.TREND_EXPANSION: "\033[92m",
    MarketRegime.TREND_EXHAUSTION: "\033[93m",
    MarketRegime.COMPRESSION: "\033[96m",
    MarketRegime.MEAN_REVERSION: "\033[95m",
    MarketRegime.CHAOTIC: "\033[91m",
    MarketRegime.UNKNOWN: "\033[90m",
}
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"


def print_state(state: MarketState, candle_idx: int, phase_label: str):
    color = REGIME_COLORS.get(state.regime, RESET)
    rm = state.regime_metrics
    cm = state.compression

    transition = ""
    if rm.transition_from:
        transition = f" (was {rm.transition_from.value})"

    comp_flag = f"{CYAN}■ COMPRESSED{RESET}" if cm.is_compressed else "  ---"

    half_life_str = (
        f"{cm.compression_half_life:.0f}"
        if cm.compression_half_life != float("inf")
        else "inf"
    )

    print(
        f"  {candle_idx:>4d} | "
        f"{state.last_price:>10.2f} | "
        f"{color}{state.regime.value:<18s}{RESET} "
        f"conf={rm.regime_confidence:.2f} "
        f"bias={state.bias.value:<8s} | "
        f"comp={cm.compression_score:.2f} "
        f"vel={cm.compression_velocity:+.5f} "
        f"hl={half_life_str:>5s} "
        f"dur={cm.candles_compressed:>3d} "
        f"{comp_flag:>16s}"
        f"{transition}"
        f"  [{phase_label}]"
    )


def main():
    print("\n" + "=" * 120)
    print("  KAIROS ENGINE — Regime V1 + Compression V1")
    print("  Synthetic NIFTY 1-min candles")
    print("=" * 120)
    print(
        f"  {'#':>4s} | {'Price':>10s} | {'Regime':<18s} "
        f"{'':4s} {'Bias':<8s} | "
        f"{'Comp':>4s}  {'Vel':>10s}  {'HL':>5s}  {'Dur':>3s}  {'Status':>16s}"
    )
    print("-" * 120)

    data = generate_synthetic_nifty(n_candles=600)
    classifier = RegimeClassifier()
    comp_detector = CompressionDetector()
    config = RegimeConfig()

    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    labels = data["phase_labels"]

    min_candles = config.vol_percentile_window + config.lookback
    step = 5

    for i in range(min_candles, len(closes), step):
        result = classifier.classify(
            closes=closes[:i],
            highs=highs[:i],
            lows=lows[:i],
        )

        comp = comp_detector.detect(
            closes=closes[:i],
            highs=highs[:i],
            lows=lows[:i],
        )

        state = MarketState(
            symbol="NIFTY",
            timestamp=datetime(2025, 1, 6, 9, 15) + timedelta(minutes=i),
            last_price=float(closes[i - 1]),
            candle_close=float(closes[i - 1]),
            candle_high=float(highs[i - 1]),
            candle_low=float(lows[i - 1]),
            candle_open=float(data["opens"][i - 1]),
            candle_volume=float(data["volumes"][i - 1]),
            regime=result.regime,
            bias=result.bias,
            regime_metrics=RegimeMetrics(
                efficiency_ratio=result.efficiency_ratio,
                directional_persistence=result.directional_persistence,
                normalized_slope=result.normalized_slope,
                directional_entropy=result.directional_entropy,
                realized_volatility=result.realized_volatility,
                volatility_percentile=result.volatility_percentile,
                atr_contraction=result.atr_contraction,
                log_return_mean=result.log_return_mean,
                regime_confidence=result.confidence,
                regime_age=result.regime_age,
                transition_from=result.transition_from,
            ),
            compression=CompressionMetrics(
                is_compressed=comp.is_compressed,
                compression_score=comp.compression_score,
                atr_contraction=comp.atr_contraction,
                range_contraction=comp.range_contraction,
                rv_decay=comp.rv_decay,
                bbw_percentile=comp.bbw_percentile,
                compression_velocity=comp.compression_velocity,
                compression_half_life=comp.compression_half_life,
                candles_compressed=comp.candles_compressed,
                pre_compression_vol=comp.pre_compression_vol,
            ),
        )

        print_state(state, i, labels[i - 1])

    print("=" * 120)
    print("  Pipeline complete: Regime V1 + Compression V1")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
