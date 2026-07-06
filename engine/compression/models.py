"""Kairos Engine — Compression models."""

from dataclasses import dataclass


@dataclass
class CompressionResult:
    is_compressed: bool
    compression_score: float  # 0.0 - 1.0

    # Individual metrics
    atr_contraction: float
    range_contraction: float
    rv_decay: float  # realized vol now / realized vol recent median
    bbw_percentile: float  # 0-100

    # Advanced
    compression_velocity: (
        float  # rate of contraction per candle (negative = compressing)
    )
    compression_half_life: (
        float  # estimated candles until range halves again (inf if expanding)
    )

    # Context
    candles_compressed: int  # how many consecutive candles in compression
    pre_compression_vol: float  # vol level before compression started
