"""Kairos Engine — Market Pipeline V3. ATR-based moves, MTF, regime_age to state machine."""

import numpy as np
from datetime import datetime

from engine.core.config import RegimeConfig
from engine.core.types import FloatArray
from engine.regime.classifier import RegimeClassifier
from engine.compression.detector import CompressionDetector
from engine.structure.levels import StructureAnalyzer
from engine.volatility.iv_state import IVClassifier
from engine.options.efficiency import OptionEfficiencyEngine
from engine.flow.pressure import PressureAnalyzer
from engine.scoring.thesis import ThesisEngine
from engine.state_machine.machine import EntryStateMachine
from engine.features.multi_timeframe import MultiTimeframe
from engine.features.volatility import atr
from data.models.market_state import (
    MarketState, RegimeMetrics, CompressionMetrics, StructureMetrics,
    IVMetrics, OptionMetrics, FlowMetrics, ThesisMetrics,
    StateMachineMetrics, GateInfo,
)


# Fix 4: ATR multiplier by regime
ATR_MULTIPLIER = {
    "TREND_EXPANSION": 2.0,   # moves are extended
    "COMPRESSION": 3.0,       # breakouts are explosive
    "TREND_EXHAUSTION": 1.5,  # moves are fading
    "MEAN_REVERSION": 1.0,    # moves are limited
    "CHAOTIC": 1.5,           # unpredictable
    "UNKNOWN": 1.5,
}


class MarketPipeline:
    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self.regime_clf = RegimeClassifier(self.config)
        self.comp_det = CompressionDetector()
        self.struct_ana = StructureAnalyzer()
        self.iv_clf = IVClassifier()
        self.opt_eng = OptionEfficiencyEngine()
        self.press_ana = PressureAnalyzer()
        self.thesis_eng = ThesisEngine()
        self.state_machine = EntryStateMachine()
        self.mtf = MultiTimeframe(resample_factor=7)

    @property
    def min_candles(self) -> int:
        return self.config.vol_percentile_window + self.config.lookback

    def process(
        self,
        closes: FloatArray,
        highs: FloatArray,
        lows: FloatArray,
        opens: FloatArray,
        volumes: FloatArray,
        iv_series: FloatArray | None = None,
        symbol: str = "NIFTY",
        timestamp: datetime | None = None,
    ) -> MarketState:
        n = len(closes)
        price = float(closes[-1])
        ts = timestamp or datetime.now()

        if iv_series is None:
            rets = np.diff(np.log(closes)) if n > 1 else np.array([0.0])
            rolling = np.array([
                np.std(rets[max(0, i - 20):i], ddof=1) if i > 1 else 0.01
                for i in range(1, len(rets) + 1)
            ])
            iv_series = rolling * np.sqrt(93750)

        # 1. Regime
        reg = self.regime_clf.classify(closes, highs, lows)

        # 2. Compression
        comp = self.comp_det.detect(closes, highs, lows)

        # 3. Structure
        struct = self.struct_ana.analyze(closes, highs, lows, volumes, price)

        # 4. IV
        iv = self.iv_clf.classify(iv_series)

        # Fix 4: ATR-based expected move
        true_ranges = atr(highs, lows, closes)
        if len(true_ranges) >= 14:
            current_atr = float(np.mean(true_ranges[-14:]))
            multiplier = ATR_MULTIPLIER.get(reg.regime.value, 1.5)
            expected_move = current_atr * multiplier
        else:
            expected_move = float(np.std(np.diff(closes[-20:]))) * 3 if n > 21 else 10.0

        # 5. Options Efficiency
        strike = round(price / 50) * 50
        dte_min = max(60.0, 375.0)
        moneyness = abs(price - strike) / price
        syn_delta = max(0.1, 0.5 - moneyness * 5)
        syn_gamma = max(0.001, 0.05 - moneyness)
        syn_theta = -(0.5 + iv.current_iv * 2)
        syn_vega = max(0.1, 5.0 - moneyness * 20)
        req_move = abs(price - strike) + 5.0

        opt = self.opt_eng.evaluate(
            spot=price, strike=strike, delta=syn_delta,
            gamma=syn_gamma, theta=syn_theta, vega=syn_vega,
            expected_move=expected_move, required_move=req_move,
            dte_minutes=dte_min,
        )

        # 6. Flow
        target = struct.nearest_zone.center if struct.nearest_zone else price
        flow = self.press_ana.analyze(closes, highs, lows, opens, target)

        # 7. Thesis
        thesis = self.thesis_eng.score(
            regime=reg.regime, bias=reg.bias,
            regime_confidence=reg.confidence,
            compression_score=comp.compression_score,
            is_compressed=comp.is_compressed,
            structure_score=struct.structure_score,
            nearest_zone_dist_pct=struct.nearest_zone_distance_pct,
            pressure_score=flow.pressure_score,
            aggression_ratio=flow.aggression_ratio,
            iv_percentile=iv.iv_percentile,
            option_efficient=opt.is_efficient,
        )

        # Fix 5/7: Multi-timeframe alignment
        mtf_result = self.mtf.analyze(
            closes, highs, lows, reg.regime, reg.bias,
        )

        # 8. State machine with regime_age + MTF
        sm = self.state_machine.evaluate(
            regime=reg.regime,
            bias=reg.bias,
            regime_confidence=reg.confidence,
            efficiency_ratio=reg.efficiency_ratio,
            regime_age=reg.regime_age,
            mtf_aligned=mtf_result.aligned,
            structure_score=struct.structure_score,
            nearest_zone_dist_pct=struct.nearest_zone_distance_pct,
            compression_score=comp.compression_score,
            is_compressed=comp.is_compressed,
            pressure_score=flow.pressure_score,
            option_efficient=opt.is_efficient,
            theta_survival=opt.theta_survival_minutes,
            thesis_valid=thesis.thesis_valid,
            thesis_separation=thesis.separation,
        )

        return MarketState(
            symbol=symbol, timestamp=ts, last_price=price,
            candle_close=price, candle_high=float(highs[-1]),
            candle_low=float(lows[-1]), candle_open=float(opens[-1]),
            candle_volume=float(volumes[-1]),
            regime=reg.regime, bias=reg.bias,
            regime_metrics=RegimeMetrics(
                efficiency_ratio=reg.efficiency_ratio,
                directional_persistence=reg.directional_persistence,
                normalized_slope=reg.normalized_slope,
                directional_entropy=reg.directional_entropy,
                realized_volatility=reg.realized_volatility,
                volatility_percentile=reg.volatility_percentile,
                atr_contraction=reg.atr_contraction,
                log_return_mean=reg.log_return_mean,
                regime_confidence=reg.confidence,
                regime_age=reg.regime_age,
                transition_from=reg.transition_from,
            ),
            compression=CompressionMetrics(
                is_compressed=comp.is_compressed,
                compression_score=comp.compression_score,
                atr_contraction=comp.atr_contraction,
                range_contraction=comp.range_contraction,
                rv_decay=comp.rv_decay, bbw_percentile=comp.bbw_percentile,
                compression_velocity=comp.compression_velocity,
                compression_half_life=comp.compression_half_life,
                candles_compressed=comp.candles_compressed,
                pre_compression_vol=comp.pre_compression_vol,
            ),
            structure=StructureMetrics(
                structure_score=struct.structure_score,
                nearest_zone_distance=struct.nearest_zone_distance,
                nearest_zone_distance_pct=struct.nearest_zone_distance_pct,
                nearest_zone_center=struct.nearest_zone.center if struct.nearest_zone else 0,
                nearest_zone_confluence=struct.nearest_zone.confluence if struct.nearest_zone else 0,
                above_zones=struct.above_zones, below_zones=struct.below_zones,
                inside_zone=struct.inside_zone,
                total_levels=len(struct.levels), total_zones=len(struct.zones),
            ),
            iv=IVMetrics(
                state=iv.state, current_iv=iv.current_iv,
                iv_percentile=iv.iv_percentile, iv_velocity=iv.iv_velocity,
            ),
            option=OptionMetrics(
                is_efficient=opt.is_efficient,
                delta_acceleration=opt.delta_acceleration,
                gamma_theta_ratio=opt.gamma_theta_ratio,
                theta_survival_minutes=opt.theta_survival_minutes,
                move_feasibility=opt.move_feasibility,
                strike=opt.strike, delta=opt.delta,
                gamma=opt.gamma, theta=opt.theta,
            ),
            flow=FlowMetrics(
                pressure_score=flow.pressure_score,
                level_tests=flow.level_tests,
                test_frequency=flow.test_frequency,
                pullback_shrinking=flow.pullback_shrinking,
                aggression_ratio=flow.aggression_ratio,
                pressure_asymmetry=flow.pressure_asymmetry,
                liquidity_thinning=flow.liquidity_thinning,
            ),
            thesis=ThesisMetrics(
                primary_bias=thesis.primary_bias,
                primary_score=thesis.primary_score,
                counter_score=thesis.counter_score,
                separation=thesis.separation,
                thesis_valid=thesis.thesis_valid,
            ),
            state_machine=StateMachineMetrics(
                gates=[GateInfo(name=g.name, ready=g.ready, reason=g.reason) for g in sm.gates],
                estimated_window_seconds=sm.estimated_window_seconds,
                thesis_survival_minutes=sm.thesis_survival_minutes,
            ),
            trade_state=sm.state, entry_window=sm.entry_window,
        )