"""
Kairos Engine — Strike Selector V3

Adapts to any index level (NIFTY 25K, SENSEX 77K, BTC 60K).
No hard premium range — picks best gamma-per-rupee in the
₹50-200 zone when possible, falls back to best available.
"""

import math
from dataclasses import dataclass
from engine.core.enums import MarketBias


@dataclass
class StrikeCandidate:
    strike: float
    option_type: str
    moneyness: str
    estimated_premium: float
    estimated_delta: float
    estimated_gamma: float
    estimated_theta: float
    move_feasibility: float
    gamma_theta_ratio: float
    theta_survival_minutes: float
    gamma_per_rupee: float
    score: float


@dataclass
class TradeSignal:
    action: str
    symbol: str
    strike: float
    option_type: str
    spot: float
    estimated_premium: float
    stoploss_premium: float
    target_premium: float
    risk_reward: float
    survival_minutes: float
    confidence: float
    reason: str
    all_candidates: list[StrikeCandidate]


class StrikeSelector:
    def __init__(
        self,
        strike_step: float = 50.0,
        n_candidates: int = 8,
        risk_pct: float = 0.30,
        target_multiplier: float = 2.0,
        min_survival: float = 5.0,
    ):
        self.strike_step = strike_step
        self.n_candidates = n_candidates
        self.risk_pct = risk_pct
        self.target_multiplier = target_multiplier
        self.min_survival = min_survival

    def select(
        self,
        spot: float,
        bias: MarketBias,
        expected_move: float,
        iv: float = 0.15,
        dte_minutes: float = 375.0,
        regime_confidence: float = 0.5,
        thesis_separation: float = 0.0,
        symbol: str = "NIFTY",
    ) -> TradeSignal | None:

        if bias == MarketBias.NEUTRAL:
            return None

        option_type = "CE" if bias == MarketBias.BULLISH else "PE"
        atm = round(spot / self.strike_step) * self.strike_step

        candidates = []
        for i in range(-2, self.n_candidates):
            if option_type == "CE":
                strike = atm + i * self.strike_step
            else:
                strike = atm - i * self.strike_step

            if strike <= 0:
                continue

            cand = self._evaluate(
                spot, strike, option_type, expected_move, iv, dte_minutes
            )
            if cand:
                candidates.append(cand)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        if best.theta_survival_minutes < self.min_survival:
            return None

        sl_premium = best.estimated_premium * (1 - self.risk_pct)
        risk_amount = best.estimated_premium - sl_premium
        target_premium = best.estimated_premium + risk_amount * self.target_multiplier
        rr = self.target_multiplier

        confidence = min(
            1.0,
            (
                0.25 * min(best.move_feasibility, 2.0) / 2.0
                + 0.20 * min(best.gamma_theta_ratio / 10, 1.0)
                + 0.20 * regime_confidence
                + 0.20 * min(thesis_separation / 0.3, 1.0)
                + 0.15 * min(best.gamma_per_rupee * 500, 1.0)
            ),
        )

        reason = (
            f"{best.moneyness} ₹{best.estimated_premium:.0f}, "
            f"feas={best.move_feasibility:.2f}, "
            f"γ/θ={best.gamma_theta_ratio:.1f}"
        )

        return TradeSignal(
            action="BUY",
            symbol=symbol,
            strike=best.strike,
            option_type=option_type,
            spot=spot,
            estimated_premium=round(best.estimated_premium, 2),
            stoploss_premium=round(sl_premium, 2),
            target_premium=round(target_premium, 2),
            risk_reward=round(rr, 2),
            survival_minutes=round(best.theta_survival_minutes, 1),
            confidence=round(confidence, 3),
            reason=reason,
            all_candidates=candidates[:5],
        )

    def _evaluate(self, spot, strike, option_type, expected_move, iv, dte_minutes):
        dte_years = dte_minutes / (375 * 365)
        if dte_years <= 0 or iv <= 0:
            return None

        distance = abs(spot - strike)
        distance_pct = distance / spot
        if distance_pct > 0.04:
            return None

        if option_type == "CE":
            itm_amount = spot - strike
        else:
            itm_amount = strike - spot

        if abs(itm_amount) < self.strike_step * 0.5:
            moneyness = "ATM"
        elif itm_amount > 0:
            moneyness = "ITM"
        elif distance <= self.strike_step * 1.5:
            moneyness = "OTM1"
        elif distance <= self.strike_step * 2.5:
            moneyness = "OTM2"
        else:
            moneyness = "OTM3"

        d1 = (math.log(spot / strike) + 0.5 * iv**2 * dte_years) / (
            iv * math.sqrt(dte_years)
        )
        d2 = d1 - iv * math.sqrt(dte_years)

        nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))

        if option_type == "CE":
            delta = nd1
            premium = spot * nd1 - strike * nd2
        else:
            delta = nd1 - 1
            premium = strike * (1 - nd2) - spot * (1 - nd1)

        premium = max(premium, 0.5)

        gamma = math.exp(-0.5 * d1**2) / (
            spot * iv * math.sqrt(2 * math.pi * dte_years)
        )

        theta = -(spot * iv * math.exp(-0.5 * d1**2)) / (
            2 * math.sqrt(2 * math.pi * dte_years) * 365
        )

        # Move feasibility
        if option_type == "CE":
            needed = max(0.1, strike - spot)
        else:
            needed = max(0.1, spot - strike)

        if moneyness in ("ATM", "ITM"):
            move_feas = min(expected_move / max(needed, 1.0), 5.0)
        else:
            move_feas = expected_move / needed if needed > 0 else 0.0

        gt_ratio = abs(gamma / theta) * 100 if abs(theta) > 1e-10 else 999

        theta_per_min = abs(theta) / 375 if theta != 0 else 0
        edge = premium * 0.10
        survival = edge / theta_per_min if theta_per_min > 1e-10 else 9999

        gamma_per_rupee = gamma / premium if premium > 0 else 0

        score = (
            0.30 * min(move_feas / 2.0, 1.0)
            + 0.25 * min(gt_ratio / 15, 1.0)
            + 0.15 * min(survival / 20, 1.0)
            + 0.15 * min(gamma_per_rupee * 500, 1.0)
            + 0.15 * (1.0 if moneyness in ("ATM", "OTM1") else 0.5)
        )

        return StrikeCandidate(
            strike=strike,
            option_type=option_type,
            moneyness=moneyness,
            estimated_premium=round(premium, 2),
            estimated_delta=round(abs(delta), 4),
            estimated_gamma=round(gamma, 6),
            estimated_theta=round(theta, 4),
            move_feasibility=round(move_feas, 3),
            gamma_theta_ratio=round(gt_ratio, 2),
            theta_survival_minutes=round(min(survival, 9999), 1),
            gamma_per_rupee=round(gamma_per_rupee, 6),
            score=round(score, 4),
        )
