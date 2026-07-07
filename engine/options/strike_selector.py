"""
Kairos Engine — Optimal Strike Selector

Given spot price, bias, expected move, and IV:
picks the best CE or PE strike for BUYING.

Strike selection logic:
  - ATM: highest gamma, fastest delta acceleration, most expensive
  - 1-strike OTM: cheaper, still good gamma, better risk/reward
  - 2-strike OTM: cheap but needs a bigger move to pay off

We pick the strike where:
  move_feasibility is highest AND theta_survival is adequate

For BUY-only (no selling), we want:
  - Enough gamma to accelerate delta on the move
  - Low enough premium that theta doesn't kill us in the survival window
  - Strike close enough that the expected move reaches it
"""

import math
from dataclasses import dataclass
from engine.core.enums import MarketBias


@dataclass
class StrikeCandidate:
    strike: float
    option_type: str  # "CE" or "PE"
    moneyness: str  # "ITM", "ATM", "OTM1", "OTM2"
    estimated_premium: float
    estimated_delta: float
    estimated_gamma: float
    estimated_theta: float  # per day, negative
    move_feasibility: float  # expected_move / distance_to_strike
    gamma_theta_ratio: float
    theta_survival_minutes: float
    score: float  # composite ranking score


@dataclass
class TradeSignal:
    action: str  # "BUY"
    symbol: str
    strike: float
    option_type: str  # "CE" or "PE"
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
        strike_step: float = 50.0,  # NIFTY strike gap
        n_candidates: int = 5,  # check 5 strikes each side
        risk_pct: float = 0.30,  # SL at 30% of premium
        target_multiplier: float = 2.0,  # target = 2x risk
        min_survival: float = 8.0,  # minimum 8 minutes survival
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
        expected_move: float,  # expected points move in bias direction
        iv: float = 0.15,  # annualized IV as decimal
        dte_minutes: float = 375.0,  # minutes to expiry
        regime_confidence: float = 0.5,
        thesis_separation: float = 0.0,
        symbol: str = "NIFTY",
    ) -> TradeSignal | None:

        if bias == MarketBias.NEUTRAL:
            return None

        option_type = "CE" if bias == MarketBias.BULLISH else "PE"
        atm = round(spot / self.strike_step) * self.strike_step

        # Generate candidate strikes
        candidates = []
        for i in range(-2, self.n_candidates):
            if option_type == "CE":
                strike = atm + i * self.strike_step
            else:
                strike = atm - i * self.strike_step

            if strike <= 0:
                continue

            cand = self._evaluate_strike(
                spot,
                strike,
                option_type,
                expected_move,
                iv,
                dte_minutes,
            )
            if cand:
                candidates.append(cand)

        if not candidates:
            return None

        # Sort by composite score (highest first)
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        # Check minimum survival
        if best.theta_survival_minutes < self.min_survival:
            return None

        # Calculate SL and target
        sl_premium = best.estimated_premium * (1 - self.risk_pct)
        risk_amount = best.estimated_premium - sl_premium
        target_premium = best.estimated_premium + risk_amount * self.target_multiplier
        rr = self.target_multiplier if risk_amount > 0 else 0

        # Confidence based on multiple factors
        confidence = min(
            1.0,
            (
                0.30 * min(best.move_feasibility, 1.5) / 1.5
                + 0.25 * min(best.gamma_theta_ratio / 10, 1.0)
                + 0.25 * regime_confidence
                + 0.20 * min(thesis_separation / 0.3, 1.0)
            ),
        )

        # Build reason string
        reason = (
            f"{best.moneyness} strike, "
            f"feasibility={best.move_feasibility:.2f}, "
            f"γ/θ={best.gamma_theta_ratio:.1f}, "
            f"survival={best.theta_survival_minutes:.0f}m"
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
            all_candidates=candidates,
        )

    def _evaluate_strike(
        self,
        spot,
        strike,
        option_type,
        expected_move,
        iv,
        dte_minutes,
    ) -> StrikeCandidate | None:

        dte_years = dte_minutes / (375 * 365)
        if dte_years <= 0 or iv <= 0:
            return None

        # Moneyness
        if option_type == "CE":
            itm_amount = spot - strike
        else:
            itm_amount = strike - spot

        distance = abs(spot - strike)
        distance_pct = distance / spot

        if distance_pct > 0.03:  # skip strikes >3% away
            return None

        # Label
        if abs(itm_amount) < self.strike_step * 0.5:
            moneyness = "ATM"
        elif itm_amount > 0:
            moneyness = "ITM"
        elif distance <= self.strike_step * 1.5:
            moneyness = "OTM1"
        else:
            moneyness = "OTM2"

        # Approximate BS greeks
        d1 = (math.log(spot / strike) + 0.5 * iv**2 * dte_years) / (
            iv * math.sqrt(dte_years)
        )
        d2 = d1 - iv * math.sqrt(dte_years)

        # Normal CDF approximation
        from math import erf

        nd1 = 0.5 * (1 + erf(d1 / math.sqrt(2)))
        nd2 = 0.5 * (1 + erf(d2 / math.sqrt(2)))

        if option_type == "CE":
            delta = nd1
            premium = spot * nd1 - strike * nd2
        else:
            delta = nd1 - 1
            premium = strike * (1 - nd2) - spot * (1 - nd1)

        premium = max(premium, 0.5)  # floor at 0.5

        # Gamma
        gamma = math.exp(-0.5 * d1**2) / (
            spot * iv * math.sqrt(2 * math.pi * dte_years)
        )

        # Theta (per day)
        theta_component = -(spot * iv * math.exp(-0.5 * d1**2)) / (
            2 * math.sqrt(2 * math.pi * dte_years) * 365
        )
        theta = theta_component  # negative

        # Metrics
        # Move feasibility: can the expected move cover the distance?
        if option_type == "CE":
            needed = max(0.1, strike - spot)
        else:
            needed = max(0.1, spot - strike)

        if moneyness == "ATM" or moneyness == "ITM":
            move_feas = expected_move / max(needed, 1.0) if needed > 0 else 5.0
            move_feas = min(move_feas, 5.0)
        else:
            move_feas = expected_move / needed if needed > 0 else 0.0

        # Gamma/Theta ratio
        gt_ratio = abs(gamma / theta) * 100 if abs(theta) > 1e-10 else 999

        # Theta survival
        theta_per_min = abs(theta) / 375 if theta != 0 else 0
        edge = premium * 0.15  # assume 15% of premium as edge
        survival = edge / theta_per_min if theta_per_min > 1e-10 else 9999

        # Composite score — what matters for a BUY
        # High feasibility + high gamma/theta + reasonable premium
        score = (
            0.35 * min(move_feas / 2.0, 1.0)  # can the move happen?
            + 0.25 * min(gt_ratio / 15, 1.0)  # gamma bang per theta buck
            + 0.20 * min(survival / 30, 1.0)  # enough time?
            + 0.10
            * (1.0 if moneyness in ("ATM", "OTM1") else 0.5)  # prefer ATM/slight OTM
            + 0.10 * max(0, 1.0 - premium / 300)  # cheaper is better (for risk)
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
            score=round(score, 4),
        )
