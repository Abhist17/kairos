"""
Kairos Engine — Gamma Gravity Map (ESTIMATED)

Estimates gamma concentration across strikes.

CRITICAL DISCLAIMER:
Dealer gamma positioning is NOT publicly observable.
This is an ESTIMATE based on OI distribution and standard
Black-Scholes gamma curves. Actual dealer positions depend on
whether they are long or short the options, which we cannot know.

The estimate is useful for identifying ZONES where gamma effects
may amplify moves, not for exact positioning.

Positive gamma zone: dealers hedge BY buying dips/selling rallies → stabilizing
Negative gamma zone: dealers hedge BY selling dips/buying rallies → destabilizing
"""

import numpy as np
from dataclasses import dataclass, field
from engine.core.types import FloatArray


@dataclass
class StrikeGamma:
    strike: float
    estimated_gamma: float  # always labelled ESTIMATED
    call_gamma_exposure: float
    put_gamma_exposure: float
    net_gamma: float


@dataclass
class GammaMapResult:
    """All gamma values are ESTIMATED — exact dealer positioning is unknown."""
    total_gamma_exposure: float
    gamma_flip_strike: float  # price where net gamma flips sign
    max_gamma_strike: float
    gamma_concentration: list[StrikeGamma] = field(default_factory=list)
    regime_note: str = ""  # "ESTIMATED: positive gamma environment" etc.


class GammaGravityMap:
    def __init__(self, strike_step: float = 50.0, n_strikes: int = 21):
        self.strike_step = strike_step
        self.n_strikes = n_strikes

    def estimate(
        self,
        spot: float,
        strikes: FloatArray | None = None,
        call_oi: FloatArray | None = None,
        put_oi: FloatArray | None = None,
        iv: float = 0.15,
        dte_years: float = 1 / 365,
    ) -> GammaMapResult:
        if strikes is None or call_oi is None or put_oi is None:
            atm = round(spot / self.strike_step) * self.strike_step
            half = self.n_strikes // 2
            strikes = np.array([
                atm + (i - half) * self.strike_step for i in range(self.n_strikes)
            ])
            call_oi = np.exp(
                -0.5 * ((strikes - (atm + self.strike_step)) / (3 * self.strike_step)) ** 2
            ) * 50000
            put_oi = np.exp(
                -0.5 * ((strikes - (atm - self.strike_step)) / (3 * self.strike_step)) ** 2
            ) * 45000

        n = len(strikes)
        concentration = []
        total_gex = 0.0
        max_gex = 0.0
        max_gex_strike = spot
        flip_strike = spot

        prev_net = None

        for i in range(n):
            k = float(strikes[i])
            g = self._bs_gamma(spot, k, iv, dte_years)

            # GEX = gamma * OI * contract_multiplier * spot
            # Assume contract multiplier = 1 for index
            call_gex = g * float(call_oi[i]) * spot * 0.01
            # Put gamma exposure is negative (dealers short puts → negative gamma)
            put_gex = -g * float(put_oi[i]) * spot * 0.01
            net = call_gex + put_gex

            total_gex += net

            if abs(net) > max_gex:
                max_gex = abs(net)
                max_gex_strike = k

            # Track gamma flip
            if prev_net is not None and prev_net * net < 0:
                flip_strike = k

            prev_net = net

            concentration.append(StrikeGamma(
                strike=k,
                estimated_gamma=round(g, 6),
                call_gamma_exposure=round(call_gex, 2),
                put_gamma_exposure=round(put_gex, 2),
                net_gamma=round(net, 2),
            ))

        if total_gex > 0:
            note = "ESTIMATED: positive gamma environment (stabilizing)"
        else:
            note = "ESTIMATED: negative gamma environment (destabilizing)"

        return GammaMapResult(
            total_gamma_exposure=round(total_gex, 2),
            gamma_flip_strike=round(flip_strike, 2),
            max_gamma_strike=round(max_gex_strike, 2),
            gamma_concentration=concentration,
            regime_note=note,
        )

    def _bs_gamma(
        self, spot: float, strike: float, iv: float, dte_years: float
    ) -> float:
        """Black-Scholes gamma approximation."""
        if iv <= 0 or dte_years <= 0 or spot <= 0:
            return 0.0

        d1 = (np.log(spot / strike) + 0.5 * iv ** 2 * dte_years) / (iv * np.sqrt(dte_years))
        gamma = np.exp(-0.5 * d1 ** 2) / (spot * iv * np.sqrt(2 * np.pi * dte_years))
        return float(gamma)