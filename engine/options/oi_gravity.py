"""
Kairos Engine — OI Center of Gravity

Tracks movement of option-chain OI concentration through time
instead of treating highest OI as static support/resistance.

Most traders look at "max OI at 25000 PE = support" — this is
static and wrong. OI SHIFTS as dealers adjust positions.

OI Center of Gravity tracks:
- Weighted average strike by OI (call-side and put-side separately)
- How fast the center is moving (OI gravity velocity)
- Put-Call OI ratio at each strike
- Where the gravity is pulling price toward

Without real option chain data, this uses synthetic OI.
When connected to a broker API, swap in real chain data.
"""

import numpy as np
from dataclasses import dataclass, field
from engine.core.types import FloatArray


@dataclass
class StrikeOI:
    strike: float
    call_oi: float
    put_oi: float
    pcr: float  # put/call ratio at this strike


@dataclass
class OIGravityResult:
    call_gravity: float  # OI-weighted avg strike for calls
    put_gravity: float  # OI-weighted avg strike for puts
    combined_gravity: float  # overall center
    gravity_velocity: float  # change per update (positive = moving up)
    max_pain: float  # strike where total OI loss is minimized
    pcr_overall: float  # total put OI / total call OI
    strike_data: list[StrikeOI] = field(default_factory=list)


class OIGravityTracker:
    def __init__(self, strike_step: float = 50.0, n_strikes: int = 21):
        self.strike_step = strike_step
        self.n_strikes = n_strikes
        self._prev_gravity: float = 0.0
        self._update_count: int = 0

    def analyze(
        self,
        spot: float,
        call_oi: FloatArray | None = None,
        put_oi: FloatArray | None = None,
        strikes: FloatArray | None = None,
    ) -> OIGravityResult:
        """
        Analyze OI distribution. If no real data provided,
        generates synthetic OI centered around spot.
        """
        if strikes is None or call_oi is None or put_oi is None:
            strikes, call_oi, put_oi = self._synthetic_oi(spot)

        n = len(strikes)
        if n == 0:
            return OIGravityResult(
                call_gravity=spot,
                put_gravity=spot,
                combined_gravity=spot,
                gravity_velocity=0.0,
                max_pain=spot,
                pcr_overall=1.0,
            )

        # Call gravity: OI-weighted average strike
        total_call = np.sum(call_oi)
        total_put = np.sum(put_oi)

        call_grav = (
            float(np.average(strikes, weights=call_oi)) if total_call > 0 else spot
        )
        put_grav = float(np.average(strikes, weights=put_oi)) if total_put > 0 else spot

        total_oi = call_oi + put_oi
        combined = (
            float(np.average(strikes, weights=total_oi))
            if np.sum(total_oi) > 0
            else spot
        )

        # Max pain: strike minimizing total intrinsic value
        max_pain = self._calc_max_pain(strikes, call_oi, put_oi)

        # Velocity
        velocity = 0.0
        if self._update_count > 0 and self._prev_gravity != 0:
            velocity = combined - self._prev_gravity
        self._prev_gravity = combined
        self._update_count += 1

        # PCR
        pcr = float(total_put / total_call) if total_call > 0 else 1.0

        # Strike-level data
        strike_data = []
        for i in range(n):
            s_pcr = float(put_oi[i] / call_oi[i]) if call_oi[i] > 0 else 99.0
            strike_data.append(
                StrikeOI(
                    strike=float(strikes[i]),
                    call_oi=float(call_oi[i]),
                    put_oi=float(put_oi[i]),
                    pcr=round(s_pcr, 2),
                )
            )

        return OIGravityResult(
            call_gravity=round(call_grav, 2),
            put_gravity=round(put_grav, 2),
            combined_gravity=round(combined, 2),
            gravity_velocity=round(velocity, 2),
            max_pain=round(max_pain, 2),
            pcr_overall=round(pcr, 2),
            strike_data=strike_data,
        )

    def _calc_max_pain(
        self, strikes: FloatArray, call_oi: FloatArray, put_oi: FloatArray
    ) -> float:
        """Find strike that minimizes total option buyer losses."""
        min_pain = float("inf")
        max_pain_strike = float(strikes[len(strikes) // 2])

        for i, s in enumerate(strikes):
            # If expiry at this strike, call buyers lose on ITM calls below s
            call_loss = sum(
                max(0, s - strikes[j]) * call_oi[j] for j in range(len(strikes))
            )
            put_loss = sum(
                max(0, strikes[j] - s) * put_oi[j] for j in range(len(strikes))
            )
            total = call_loss + put_loss
            if total < min_pain:
                min_pain = total
                max_pain_strike = float(s)

        return max_pain_strike

    def _synthetic_oi(self, spot: float):
        """Generate synthetic OI distribution for testing."""
        atm = round(spot / self.strike_step) * self.strike_step
        half = self.n_strikes // 2
        strikes = np.array(
            [atm + (i - half) * self.strike_step for i in range(self.n_strikes)]
        )

        # Calls: OI peaks slightly above spot
        call_center = atm + self.strike_step
        call_oi = (
            np.exp(-0.5 * ((strikes - call_center) / (3 * self.strike_step)) ** 2)
            * 50000
        )

        # Puts: OI peaks slightly below spot
        put_center = atm - self.strike_step
        put_oi = (
            np.exp(-0.5 * ((strikes - put_center) / (3 * self.strike_step)) ** 2)
            * 45000
        )

        return strikes, call_oi, put_oi
