"""
Kairos Engine — Option Efficiency Calculator

The core question: is THIS option contract mathematically efficient
to enter RIGHT NOW?

Delta Acceleration Window:
  gamma * expected_spot_move / current_delta
  High = delta will accelerate significantly if the move happens.
  Low = delta barely changes, you're paying for nothing.

Gamma/Theta Efficiency:
  gamma / |theta|
  How much acceleration do you get per unit of time decay?
  High = good. Low = theta is eating you alive relative to what
  gamma gives you if the move happens.

Theta Survival:
  estimated_edge / |theta_per_minute|
  How many minutes before theta decay makes this trade negative EV?

Move Feasibility:
  expected_underlying_move / required_move_for_target
  >1 = the expected move is large enough to hit the target.
  <1 = you need a bigger move than expected — bad entry.
"""

from engine.options.models import OptionEfficiencyResult


class OptionEfficiencyEngine:
    def __init__(
        self,
        min_gamma_theta: float = 0.5,
        min_move_feasibility: float = 0.8,
        min_theta_survival: float = 5.0,  # minutes
        min_delta_accel: float = 0.1,
    ):
        self.min_gamma_theta = min_gamma_theta
        self.min_move_feasibility = min_move_feasibility
        self.min_theta_survival = min_theta_survival
        self.min_delta_accel = min_delta_accel

    def evaluate(
        self,
        spot: float,
        strike: float,
        delta: float,
        gamma: float,
        theta: float,  # per day, negative for long
        vega: float,
        expected_move: float,  # expected underlying points move
        required_move: float,  # points needed for option target
        dte_minutes: float = 375.0,  # minutes to expiry
        edge_points: float = 2.0,  # estimated edge in option price points
    ) -> OptionEfficiencyResult:

        # Delta acceleration: how much will delta change per expected move
        delta_accel = 0.0
        if abs(delta) > 0.001:
            delta_accel = abs(gamma * expected_move / delta)

        # Gamma/Theta efficiency
        theta_per_min = abs(theta) / 375.0 if theta != 0 else 0.0
        gamma_theta = 0.0
        if abs(theta) > 1e-8:
            gamma_theta = abs(gamma) / abs(theta) * 100  # scale for readability

        # Theta survival: minutes until theta eats the edge
        theta_survival = float("inf")
        if theta_per_min > 1e-8:
            theta_survival = edge_points / theta_per_min

        # Move feasibility
        move_feas = 0.0
        if required_move > 0:
            move_feas = expected_move / required_move

        is_efficient = (
            delta_accel >= self.min_delta_accel
            and gamma_theta >= self.min_gamma_theta
            and theta_survival >= self.min_theta_survival
            and move_feas >= self.min_move_feasibility
        )

        return OptionEfficiencyResult(
            delta_acceleration=round(delta_accel, 4),
            gamma_theta_ratio=round(gamma_theta, 4),
            theta_survival_minutes=round(min(theta_survival, 9999), 2),
            move_feasibility=round(move_feas, 4),
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            strike=strike,
            spot=spot,
            dte_minutes=dte_minutes,
            is_efficient=is_efficient,
        )
