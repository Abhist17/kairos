"""Kairos Engine — Options efficiency models."""

from dataclasses import dataclass


@dataclass
class OptionEfficiencyResult:
    # Delta acceleration: where delta changes fastest for spot movement
    delta_acceleration: float  # gamma * expected_move / delta
    gamma_theta_ratio: float  # gamma / abs(theta) — acceleration per unit decay
    theta_survival_minutes: float  # minutes before theta eats the edge
    move_feasibility: float  # expected_move / required_move (>1 = feasible)

    # Raw greeks used
    delta: float
    gamma: float
    theta: float
    vega: float

    # Context
    strike: float
    spot: float
    dte_minutes: float
    is_efficient: bool  # all checks pass
