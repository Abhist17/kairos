"""
Kairos Engine — Risk Manager

Calculates position size based on account size and max risk per trade.
Prevents blowing up on a single trade.

Usage:
    rm = RiskManager(account_size=100000)
    size = rm.calculate_lots(premium=150, sl_premium=105, lot_size=25)
    # → 1 lot (risk = (150-105)*25 = ₹1125, within 2% of 1L)
"""

from dataclasses import dataclass
from engine.core.config import RiskConfig


@dataclass
class PositionSize:
    lots: int
    quantity: int  # lots * lot_size
    risk_per_lot: float  # (premium - SL) * lot_size
    total_risk: float  # risk_per_lot * lots
    total_premium: float  # premium * quantity
    risk_pct: float  # total_risk / account_size * 100
    approved: bool
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    def calculate(
        self,
        premium: float,
        sl_premium: float,
        lot_size: int = 25,
    ) -> PositionSize:
        cfg = self.config

        risk_per_unit = premium - sl_premium
        if risk_per_unit <= 0:
            return PositionSize(
                lots=0, quantity=0, risk_per_lot=0, total_risk=0,
                total_premium=0, risk_pct=0, approved=False,
                reason="SL >= premium (invalid)",
            )

        risk_per_lot = risk_per_unit * lot_size
        max_risk = cfg.account_size * cfg.max_risk_pct
        max_lots_by_risk = int(max_risk / risk_per_lot) if risk_per_lot > 0 else 0

        max_capital = cfg.account_size * cfg.max_premium_pct
        premium_per_lot = premium * lot_size
        max_lots_by_capital = int(max_capital / premium_per_lot) if premium_per_lot > 0 else 0

        lots = max(1, min(max_lots_by_risk, max_lots_by_capital))
        quantity = lots * lot_size
        total_risk = risk_per_lot * lots
        total_premium = premium * quantity
        risk_pct = total_risk / cfg.account_size * 100

        approved = True
        reason = "OK"

        if total_risk > max_risk:
            lots = max(1, int(max_risk / risk_per_lot))
            quantity = lots * lot_size
            total_risk = risk_per_lot * lots
            total_premium = premium * quantity
            risk_pct = total_risk / cfg.account_size * 100
            reason = f"Capped to {lots} lot(s) by risk limit"

        if total_premium > cfg.account_size:
            approved = False
            reason = f"Premium ₹{total_premium:,.0f} exceeds account"

        return PositionSize(
            lots=lots,
            quantity=quantity,
            risk_per_lot=round(risk_per_lot, 2),
            total_risk=round(total_risk, 2),
            total_premium=round(total_premium, 2),
            risk_pct=round(risk_pct, 2),
            approved=approved,
            reason=reason,
        )