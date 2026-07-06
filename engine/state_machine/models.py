"""Kairos Engine — State Machine models."""

from dataclasses import dataclass, field
from engine.core.enums import TradeState, EntryWindow


@dataclass
class GateStatus:
    name: str
    ready: bool
    reason: str = ""


@dataclass
class StateMachineResult:
    state: TradeState
    entry_window: EntryWindow
    gates: list[GateStatus] = field(default_factory=list)
    estimated_window_seconds: int = 0
    thesis_survival_minutes: float = 0.0
