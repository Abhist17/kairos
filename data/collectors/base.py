"""
Kairos Engine — Base data collector interface.

All broker connectors implement this.
"""

from abc import ABC, abstractmethod
from data.models.candle import Candle


class BaseCollector(ABC):
    """Interface for all market data collectors."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source. Returns True on success."""
        ...

    @abstractmethod
    def get_candles(self, symbol: str, interval: str, count: int) -> list[Candle]:
        """Fetch historical candles."""
        ...

    @abstractmethod
    def get_ltp(self, symbol: str) -> float:
        """Get last traded price."""
        ...

    @abstractmethod
    def disconnect(self): ...

    @property
    @abstractmethod
    def name(self) -> str: ...
