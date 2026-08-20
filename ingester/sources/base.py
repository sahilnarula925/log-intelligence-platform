"""Base class for log sources."""

from __future__ import annotations

import abc
from typing import AsyncIterator

from shared.events import LogEvent


class LogSource(abc.ABC):
    name: str

    @abc.abstractmethod
    async def stream(self) -> AsyncIterator[LogEvent]:
        """Yield normalized log events from this source."""
