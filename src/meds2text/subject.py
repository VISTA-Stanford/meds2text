"""Lightweight, dependency-free MEDS subject/event types.

These replace ``meds_reader.transform.MutableSubject`` / ``MutableEvent`` so the
package has no dependency on ``meds-reader``. A MEDS event is just a timestamp,
a code, and an arbitrary bag of properties (the remaining MEDS columns).

Property access is intentionally permissive: ``event.<column>`` returns ``None``
when the column is absent, mirroring how nullable MEDS columns behave. This keeps
the rendering code (which probes optional columns like ``visit_id``,
``provider_id``, ``end``) simple.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple


class Event:
    """A single MEDS event: a time, a code, and a bag of MEDS properties."""

    __slots__ = ("time", "code", "properties")

    def __init__(
        self,
        time: Optional[datetime],
        code: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.time = time
        self.code = code
        self.properties: Dict[str, Any] = properties or {}

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails (i.e. not time/code/
        # properties and not a dunder). Treat any other name as a MEDS column,
        # returning None when absent. Dunders must raise so pickling/copy work.
        if name.startswith("_"):
            raise AttributeError(name)
        return self.properties.get(name)

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Yield ``(key, value)`` pairs for time, code, and every property.

        Renderers iterate events to materialize XML attributes and filter out
        unwanted keys, so time and code are included here and excluded downstream.
        """
        yield ("time", self.time)
        yield ("code", self.code)
        yield from self.properties.items()

    def __repr__(self) -> str:
        return f"Event(time={self.time!r}, code={self.code!r}, properties={self.properties!r})"


class Subject:
    """A MEDS subject: an id and a time-ordered list of events."""

    __slots__ = ("subject_id", "events")

    def __init__(self, subject_id: int, events: Optional[List[Event]] = None) -> None:
        self.subject_id = subject_id
        self.events: List[Event] = events or []

    def __repr__(self) -> str:
        return f"Subject(subject_id={self.subject_id!r}, n_events={len(self.events)})"
