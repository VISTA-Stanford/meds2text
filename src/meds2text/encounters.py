"""Group a subject's events into encounters (time-bounded bins).

MEDS extracts do not always carry clean visit boundaries, so encounters are
*derived* from the event stream rather than read directly from visit rows. The
binning policy is pluggable via :class:`EncounterStrategy`; the default
:class:`FuzzyVisitStrategy` reproduces the long-standing heuristic:

1. Partition ``(timestamp, visit_id)`` rows into groups dominated by a single
   ``visit_id``, tolerating a small number of "glitch" rows with a different id
   and force-merging rows that are close together in time.
2. Turn each group into a ``(start, end)`` interval and assign every event to
   the interval that contains it.
3. Events that fall outside every interval are collected into contiguous
   "pseudo" encounters so nothing is dropped.

Keeping this logic isolated (and free of XML/serialization concerns) makes the
binning rules legible, unit-testable, and swappable.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple

from meds2text.subject import Event

logger = logging.getLogger(__name__)

Interval = Tuple[datetime, datetime]


@dataclass
class Encounter:
    """A time-bounded bin of events."""

    start: datetime
    end: datetime
    events: List[Event] = field(default_factory=list)


class EncounterStrategy(Protocol):
    """Maps a subject's events to an ordered list of encounters."""

    def bin(
        self, events: Sequence[Event], *, excluded_tables: Optional[Set[str]] = None
    ) -> List[Encounter]: ...


def fuzzy_partition(
    rows: Sequence[Tuple[datetime, object]],
    max_mismatches: int = 1,
    max_timedelta: timedelta = timedelta(hours=24),
) -> List[List[Tuple[datetime, object]]]:
    """Partition ``(timestamp, id)`` rows into groups via fuzzy glitch merging.

    Constraints:
      1. Rows sharing a timestamp are never split, even if their id differs.
      2. If the gap between the current group's last event and the next event is
         <= ``max_timedelta``, the next row is force-merged into the group.
      3. Otherwise a contiguous block whose id differs from the group's dominant
         id is only merged while the running glitch count stays within
         ``max_mismatches``.

    ``rows`` must be sorted by timestamp.
    """
    if not rows:
        return []

    groups: List[List[Tuple[datetime, object]]] = []
    n = len(rows)
    i = 0

    while i < n:
        current_group = [rows[i]]
        dominant_id = rows[i][1]
        glitch_count = 0
        i += 1

        while i < n:
            last_ts = current_group[-1][0]
            next_ts = rows[i][0]

            # Constraint 1: never split rows with the same timestamp.
            if next_ts == last_ts:
                same_ts = next_ts
                while i < n and rows[i][0] == same_ts:
                    current_group.append(rows[i])
                    if rows[i][1] != dominant_id:
                        glitch_count += 1
                    i += 1
                continue

            # Constraint 2: do not split if the time gap is within max_timedelta.
            if next_ts - last_ts <= max_timedelta:
                current_group.append(rows[i])
                if rows[i][1] != dominant_id:
                    glitch_count += 1
                i += 1
                continue

            # Gather a contiguous block sharing the next row's id.
            candidate_id = rows[i][1]
            block_start = i
            while i < n and rows[i][1] == candidate_id:
                i += 1
            block_length = i - block_start

            if candidate_id == dominant_id:
                current_group.extend(rows[block_start:i])
            elif glitch_count + block_length <= max_mismatches:
                current_group.extend(rows[block_start:i])
                glitch_count += block_length
            else:
                # Too many glitches; end the group before this block.
                i = block_start
                break

        groups.append(current_group)

    return groups


def bin_events(
    intervals: Sequence[Interval],
    events: Iterable[Event],
    excluded_tables: Optional[Set[str]] = None,
) -> Dict[Interval, List[Event]]:
    """Assign each event to exactly one interval.

    Events that do not fall inside any interval are grouped into contiguous
    pseudo intervals so they are still emitted. The returned dict preserves the
    order in which intervals first receive an event, with pseudo intervals last.
    """
    excluded_tables = excluded_tables or set()
    ordered_events = sorted(
        (e for e in events if e.table not in excluded_tables), key=lambda e: e.time
    )

    result: Dict[Interval, List[Event]] = collections.defaultdict(list)
    interval_indices = [-1] * len(ordered_events)
    interval_idx = 0

    for i, event in enumerate(ordered_events):
        while interval_idx < len(intervals) and event.time > intervals[interval_idx][1]:
            interval_idx += 1
        if (
            interval_idx < len(intervals)
            and intervals[interval_idx][0] <= event.time <= intervals[interval_idx][1]
        ):
            result[intervals[interval_idx]].append(event)
            interval_indices[i] = interval_idx

    current_pseudo: List[Event] = []
    for i, event in enumerate(ordered_events):
        if interval_indices[i] == -1:
            current_pseudo.append(event)
        elif current_pseudo:
            pseudo = (current_pseudo[0].time, current_pseudo[-1].time)
            result[pseudo] = current_pseudo
            current_pseudo = []
    if current_pseudo:
        pseudo = (current_pseudo[0].time, current_pseudo[-1].time)
        result[pseudo] = current_pseudo

    return dict(result)


def bin_by_time(events: Iterable[Event]) -> Dict[datetime, List[Event]]:
    """Group events by exact timestamp, preserving first-seen order."""
    time_bins: Dict[datetime, List[Event]] = collections.defaultdict(list)
    for event in events:
        time_bins[event.time].append(event)
    return time_bins


def is_non_overlapping(intervals: Sequence[Interval]) -> bool:
    """True if no two intervals overlap (touching endpoints are allowed)."""
    ordered = sorted(intervals, key=lambda iv: iv[0])
    for prev, cur in zip(ordered, ordered[1:]):
        if cur[0] < prev[1]:
            return False
    return True


class FuzzyVisitStrategy:
    """Default encounter binning: fuzzy ``visit_id`` grouping with time merging."""

    def __init__(
        self,
        max_mismatches: int = 1,
        max_timedelta: timedelta = timedelta(hours=24),
    ) -> None:
        self.max_mismatches = max_mismatches
        self.max_timedelta = max_timedelta

    def bin(
        self, events: Sequence[Event], *, excluded_tables: Optional[Set[str]] = None
    ) -> List[Encounter]:
        excluded_tables = excluded_tables or set()
        rows = [
            (event.time, event.visit_id)
            for event in events
            if event.table not in excluded_tables
        ]
        groups = fuzzy_partition(
            rows, max_mismatches=self.max_mismatches, max_timedelta=self.max_timedelta
        )
        intervals = [
            (min(times), max(times)) for times, _ in (zip(*group) for group in groups)
        ]
        if not is_non_overlapping(intervals):
            logger.warning("Encounter intervals overlap; binning may be ambiguous")

        binned = bin_events(intervals, events, excluded_tables=excluded_tables)
        return [
            Encounter(start=start, end=end, events=evs)
            for (start, end), evs in binned.items()
        ]


def bin_encounters(
    events: Sequence[Event],
    *,
    strategy: Optional[EncounterStrategy] = None,
    excluded_tables: Optional[Set[str]] = None,
) -> List[Encounter]:
    """Bin ``events`` into encounters using ``strategy`` (default fuzzy/visit)."""
    strategy = strategy or FuzzyVisitStrategy()
    return strategy.bin(events, excluded_tables=excluded_tables)
