"""Read standard MEDS Parquet shards (data/**/*.parquet) and yield subjects as MutableSubject."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from meds_reader.transform import MutableEvent, MutableSubject

logger = logging.getLogger(__name__)


def discover_parquet_shards(meds_root: str) -> List[str]:
    pattern = os.path.join(meds_root, "data", "**", "*.parquet")
    shards = sorted(glob.glob(pattern, recursive=True))
    if not shards:
        raise FileNotFoundError(
            f"No Parquet shards found under {meds_root!r} (expected data/**/*.parquet)"
        )
    return shards


def partition_shards_across_workers(
    shards: Sequence[str], n_processes: int
) -> List[List[str]]:
    """Assign shards round-robin so workers get similar counts when file sizes vary."""
    if n_processes <= 1:
        return [list(shards)]
    buckets: List[List[str]] = [[] for _ in range(n_processes)]
    for i, path in enumerate(shards):
        buckets[i % n_processes].append(path)
    return buckets


def collect_subject_ids_from_shards(shards: Sequence[str]) -> Set[int]:
    """Unique subject_id values across shards (subject_id column only)."""
    ids: Set[int] = set()
    for path in shards:
        table = pq.read_table(path, columns=["subject_id"])
        col = table.column(0).combine_chunks()
        uniq = pc.unique(col)
        for v in uniq.to_pylist():
            if v is None:
                continue
            ids.add(int(v))
    return ids


def rows_to_mutable_subject(
    subject_id: int, rows: List[Dict[str, Any]]
) -> MutableSubject:
    """Build MutableSubject from MEDS row dicts (same contract as meds_reader.transform)."""
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r.get("time") is None,
            r.get("time"),
        ),
    )
    events: List[MutableEvent] = []
    for event_dict in sorted_rows:
        t = event_dict["time"]
        code = event_dict["code"]
        if code is None:
            raise ValueError(f"Null code for subject_id={subject_id}")
        code_str = str(code)
        properties = {k: v for k, v in event_dict.items() if k not in ("time", "code")}
        events.append(MutableEvent(t, code_str, properties))
    return MutableSubject(subject_id=subject_id, events=events)


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _row_from_pylist_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _normalize_cell(v) for k, v in d.items()}


def iter_subjects_from_parquet_files(
    shard_paths: Sequence[str],
    *,
    allowed_subject_ids: Optional[Set[int]] = None,
    batch_size: int = 65536,
) -> Iterator[Tuple[int, MutableSubject]]:
    """
    Stream Parquet row batches; for each contiguous subject_id block emit one MutableSubject.

    Enforces MEDS per-file contiguity: a subject_id may not reappear after another subject
    was started in the same worker stream.
    """
    completed: Set[int] = set()
    current_id: Optional[int] = None
    buffer: List[Dict[str, Any]] = []

    def flush_buffer() -> Optional[Tuple[int, MutableSubject]]:
        nonlocal buffer, current_id
        if current_id is None or not buffer:
            buffer = []
            current_id = None
            return None
        sid = current_id
        rows = buffer
        buffer = []
        current_id = None
        if allowed_subject_ids is not None and sid not in allowed_subject_ids:
            return None
        return sid, rows_to_mutable_subject(sid, rows)

    for path in shard_paths:
        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        if "subject_id" not in schema_names:
            raise ValueError(
                f"Parquet file {path} missing required column 'subject_id'"
            )

        for batch in pf.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            for raw in table.to_pylist():
                row = _row_from_pylist_dict(raw)
                if "subject_id" not in row or row["subject_id"] is None:
                    raise ValueError(f"Row missing subject_id in {path}")
                sid = int(row["subject_id"])

                if current_id is None:
                    current_id = sid
                elif sid != current_id:
                    if sid in completed:
                        raise ValueError(
                            f"MEDS contiguity violated in {path}: subject_id={sid} "
                            "reappears after a later subject block was already closed."
                        )
                    completed.add(current_id)
                    out = flush_buffer()
                    if out is not None:
                        yield out
                    current_id = sid

                buffer.append(row)

        if buffer and current_id is not None:
            completed.add(current_id)
            out = flush_buffer()
            if out is not None:
                yield out
            buffer = []
            current_id = None


__all__ = [
    "collect_subject_ids_from_shards",
    "discover_parquet_shards",
    "iter_subjects_from_parquet_files",
    "partition_shards_across_workers",
    "rows_to_mutable_subject",
]
