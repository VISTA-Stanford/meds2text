"""Read standard MEDS parquet shards (``data/**/*.parquet``) and yield Subjects.

The input is assumed to be an already-transformed MEDS extract. Shards are
streamed lazily so memory stays bounded regardless of dataset size.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from meds2text.subject import Event, Subject

logger = logging.getLogger(__name__)


def discover_parquet_shards(meds_root: str) -> List[str]:
    """Return sorted parquet shard paths under ``<meds_root>/data/**/*.parquet``."""
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
    """Assign shards round-robin so workers get similar counts when sizes vary."""
    if n_processes <= 1:
        return [list(shards)]
    buckets: List[List[str]] = [[] for _ in range(n_processes)]
    for i, path in enumerate(shards):
        buckets[i % n_processes].append(path)
    return buckets


def collect_subject_ids_from_shards(shards: Sequence[str]) -> Set[int]:
    """Return the set of unique ``subject_id`` values across shards."""
    ids: Set[int] = set()
    for path in shards:
        table = pq.read_table(path, columns=["subject_id"])
        col = table.column(0).combine_chunks()
        for v in pc.unique(col).to_pylist():
            if v is not None:
                ids.add(int(v))
    return ids


def rows_to_subject(subject_id: int, rows: List[Dict[str, Any]]) -> Subject:
    """Build a :class:`Subject` from MEDS row dicts, sorted by event time."""
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get("time") is None, r.get("time")),
    )
    events: List[Event] = []
    for row in sorted_rows:
        code = row["code"]
        if code is None:
            raise ValueError(f"Null code for subject_id={subject_id}")
        properties = {k: v for k, v in row.items() if k not in ("time", "code")}
        events.append(Event(row["time"], str(code), properties))
    return Subject(subject_id=subject_id, events=events)


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
) -> Iterator[Tuple[int, Subject]]:
    """Stream parquet row batches, emitting one :class:`Subject` per id block.

    Enforces MEDS per-file contiguity: a ``subject_id`` may not reappear after a
    later subject block has already been closed in the same worker stream.
    """
    completed: Set[int] = set()
    current_id: Optional[int] = None
    buffer: List[Dict[str, Any]] = []

    def flush_buffer() -> Optional[Tuple[int, Subject]]:
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
        return sid, rows_to_subject(sid, rows)

    for path in shard_paths:
        pf = pq.ParquetFile(path)
        if "subject_id" not in set(pf.schema_arrow.names):
            raise ValueError(
                f"Parquet file {path} missing required column 'subject_id'"
            )

        for batch in pf.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            for raw in table.to_pylist():
                row = _row_from_pylist_dict(raw)
                if row.get("subject_id") is None:
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


def load_and_validate_person_ids(
    person_ids_file: str, shard_paths: Sequence[str]
) -> List[int]:
    """Load person IDs from a CSV/TSV file, keeping those present in the shards.

    The file must have a ``person_id`` column header. Separator is inferred from
    the file extension (``.tsv`` vs ``.csv``) or sniffed from the header line.
    """
    if not os.path.exists(person_ids_file):
        raise FileNotFoundError(f"Person IDs file not found: {person_ids_file}")

    if person_ids_file.lower().endswith(".tsv"):
        separator = "\t"
    elif person_ids_file.lower().endswith(".csv"):
        separator = ","
    else:
        with open(person_ids_file, "r") as f:
            separator = "\t" if "\t" in f.readline() else ","

    try:
        df = pd.read_csv(person_ids_file, sep=separator)
    except Exception as e:
        raise ValueError(f"Error reading file {person_ids_file}: {e}")

    if "person_id" not in df.columns:
        raise ValueError(
            f"'person_id' column not found in {person_ids_file}. "
            f"Available columns: {list(df.columns)}"
        )

    requested = df["person_id"].astype(str).dropna().tolist()
    if not requested:
        raise ValueError(
            f"No person IDs found in 'person_id' column of {person_ids_file}"
        )

    available = {str(sid) for sid in collect_subject_ids_from_shards(shard_paths)}

    valid: List[int] = []
    missing: List[str] = []
    for person_id in requested:
        (
            valid.append(int(person_id))
            if person_id in available
            else missing.append(person_id)
        )

    logger.info(f"Loaded {len(requested)} person IDs from {person_ids_file}")
    logger.info(
        f"Found {len(valid)} valid person IDs in Parquet shards "
        f"({len(shard_paths)} files)"
    )
    if missing:
        logger.warning(
            f"Missing {len(missing)} person IDs from Parquet data: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if not valid:
        raise ValueError("No valid person IDs found in Parquet shards")
    return valid
