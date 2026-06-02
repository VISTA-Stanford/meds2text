"""Tests for raw MEDS Parquet shard streaming."""

from __future__ import annotations

from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from meds2text.io.parquet import (
    collect_subject_ids_from_shards,
    discover_parquet_shards,
    iter_subjects_from_parquet_files,
    partition_shards_across_workers,
    rows_to_subject,
)


def _write_meds_shard(path, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def test_discover_parquet_shards(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_meds_shard(
        str(data_dir / "a.parquet"),
        [
            {
                "subject_id": 1,
                "time": datetime(2020, 1, 1, 12, 0, 0),
                "code": "A",
                "table": "visit",
            }
        ],
    )
    shards = discover_parquet_shards(str(tmp_path))
    assert len(shards) == 1
    assert shards[0].endswith("a.parquet")


def test_discover_raises_when_no_data(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        discover_parquet_shards(str(tmp_path))


def test_partition_shards_round_robin():
    paths = ["a", "b", "c", "d", "e"]
    p3 = partition_shards_across_workers(paths, 3)
    assert len(p3) == 3
    assert sum(len(x) for x in p3) == 5


def test_iter_subjects_two_contiguous_subjects(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {
            "subject_id": 10,
            "time": datetime(2020, 1, 1, 8, 0, 0),
            "code": "X",
            "table": "observation",
        },
        {
            "subject_id": 10,
            "time": datetime(2020, 1, 1, 9, 0, 0),
            "code": "Y",
            "table": "observation",
        },
        {
            "subject_id": 20,
            "time": datetime(2020, 1, 2, 8, 0, 0),
            "code": "Z",
            "table": "observation",
        },
    ]
    _write_meds_shard(str(data_dir / "s.parquet"), rows)

    out = list(iter_subjects_from_parquet_files([str(data_dir / "s.parquet")]))
    assert len(out) == 2
    assert out[0][0] == 10
    assert [e.code for e in out[0][1].events] == ["X", "Y"]
    assert out[1][0] == 20
    assert [e.code for e in out[1][1].events] == ["Z"]


def test_iter_subjects_allowlist_filters(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {
            "subject_id": 1,
            "time": datetime(2020, 1, 1),
            "code": "A",
            "table": "observation",
        },
        {
            "subject_id": 2,
            "time": datetime(2020, 1, 2),
            "code": "B",
            "table": "observation",
        },
    ]
    _write_meds_shard(str(data_dir / "s.parquet"), rows)
    out = list(
        iter_subjects_from_parquet_files(
            [str(data_dir / "s.parquet")], allowed_subject_ids={2}
        )
    )
    assert len(out) == 1
    assert out[0][0] == 2


def test_collect_subject_ids_from_shards(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_meds_shard(
        str(data_dir / "x.parquet"),
        [
            {"subject_id": 5, "time": datetime(2020, 1, 1), "code": "a", "table": "t"},
            {"subject_id": 5, "time": datetime(2020, 1, 2), "code": "b", "table": "t"},
        ],
    )
    _write_meds_shard(
        str(data_dir / "y.parquet"),
        [
            {"subject_id": 7, "time": datetime(2020, 1, 1), "code": "c", "table": "t"},
        ],
    )
    ids = collect_subject_ids_from_shards(
        [str(data_dir / "x.parquet"), str(data_dir / "y.parquet")]
    )
    assert ids == {5, 7}


def test_rows_to_mutable_subject_sorts_by_time():
    rows = [
        {
            "subject_id": 1,
            "time": datetime(2020, 1, 2),
            "code": "late",
            "table": "observation",
        },
        {
            "subject_id": 1,
            "time": datetime(2020, 1, 1),
            "code": "early",
            "table": "observation",
        },
    ]
    subj = rows_to_subject(1, rows)
    assert [e.code for e in subj.events] == ["early", "late"]
