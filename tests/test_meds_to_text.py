"""End-to-end characterization + encounter-binning unit tests.

The characterization test renders a tiny synthetic (already-transformed) MEDS
parquet extract to LUMIA XML and compares it against a committed golden file,
guarding the rendering pipeline against accidental output changes.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import marisa_trie
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from meds2text import pipeline
from meds2text.config import TextifyConfig
from meds2text.encounters import FuzzyVisitStrategy, bin_encounters, fuzzy_partition
from meds2text.subject import Event

GOLDEN_DIR = Path(__file__).parent / "golden"

# --- synthetic MEDS extract ------------------------------------------------

_DT = datetime
_SUBJECT_ID = 100
_MEDS_ROWS = [
    # person / demographics
    {"code": "MEDS_BIRTH", "time": _DT(1980, 5, 15), "table": "person"},
    {"code": "Gender/8507", "time": _DT(1980, 5, 15), "table": "person"},
    {"code": "Race/8527", "time": _DT(1980, 5, 15), "table": "person"},
    {"code": "Ethnicity/38003564", "time": _DT(1980, 5, 15), "table": "person"},
    # encounter 1 (Jan 2020)
    {
        "code": "Visit/IP",
        "time": _DT(2020, 1, 1, 8, 0),
        "end": _DT(2020, 1, 3, 8, 0),
        "table": "visit",
        "visit_id": 500,
        "provider_id": 1,
        "care_site_id": 10,
    },
    {
        "code": "LOINC/8867-4",
        "time": _DT(2020, 1, 1, 9, 30),
        "table": "measurement",
        "visit_id": 500,
        "provider_id": 1,
        "care_site_id": 10,
        "numeric_value": 72.0,
    },
    {
        "code": "Note/11506-3",
        "time": _DT(2020, 1, 1, 10, 0),
        "table": "note",
        "visit_id": 500,
        "provider_id": 1,
        "text_value": "Patient stable.",
    },
    {
        "code": "ICD10CM/I10",
        "time": _DT(2020, 1, 2, 9, 0),
        "table": "condition",
        "visit_id": 500,
    },
    # encounter 2 (Jun 2021): > 24h gap AND >1 event with a new visit_id, so it
    # exceeds the fuzzy glitch budget and forms a distinct encounter.
    {
        "code": "LOINC/2345-7",
        "time": _DT(2021, 6, 1, 12, 0),
        "table": "measurement",
        "visit_id": 600,
        "provider_id": 2,
        "care_site_id": 20,
        "numeric_value": 105.0,
    },
    {
        "code": "ICD10CM/E11",
        "time": _DT(2021, 6, 1, 13, 0),
        "table": "condition",
        "visit_id": 600,
        "provider_id": 2,
        "care_site_id": 20,
    },
]

_ONTOLOGY = {
    "Gender/8507": "MALE",
    "Race/8527": "White",
    "Ethnicity/38003564": "Not Hispanic or Latino",
    "Visit/IP": "Inpatient Visit",
    "LOINC/8867-4": "Heart rate",
    "Note/11506-3": "Progress note",
    "ICD10CM/I10": "Essential hypertension",
    "LOINC/2345-7": "Glucose",
    "ICD10CM/E11": "Type 2 diabetes mellitus",
}

_ALL_COLUMNS = [
    "subject_id",
    "time",
    "code",
    "table",
    "visit_id",
    "numeric_value",
    "text_value",
    "end",
    "provider_id",
    "care_site_id",
]


def _build_meds_extract(root: Path) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    rows = []
    for row in _MEDS_ROWS:
        full = {col: None for col in _ALL_COLUMNS}
        full.update(row)
        full["subject_id"] = _SUBJECT_ID
        rows.append(full)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(data_dir / "shard_0.parquet"))


def _build_ontology(root: Path) -> None:
    root.mkdir(parents=True)
    trie = marisa_trie.BytesTrie(
        (code, desc.encode("utf-8")) for code, desc in _ONTOLOGY.items()
    )
    trie.save(str(root / "descriptions.trie"))


def _build_metadata(root: Path) -> None:
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "provider_id": 1,
                "gender_concept_id": 8507,
                "specialty_source_value": "Cardiology",
                "year_of_birth": 1975,
                "care_site_id": 10,
            },
            {
                "provider_id": 2,
                "gender_concept_id": 8532,
                "specialty_source_value": "Endocrinology",
                "year_of_birth": 1980,
                "care_site_id": 20,
            },
        ]
    ).to_csv(root / "provider.csv", index=False)
    pd.DataFrame(
        [
            {"care_site_id": 10, "care_site_name": "Main Hospital"},
            {"care_site_id": 20, "care_site_name": "Outpatient Clinic"},
        ]
    ).to_csv(root / "care_site.csv", index=False)
    pd.DataFrame(
        [
            {
                "person_id": _SUBJECT_ID,
                "payer_plan_period_start_date": "2019-01-01",
                "payer_plan_period_end_date": "2022-12-31",
                "payer_source_value": "Medicare",
            }
        ]
    ).to_csv(root / "payer_plan_period.csv", index=False)


@pytest.fixture
def textify_env(tmp_path):
    meds_root = tmp_path / "meds"
    ontology_root = tmp_path / "ontology"
    metadata_root = tmp_path / "metadata"
    output_root = tmp_path / "output"
    _build_meds_extract(meds_root)
    _build_ontology(ontology_root)
    _build_metadata(metadata_root)
    return {
        "meds": str(meds_root),
        "ontology": str(ontology_root),
        "metadata": str(metadata_root),
        "output": str(output_root),
    }


def _assert_matches_golden(name: str, content: str) -> None:
    GOLDEN_DIR.mkdir(exist_ok=True)
    golden_path = GOLDEN_DIR / name
    if not golden_path.exists():
        golden_path.write_text(content, encoding="utf-8")
        pytest.skip(f"Golden file {name} created; re-run to compare.")
    assert content == golden_path.read_text(encoding="utf-8")


def test_textify_lumia_xml_golden(textify_env):
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        path_to_metadata=textify_env["metadata"],
        output_format="lumia_xml",
        include_contexts=("person", "providers", "care_sites"),
    )
    pipeline.run(config)

    out_file = os.path.join(textify_env["output"], f"{_SUBJECT_ID}.xml")
    assert os.path.exists(out_file)
    content = Path(out_file).read_text(encoding="utf-8")
    _assert_matches_golden("subject_100.lumia.xml", content)


def test_textify_two_encounters_and_event_names(textify_env):
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        output_format="lumia_xml",
    )
    pipeline.run(config)

    content = Path(os.path.join(textify_env["output"], f"{_SUBJECT_ID}.xml")).read_text(
        encoding="utf-8"
    )
    # Two derived encounters (Jan 2020 + Jun 2021).
    assert content.count("<encounter>") == 2
    # Ontology-resolved names appear as attributes.
    assert 'name="Heart rate"' in content
    assert 'name="Essential hypertension"' in content


def test_textify_event_type_filter(textify_env):
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        output_format="lumia_xml",
        event_types=("note",),
    )
    pipeline.run(config)

    content = Path(os.path.join(textify_env["output"], f"{_SUBJECT_ID}.xml")).read_text(
        encoding="utf-8"
    )
    assert 'table="note"' in content
    assert 'table="measurement"' not in content
    assert 'table="condition"' not in content


def test_textify_json_format(textify_env):
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        output_format="lumia_json",
    )
    pipeline.run(config)
    assert os.path.exists(os.path.join(textify_env["output"], f"{_SUBJECT_ID}.json"))


def test_person_static_blocks_only_in_first_encounter(textify_env):
    """By default birthdate/demographics appear once; age/payerplan every visit."""
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        path_to_metadata=textify_env["metadata"],
        output_format="lumia_xml",
        include_contexts=("person", "providers", "care_sites"),
    )
    pipeline.run(config)
    content = Path(os.path.join(textify_env["output"], f"{_SUBJECT_ID}.xml")).read_text(
        encoding="utf-8"
    )

    assert content.count("<encounter>") == 2
    assert content.count("<birthdate>") == 1
    assert content.count("<demographics>") == 1
    # Dynamic fields repeat in both encounters.
    assert content.count("<age>") == 2
    assert content.count("<payerplan>") == 2


def test_person_fields_can_repeat_static_blocks(textify_env):
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        path_to_metadata=textify_env["metadata"],
        output_format="lumia_xml",
        include_contexts=("person",),
        person_fields_every_encounter=("birthdate", "age"),
        person_fields_first_encounter=(),
    )
    pipeline.run(config)
    content = Path(os.path.join(textify_env["output"], f"{_SUBJECT_ID}.xml")).read_text(
        encoding="utf-8"
    )

    # birthdate now repeats; demographics/payerplan dropped entirely.
    assert content.count("<birthdate>") == 2
    assert content.count("<demographics>") == 0
    assert content.count("<payerplan>") == 0


def test_person_fields_can_drop_all_blocks(textify_env):
    config = TextifyConfig(
        path_to_meds=textify_env["meds"],
        path_to_output=textify_env["output"],
        path_to_ontology=textify_env["ontology"],
        path_to_metadata=textify_env["metadata"],
        output_format="lumia_xml",
        include_contexts=("person",),
        person_fields_every_encounter=(),
        person_fields_first_encounter=(),
    )
    pipeline.run(config)
    content = Path(os.path.join(textify_env["output"], f"{_SUBJECT_ID}.xml")).read_text(
        encoding="utf-8"
    )
    # No <person> blocks emitted when no fields are requested.
    assert "<person>" not in content


# --- encounter binning unit tests -----------------------------------------


def _evt(time, visit_id, table="measurement"):
    return Event(time, "C", {"table": table, "visit_id": visit_id})


def test_fuzzy_partition_same_timestamp_never_split():
    t0 = datetime(2020, 1, 1)
    rows = [(t0, "A"), (t0, "B")]
    groups = fuzzy_partition(rows, max_mismatches=0, max_timedelta=timedelta(0))
    assert len(groups) == 1


def test_fuzzy_partition_time_gap_force_merge():
    t0 = datetime(2020, 1, 1)
    # Different ids but within the 24h gap window -> single group.
    rows = [(t0, "A"), (t0 + timedelta(hours=1), "B")]
    groups = fuzzy_partition(rows, max_mismatches=0, max_timedelta=timedelta(hours=24))
    assert len(groups) == 1


def test_fuzzy_partition_glitch_merge_within_budget():
    t0 = datetime(2020, 1, 1)
    rows = [
        (t0, "A"),
        (t0 + timedelta(days=2), "B"),  # glitch (1) merged within budget
        (t0 + timedelta(days=4), "A"),
    ]
    groups = fuzzy_partition(rows, max_mismatches=1, max_timedelta=timedelta(hours=24))
    assert len(groups) == 1


def test_fuzzy_partition_splits_on_large_gap_and_glitch_overflow():
    t0 = datetime(2020, 1, 1)
    rows = [
        (t0, "A"),
        (t0 + timedelta(hours=1), "A"),
        (t0 + timedelta(days=2), "B"),
        (t0 + timedelta(days=2, hours=1), "B"),
    ]
    groups = fuzzy_partition(rows, max_mismatches=1, max_timedelta=timedelta(hours=24))
    assert len(groups) == 2
    assert [r[1] for r in groups[0]] == ["A", "A"]
    assert [r[1] for r in groups[1]] == ["B", "B"]


def test_bin_encounters_excludes_person_and_groups():
    base = datetime(2020, 1, 1, 8, 0)
    far = base + timedelta(days=400)
    events = [
        _evt(datetime(1980, 1, 1), None, table="person"),
        _evt(base, 1),
        _evt(base + timedelta(hours=1), 1),
        # Two far-future events with a new visit_id exceed the glitch budget.
        _evt(far, 2),
        _evt(far + timedelta(hours=1), 2),
    ]
    encounters = bin_encounters(events, excluded_tables={"person"})
    assert len(encounters) == 2
    # Person events are excluded from binning.
    assert all(e.table != "person" for enc in encounters for e in enc.events)


def test_bin_encounters_emits_all_events():
    base = datetime(2020, 1, 1)
    # A lone far event with a new visit_id is glitch-merged, but still emitted.
    events = [_evt(base, 1), _evt(base + timedelta(days=500), 2)]
    strategy = FuzzyVisitStrategy()
    encounters = strategy.bin(events, excluded_tables={"person"})
    assert sum(len(enc.events) for enc in encounters) == 2
