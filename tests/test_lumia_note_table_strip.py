"""Tests for strip_note_table_blocks in lumia_compression_benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lcb():
    return _load_module(
        "lumia_compression_benchmark",
        _ROOT / "scripts" / "lumia_compression_benchmark.py",
    )


@pytest.fixture(scope="module")
def excerpts():
    return _load_module(
        "lumia_note_table_excerpts",
        Path(__file__).resolve().parent / "fixtures" / "lumia_note_table_excerpts.py",
    )


def test_strip_procedure_ascii_table(lcb, excerpts):
    s = "Intro text. " + excerpts.PROCEDURE_NOTE_ASCII + " Trailing prose."
    out = lcb.strip_note_table_blocks(s)
    assert "Please view results" not in out
    assert "Abnormality" not in out
    assert "Intro text." in out
    assert "Trailing prose." in out


def test_strip_recent_results_until_reviewed(lcb, excerpts):
    out = lcb.strip_note_table_blocks(excerpts.RECENT_RESULTS_EMBEDDED)
    assert "Recent Results (from the past" not in out
    assert "Result Value Ref Range" not in out
    assert "I have reviewed the labs." in out
    assert "Echo:" in out


def test_strip_recent_results_until_radiology(lcb, excerpts):
    out = lcb.strip_note_table_blocks(excerpts.RECENT_RESULTS_TO_RADIOLOGY)
    assert "Recent Results" not in out
    assert "Radiology & Imaging Studies" in out


def test_strip_ecg_banner_keeps_interpretation(lcb, excerpts):
    out = lcb.strip_note_table_blocks(excerpts.ECG_NOTE)
    assert "Pediatric ECG interpretation" not in out
    assert "FINDINGS:" not in out
    assert "Sinus rhythm" not in out
    assert "INTERPRETATION:" in out
    assert "ABNORMAL ECG" in out


def test_strip_lines_drains_prefix(lcb, excerpts):
    out = lcb.strip_note_table_blocks(excerpts.LINES_DRAINS_TO_VITALS)
    assert "Patient Lines/Drains/Airways Status" not in out
    assert "Vital Signs:" in out
    assert "Temp: 36.6" in out


def test_strip_pipe_metadata_fields(lcb, excerpts):
    out = lcb.strip_note_table_blocks(excerpts.PIPE_METADATA_HEADER)
    assert "Admitting Service: Neonatology |" not in out
    assert "Admission Date:" not in out
    assert "Hospital Day:" not in out
    assert "Day of Life:" not in out
    assert "NAME: Patient A" in out
    assert "Date of Service:" in out


def test_strip_idempotent_small_string(lcb):
    out = lcb.strip_note_table_blocks("No tables here.")
    assert out == "No tables here."
