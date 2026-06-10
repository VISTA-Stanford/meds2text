"""Unit tests for clinical-note compression (``--compress_notes``)."""

from __future__ import annotations

from datetime import datetime

import pytest

from meds2text.render.note_compress import (
    NoteCompressor,
    collapse_repeated_symbols,
    collapse_whitespace,
    compress_subject_notes,
    dedup_ngrams,
    mark_duplicate_token_indices,
    normalize_unicode_whitespace,
    scrub_deid,
)
from meds2text.subject import Event, Subject


def test_scrub_deid_removes_placeholders():
    assert "999-999-9999" not in scrub_deid("call (999-999-9999) now")
    assert "[0000]" not in scrub_deid("MRN: [0000]")
    assert "@NAME@" not in scrub_deid("Patient @NAME@ seen")


@pytest.mark.parametrize(
    "text",
    [
        "one two three",
        "one    two\n\tthree",
        "\n\nalpha\nbeta\n\n",
        "x " * 15,
    ],
)
def test_collapse_whitespace_preserves_token_list(text: str):
    """Collapse normalizes spacing; ``str.split()`` token sequence is unchanged."""
    collapsed = collapse_whitespace(text)
    assert text.split() == collapsed.split()


def test_dedup_ngrams_removes_cross_note_redundancy():
    prior: set = set()
    phrase = "one two three four five six seven eight nine ten"
    first = dedup_ngrams(phrase, 10, prior)
    assert first == phrase
    second = dedup_ngrams(f"{phrase} eleven twelve", 10, prior)
    assert second == "eleven twelve"


def test_mark_duplicate_indices_within_note():
    tokens = ("Same " * 25).split()
    prior: set = set()
    redundant = mark_duplicate_token_indices(tokens, 10, prior)
    assert len(redundant) > 0


def test_compress_subject_notes_chronological_order():
    """Later note drops tokens already seen in an earlier note's 10-grams."""
    phrase = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    subject = Subject(
        subject_id=1,
        events=[
            Event(
                datetime(2020, 1, 2),
                "Note/2",
                {"table": "note", "text_value": f"{phrase} later"},
            ),
            Event(
                datetime(2020, 1, 1),
                "Note/1",
                {"table": "note", "text_value": phrase},
            ),
        ],
    )
    compress_subject_notes(subject)
    notes = [e for e in subject.events if e.table == "note"]
    by_code = {e.code: e.text_value for e in notes}
    assert by_code["Note/1"] == phrase
    assert by_code["Note/2"] == "later"


def test_note_compressor_scrubs_and_collapses():
    raw = "  MRN: [0000]   hello   world  \n"
    out = NoteCompressor().compress(raw)
    assert "MRN" not in out
    assert out == "hello world"


def test_collapse_repeated_symbols():
    assert collapse_repeated_symbols("Section *************** end") == "Section * end"
    assert collapse_repeated_symbols("----") == "----"
    assert collapse_repeated_symbols("----------------") == "-"
    assert collapse_repeated_symbols("=====RESULT=====") == "=RESULT="


def test_normalize_unicode_whitespace():
    assert normalize_unicode_whitespace("a\u00a0\u00a0b") == "a  b"
    assert collapse_whitespace(normalize_unicode_whitespace("a\u00a0\u00a0b")) == "a b"
