"""Whitespace collapse vs 10-gram tokenization in lumia_compression_benchmark."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from lxml import etree

_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = _ROOT / "scripts" / "lumia_compression_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "lumia_compression_benchmark_ws", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lumia_compression_benchmark_ws"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def L():
    return _load_benchmark()


@pytest.mark.parametrize(
    "text",
    [
        "one two three",
        "one    two\n\tthree",
        "\n\nalpha\nbeta\n\n",
        "x " * 15,
        "Same phrase repeated. Same phrase repeated.",
    ],
)
def test_collapse_does_not_change_token_list_for_10grams(L, text: str):
    """str.split() already merges whitespace runs; collapse + split yields same tokens."""
    collapsed = L._collapse_note_whitespace(text)
    assert text.split() == collapsed.split()


def test_short_note_returns_collapsed_when_flag_on(L):
    """Notes with <10 tokens return original text unless collapse is enabled."""
    prior: set = set()
    raw = "  hello   world  \n"
    assert (
        L.compress_note_text_pmc9513649(raw, 10, prior, collapse_note_whitespace=False)
        == raw
    )
    prior.clear()
    assert (
        L.compress_note_text_pmc9513649(raw, 10, prior, collapse_note_whitespace=True)
        == "hello world"
    )


def test_long_note_duplicate_mask_same_after_collapse(L):
    """Same token stream → same duplicate indices (collapse does not alter grams here)."""
    text = "Same " * 25
    tokens_plain = text.split()
    tokens_collapsed = L._collapse_note_whitespace(text).split()
    assert tokens_plain == tokens_collapsed
    prior: set = set()
    r0 = L.mark_duplicate_token_indices_pmc9513649(tokens_plain, 10, prior)
    prior2: set = set()
    r1 = L.mark_duplicate_token_indices_pmc9513649(tokens_collapsed, 10, prior2)
    assert r0 == r1


@pytest.mark.skipif(
    not (_ROOT / "data/inspect_lumia_xml/128492363.xml").is_file(),
    reason="fixture Lumia XML not present",
)
def test_real_file_stage2_size_collapse_no_worse_than_baseline(L):
    """Empirical: collapse is same or slightly smaller after notes (never larger here)."""
    path = str(_ROOT / "data/inspect_lumia_xml/128492363.xml")
    off = L.BenchmarkCompressionOptions(collapse_note_whitespace=False)
    on = L.BenchmarkCompressionOptions(collapse_note_whitespace=True)
    _, _, stages_off, _, _, _ = L.process_pipeline_all_rules(path, off)
    _, _, stages_on, _, _, _ = L.process_pipeline_all_rules(path, on)
    # index 2 = after_notes_10gram
    assert stages_on[2][2] <= stages_off[2][2]
    assert stages_on[2][1] == stages_off[2][1]


def test_pipeline_short_note_whitespace_only_collapsed_when_flag(L):
    """Short notes (<10 tokens) keep original spacing unless collapse is on."""
    xml = b"""<eventstream person_id="p">
  <encounter><events>
    <entry timestamp="2020-01-01 00:00">
      <event type="note">   hello   world   \n</event>
    </entry>
  </events></encounter>
</eventstream>"""
    off = L.BenchmarkCompressionOptions(collapse_note_whitespace=False)
    on = L.BenchmarkCompressionOptions(collapse_note_whitespace=True)
    r_off = etree.fromstring(xml)
    r_on = etree.fromstring(xml)
    L.run_lumia_compression_pipeline(r_off, off)
    L.run_lumia_compression_pipeline(r_on, on)
    s_off = L._xml_to_string(r_off)
    s_on = L._xml_to_string(r_on)
    assert "   hello   world" in s_off
    assert "   hello   world" not in s_on
    assert "hello world" in s_on
    assert len(s_on.encode("utf-8")) < len(s_off.encode("utf-8"))
