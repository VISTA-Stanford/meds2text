"""Tests for --emit-final rows/xml (after full compression pipeline)."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest
from lxml import etree

_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = _ROOT / "scripts" / "lumia_compression_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "lumia_compression_benchmark_emit2", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lumia_compression_benchmark_emit2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def L():
    return _load_benchmark()


def _parse_run_emit_rows(L, xml: bytes) -> str:
    root = etree.fromstring(xml)
    opts = L.BenchmarkCompressionOptions()
    L.run_lumia_compression_pipeline(root, opts, stages=None)
    buf = io.StringIO()
    L.emit_minimal_rows_from_root(root, buf)
    return buf.getvalue()


def test_emit_rows_after_pipeline(L, tmp_path):
    xml = b"""<eventstream person_id="99">
  <encounter>
    <person>
      <birthdate>2014-01-01</birthdate>
      <age><days>1</days><years>0</years></age>
      <payerplan>X</payerplan>
    </person>
    <events>
      <entry timestamp="2014-10-06 10:54">
        <event type="visit" code="STANFORD_VISIT/A" name="N">start</event>
        <event type="measurement" code="LOINC/1" name="W">3.0</event>
      </entry>
    </events>
  </encounter>
</eventstream>"""
    text = _parse_run_emit_rows(L, xml)
    lines = text.strip().split("\n")
    assert lines[0].startswith("[H]\tperson_id=99")
    assert lines[1] == "[R]\tencounter=0"
    assert lines[2].startswith("[P]\t")
    assert "birthdate=2014-01-01" in lines[2]
    assert lines[3].startswith("[T]\t2014-10-06 10:54")
    assert lines[4].startswith("[V]\t")
    assert "type=visit" in lines[4]
    assert lines[5].startswith("[E]\t")
    assert "type=measurement" in lines[5]


def test_visit_detail_uses_v_prefix(L):
    xml = b"""<eventstream person_id="1"><encounter><person/><events>
      <entry timestamp="2015-01-01 00:00">
        <event type="visit_detail" code="C" name="N">x</event>
      </entry>
    </events></encounter></eventstream>"""
    text = _parse_run_emit_rows(L, xml)
    assert "[V]\t" in text
    assert "type=visit_detail" in text


def test_collapsed_condition_emits_c_row(L):
    xml = b"""<eventstream person_id="1"><encounter><person/><events>
      <entry timestamp="2016-01-01 00:00">
        <event type="condition" code="ICD10CM/R05" name="Cough"/>
      </entry>
    </events></encounter></eventstream>"""
    text = _parse_run_emit_rows(L, xml)
    assert "[C]\t" in text
    assert "code=ICD10CM/R05" in text
    assert "name=Cough" in text


def test_emit_xml_after_pipeline(L):
    xml = b"""<eventstream person_id="1"><encounter><person/><events>
      <entry timestamp="2016-01-01 00:00">
        <event type="measurement" code="LOINC/1" name="W">1</event>
      </entry>
    </events></encounter></eventstream>"""
    root = etree.fromstring(xml)
    L.run_lumia_compression_pipeline(root, L.BenchmarkCompressionOptions(), stages=None)
    buf = io.StringIO()
    L.emit_compressed_xml_to_stream(root, buf)
    out = buf.getvalue()
    assert "<eventstream" in out
    assert "person_id" in out
    assert "LOINC/1" in out
