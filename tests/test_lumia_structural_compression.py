"""Tests for optional LUMIA structural compression (person slim, drop event type)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from lxml import etree

_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = _ROOT / "scripts" / "lumia_compression_benchmark.py"
    spec = importlib.util.spec_from_file_location("lumia_compression_benchmark", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lumia_compression_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def L():
    return _load_benchmark()


def test_slim_person_keeps_first_encounter_full(L):
    xml = b"""<eventstream person_id="1">
  <encounter>
    <person>
      <birthdate>2014-09-23</birthdate>
      <age><days>12</days><years>0</years></age>
      <demographics><gender>MALE</gender></demographics>
      <payerplan>COMMERCIAL</payerplan>
    </person>
    <events/>
  </encounter>
  <encounter>
    <person>
      <birthdate>2014-09-23</birthdate>
      <age><days>580</days><years>1</years></age>
      <demographics><gender>MALE</gender></demographics>
      <payerplan>COMMERCIAL</payerplan>
    </person>
    <events/>
  </encounter>
</eventstream>"""
    root = etree.fromstring(xml)
    L.apply_person_slim_followup_encounters(root)
    encs = [c for c in root if c.tag == "encounter"]
    p0 = encs[0].find("person")
    assert p0 is not None
    assert p0.find("birthdate") is not None
    assert p0.find("demographics") is not None
    p1 = encs[1].find("person")
    assert p1 is not None
    assert list(p1) == [p1.find("age"), p1.find("payerplan")]
    assert p1.find("birthdate") is None
    assert p1.find("demographics") is None
    assert (p1.find("age").findtext("years")) == "1"


def test_slim_person_single_encounter_noop(L):
    xml = b"""<eventstream><encounter><person><age><years>1</years></age></person><events/></encounter></eventstream>"""
    root = etree.fromstring(xml)
    L.apply_person_slim_followup_encounters(root)
    p = root.find("encounter").find("person")
    assert p.find("age") is not None


def test_drop_event_type_attr(L):
    xml = b"""<eventstream><encounter><person/><events>
      <entry><event type="measurement" code="LOINC/1" name="x">3</event></entry>
    </events></encounter></eventstream>"""
    root = etree.fromstring(xml)
    ev = root.find(".//event")
    assert ev.get("type") == "measurement"
    L.apply_drop_event_type_attr(root)
    assert ev.get("type") is None
    assert ev.get("code") == "LOINC/1"


def test_clone_root_with_optional_person_slim(L):
    opts = L.BenchmarkCompressionOptions(slim_person_followup_encounters=True)
    xml = b"""<eventstream><encounter><person><birthdate>x</birthdate><age><years>0</years></age><payerplan>p</payerplan></person><events/></encounter><encounter><person><birthdate>x</birthdate><age><years>1</years></age><payerplan>p</payerplan></person><events/></encounter></eventstream>"""
    root = etree.fromstring(xml)
    c = L.clone_root_with_optional_person_slim(root, opts)
    encs = [x for x in c if x.tag == "encounter"]
    assert encs[0].find("person").find("birthdate") is not None
    assert encs[1].find("person").find("birthdate") is None
