#!/usr/bin/env python3
"""
Rule-based compression benchmark for LUMIA XML exports.

Rules:
  1) Notes (PMC9513649-style 10-grams): slide a window of 10 adjacent
     whitespace-separated tokens (Python str.split(); not tiktoken) one token
     at a time. If a 10-gram appears in
     any *previous* note (chronological order by <entry timestamp>) or earlier
     in the *same* note, every token in that window is marked duplicate and
     dropped from the compressed note. Prior notes contribute ngrams from their
     *original* text before compression (same spirit as the paper's cross-note
     duplication). See https://pmc.ncbi.nlm.nih.gov/articles/PMC9513649/
     Optional --strip-note-tables removes table-like blocks (procedure ASCII
     tables, embedded Recent Results lab grids, Lines/Drains status, ECG banner
     findings, pipe-separated header fields) before 10-gram compression.
     Optional --collapse-note-whitespace normalizes runs of whitespace to a
     single space before tokenization; the token list usually matches plain
     split() (runs are already one separator), but short notes (<10 tokens) can
     shrink because they otherwise return the original string unchanged.
  2) Labs (measurement): per (code, calendar day), if there are *more than two*
     same-code measurements that day (chronologically ordered in the bin), keep
     the first as baseline, drop intermediates, rewrite the last as min/max/median
     summary. Pairs (≤2 that day) are left unchanged.
  3) Conditions: collapse all <event type="condition"/> with the same code
     into one <condition .../> interval under <collapsed_conditions>.
  4) Optional --slim-person-followup-encounters: keep full <person> (birthdate,
     age, demographics, payerplan) on the first <encounter> only; on later
     encounters replace <person> with just <age> and <payerplan>.
  5) Optional --drop-event-type-attr: after notes/labs/conditions, remove the
     type="..." attribute from every <event> (earlier steps still require type).

Reports token and UTF-8 character savings. Default output is human-readable
percentages plus a stacked table (size after each rule in order); use --tsv for
a compact tab-separated table (--tsv omits the composition table). Use
--no-composition to hide the stacked table in human-readable mode.

Headline byte metrics (human mode and TSV): baseline is on-disk raw XML UTF-8
bytes; after the pipeline, compressed XML UTF-8 bytes (step 1: structural
compression %) and minimal rows UTF-8 bytes (step 2: minimal-document % vs raw).

Optional --save-compression-artifacts PREFIX (human mode only) writes
PREFIX_breakdown.txt (same text as the console report plus option summary and
per-file byte breakdown), and PREFIX[_STEM]_final.xml / _final.rows per input file.

Optional reference path (e.g. medalign instructions) is accepted for logging only.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, TextIO, Tuple

from lxml import etree

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    TOKENIZER_LABEL = "tiktoken cl100k_base"
except ImportError:

    def count_tokens(text: str) -> int:
        """Whitespace-token proxy when tiktoken is not installed."""
        return len(re.findall(r"\S+", text)) or (len(text) // 4 + 1)

    TOKENIZER_LABEL = "whitespace pseudo-tokens"


def _xml_to_string(root: etree._Element) -> str:
    return etree.tostring(root, encoding="unicode")


def _parse_ts(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    t = text.strip()
    if not t or t == "_":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _note_plain_text(el: etree._Element) -> str:
    parts = [el.text or ""]
    parts.extend(etree.tostring(c, encoding="unicode") for c in el)
    parts.append(el.tail or "")
    return "".join(parts)


# Table-like blocks in flattened LUMIA note text (single-line friendly; use DOTALL).
_RE_PROCEDURE_ASCII_TABLE = re.compile(
    r"The following orders were created for panel order\s+.+?Procedure\s+Abnormality\s+Status\s+"
    r".*?Please view results for these tests on the individual orders\.",
    re.DOTALL,
)
_RE_RECENT_RESULTS_BLOCK = re.compile(
    r"Recent Results \(from the past \d+ hour\(s\)\).*?"
    r"(?=(?:Radiology|Echo:|I have reviewed the labs\.|Outside Labs:|ASSESSMENT|IMPRESSION|PHYSICAL EXAM:))",
    re.DOTALL,
)
_RE_LINES_DRAINS_TO_VITALS = re.compile(
    r"Patient Lines/Drains/Airways Status.*?Vital Signs:",
    re.DOTALL,
)
_RE_ECG_FINDINGS_BANNER = re.compile(
    r"-{5,}\s*Pediatric ECG interpretation\s*-{5,}.*?(?=INTERPRETATION:)",
    re.DOTALL,
)
_RE_PIPE_METADATA_FIELD = re.compile(
    r"(?:Admitting Service|Admission Date|Hospital Day|Day of Life|LOC):\s*[^|]+\s*\|\s*"
)


def strip_note_table_blocks(text: str) -> str:
    """
    Remove common table-like / grid fragments from LUMIA note prose.

    Order: procedure ASCII table, embedded Recent Results, Lines/Drains block,
    ECG findings banner, then repeated pipe-separated header fields.
    """
    s = _RE_PROCEDURE_ASCII_TABLE.sub(" ", text)
    s = _RE_RECENT_RESULTS_BLOCK.sub(" ", s)
    s = _RE_LINES_DRAINS_TO_VITALS.sub(" Vital Signs:", s)
    s = _RE_ECG_FINDINGS_BANNER.sub(" ", s)
    prev: Optional[str] = None
    while prev != s:
        prev = s
        s = _RE_PIPE_METADATA_FIELD.sub(" ", s)
    return s


def _entry_timestamp_for_event(ev: etree._Element) -> Optional[datetime]:
    p = ev.getparent()
    while p is not None:
        if p.tag == "entry":
            return _parse_ts(p.get("timestamp"))
        p = p.getparent()
    return None


def _iter_note_events_chronological(root: etree._Element) -> List[etree._Element]:
    """Note events sorted by ancestor <entry> timestamp, then document order."""
    keyed: List[Tuple[datetime, int, etree._Element]] = []
    for doc_idx, el in enumerate(root.iter("event")):
        if el.get("type") != "note":
            continue
        ts = _entry_timestamp_for_event(el) or datetime.min
        keyed.append((ts, doc_idx, el))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [el for _ts, _i, el in keyed]


def mark_duplicate_token_indices_pmc9513649(
    tokens: List[str],
    n: int,
    prior_notes_ngrams: Set[Tuple[str, ...]],
) -> Set[int]:
    """
    PMC9513649 / eAppendix 1: step window by 1 token; a 10-gram is duplicate if it
    appeared in any prior note (prior_notes_ngrams) or earlier in this note.
    Tokens in any duplicate window are marked duplicate.
    """
    redundant: Set[int] = set()
    L = len(tokens)
    if L < n:
        return redundant
    seen_in_this_note: Set[Tuple[str, ...]] = set()
    for i in range(0, L - n + 1):
        gram = tuple(tokens[i : i + n])
        if gram in prior_notes_ngrams or gram in seen_in_this_note:
            for j in range(i, i + n):
                redundant.add(j)
        seen_in_this_note.add(gram)
    return redundant


def add_all_ngrams(tokens: List[str], n: int, target: Set[Tuple[str, ...]]) -> None:
    if len(tokens) < n:
        return
    for i in range(0, len(tokens) - n + 1):
        target.add(tuple(tokens[i : i + n]))


_RE_NOTE_WS_RUNS = re.compile(r"\s+")


def _collapse_note_whitespace(text: str) -> str:
    """Strip ends and replace each run of whitespace with a single ASCII space."""
    return _RE_NOTE_WS_RUNS.sub(" ", text.strip())


def compress_note_text_pmc9513649(
    text: str,
    n: int,
    prior_notes_ngrams: Set[Tuple[str, ...]],
    *,
    collapse_note_whitespace: bool = False,
) -> str:
    """
    Remove duplicate 10-gram tokens, then register all 10-grams from the
    *original* (pre-drop) token sequence into prior_notes_ngrams for later notes.
    """
    if not text.strip():
        return text
    work = _collapse_note_whitespace(text) if collapse_note_whitespace else text
    tokens = work.split()
    if len(tokens) < n:
        add_all_ngrams(tokens, n, prior_notes_ngrams)
        return work if collapse_note_whitespace else text
    redundant = mark_duplicate_token_indices_pmc9513649(tokens, n, prior_notes_ngrams)
    out_tokens = [tokens[i] for i in range(len(tokens)) if i not in redundant]
    add_all_ngrams(tokens, n, prior_notes_ngrams)
    return " ".join(out_tokens)


@dataclass
class Savings:
    """Token delta (positive = fewer tokens after compression)."""

    notes: int = 0
    labs: int = 0
    conditions: int = 0

    def total(self) -> int:
        return self.notes + self.labs + self.conditions


def apply_note_compression(
    root: etree._Element,
    *,
    strip_note_tables: bool = False,
    collapse_note_whitespace: bool = False,
) -> int:
    """Compress note events (PMC9513649 10-grams). Returns full-document token savings."""
    before = count_tokens(_xml_to_string(root))
    prior_notes_ngrams: Set[Tuple[str, ...]] = set()
    for el in _iter_note_events_chronological(root):
        original = _note_plain_text(el)
        if not original.strip():
            continue
        base = strip_note_table_blocks(original) if strip_note_tables else original
        compressed = compress_note_text_pmc9513649(
            base,
            n=10,
            prior_notes_ngrams=prior_notes_ngrams,
            collapse_note_whitespace=collapse_note_whitespace,
        )
        if compressed != original:
            el.clear()
            el.text = compressed
    after = count_tokens(_xml_to_string(root))
    return max(0, before - after)


def apply_lab_daily_compression(root: etree._Element) -> int:
    """
    Group measurement events by (code, calendar date), ordered by time within
    the day. Only when there are more than two rows in a bin (strictly > 2):
    keep first as baseline, replace last with min/max/median summary, remove
    others. Bins with one or two measurements are unchanged.
    """
    before = count_tokens(_xml_to_string(root))

    rows: List[Tuple[datetime, str, etree._Element, etree._Element]] = []
    for entry in root.iter("entry"):
        ts_raw = entry.get("timestamp")
        ts = _parse_ts(ts_raw)
        if ts is None:
            continue
        for ev in entry.findall("event"):
            if ev.get("type") != "measurement":
                continue
            code = ev.get("code") or ""
            rows.append((ts, code, ev, entry))

    by_key: Dict[
        Tuple[str, str], List[Tuple[datetime, etree._Element, etree._Element]]
    ] = {}
    for ts, code, ev, entry in rows:
        day = ts.date().isoformat()
        by_key.setdefault((code, day), []).append((ts, ev, entry))

    removed: List[etree._Element] = []
    for (_code, _day), items in by_key.items():
        items.sort(key=lambda x: x[0])
        if len(items) <= 2:
            continue
        first_ts, first_ev, _ = items[0]
        last_ts, last_ev, _ = items[-1]
        values: List[float] = []
        for _t, ev, _e in items:
            v = _parse_float((ev.text or "").strip() if ev.text else None)
            if v is not None:
                values.append(v)

        first_ev.set("enc", "lab_baseline")
        first_ev.set("d", first_ts.date().isoformat())

        for _t, ev, _ent in items[1:-1]:
            removed.append(ev)

        if values:
            lo, hi = min(values), max(values)
            med = statistics.median(values)
            last_ev.text = f"{lo}/{hi}/{med}/{len(values)}"
        else:
            last_ev.text = f"nonnum/{len(items)}"
        last_ev.set("enc", "lab_day_summary")
        last_ev.set("d", last_ts.date().isoformat())

    for ev in removed:
        parent = ev.getparent()
        if parent is not None:
            parent.remove(ev)

    after = count_tokens(_xml_to_string(root))
    return max(0, before - after)


def apply_condition_interval_collapse(root: etree._Element) -> int:
    """
    Remove all type=condition events; append <collapsed_conditions> with
    <condition code=... name=... first_seen=YYYY-MM last_seen=... evidence_count=/>.
    """
    before = count_tokens(_xml_to_string(root))

    events: List[Tuple[datetime, str, str, etree._Element]] = []
    for entry in root.iter("entry"):
        ts = _parse_ts(entry.get("timestamp"))
        if ts is None:
            continue
        for ev in entry.findall("event"):
            if ev.get("type") != "condition":
                continue
            code = ev.get("code") or ""
            name = ev.get("name") or ""
            if not _is_chronic_vocab_code(code):
                continue
            events.append((ts, code, name, ev))

    by_code: Dict[str, List[Tuple[datetime, str, etree._Element]]] = {}
    for ts, code, name, ev in events:
        by_code.setdefault(code, []).append((ts, name, ev))

    # Remove condition elements from tree
    for _ts, _code, _name, ev in events:
        parent = ev.getparent()
        if parent is not None:
            parent.remove(ev)

    collapsed = etree.Element("collapsed_conditions")
    for code, lst in sorted(by_code.items(), key=lambda x: x[0]):
        lst.sort(key=lambda x: x[0])
        times = [t for t, _n, _e in lst]
        name = lst[-1][1] or code
        first = times[0].strftime("%Y-%m")
        last = times[-1].strftime("%Y-%m")
        count = len(lst)
        etree.SubElement(
            collapsed,
            "condition",
            name=name,
            code=code,
            first_seen=first,
            last_seen=last,
            evidence_count=str(count),
        )

    if len(collapsed) > 0:
        root.append(collapsed)

    after = count_tokens(_xml_to_string(root))
    return max(0, before - after)


def _is_chronic_vocab_code(code: str) -> bool:
    c = code.upper()
    return (
        c.startswith("ICD9CM/")
        or c.startswith("ICD10CM/")
        or c.startswith("ICD10/")
        or c.startswith("SNOMED/")
    )


@dataclass
class BenchmarkCompressionOptions:
    """Optional transforms applied before note/lab/condition rules."""

    strip_note_tables: bool = False
    collapse_note_whitespace: bool = False
    slim_person_followup_encounters: bool = False
    drop_event_type_attr: bool = False


def clone_root(root: etree._Element) -> etree._Element:
    return etree.fromstring(etree.tostring(root, encoding="utf-8"))


def _clone_element(el: etree._Element) -> etree._Element:
    return etree.fromstring(etree.tostring(el, encoding="utf-8"))


def apply_person_slim_followup_encounters(root: etree._Element) -> None:
    """
    Keep the first <encounter>'s <person> subtree intact. For each later
    <encounter>, replace <person> with only <age> and <payerplan> (copied).
    """
    encounters = [c for c in root if c.tag == "encounter"]
    for enc in encounters[1:]:
        person = enc.find("person")
        if person is None:
            continue
        age_el = person.find("age")
        pay_el = person.find("payerplan")
        new_person = etree.Element("person")
        if age_el is not None:
            new_person.append(_clone_element(age_el))
        if pay_el is not None:
            new_person.append(_clone_element(pay_el))
        parent = person.getparent()
        if parent is not None:
            parent.replace(person, new_person)


def apply_drop_event_type_attr(root: etree._Element) -> None:
    """Remove type=... from all <event> elements (mutates tree)."""
    for ev in root.iter("event"):
        if "type" in ev.attrib:
            del ev.attrib["type"]


def clone_root_with_optional_person_slim(
    root: etree._Element, opts: BenchmarkCompressionOptions
) -> etree._Element:
    """Clone and optionally slim <person> on follow-up encounters (labs/notes still need type=)."""
    c = clone_root(root)
    if opts.slim_person_followup_encounters:
        apply_person_slim_followup_encounters(c)
    return c


def aggregate_pipeline_stages(
    per_file_stages: List[List[Tuple[str, int, int]]],
) -> List[Tuple[str, int, int]]:
    """Sum tokens and serialized string length at each stage (same labels in each row)."""
    if not per_file_stages:
        return []
    n = len(per_file_stages[0])
    out: List[Tuple[str, int, int]] = []
    for i in range(n):
        label = per_file_stages[0][i][0]
        tok = sum(row[i][1] for row in per_file_stages)
        size = sum(row[i][2] for row in per_file_stages)
        out.append((label, tok, size))
    return out


def print_composition_table(
    title: str, stages: List[Tuple[str, int, int]], *, sink: TextIO = sys.stdout
) -> None:
    """Print stacked pipeline sizes; last row is the smallest full-stack representation."""
    print(title, file=sink)
    t0, s0 = stages[0][1], stages[0][2]
    print(
        f"  {'stage':<36} {'tokens':>12} {'str_len':>12} "
        f"{'%tokens':>10} {'%str_len':>10}",
        file=sink,
    )
    print(f"  {'-'*36} {'-'*12} {'-'*12} {'-'*10} {'-'*10}", file=sink)
    for lab, tok, sz in stages:
        pt = 100.0 * tok / t0 if t0 else 0.0
        ps = 100.0 * sz / s0 if s0 else 0.0
        print(f"  {lab:<36} {tok:>12,} {sz:>12,} {pt:>9.1f}% {ps:>9.1f}%", file=sink)
    ft, fs = stages[-1][1], stages[-1][2]
    print(f"  {'→ smallest (full stack)':<36} {ft:>12,} {fs:>12,}", file=sink)
    rt = 100.0 * (t0 - ft) / t0 if t0 else 0.0
    rs = 100.0 * (s0 - fs) / s0 if s0 else 0.0
    print(
        f"  {'total reduction vs original':<36} {t0 - ft:>12,} {s0 - fs:>12,} "
        f"{rt:>9.1f}% {rs:>9.1f}%",
        file=sink,
    )


# --- Final emit: compressed tree as XML or tagless rows ---


def _kv_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value).replace("\t", " ").strip()


def _flatten_person_tab_kv(person: etree._Element) -> str:
    """Serialize <person> children as tab-separated key=value (no XML tags)."""
    cells: List[str] = []
    for child in person:
        tag = child.tag
        if len(child) == 0:
            t = (child.text or "").strip()
            if t:
                cells.append(f"{tag}={_kv_cell(t)}")
        else:
            for gc in child:
                sub = (gc.text or "").strip()
                if sub:
                    cells.append(f"{tag}.{gc.tag}={_kv_cell(sub)}")
    return "\t".join(cells)


def _is_visit_row_event(ev: etree._Element) -> bool:
    t = (ev.get("type") or "").lower()
    return t in ("visit", "visit_detail")


def _event_row_tab_kv(ev: etree._Element) -> str:
    parts: List[str] = []
    for key in ("type", "visit_id", "unit", "code", "name", "note_id"):
        v = ev.get(key)
        if v:
            parts.append(f"{key}={_kv_cell(v)}")
    body = _note_plain_text(ev).strip()
    if body:
        parts.append(f"text={_kv_cell(body)}")
    return "\t".join(parts)


def emit_minimal_rows_from_root(root: etree._Element, out: TextIO) -> None:
    """
    Emit one logical row per line from a (possibly compressed) LUMIA tree.

    Prefixes: [H] subject, [R] encounter, [P] person key-values, [T] entry
    timestamp, [V] visit / visit_detail, [E] other events, [C] collapsed conditions.
    """
    pid = root.get("person_id") or ""
    out.write(f"[H]\tperson_id={_kv_cell(pid)}\n")
    encounters = [c for c in root if c.tag == "encounter"]
    for ei, enc in enumerate(encounters):
        out.write(f"[R]\tencounter={ei}\n")
        person = enc.find("person")
        if person is not None and len(person) > 0:
            pk = _flatten_person_tab_kv(person)
            if pk:
                out.write(f"[P]\t{pk}\n")
        events_el = enc.find("events")
        if events_el is None:
            continue
        for entry in events_el.findall("entry"):
            ts = entry.get("timestamp") or ""
            out.write(f"[T]\t{_kv_cell(ts)}\n")
            for ev in entry.findall("event"):
                row = _event_row_tab_kv(ev)
                prefix = "[V]" if _is_visit_row_event(ev) else "[E]"
                out.write(f"{prefix}\t{row}\n")

    cc = root.find("collapsed_conditions")
    if cc is not None:
        for cond in cc.findall("condition"):
            parts: List[str] = []
            for key in ("code", "name", "first_seen", "last_seen", "evidence_count"):
                v = cond.get(key)
                if v:
                    parts.append(f"{key}={_kv_cell(v)}")
            if parts:
                out.write("[C]\t" + "\t".join(parts) + "\n")


def _post_pipeline_xml_and_rows_utf8_bytes(root: etree._Element) -> Tuple[int, int]:
    """UTF-8 byte length of compressed tree as XML, then as minimal rows."""
    xml_text = _xml_to_string(root)
    row_buf = io.StringIO()
    emit_minimal_rows_from_root(root, row_buf)
    rows_text = row_buf.getvalue()
    return len(xml_text.encode("utf-8")), len(rows_text.encode("utf-8"))


def emit_compressed_xml_to_stream(root: etree._Element, out: TextIO) -> None:
    """Write the in-memory tree as one UTF-8 XML document."""
    out.write(etree.tostring(root, encoding="unicode"))
    out.write("\n")


class _TeeText:
    """Duplicate writes to two text streams (for capturing the human report)."""

    __slots__ = ("_a", "_b")

    def __init__(self, a: TextIO, b: TextIO) -> None:
        self._a = a
        self._b = b

    def write(self, s: str) -> int:
        self._a.write(s)
        self._b.write(s)
        return len(s)

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()


def format_compression_options_summary(opts: BenchmarkCompressionOptions) -> str:
    return (
        "Compression options (this run)\n"
        f"  strip_note_tables:                {opts.strip_note_tables}\n"
        f"  collapse_note_whitespace:         {opts.collapse_note_whitespace}\n"
        f"  slim_person_followup_encounters:  {opts.slim_person_followup_encounters}\n"
        f"  drop_event_type_attr:             {opts.drop_event_type_attr}\n"
        "Always applied: PMC 10-gram notes → lab-by-day (≥3 same-code draws/day) "
        "→ chronic ICD9/10/SNOMED condition collapse.\n"
    )


def run_lumia_compression_pipeline(
    root: etree._Element,
    opts: BenchmarkCompressionOptions,
    *,
    stages: Optional[List[Tuple[str, int, int]]] = None,
) -> None:
    """
    Apply the full benchmark transform sequence in place.

    If ``stages`` is a list, append (label, token_count, str_len) after each step.
    """

    def snap(label: str) -> None:
        if stages is not None:
            s = _xml_to_string(root)
            stages.append((label, count_tokens(s), len(s)))

    snap("0_original")
    if opts.slim_person_followup_encounters:
        apply_person_slim_followup_encounters(root)
    snap("1_after_person_followup_slim")
    apply_note_compression(
        root,
        strip_note_tables=opts.strip_note_tables,
        collapse_note_whitespace=opts.collapse_note_whitespace,
    )
    snap("2_after_notes_10gram")
    apply_lab_daily_compression(root)
    snap("3_after_labs_daily")
    apply_condition_interval_collapse(root)
    snap("4_after_conditions_collapsed")
    if opts.drop_event_type_attr:
        apply_drop_event_type_attr(root)
    snap("5_after_drop_event_type")


def process_one_file(path: str, opts: BenchmarkCompressionOptions) -> Savings:
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    tree = etree.parse(path, parser)
    root = tree.getroot()

    s = Savings()
    s.notes = apply_note_compression(
        clone_root_with_optional_person_slim(root, opts),
        strip_note_tables=opts.strip_note_tables,
        collapse_note_whitespace=opts.collapse_note_whitespace,
    )
    s.labs = apply_lab_daily_compression(
        clone_root_with_optional_person_slim(root, opts)
    )
    s.conditions = apply_condition_interval_collapse(
        clone_root_with_optional_person_slim(root, opts)
    )

    return s


def process_pipeline_all_rules(
    path: str, opts: BenchmarkCompressionOptions
) -> Tuple[int, int, List[Tuple[str, int, int]], int, int, etree._Element]:
    """
    Apply optional person slim, then notes → labs → conditions, then optional
    drop of event type= (type is required for earlier steps).

    Returns:
        before_tok, after_tok, stages, before_chars, after_chars, root
        where stages is (label, tokens, len(serialized_xml)) at each checkpoint,
        and root is the mutated tree (for --emit-final).
    """
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    tree = etree.parse(path, parser)
    root = tree.getroot()
    stages: List[Tuple[str, int, int]] = []
    run_lumia_compression_pipeline(root, opts, stages=stages)

    before_tok, before_chars = stages[0][1], stages[0][2]
    after_tok, after_chars = stages[-1][1], stages[-1][2]
    return before_tok, after_tok, stages, before_chars, after_chars, root


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/inspect_lumia_xml",
        help="Directory of LUMIA *.xml files or a single file",
    )
    p.add_argument(
        "--glob",
        dest="glob_pat",
        default="*.xml",
        help="Glob under directory (default: *.xml)",
    )
    p.add_argument(
        "--reference-ehrs",
        default="",
        help="Optional path (e.g. data/medalign_instructions_v1_3/ehrs) for future alignment; logged if present",
    )
    p.add_argument(
        "--show-stages",
        action="store_true",
        help="Print token count after each pipeline stage (per file)",
    )
    p.add_argument(
        "--tsv",
        action="store_true",
        help="Emit dense tab-separated table (default is human-readable %% summary)",
    )
    p.add_argument(
        "--composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print stacked pipeline size after each technique (default: on; use --no-composition to hide)",
    )
    p.add_argument(
        "--strip-note-tables",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strip table-like blocks from note text before PMC 10-gram compression (default: off)",
    )
    p.add_argument(
        "--collapse-note-whitespace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize note whitespace to single spaces before split/10-gram (default: off)",
    )
    p.add_argument(
        "--slim-person-followup-encounters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep full <person> on first <encounter> only; later encounters keep <age> and <payerplan> only",
    )
    p.add_argument(
        "--drop-event-type-attr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After notes/labs/conditions, remove type=... from each <event> (default: off)",
    )
    p.add_argument(
        "--emit-final",
        choices=("xml", "rows"),
        default=None,
        help="After metrics, also write compressed XML or tagless rows (--emit-final-out or stdout)",
    )
    p.add_argument(
        "--emit-final-out",
        default="",
        metavar="PATH",
        help="Destination for --emit-final (default: stdout)",
    )
    p.add_argument(
        "--save-compression-artifacts",
        default="",
        metavar="PREFIX",
        help="Write PREFIX_breakdown.txt and PREFIX[_STEM]_final.xml/_final.rows; omit --tsv (human report only)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    opts = BenchmarkCompressionOptions(
        strip_note_tables=args.strip_note_tables,
        collapse_note_whitespace=args.collapse_note_whitespace,
        slim_person_followup_encounters=args.slim_person_followup_encounters,
        drop_event_type_attr=args.drop_event_type_attr,
    )
    artifact_prefix = (args.save_compression_artifacts or "").strip()

    if args.reference_ehrs and os.path.isdir(args.reference_ehrs):
        print(f"Reference EHRs dir present: {args.reference_ehrs}", file=sys.stderr)

    paths: List[str]
    if os.path.isfile(args.input):
        paths = [args.input]
    elif os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, args.glob_pat)))
    else:
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    if not paths:
        print("No XML files matched.", file=sys.stderr)
        return 1

    if artifact_prefix and args.tsv:
        print(
            "lumia_compression_benchmark: omit --tsv when using "
            "--save-compression-artifacts (breakdown is human-readable).",
            file=sys.stderr,
        )
        return 1

    if args.emit_final and not args.emit_final_out and args.tsv:
        print(
            "lumia_compression_benchmark: use --emit-final-out PATH when combining "
            "--emit-final with --tsv (stdout is reserved for the TSV table).",
            file=sys.stderr,
        )
        return 1

    emit_fp: Optional[TextIO] = None
    emit_close = False
    if args.emit_final:
        if args.emit_final_out:
            emit_fp = open(args.emit_final_out, "w", encoding="utf-8")
            emit_close = True
        else:
            emit_fp = sys.stdout

    def rel_path(path: str) -> str:
        return (
            os.path.relpath(path, start=os.getcwd())
            if path.startswith(os.getcwd())
            else path
        )

    def pct_reduction(before: int, after: int) -> float:
        if before <= 0:
            return 0.0
        return 100.0 * (before - after) / before

    def pct_remaining(before: int, after: int) -> float:
        if before <= 0:
            return 0.0
        return 100.0 * after / before

    agg = Savings()
    combined_before = 0
    combined_after = 0
    combined_chars_before = 0
    combined_chars_after = 0
    combined_raw_utf8 = 0
    combined_post_xml_utf8 = 0
    combined_post_rows_utf8 = 0

    per_file_rows: List[Tuple[str, Savings, int, int, int, int, int, int, int]] = []
    all_stage_rows: List[List[Tuple[str, int, int]]] = []
    compressed_roots: List[etree._Element] = []

    for path in paths:
        s = process_one_file(path, opts)
        agg.notes += s.notes
        agg.labs += s.labs
        agg.conditions += s.conditions
        b, a, stages, cb, ca, root = process_pipeline_all_rules(path, opts)
        raw_utf8 = os.path.getsize(path)
        post_xml_utf8, post_rows_utf8 = _post_pipeline_xml_and_rows_utf8_bytes(root)
        all_stage_rows.append(stages)
        combined_before += b
        combined_after += a
        combined_chars_before += cb
        combined_chars_after += ca
        combined_raw_utf8 += raw_utf8
        combined_post_xml_utf8 += post_xml_utf8
        combined_post_rows_utf8 += post_rows_utf8
        per_file_rows.append(
            (rel_path(path), s, b, a, cb, ca, raw_utf8, post_xml_utf8, post_rows_utf8)
        )
        compressed_roots.append(root)
        if args.show_stages:
            for label, tok, ch in stages:
                print(
                    f"  {rel_path(path)}\t{label}\t{tok}\t{ch}",
                    file=sys.stderr,
                )

    stats_stream: TextIO = (
        sys.stderr
        if (args.emit_final and not args.emit_final_out and not args.tsv)
        else sys.stdout
    )

    report_buf: Optional[io.StringIO] = None
    if artifact_prefix:
        report_buf = io.StringIO()
        report_sink: TextIO = _TeeText(stats_stream, report_buf)
    else:
        report_sink = stats_stream

    if args.tsv:
        print(
            "file\tnotesΔtok\tlabsΔtok\tcondΔtok\trule_sumΔtok\t"
            "pipeline_before_tok\tpipeline_after_tok\tpipelineΔtok\t"
            "pipeline_chars_before\tpipeline_chars_after\tpipelineΔchars\t"
            "raw_xml_utf8_bytes\tpost_pipeline_xml_utf8_bytes\tpost_pipeline_rows_utf8_bytes"
        )
        for rel, s, b, a, cb, ca, raw_u8, xml_u8, rows_u8 in per_file_rows:
            print(
                f"{rel}\t{s.notes}\t{s.labs}\t{s.conditions}\t{s.total()}\t"
                f"{b}\t{a}\t{b - a}\t{cb}\t{ca}\t{cb - ca}\t"
                f"{raw_u8}\t{xml_u8}\t{rows_u8}"
            )
        print(
            f"TOTAL\t{agg.notes}\t{agg.labs}\t{agg.conditions}\t{agg.total()}\t"
            f"{combined_before}\t{combined_after}\t{combined_before - combined_after}\t"
            f"{combined_chars_before}\t{combined_chars_after}\t{combined_chars_before - combined_chars_after}\t"
            f"{combined_raw_utf8}\t{combined_post_xml_utf8}\t{combined_post_rows_utf8}"
        )
    else:
        print(
            "Per file (pipeline: notes → labs → conditions on one tree)\n",
            file=report_sink,
        )
        for rel, _s, b, a, cb, ca, raw_u8, xml_u8, rows_u8 in per_file_rows:
            tr, tw = pct_reduction(b, a), pct_remaining(b, a)
            cr, cw = pct_reduction(cb, ca), pct_remaining(cb, ca)
            xml_vs_raw = pct_reduction(raw_u8, xml_u8)
            rows_vs_raw = pct_reduction(raw_u8, rows_u8)
            rows_vs_xml = pct_reduction(xml_u8, rows_u8) if xml_u8 else 0.0
            print(
                f"  {rel}\n"
                f"      tokens:  {tr:5.1f}% smaller  →  {tw:5.1f}% of original count remains  "
                f"(saved {b - a:,} of {b:,})\n"
                f"      chars:   {cr:5.1f}% smaller  →  {cw:5.1f}% of original size remains  "
                f"(saved {cb - ca:,} of {cb:,})\n"
                f"      FINAL bytes (UTF-8): raw on disk {raw_u8:,}  →  compressed XML {xml_u8:,}  "
                f"({xml_vs_raw:.1f}% smaller than raw)\n"
                f"                           →  minimal rows {rows_u8:,}  "
                f"({rows_vs_raw:.1f}% smaller than raw; {rows_vs_xml:.1f}% smaller than compressed XML)",
                file=report_sink,
            )
            print(file=report_sink)

        tr_o = pct_reduction(combined_before, combined_after)
        tw_o = pct_remaining(combined_before, combined_after)
        cr_o = pct_reduction(combined_chars_before, combined_chars_after)
        cw_o = pct_remaining(combined_chars_before, combined_chars_after)
        print("─" * 58, file=report_sink)
        print(
            f"OVERALL ({len(paths)} file(s), concatenated totals)\n\n"
            f"  Tokens\n"
            f"    Reduction:     {tr_o:.1f}%  ({combined_before - combined_after:,} fewer than original)\n"
            f"    Still there:   {tw_o:.1f}%  of original token count (final / before)\n\n"
            f"  Serialized tree length (Python char count of in-memory XML)\n"
            f"    Reduction:     {cr_o:.1f}%  ({combined_chars_before - combined_chars_after:,} fewer characters)\n"
            f"    Still there:   {cw_o:.1f}%  of original (tokenizer / stacked table baseline)\n\n"
            f"  FINAL UTF-8 DOCUMENT BYTES — baseline: on-disk raw XML per file\n"
            f"    Step 1 — pipeline, serialize as XML:\n"
            f"      Raw on disk:              {combined_raw_utf8:,} bytes\n"
            f"      Compressed XML (UTF-8):   {combined_post_xml_utf8:,} bytes  →  "
            f"{pct_reduction(combined_raw_utf8, combined_post_xml_utf8):.1f}% smaller than raw\n"
            f"    Step 2 — same tree as minimal rows:\n"
            f"      Minimal rows (UTF-8):     {combined_post_rows_utf8:,} bytes  →  "
            f"{pct_reduction(combined_raw_utf8, combined_post_rows_utf8):.1f}% smaller than raw\n"
            f"      (rows vs compressed XML alone: "
            f"{pct_reduction(combined_post_xml_utf8, combined_post_rows_utf8):.1f}% smaller)\n",
            file=report_sink,
        )
        print("─" * 58, file=report_sink)

        if args.composition:
            agg_st = aggregate_pipeline_stages(all_stage_rows)
            print_composition_table(
                "\nSTACKED COMPOSITION — all techniques applied in order\n"
                "  (values summed over all input files; last row is the smallest total representation)\n"
                "  Order: original → optional person slim (follow-up encounters) → PMC notes → "
                "lab-by-day rollup → condition collapse → optional drop event type\n",
                agg_st,
                sink=report_sink,
            )
            print(file=report_sink)

    if args.emit_final and emit_fp is not None:
        for er in compressed_roots:
            if args.emit_final == "xml":
                emit_compressed_xml_to_stream(er, emit_fp)
            else:
                emit_minimal_rows_from_root(er, emit_fp)

    if emit_close and emit_fp is not None:
        emit_fp.close()

    print(f"Tokenizer: {TOKENIZER_LABEL}", file=report_sink)
    if not args.tsv:
        msg = "Use --tsv for the compact tab-separated table."
        if args.strip_note_tables:
            msg += " Note table stripping is on (--strip-note-tables)."
        if args.collapse_note_whitespace:
            msg += " Note whitespace collapse is on (--collapse-note-whitespace)."
        if args.slim_person_followup_encounters:
            msg += " Person slim on follow-up encounters is on."
        if args.drop_event_type_attr:
            msg += " Event type attribute stripping is on (--drop-event-type-attr)."
        if args.composition:
            msg += " Use --no-composition to hide the stacked stage table."
        if args.emit_final and not args.emit_final_out and not args.tsv:
            msg += (
                " Compression stats above were written to stderr because "
                "--emit-final payload goes to stdout."
            )
        if artifact_prefix:
            msg += f" Wrote {artifact_prefix}_breakdown.txt plus per-file *_final.xml and *_final.rows."
        print(msg, file=report_sink)

    if artifact_prefix and report_buf is not None:
        parent = os.path.dirname(artifact_prefix)
        if parent:
            os.makedirs(parent, exist_ok=True)
        breakdown_path = f"{artifact_prefix}_breakdown.txt"
        with open(breakdown_path, "w", encoding="utf-8") as bf:
            bf.write(format_compression_options_summary(opts))
            bf.write("\n")
            bf.write(report_buf.getvalue())
        size_chunks: List[str] = []
        for path, root, prow in zip(paths, compressed_roots, per_file_rows):
            rel, _s, _b, _a, _cb, _ca, raw_u8, xml_u8, rows_u8 = prow
            stem = os.path.splitext(os.path.basename(path))[0]
            base = artifact_prefix if len(paths) == 1 else f"{artifact_prefix}_{stem}"
            xml_text = _xml_to_string(root)
            row_buf = io.StringIO()
            emit_minimal_rows_from_root(root, row_buf)
            rows_text = row_buf.getvalue()
            with open(base + "_final.xml", "w", encoding="utf-8") as xf:
                xf.write(xml_text)
            with open(base + "_final.rows", "w", encoding="utf-8") as rf:
                rf.write(rows_text)
            xml_vs_raw = pct_reduction(raw_u8, xml_u8)
            rows_vs_raw = pct_reduction(raw_u8, rows_u8)
            rows_vs_xml = pct_reduction(xml_u8, rows_u8) if xml_u8 else 0.0
            size_chunks.append(
                f"  {rel}\n"
                f"      raw on disk (UTF-8):        {raw_u8:,}\n"
                f"      compressed XML (UTF-8):     {xml_u8:,}  ({xml_vs_raw:.1f}% smaller than raw)\n"
                f"      minimal rows (UTF-8):       {rows_u8:,}  ({rows_vs_raw:.1f}% smaller than raw; "
                f"{rows_vs_xml:.1f}% smaller than compressed XML)\n"
            )
        overall_lines: List[str] = []
        if len(paths) > 1:
            ox = pct_reduction(combined_raw_utf8, combined_post_xml_utf8)
            orow = pct_reduction(combined_raw_utf8, combined_post_rows_utf8)
            oenc = pct_reduction(combined_post_xml_utf8, combined_post_rows_utf8)
            overall_lines.append(
                "  OVERALL (concatenated)\n"
                f"      raw on disk (UTF-8):        {combined_raw_utf8:,}\n"
                f"      compressed XML (UTF-8):     {combined_post_xml_utf8:,}  ({ox:.1f}% smaller than raw)\n"
                f"      minimal rows (UTF-8):       {combined_post_rows_utf8:,}  ({orow:.1f}% smaller than raw; "
                f"{oenc:.1f}% smaller than compressed XML)\n"
            )
        with open(breakdown_path, "a", encoding="utf-8") as bf:
            bf.write(
                "\n--- Per-file FINAL UTF-8 bytes (vs on-disk raw; same as console OVERALL) ---\n"
            )
            bf.write("\n".join(size_chunks + overall_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
