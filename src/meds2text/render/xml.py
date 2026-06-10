"""LUMIA XML rendering: build the canonical event-stream tree from a subject.

The XML tree produced here is the canonical intermediate representation; the
JSON and FHIR-like renderers derive their output from it.
"""

from __future__ import annotations

import fnmatch
import re
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set

from lxml.etree import Element, SubElement, tostring

from meds2text.config import VALID_PERSON_FIELDS, TextifyConfig
from meds2text.encounters import Encounter, bin_by_time
from meds2text.metadata import (
    MissTracker,
    calculate_age,
    get_payer_plan_coverage,
    get_person_values,
)
from meds2text.render.note_compress import compress_subject_notes
from meds2text.subject import Event, Subject

BASE_EXCLUDED_PROPS = {"time", "end", "subject_id"}
DEFAULT_ATTRIBUTE_ORDER = ("table", "code", "name")

# Tables whose events are kept verbatim (never summarized or de-duplicated) when
# collapsing: their content (free text / per-event imaging name) is not a
# function of the code, so collapsing would lose information.
COLLAPSE_VERBATIM_TABLES = {"note", "image"}

# High-frequency tag/attribute renames applied by `minify_tags`. Structural
# wrappers (eventstream, legend, code, person, ...) are left intact.
MINIFY_TAG_MAP = {"encounter": "enc", "events": "es", "entry": "g", "event": "e"}
MINIFY_ATTR_MAP = {"timestamp": "t", "unit": "u"}

# STARR tobacco-screening codes that repeat together; collapse to one pseudo event.
SMOKING_HISTORY_PSEUDO_CODE = "STANFORD_OBS/SmokingHistory"
SMOKING_HISTORY_LEGEND_NAME = "Smoking history (screening panel)"
SMOKING_HISTORY_CODES = frozenset(
    {
        "LOINC/72166-2",
        "SNOMED/110483000",
        "SNOMED/228490006",
        "SNOMED/228510007",
        "SNOMED/230056004",
        "SNOMED/230057008",
        "SNOMED/230058003",
        "SNOMED/713914004",
    }
)
SMOKING_CODE_LABELS = {
    "LOINC/72166-2": "smoking",
    "SNOMED/110483000": "tobacco_user",
    "SNOMED/228490006": "snuff",
    "SNOMED/228510007": "chew",
    "SNOMED/230056004": "tobacco",
    "SNOMED/230057008": "chew_status",
    "SNOMED/230058003": "pipe",
    "SNOMED/713914004": "smokeless",
}


def sanitize_xml_text(text: Any) -> str:
    """Strip characters that are illegal in XML and fix Unicode issues."""
    if not isinstance(text, str):
        text = str(text)
    # Control chars except tab/newline/carriage-return.
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)
    # Invalid Unicode surrogate pairs.
    text = re.sub(r"[\uD800-\uDFFF]", "", text)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="xmlcharrefreplace").decode("utf-8")
    return text


def datetime_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def _resolve_excluded_props(config: TextifyConfig) -> Set[str]:
    excluded = set(BASE_EXCLUDED_PROPS)
    excluded.update(config.exclude_props)
    if "providers" not in config.include_contexts:
        excluded.add("provider_id")
    if "care_sites" not in config.include_contexts:
        excluded.add("care_site_id")
        excluded.add("care_site_name")
    return excluded


def event_to_xml(
    event: Event,
    ontology: Any,
    excluded_props: Optional[Set[str]] = None,
    attribute_order: Optional[Sequence[str]] = None,
    *,
    emit_name: bool = True,
) -> Element:
    """Render a single event as an ``<event>`` element.

    ``excluded_props`` only filters which attributes appear in the output. When
    ``emit_name`` is False, the per-event ``name`` attribute is omitted (the
    code -> name mapping is expected to live in a document-level ``<legend>``);
    image events still receive a name later via :func:`fix_image_events`.
    """
    excluded_props = excluded_props or set()
    attribute_order = attribute_order or DEFAULT_ATTRIBUTE_ORDER

    name = ontology.get_description(event.code) or ""
    attributes = {key: value for key, value in event if key not in excluded_props}
    value = attributes.get("numeric_value") or attributes.get("text_value") or None

    if emit_name:
        attributes.setdefault("name", name)
    attributes.pop("numeric_value", None)
    attributes.pop("text_value", None)
    unit = _event_unit(event)
    if unit and getattr(event, "table", None) == "measurement":
        attributes["unit"] = unit

    # Omit null / empty / "_" sentinel values — never emit placeholder attrs.
    xml_attributes = {
        key: sanitize_xml_text(str(attr_value))
        for key, attr_value in attributes.items()
        if attr_value not in (None, "", "_")
        and not (key == "name" and not str(attr_value).strip())
    }

    event_element = Element("event")
    for key in attribute_order:
        if key in xml_attributes:
            event_element.set(key, xml_attributes[key])
    for key, attr_val in xml_attributes.items():
        if key not in attribute_order:
            event_element.set(key, attr_val)

    if value is None:
        event_element.text = None
    elif isinstance(value, float) and value.is_integer():
        event_element.text = str(int(value))
    elif isinstance(value, (float, int)):
        event_element.text = f"{float(value):.2f}"
    else:
        event_element.text = sanitize_xml_text(str(value))

    # Some text values are themselves concept codes; resolve to a description.
    if value is not None:
        code_value_description = ontology.get_description(str(value))
        if code_value_description:
            event_element.text = code_value_description

    return event_element


def entry_to_xml(timestamp: datetime, events: Sequence[Element]) -> Element:
    entry_elem = Element("entry", timestamp=datetime_to_str(timestamp))
    for event in events:
        entry_elem.append(event)
    return entry_elem


def person_to_xml(person: Dict[str, Any], fields: Optional[Set[str]] = None) -> Element:
    """Render the ``<person>`` block, emitting only the requested sub-blocks.

    ``fields`` is a subset of ``birthdate``, ``age``, ``demographics``,
    ``payerplan``; ``None`` emits all of them.
    """
    if fields is None:
        fields = set(VALID_PERSON_FIELDS)
    person_elem = Element("person")

    if "birthdate" in fields and "birth" in person:
        birthdate_elem = SubElement(person_elem, "birthdate")
        birthdate_elem.text = person["birth"].strftime("%Y-%m-%d")

    if "age" in fields and ("age_in_days" in person or "age_in_years" in person):
        age_elem = SubElement(person_elem, "age")
        if "age_in_days" in person:
            SubElement(age_elem, "days").text = str(person["age_in_days"])
        if "age_in_years" in person:
            SubElement(age_elem, "years").text = str(person["age_in_years"])

    demographics = [
        (tag, person[tag]) for tag in ("ethnicity", "gender", "race") if tag in person
    ]

    def _description(data: Any) -> Optional[str]:
        return data.get("description") if isinstance(data, dict) else (data or None)

    if "demographics" in fields and any(_description(data) for _, data in demographics):
        demographics_elem = SubElement(person_elem, "demographics")
        for tag, data in demographics:
            description = _description(data)
            if not description:
                continue
            elem = SubElement(demographics_elem, tag)
            if isinstance(data, dict):
                elem.set("code", str(data.get("code", "")))
            elem.text = str(description).strip()

    if "payerplan" in fields and "payer_plan" in person:
        payerplan_elem = SubElement(person_elem, "payerplan")
        payerplan_elem.text = (
            str(person["payer_plan"]).strip() if person["payer_plan"] else ""
        )

    return person_elem


def dict_to_xml(data: List[Dict[str, Any]], parent_tag: str, child_tag: str) -> Element:
    parent_elem = Element(parent_tag)
    for item in data:
        child_elem = SubElement(parent_elem, child_tag)
        for key, value in item.items():
            child_elem.set(key, str(value) if value is not None else "_")
    return parent_elem


def _is_usable_care_site_name(name: Optional[str]) -> bool:
    """True when a care-site label is worth emitting (filters DONOTUSE / empty)."""
    if not name or name == "_":
        return False
    if "DONOTUSE" in name.upper():
        return False
    return bool(name.replace("|", "").strip())


def care_sites_to_xml(care_site_names: Sequence[str]) -> Element:
    """Compact ``<caresites>``: name only, de-duplicated, bad sites filtered."""
    parent_elem = Element("caresites")
    seen: Set[str] = set()
    for raw in sorted(care_site_names):
        name = raw.strip()
        if not _is_usable_care_site_name(name) or name in seen:
            continue
        seen.add(name)
        SubElement(parent_elem, "caresite", name=sanitize_xml_text(name))
    return parent_elem


def providers_to_xml(providers: Sequence[Dict[str, Any]]) -> Optional[Element]:
    """Compact ``<providers>``: non-empty speciality only (no ids or sentinels)."""
    children: List[Element] = []
    for provider in providers:
        speciality = provider.get("speciality")
        if not speciality or speciality in ("_", ""):
            continue
        child = Element("provider", speciality=sanitize_xml_text(str(speciality)))
        children.append(child)
    if not children:
        return None
    parent_elem = Element("providers")
    for child in children:
        parent_elem.append(child)
    return parent_elem


def _event_unit(event: Event) -> Optional[str]:
    """Return the measurement unit string when present on the MEDS row."""
    raw = getattr(event, "unit_source_value", None)
    if raw is None or raw in ("_", ""):
        return None
    return str(raw)


def _event_value_text(event: Event) -> Optional[str]:
    """Best-effort text/numeric value for an event."""
    text = getattr(event, "text_value", None)
    if text is not None and str(text).strip():
        return str(text)
    num = getattr(event, "numeric_value", None)
    if num is not None:
        return _format_number(float(num))
    return None


def collapse_smoking_events(
    events: Sequence[Event],
    *,
    legend: Optional[Dict[str, str]] = None,
) -> List[Event]:
    """Replace a cluster of tobacco-screening codes with one pseudo event.

    No-op when fewer than two smoking codes appear in ``events``.
    """
    smoking = [e for e in events if str(e.code) in SMOKING_HISTORY_CODES]
    if len(smoking) < 2:
        return list(events)
    rest = [e for e in events if str(e.code) not in SMOKING_HISTORY_CODES]
    parts: List[str] = []
    for event in smoking:
        label = SMOKING_CODE_LABELS.get(str(event.code), str(event.code).split("/")[-1])
        value = _event_value_text(event)
        if not value:
            continue
        parts.append(f"{label}={sanitize_xml_text(value)}")
    if not parts:
        return list(events)
    pseudo = Event(
        smoking[0].time,
        SMOKING_HISTORY_PSEUDO_CODE,
        {"table": "observation", "text_value": "; ".join(parts)},
    )
    if legend is not None:
        legend.setdefault(SMOKING_HISTORY_PSEUDO_CODE, SMOKING_HISTORY_LEGEND_NAME)
    return rest + [pseudo]


def fix_image_events(xml_root: Element) -> Element:
    """Set a readable ``name`` on image events from anatomic site + modality."""
    for event in xml_root.xpath(".//event[@table='image']"):
        anatomic_site = event.get("anatomic_site_source_value", "")
        modality = event.get("modality_source_value", "")
        event.set("name", f"{anatomic_site} {modality}".strip())
    return xml_root


def remove_null_elements(xml_root: Element) -> Element:
    """Drop elements whose every attribute is the ``"_"`` null sentinel."""
    to_remove = [
        elem
        for elem in xml_root.iter()
        if elem.attrib and all(value == "_" for value in elem.attrib.values())
    ]
    for elem in reversed(to_remove):
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)
    return xml_root


def build_legend_element(
    code_to_name: Dict[str, str],
    code_to_unit: Optional[Dict[str, str]] = None,
) -> Element:
    """Build a ``<legend>`` of ``<code id=.. name=.. [unit=..]>`` entries.

    The legend is the document-level home for code -> name strings so the
    (otherwise repeated) per-event ``name`` attribute can be dropped losslessly.
    Optional ``unit`` carries the canonical UCUM/source unit per measurement code.
    """
    code_to_unit = code_to_unit or {}
    legend_elem = Element("legend")
    for code in sorted(code_to_name):
        name = code_to_name[code]
        if not name:
            continue
        attrs = {"id": code, "name": sanitize_xml_text(name)}
        unit = code_to_unit.get(code)
        if unit:
            attrs["unit"] = sanitize_xml_text(unit)
        SubElement(legend_elem, "code", **attrs)
    return legend_elem


def strip_table_attr(xml_root: Element) -> Element:
    """Remove the ``table`` attribute from every ``<event>``.

    Run only after :func:`fix_image_events`, which selects image events via the
    ``table`` attribute.
    """
    for event in xml_root.xpath(".//event[@table]"):
        del event.attrib["table"]
    return xml_root


def _encounter_context_elements(
    encounter: Encounter,
    *,
    subject_id: int,
    person_values: Dict[str, Any],
    person_fields: Set[str],
    metadata: Dict[str, Any],
    config: TextifyConfig,
    miss: MissTracker,
) -> List[Element]:
    """Build the optional person/providers/care_sites elements for an encounter."""
    providers: Set[str] = set()
    care_sites: Set[str] = set()
    for event in encounter.events:
        provider_id = getattr(event, "provider_id", None)
        if provider_id is not None:
            provider_id_str = str(provider_id)
            providers.add(provider_id_str)
            miss.see_provider(provider_id_str, provider_id_str in metadata["provider"])
        care_site_id = getattr(event, "care_site_id", None)
        if care_site_id is not None:
            care_site_id_str = str(care_site_id)
            care_sites.add(care_site_id_str)
            miss.see_care_site(
                care_site_id_str, care_site_id_str in metadata["care_site"]
            )

    payer_plan = (
        get_payer_plan_coverage(subject_id, encounter.start, metadata["payer_plan"])
        or ""
    )
    age = calculate_age(person_values["birth"], encounter.start)
    person_values.update(
        {
            "payer_plan": payer_plan,
            "age_in_years": age["age_in_years"],
            "age_in_days": age["age_in_days"],
        }
    )

    care_sites.update(
        metadata["provider"][p]["care_site_id"]
        for p in providers
        if p in metadata["provider"] and metadata["provider"][p].get("care_site_id")
    )

    person_elem = person_to_xml(person_values, person_fields)
    care_site_names = [
        metadata["care_site"][cs] for cs in care_sites if cs in metadata["care_site"]
    ]
    care_sites_xml = care_sites_to_xml(care_site_names)
    providers_xml = providers_to_xml(
        [metadata["provider"][p] for p in providers if p in metadata["provider"]]
    )

    elements: List[Element] = []
    # Skip an empty <person> (e.g. a later encounter with no repeated fields).
    if "person" in config.include_contexts and len(person_elem):
        elements.append(person_elem)
    if "care_sites" in config.include_contexts and len(care_sites_xml):
        elements.append(care_sites_xml)
    if "providers" in config.include_contexts and providers_xml is not None:
        elements.append(providers_xml)
    return elements


def _should_keep_event(
    event: Event,
    *,
    allowed_event_types: Optional[Set[str]],
    exclude_code_patterns: Sequence[str],
) -> bool:
    if allowed_event_types is not None and event.table not in allowed_event_types:
        return False
    code = str(event.code) if getattr(event, "code", None) is not None else ""
    return not any(fnmatch.fnmatch(code, pattern) for pattern in exclude_code_patterns)


def _collect_legend_codes(
    events: Sequence[Event],
    ontology: Any,
    legend: Dict[str, str],
    legend_units: Optional[Dict[str, str]] = None,
) -> None:
    """Record code -> ontology name (and optional unit) for legend-eligible events."""
    for event in events:
        if getattr(event, "table", None) == "image":
            continue
        code = str(event.code)
        if code == SMOKING_HISTORY_PSEUDO_CODE:
            legend.setdefault(code, SMOKING_HISTORY_LEGEND_NAME)
            continue
        name = ontology.get_description(event.code) or ""
        if name:
            legend.setdefault(code, name)
        if legend_units is not None:
            unit = _event_unit(event)
            if unit:
                legend_units.setdefault(code, unit)


def _format_number(value: float) -> str:
    """Format a numeric value the same way single events render their text."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{float(value):.2f}"


def _numeric_value(event: Event) -> Optional[float]:
    """Return the event's numeric measurement value, or None if non-numeric."""
    if getattr(event, "table", None) != "measurement":
        return None
    raw = getattr(event, "numeric_value", None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _bucket_key(event: Event, resolution: str) -> Any:
    """Key an event into a day or visit bucket within its encounter."""
    if resolution == "visit":
        visit_id = getattr(event, "visit_id", None)
        if visit_id not in (None, "_", ""):
            return ("v", visit_id)
    day = event.time.date() if getattr(event, "time", None) else None
    return ("d", day)


def _bucket_label(events: Sequence[Event]) -> str:
    """Build the entry timestamp label (single date or ``start..end`` span)."""
    dates = sorted({e.time.date() for e in events if getattr(e, "time", None)})
    if not dates:
        return "?"
    if len(dates) == 1:
        return dates[0].isoformat()
    return f"{dates[0].isoformat()}..{dates[-1].isoformat()}"


def _summarize_measurement(
    code: str,
    name: str,
    values: List[float],
    *,
    emit_name: bool,
    unit: Optional[str] = None,
) -> Element:
    """Render a numeric measurement group as min/max/mean/count (or verbatim)."""
    el = Element("event")
    el.set("code", code)
    if emit_name and name:
        el.set("name", name)
    if unit:
        el.set("unit", sanitize_xml_text(unit))
    if len(values) == 1:
        el.text = _format_number(values[0])
        return el
    el.set("n", str(len(values)))
    lo, hi = min(values), max(values)
    if lo == hi:
        el.text = _format_number(lo)
    else:
        el.set("min", _format_number(lo))
        el.set("max", _format_number(hi))
        el.set("mean", f"{statistics.mean(values):.2f}")
    return el


def _collapsed_entries(
    encounter: Encounter,
    *,
    resolution: str,
    ontology: Any,
    excluded_props: Set[str],
    attribute_order: Optional[Sequence[str]],
    emit_name: bool,
    allowed_event_types: Optional[Set[str]],
    exclude_code_patterns: Sequence[str],
    legend: Optional[Dict[str, str]],
    legend_units: Optional[Dict[str, str]] = None,
) -> List[Element]:
    """Group an encounter's events into day/visit buckets and summarize them.

    Within each bucket: numeric measurements collapse to a min/max/mean/count
    summary per code; notes and imaging stay verbatim; every other repeated
    (code, value) collapses to one event carrying an ``n`` repeat count.
    """
    kept = [
        e
        for e in encounter.events
        if _should_keep_event(
            e,
            allowed_event_types=allowed_event_types,
            exclude_code_patterns=exclude_code_patterns,
        )
    ]
    if legend is not None:
        _collect_legend_codes(kept, ontology, legend, legend_units)

    buckets: Dict[Any, List[Event]] = {}
    for event in kept:
        buckets.setdefault(_bucket_key(event, resolution), []).append(event)

    entries: List[Element] = []
    for bucket_events in buckets.values():
        bucket_events = collapse_smoking_events(bucket_events, legend=legend)
        measurements: Dict[str, List[float]] = {}
        meas_names: Dict[str, str] = {}
        meas_units: Dict[str, str] = {}
        verbatim: List[Event] = []
        # other repeated events keyed by (code, value) -> (representative, count)
        others: Dict[Any, List] = {}

        for event in bucket_events:
            if getattr(event, "table", None) in COLLAPSE_VERBATIM_TABLES:
                verbatim.append(event)
                continue
            num = _numeric_value(event)
            if num is not None:
                code = str(event.code)
                measurements.setdefault(code, []).append(num)
                meas_names.setdefault(code, ontology.get_description(event.code) or "")
                unit = _event_unit(event)
                if unit:
                    meas_units.setdefault(code, unit)
                continue
            value = getattr(event, "text_value", None)
            key = (str(event.code), str(value) if value is not None else None)
            if key in others:
                others[key][1] += 1
            else:
                others[key] = [event, 1]

        children: List[Element] = []
        for code in sorted(measurements):
            children.append(
                _summarize_measurement(
                    code,
                    meas_names.get(code, ""),
                    measurements[code],
                    emit_name=emit_name,
                    unit=meas_units.get(code),
                )
            )
        for rep_event, count in others.values():
            el = event_to_xml(
                rep_event,
                ontology,
                excluded_props,
                attribute_order,
                emit_name=emit_name,
            )
            if count > 1:
                el.set("n", str(count))
            children.append(el)
        for note_event in verbatim:
            children.append(
                event_to_xml(
                    note_event,
                    ontology,
                    excluded_props,
                    attribute_order,
                    emit_name=emit_name,
                )
            )

        if children:
            entries.append(
                entry_to_xml_with_label(_bucket_label(bucket_events), children)
            )
    return entries


def entry_to_xml_with_label(label: str, events: Sequence[Element]) -> Element:
    """Like :func:`entry_to_xml` but the timestamp is a precomputed label str."""
    entry_elem = Element("entry", timestamp=label)
    for event in events:
        entry_elem.append(event)
    return entry_elem


def minify_tree(root: Element) -> Element:
    """Rename high-frequency tags/attrs in place (see MINIFY_* maps)."""
    for elem in root.iter():
        new_tag = MINIFY_TAG_MAP.get(elem.tag)
        if new_tag is not None:
            elem.tag = new_tag
        for old_attr, new_attr in MINIFY_ATTR_MAP.items():
            if old_attr in elem.attrib:
                elem.set(new_attr, elem.attrib.pop(old_attr))
    return root


def build_subject_xml(
    subject: Subject,
    encounters: Sequence[Encounter],
    *,
    ontology: Any,
    metadata: Optional[Dict[str, Any]],
    config: TextifyConfig,
    miss: MissTracker,
) -> Element:
    """Assemble the canonical ``<eventstream>`` tree for one subject."""
    if config.compress_notes:
        compress_subject_notes(subject)
    excluded_props = _resolve_excluded_props(config)
    person_values = get_person_values(subject, ontology, excluded_props)

    allowed_event_types = (
        None if config.event_types == ("*",) else set(config.event_types)
    )
    exclude_code_patterns = list(config.exclude_codes)

    root = Element("eventstream", person_id=str(subject.subject_id))

    # When emitting a legend, gather code -> name for every rendered (non-image)
    # event so the per-event name attribute can be dropped. Image-event names are
    # derived per-event (anatomic site + modality), not from the code, so they
    # stay inline and are excluded from the legend.
    legend: Dict[str, str] = {}
    legend_units: Dict[str, str] = {}

    for index, encounter in enumerate(encounters):
        enc_elem = Element("encounter")
        if config.include_contexts and metadata is not None:
            person_fields = config.person_fields_for_encounter(is_first=index == 0)
            for elem in _encounter_context_elements(
                encounter,
                subject_id=subject.subject_id,
                person_values=person_values,
                person_fields=set(person_fields),
                metadata=metadata,
                config=config,
                miss=miss,
            ):
                enc_elem.append(elem)

        events_elem = SubElement(enc_elem, "events")
        if config.collapse_events:
            for entry in _collapsed_entries(
                encounter,
                resolution=config.collapse_events,
                ontology=ontology,
                excluded_props=excluded_props,
                attribute_order=config.attribute_order,
                emit_name=not config.emit_code_legend,
                allowed_event_types=allowed_event_types,
                exclude_code_patterns=exclude_code_patterns,
                legend=legend if config.emit_code_legend else None,
                legend_units=legend_units if config.emit_code_legend else None,
            ):
                events_elem.append(entry)
            root.append(enc_elem)
            continue
        for timestamp, entries in bin_by_time(encounter.events).items():
            kept = [
                event
                for event in entries
                if _should_keep_event(
                    event,
                    allowed_event_types=allowed_event_types,
                    exclude_code_patterns=exclude_code_patterns,
                )
            ]
            if not kept:
                continue
            kept = collapse_smoking_events(
                kept, legend=legend if config.emit_code_legend else None
            )
            if config.emit_code_legend:
                _collect_legend_codes(kept, ontology, legend, legend_units)
            events_elem.append(
                entry_to_xml(
                    timestamp,
                    [
                        event_to_xml(
                            event,
                            ontology,
                            excluded_props,
                            config.attribute_order,
                            emit_name=not config.emit_code_legend,
                        )
                        for event in kept
                    ],
                )
            )
        root.append(enc_elem)

    # `fix_image_events` selects image events by their `table` attribute, so it
    # must run before `strip_table_attr`.
    root = fix_image_events(root)
    root = remove_null_elements(root)
    if config.drop_table_attr:
        root = strip_table_attr(root)
    if config.emit_code_legend:
        root.insert(0, build_legend_element(legend, legend_units))
    # Minification renames tags/attrs, so it must run last (after image fixup,
    # table stripping, and legend insertion which all rely on canonical names).
    if config.minify_tags:
        root = minify_tree(root)
    return root


def serialize(root: Element) -> str:
    """Serialize the canonical tree to pretty-printed LUMIA XML."""
    return tostring(
        root, encoding="utf-8", pretty_print=True, xml_declaration=True
    ).decode("utf-8")
