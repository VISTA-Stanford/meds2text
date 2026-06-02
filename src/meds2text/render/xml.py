"""LUMIA XML rendering: build the canonical event-stream tree from a subject.

The XML tree produced here is the canonical intermediate representation; the
JSON and FHIR-like renderers derive their output from it.
"""

from __future__ import annotations

import fnmatch
import re
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
from meds2text.subject import Event, Subject

BASE_EXCLUDED_PROPS = {"time", "end", "subject_id"}
DEFAULT_ATTRIBUTE_ORDER = ("table", "code", "name")


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
) -> Element:
    """Render a single event as an ``<event>`` element.

    ``excluded_props`` only filters which attributes appear in the output.
    """
    excluded_props = excluded_props or set()
    attribute_order = attribute_order or DEFAULT_ATTRIBUTE_ORDER

    name = ontology.get_description(event.code) or ""
    attributes = {key: value for key, value in event if key not in excluded_props}
    value = attributes.get("numeric_value") or attributes.get("text_value") or None

    attributes.setdefault("name", name)
    attributes.pop("numeric_value", None)
    attributes.pop("text_value", None)

    # "_" is the sentinel for a missing attribute value.
    xml_attributes = {
        key: sanitize_xml_text(str(attr_value)) if attr_value is not None else "_"
        for key, attr_value in attributes.items()
        if not (key == "name" and (attr_value is None or attr_value == ""))
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
    providers_xml = dict_to_xml(
        [metadata["provider"][p] for p in providers if p in metadata["provider"]],
        "providers",
        "provider",
    )
    care_sites_xml = dict_to_xml(
        [
            {"care_site_id": cs, "care_site_name": metadata["care_site"].get(cs, "_")}
            for cs in care_sites
            if cs in metadata["care_site"]
        ],
        "caresites",
        "caresite",
    )

    elements: List[Element] = []
    # Skip an empty <person> (e.g. a later encounter with no repeated fields).
    if "person" in config.include_contexts and len(person_elem):
        elements.append(person_elem)
    if "care_sites" in config.include_contexts:
        elements.append(care_sites_xml)
    if "providers" in config.include_contexts:
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
    excluded_props = _resolve_excluded_props(config)
    person_values = get_person_values(subject, ontology, excluded_props)

    allowed_event_types = (
        None if config.event_types == ("*",) else set(config.event_types)
    )
    exclude_code_patterns = list(config.exclude_codes)

    root = Element("eventstream", person_id=str(subject.subject_id))

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
            events_elem.append(
                entry_to_xml(
                    timestamp,
                    [
                        event_to_xml(
                            event, ontology, excluded_props, config.attribute_order
                        )
                        for event in kept
                    ],
                )
            )
        root.append(enc_elem)

    root = fix_image_events(root)
    root = remove_null_elements(root)
    return root


def serialize(root: Element) -> str:
    """Serialize the canonical tree to pretty-printed LUMIA XML."""
    return tostring(
        root, encoding="utf-8", pretty_print=True, xml_declaration=True
    ).decode("utf-8")
