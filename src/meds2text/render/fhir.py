"""FHIR-like JSON rendering (sketch only).

This is a rough, LLM-sketched approximation of a FHIR Bundle intended to capture
surface syntactic signal, not a compliant FHIR feed. Derived from the canonical
XML tree.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union

from lxml.etree import Element, tostring


def _coding(event: ET.Element) -> Dict[str, Any]:
    code = event.get("code") or "/"
    system, _, value = code.partition("/")
    return {"system": system, "code": value, "display": event.get("name")}


def xml_to_fhir_like(xml_string: Union[str, bytes]) -> Dict[str, Any]:
    root = ET.fromstring(xml_string)
    bundle: Dict[str, Any] = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [],
    }

    for encounter in root.findall(".//encounter"):
        bundle["entry"].append(
            {
                "resource": {
                    "resourceType": "Encounter",
                    "status": "finished",
                    "subject": {"reference": f"Patient/{root.get('person_id')}"},
                }
            }
        )

        person = encounter.find(".//person")
        if person is not None:
            bundle["entry"].append(
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": root.get("person_id"),
                        "birthDate": person.findtext(".//birthdate"),
                        "gender": (
                            person.findtext(".//demographics/gender") or ""
                        ).lower(),
                        "extension": [
                            {
                                "url": "ethnicity",
                                "valueString": person.findtext(
                                    ".//demographics/ethnicity"
                                ),
                            }
                        ],
                    }
                }
            )

        for entry in encounter.findall(".//entry"):
            timestamp = entry.get("timestamp")
            for event in entry.findall("event"):
                event_type = event.get("table")
                if event_type == "measurement":
                    resource = {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {"coding": [_coding(event)]},
                        "effectiveDateTime": timestamp,
                        "valueQuantity": {
                            "value": event.text,
                            "unit": event.get("unit", ""),
                        },
                    }
                elif event_type == "note":
                    resource = {
                        "resourceType": "DocumentReference",
                        "status": "current",
                        "type": {"coding": [_coding(event)]},
                        "content": [
                            {
                                "attachment": {
                                    "contentType": "text/plain",
                                    "data": event.text,
                                }
                            }
                        ],
                    }
                elif event_type == "condition":
                    resource = {
                        "resourceType": "Condition",
                        "code": {"coding": [_coding(event)]},
                        "onsetDateTime": timestamp,
                    }
                else:
                    continue
                bundle["entry"].append({"resource": resource})

    return bundle


def serialize(root: Element) -> str:
    return json.dumps(xml_to_fhir_like(tostring(root)), indent=2)
