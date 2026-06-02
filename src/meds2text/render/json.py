"""LUMIA JSON rendering: derive a nested dict from the canonical XML tree."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any, Union

from lxml.etree import Element, tostring


def _infer_type(value_str: str) -> Union[int, float, str]:
    value_str = value_str.strip()
    for caster in (int, float):
        try:
            return caster(value_str)
        except ValueError:
            continue
    return value_str


def _parse_element(element: ET.Element) -> Any:
    result: dict = {}
    for key, value in element.attrib.items():
        result[key] = _infer_type(value)

    if element.text and element.text.strip():
        if not result and not len(element):
            return _infer_type(element.text)
        result["value"] = _infer_type(element.text)

    children = list(element)
    if children:
        if all(child.tag == children[0].tag for child in children):
            result[children[0].tag] = [_parse_element(child) for child in children]
        else:
            for child in children:
                if child.tag in result:
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(_parse_element(child))
                else:
                    result[child.tag] = _parse_element(child)
    return result


def xml_to_json(xml_string: Union[str, bytes]) -> Any:
    try:
        return _parse_element(ET.fromstring(xml_string))
    except ET.ParseError as e:
        return {"error": f"Invalid XML: {str(e)}"}


def serialize(root: Element) -> str:
    return json.dumps(xml_to_json(tostring(root)), indent=2)
