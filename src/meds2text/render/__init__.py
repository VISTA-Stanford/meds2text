"""Output format renderers.

All formats share the same canonical XML tree (built by ``render.xml``); a
renderer only knows how to serialize that tree and what file extension to use.
Adding a new format is a matter of writing a ``serialize(root) -> str`` function
and registering it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from lxml.etree import Element

from meds2text.render import fhir, json, xml

# Re-export the canonical tree builder for the pipeline.
build_subject_xml = xml.build_subject_xml


@dataclass(frozen=True)
class Renderer:
    """Serializes the canonical XML tree to a given output format."""

    serialize: Callable[[Element], str]
    extension: str


_REGISTRY: Dict[str, Renderer] = {
    "lumia_xml": Renderer(serialize=xml.serialize, extension="xml"),
    "lumia_json": Renderer(serialize=json.serialize, extension="json"),
    "fhir_like_json": Renderer(serialize=fhir.serialize, extension="fhir-like.json"),
}


def get_renderer(output_format: str) -> Renderer:
    try:
        return _REGISTRY[output_format]
    except KeyError:
        raise ValueError(
            f"Invalid format: {output_format!r}; choose from {sorted(_REGISTRY)}"
        )


def available_formats() -> tuple:
    return tuple(sorted(_REGISTRY))


__all__ = ["Renderer", "get_renderer", "available_formats", "build_subject_xml"]
