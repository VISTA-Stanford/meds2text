"""Typed configuration for the MEDS-to-text pipeline.

A single immutable ``TextifyConfig`` is built once (in ``cli.py``) and passed
explicitly through the pipeline, replacing the previous practice of threading an
``argparse.Namespace`` through every function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

VALID_FORMATS = ("lumia_xml", "lumia_json", "fhir_like_json")
VALID_CONTEXTS = ("person", "providers", "care_sites")
VALID_COLLAPSE_RESOLUTIONS = ("day", "visit")
# Sub-blocks of the per-encounter <person> element.
VALID_PERSON_FIELDS = ("birthdate", "age", "demographics", "payerplan")


@dataclass(frozen=True)
class TextifyConfig:
    """All options controlling a textify run.

    The input is assumed to be an already-transformed MEDS parquet extract
    (cleaned upstream by ``medspace``); this pipeline only renders it to text.
    """

    path_to_meds: str
    path_to_output: str
    path_to_ontology: str = "omop_ontology"
    path_to_metadata: Optional[str] = None
    output_format: str = "lumia_xml"
    include_contexts: Tuple[str, ...] = ()
    # <person> sub-blocks repeated in every encounter (these change over time).
    person_fields_every_encounter: Tuple[str, ...] = ("age", "payerplan")
    # <person> sub-blocks emitted only in the first encounter (static fields).
    person_fields_first_encounter: Tuple[str, ...] = ("birthdate", "demographics")
    exclude_props: Tuple[str, ...] = ()
    exclude_codes: Tuple[str, ...] = ()
    event_types: Tuple[str, ...] = ("*",)
    attribute_order: Optional[Tuple[str, ...]] = None
    # Emit a per-document <legend> mapping each used code -> ontology name and
    # drop the (otherwise repeated) per-event ``name`` attribute. Lossless: full
    # names live in the legend. Intended for ``lumia_xml``.
    emit_code_legend: bool = False
    # Drop the per-event ``table`` attribute from the rendered output. The table
    # is still used internally for event filtering and image-name resolution.
    drop_table_attr: bool = False
    # Collapse events within each encounter to a coarser temporal resolution:
    # ``"day"`` (group by calendar day) or ``"visit"`` (group by visit_id, falling
    # back to day). Numeric measurements are summarized as min/max/mean/count;
    # repeated non-numeric events are de-duplicated; notes are kept in full. Lossy.
    collapse_events: Optional[str] = None
    # Minify the high-frequency XML tag/attribute names (event->e, entry->g,
    # timestamp->t) to shrink the output. Intended for lumia_xml.
    minify_tags: bool = False
    # Compress clinical-note ``text_value`` before render: de-id scrub, whitespace
    # collapse (lossless for tokens), and PMC 10-gram cross-note dedup (removes
    # redundant copy-paste). Intended for lumia_xml.
    compress_notes: bool = False
    batch_mode: bool = False
    batch_size: int = 2500
    test_mode: bool = False
    n_processes: int = 1
    person_ids_file: Optional[str] = None
    # Populated internally once a person-id allowlist has been resolved.
    allowed_subject_ids: Optional[FrozenSet[int]] = field(default=None)

    def __post_init__(self) -> None:
        if self.output_format not in VALID_FORMATS:
            raise ValueError(
                f"Invalid format {self.output_format!r}; choose from {VALID_FORMATS}"
            )
        if (
            self.emit_code_legend
            or self.drop_table_attr
            or self.collapse_events
            or self.minify_tags
            or self.compress_notes
        ) and (self.output_format != "lumia_xml"):
            raise ValueError(
                "--emit_code_legend / --drop_table / --collapse_events / "
                "--minify_tags / --compress_notes are only supported with "
                "--format lumia_xml (the FHIR-like renderer relies on the "
                "per-event table and name attributes and the canonical tag names)"
            )
        if self.collapse_events and self.collapse_events not in (
            VALID_COLLAPSE_RESOLUTIONS
        ):
            raise ValueError(
                f"Invalid collapse_events {self.collapse_events!r}; "
                f"choose from {VALID_COLLAPSE_RESOLUTIONS}"
            )
        for context in self.include_contexts:
            if context not in VALID_CONTEXTS:
                raise ValueError(
                    f"Invalid context {context!r}; choose from {VALID_CONTEXTS}"
                )
        if self.include_contexts and not self.path_to_metadata:
            raise ValueError(
                "--include_contexts requires --path_to_metadata (provider/care_site/"
                "payer_plan_period CSVs)"
            )
        for field_name in (
            *self.person_fields_every_encounter,
            *self.person_fields_first_encounter,
        ):
            if field_name not in VALID_PERSON_FIELDS:
                raise ValueError(
                    f"Invalid person field {field_name!r}; "
                    f"choose from {VALID_PERSON_FIELDS}"
                )

    @property
    def needs_metadata(self) -> bool:
        return bool(self.include_contexts)

    def person_fields_for_encounter(self, *, is_first: bool) -> FrozenSet[str]:
        """Which <person> sub-blocks to emit for an encounter at this position."""
        fields = set(self.person_fields_every_encounter)
        if is_first:
            fields.update(self.person_fields_first_encounter)
        return frozenset(fields)
