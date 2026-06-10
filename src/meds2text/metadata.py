"""Optional OMOP metadata: provider / care_site / payer_plan enrichment.

Only loaded when ``--include_contexts`` requests provider, care_site, or person
context. Also holds person-level attribute extraction (birth, demographics) and
age computation used when rendering the ``<person>`` block.
"""

from __future__ import annotations

import collections
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)


@dataclass
class MissTracker:
    """Tracks provider/care_site ids seen vs. missing from metadata, per worker."""

    providers_seen: Set[str] = field(default_factory=set)
    providers_missing: Set[str] = field(default_factory=set)
    care_sites_seen: Set[str] = field(default_factory=set)
    care_sites_missing: Set[str] = field(default_factory=set)

    def see_provider(self, provider_id: str, present: bool) -> None:
        self.providers_seen.add(provider_id)
        if not present:
            self.providers_missing.add(provider_id)

    def see_care_site(self, care_site_id: str, present: bool) -> None:
        self.care_sites_seen.add(care_site_id)
        if not present:
            self.care_sites_missing.add(care_site_id)

    def log_summary(self, process_id: int) -> None:
        for label, seen, missing in (
            ("provider_ids", self.providers_seen, self.providers_missing),
            ("care_site_ids", self.care_sites_seen, self.care_sites_missing),
        ):
            if not seen:
                logger.info(f"Process {process_id}: No {label} encountered")
                continue
            pct = len(missing) / len(seen) * 100
            logger.info(
                f"Process {process_id}: Missing {label}: "
                f"{len(missing)}/{len(seen)} ({pct:.2f}%)"
            )
            if missing:
                logger.warning(
                    f"Process {process_id}: Missing {label} (first 20): "
                    f"{list(missing)[:20]}"
                )


def read_metadata_file(path_to_metadata: str, base_name: str, **kwargs) -> pd.DataFrame:
    """Read ``<base_name>.csv`` or ``<base_name>.csv.gz`` from a metadata dir."""
    for ext in (".csv", ".csv.gz"):
        file_path = os.path.join(path_to_metadata, f"{base_name}{ext}")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, **kwargs)
    raise FileNotFoundError(
        f"Could not find {base_name} with extension .csv or .csv.gz in {path_to_metadata}"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    return df


def _require_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    col_lower = col_name.lower()
    if col_lower not in df.columns:
        raise KeyError(
            f"Column '{col_name}' not found. Available columns: {list(df.columns)}"
        )
    return df[col_lower]


def _load_providers(path_to_metadata: str) -> Dict[str, Dict[str, Any]]:
    df = _normalize_columns(read_metadata_file(path_to_metadata, "provider", dtype=str))
    df = df.fillna("")
    df["gender_concept_id"] = _require_column(df, "gender_concept_id").replace(
        {"8532": "FEMALE", "8507": "MALE", "0": ""}
    )
    return {str(row.provider_id): row._asdict() for row in df.itertuples()}


def _load_payer_plans(path_to_metadata: str) -> Dict[str, List[Dict[str, Any]]]:
    df = _normalize_columns(read_metadata_file(path_to_metadata, "payer_plan_period"))

    rename_map = {
        "payer_plan_period_start_date": "start_date",
        "payer_plan_period_end_date": "end_date",
    }
    required = (
        "person_id",
        "payer_plan_period_start_date",
        "payer_plan_period_end_date",
        "payer_source_value",
    )
    for col in required:
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found. Available columns: {list(df.columns)}"
            )
        if col in rename_map:
            df[col] = pd.to_datetime(df[col])

    df = df[list(required)].rename(columns=rename_map)
    payer_plan_map: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in df.itertuples():
        payer_plan_map[row.person_id].append(row._asdict())
    return payer_plan_map


def _load_care_sites(path_to_metadata: str) -> Dict[str, str]:
    df = _normalize_columns(
        read_metadata_file(path_to_metadata, "care_site", dtype=str)
    )
    df["care_site_name"] = _require_column(df, "care_site_name").fillna("_")
    return {str(row.care_site_id): row.care_site_name for row in df.itertuples()}


def _enrich_providers(
    providers: Dict[str, Dict[str, Any]], care_sites: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    enriched: Dict[str, Dict[str, Any]] = {}
    n_missing_specialty = 0
    for provider_id, provider in providers.items():
        care_site_id = provider["care_site_id"] or None
        care_site_id_str = str(care_site_id) if care_site_id else None
        props = {
            "provider_id": provider_id,
            "gender": provider["gender_concept_id"] or None,
            "speciality": provider["specialty_source_value"] or None,
            "year_of_birth": (
                int(provider["year_of_birth"]) if provider["year_of_birth"] else None
            ),
            "care_site_id": care_site_id_str,
            "care_site_name": (
                care_sites.get(care_site_id_str) if care_site_id_str else None
            ),
        }
        enriched[provider_id] = props
        if props["speciality"] is None:
            n_missing_specialty += 1
    if enriched:
        logger.info(
            "Missing provider speciality: %.1f%% (%d/%d)",
            n_missing_specialty / len(enriched) * 100,
            n_missing_specialty,
            len(enriched),
        )
    return enriched


def load_metadata(path_to_metadata: str) -> Dict[str, Any]:
    """Load OMOP provider, payer_plan_period, and care_site tables.

    Expects ``provider``, ``payer_plan_period``, and ``care_site`` CSV files
    (optionally ``.gz``) in ``path_to_metadata``.
    """
    care_sites = _load_care_sites(path_to_metadata)
    providers = _enrich_providers(_load_providers(path_to_metadata), care_sites)
    return {
        "provider": providers,
        "payer_plan": _load_payer_plans(path_to_metadata),
        "care_site": care_sites,
    }


def get_payer_plan_coverage(
    subject_id: Any,
    index_time: datetime,
    payer_plan_map: Dict[str, List[Dict[str, Any]]],
) -> Optional[str]:
    """Return the payer plan covering ``index_time`` for ``subject_id``, if any."""
    for period in payer_plan_map.get(subject_id, []):
        if period["start_date"] <= index_time <= period["end_date"]:
            return period["payer_source_value"]
    return None


def calculate_age(start_date: datetime, end_date: datetime) -> Dict[str, int]:
    """Return age in whole years and total days between two datetimes."""
    if start_date > end_date:
        logger.debug(
            "Encounter time %s precedes birth %s; reporting age 0",
            end_date,
            start_date,
        )
        return {"age_in_years": 0, "age_in_days": 0}
    return {
        "age_in_years": relativedelta(end_date, start_date).years,
        "age_in_days": (end_date - start_date).days,
    }


def get_person_values(
    subject: Any,
    ontology: Any,
    excluded_props: Optional[Set[str]] = None,
    metadata_props: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Extract birth time and race/ethnicity/gender demographics from a subject."""
    excluded_props = excluded_props or set()
    person: Dict[str, Any] = {"person_id": subject.subject_id}
    person.update(metadata_props or {})

    for event in getattr(subject, "events", []):
        if getattr(event, "table", None) != "person":
            continue
        if event.code == "MEDS_BIRTH":
            person["birth"] = event.time
            continue

        code_str = str(event.code) if event.code else ""
        match = re.match(r"^(Race|Ethnicity|Gender)[/_](.+)$", code_str, re.IGNORECASE)
        if not match:
            logger.error(f"Unexpected event code: {event.code}")
            continue

        tag = match.group(1).lower()
        concept_id = match.group(2)
        description = ontology.get_description(event.code) if event.code else None
        if not description and concept_id:
            for code_format in (
                f"OMOP_CONCEPT_ID/{concept_id}",
                concept_id,
                f"OMOP/{concept_id}",
                f"CONCEPT/{concept_id}",
                f"SNOMED/{concept_id}",
            ):
                description = ontology.get_description(code_format)
                if description:
                    break
        person[tag] = {"code": event.code, "description": description}

    return person
