"""Pipeline orchestration: stream a MEDS parquet extract and render it to text.

Discovers parquet shards, fans them out across worker processes, and for each
subject builds encounters, renders the canonical XML tree, serializes to the
requested format, and writes per-subject files (or batched JSONL).
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional

from meds2text.config import TextifyConfig
from meds2text.encounters import bin_encounters
from meds2text.io.parquet import (
    discover_parquet_shards,
    iter_subjects_from_parquet_files,
    load_and_validate_person_ids,
    partition_shards_across_workers,
)
from meds2text.metadata import MissTracker, load_metadata
from meds2text.ontology import OntologyDescriptionLookupTable
from meds2text.render import build_subject_xml, get_renderer
from meds2text.subject import Subject

logger = logging.getLogger(__name__)

EXCLUDED_TABLES = {"person"}


def load_ontology(path_to_ontology: str) -> OntologyDescriptionLookupTable:
    ontology = OntologyDescriptionLookupTable()
    ontology.load(path_to_ontology)
    return ontology


def render_subject(
    subject: Subject,
    *,
    config: TextifyConfig,
    ontology: Any,
    metadata: Optional[Dict[str, Any]],
    miss: MissTracker,
) -> str:
    """Build encounters, assemble the XML tree, and serialize for one subject."""
    encounters = bin_encounters(subject.events, excluded_tables=EXCLUDED_TABLES)
    root = build_subject_xml(
        subject,
        encounters,
        ontology=ontology,
        metadata=metadata,
        config=config,
        miss=miss,
    )
    return get_renderer(config.output_format).serialize(root)


def process_shards_chunk(
    shard_paths: List[str], config: TextifyConfig, process_id: int
) -> None:
    """Worker: stream assigned parquet shards and render each subject."""
    if not shard_paths:
        logger.info(f"Process {process_id}: no Parquet shards assigned; skipping")
        return

    ontology = load_ontology(config.path_to_ontology)
    metadata = load_metadata(config.path_to_metadata) if config.needs_metadata else None
    renderer = get_renderer(config.output_format)
    miss = MissTracker()

    batch_file = None
    if config.batch_mode:
        batch_path = os.path.join(config.path_to_output, f"batch_{process_id}.jsonl")
        batch_file = open(batch_path, "a", buffering=1, encoding="utf-8")

    allowed = set(config.allowed_subject_ids) if config.allowed_subject_ids else None

    try:
        stream = iter_subjects_from_parquet_files(
            shard_paths, allowed_subject_ids=allowed
        )
        for index, (subject_id, subject) in enumerate(stream):
            output = render_subject(
                subject,
                config=config,
                ontology=ontology,
                metadata=metadata,
                miss=miss,
            )
            if batch_file is not None:
                batch_file.write(
                    json.dumps({"subject_id": subject_id, "output": output}) + "\n"
                )
            else:
                out_path = os.path.join(
                    config.path_to_output, f"{subject_id}.{renderer.extension}"
                )
                with open(
                    out_path, "w", encoding="utf-8", errors="xmlcharrefreplace"
                ) as f:
                    f.write(output)

            if config.test_mode and index > 20:
                break

        miss.log_summary(process_id)
    finally:
        if batch_file is not None:
            batch_file.close()


def run(config: TextifyConfig) -> None:
    """Run the full textify pipeline for ``config``."""
    os.makedirs(config.path_to_output, exist_ok=True)

    shard_paths = discover_parquet_shards(config.path_to_meds)

    if config.person_ids_file:
        allowed = load_and_validate_person_ids(config.person_ids_file, shard_paths)
        config = replace(config, allowed_subject_ids=frozenset(allowed))

    partitions = partition_shards_across_workers(shard_paths, config.n_processes)

    if config.n_processes > 1:
        # spawn avoids an os.fork() issue with polars on Python < 3.14.
        multiprocessing.set_start_method("spawn", force=True)
        processes = []
        for idx, chunk in enumerate(partitions):
            p = multiprocessing.Process(
                target=process_shards_chunk, args=(chunk, config, idx)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        process_shards_chunk(partitions[0], config, process_id=0)
