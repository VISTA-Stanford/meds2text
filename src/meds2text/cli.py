"""Command-line entry point for ``meds-textify``.

Parses arguments into a :class:`TextifyConfig` and runs the pipeline. The input
is assumed to be an already-transformed MEDS parquet extract (cleaned upstream
by ``medspace``); this tool only renders it to text.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional, Sequence

from meds2text import pipeline
from meds2text.config import (
    VALID_CONTEXTS,
    VALID_FORMATS,
    VALID_PERSON_FIELDS,
    TextifyConfig,
)

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> TextifyConfig:
    parser = argparse.ArgumentParser(
        description="Render an already-transformed MEDS parquet extract to text."
    )
    parser.add_argument(
        "--path_to_meds",
        required=True,
        help="MEDS extract root containing data/**/*.parquet shards",
    )
    parser.add_argument("--path_to_output", required=True, help="Output directory")
    parser.add_argument(
        "--path_to_ontology",
        default="omop_ontology",
        help="Directory with the prebuilt ontology description trie",
    )
    parser.add_argument(
        "--path_to_metadata",
        default=None,
        help="Directory with OMOP provider/care_site/payer_plan_period CSVs "
        "(required when --include_contexts is used)",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="lumia_xml",
        choices=list(VALID_FORMATS),
    )
    parser.add_argument(
        "--include_contexts",
        nargs="+",
        default=[],
        choices=list(VALID_CONTEXTS),
        help="Metadata contexts to embed per encounter",
    )
    parser.add_argument(
        "--person_fields_every_encounter",
        nargs="*",
        default=["age", "payerplan"],
        choices=list(VALID_PERSON_FIELDS),
        help="<person> sub-blocks repeated in every encounter (default: age payerplan)",
    )
    parser.add_argument(
        "--person_fields_first_encounter",
        nargs="*",
        default=["birthdate", "demographics"],
        choices=list(VALID_PERSON_FIELDS),
        help="<person> sub-blocks emitted only in the first encounter "
        "(default: birthdate demographics)",
    )
    parser.add_argument(
        "--exclude_props",
        nargs="+",
        default=[],
        help="Event properties to omit from the output",
    )
    parser.add_argument(
        "--exclude_codes",
        nargs="+",
        default=[],
        help="Patterns to exclude events by code (supports wildcards, e.g. 'STANFORD_OBS/*')",
    )
    parser.add_argument(
        "--event_types",
        nargs="+",
        default=["*"],
        help="Event tables to include, or '*' for all (default: *)",
    )
    parser.add_argument(
        "--attribute_order",
        nargs="+",
        default=None,
        help="Preferred XML attribute order (default: table, code, name)",
    )
    parser.add_argument("--batch_mode", action="store_true", help="Write batched JSONL")
    parser.add_argument("--batch_size", type=int, default=2500)
    parser.add_argument(
        "--test_mode", action="store_true", help="Process only a few subjects"
    )
    parser.add_argument("--n_processes", type=int, default=1)
    parser.add_argument(
        "--person_ids_file",
        default=None,
        help="CSV/TSV with a 'person_id' column to restrict which subjects are processed",
    )
    args = parser.parse_args(argv)

    return TextifyConfig(
        path_to_meds=args.path_to_meds,
        path_to_output=args.path_to_output,
        path_to_ontology=args.path_to_ontology,
        path_to_metadata=args.path_to_metadata,
        output_format=args.output_format,
        include_contexts=tuple(args.include_contexts),
        person_fields_every_encounter=tuple(args.person_fields_every_encounter),
        person_fields_first_encounter=tuple(args.person_fields_first_encounter),
        exclude_props=tuple(args.exclude_props),
        exclude_codes=tuple(args.exclude_codes),
        event_types=tuple(args.event_types),
        attribute_order=tuple(args.attribute_order) if args.attribute_order else None,
        batch_mode=args.batch_mode,
        batch_size=args.batch_size,
        test_mode=args.test_mode,
        n_processes=args.n_processes,
        person_ids_file=args.person_ids_file,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = parse_args(argv)
    start_time = time.time()
    pipeline.run(config)
    logger.info(f"Elapsed time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
