import os
import collections
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Optional, Iterable, Set, Type



# Utility to preprocess Athena CSV files
def preprocess_csv(file_path):
    # Stub for preprocessing, replace with actual logic if needed.
    return file_path





class AthenaOntology:
    """
    Full OMOP Ontology with parent-child relationships.
    TODO: This uses a lot of memory, we should consider using a more memory efficient data structure
    """

    def __init__(
        self,
        description_map: Dict[str, str],
        parents_map: Dict[str, Set[str]],
        children_map: Optional[Dict[str, Set[str]]] = None,
    ):
        self.description_map = description_map
        self.parents_map = parents_map
        self.children_map = children_map or self._build_children_map(parents_map)

    def _build_children_map(
        self, parents_map: Dict[str, Set[str]]
    ) -> Dict[str, Set[str]]:
        children_map = collections.defaultdict(set)
        for code, parents in parents_map.items():
            for parent in parents:
                children_map[parent].add(code)
        return children_map

    def save_to_parquet(self, file_path: str, compression: str = "zstd"):
        """Save the ontology as Parquet files in the specified directory."""
        # Ensure the directory exists
        os.makedirs(file_path, exist_ok=True)

        # Prepare the Parquet tables
        description_table = pa.Table.from_pydict(
            {
                "code": list(self.description_map.keys()),
                "description": list(self.description_map.values()),
            }
        )
        parents_table = pa.Table.from_pydict(
            {
                "code": list(self.parents_map.keys()),
                "parents": [list(parents) for parents in self.parents_map.values()],
            }
        )

        # Write the tables to files in the specified directory
        pq.write_table(
            description_table,
            os.path.join(file_path, "descriptions.parquet"),
            compression=compression,
        )
        pq.write_table(
            parents_table,
            os.path.join(file_path, "parents.parquet"),
            compression=compression,
        )

    @classmethod
    def load_from_parquet(cls, file_path: str):
        """Load the ontology from Parquet files in the specified directory."""
        description_table = pq.read_table(
            os.path.join(file_path, "descriptions.parquet")
        )
        parents_table = pq.read_table(os.path.join(file_path, "parents.parquet"))

        description_map = {
            row["code"]: row["description"] for row in description_table.to_pylist()
        }
        parent_map = collections.defaultdict(
            set, {row["code"]: set(row["parents"]) for row in parents_table.to_pylist()}
        )
        # faster to rebuilt children_map from parent_map
        return cls(description_map, parent_map, children_map=None)

    @classmethod
    def load_from_athena_snapshot(
        cls: Type["Ontology"],
        athena_path: str,
        code_metadata: Optional[Dict[str, Dict[str, Optional[Iterable[str]]]]] = None,
    ) -> None:

        print("Load from Athena Vocabulary Snapshot...")
        description_map: Dict[str, str] = {}
        parents_map: Dict[str, Set[str]] = collections.defaultdict(set)
        code_metadata: Dict[str, Dict[str, Optional[Iterable[str]]]] = (
            code_metadata or {}
        )

        try:
            concept_path = os.path.join(athena_path, "CONCEPT.csv")
            cleaned_concept_path = preprocess_csv(concept_path)

            concept = pl.scan_csv(
                cleaned_concept_path,
                separator="\t",
                infer_schema_length=0,
                quote_char=None,
            )

            code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
            description_col = pl.col("concept_name")
            concept_id_col = pl.col("concept_id").cast(pl.Int64)

            processed_concepts = (
                concept.select(
                    [
                        code_col.alias("code"),
                        concept_id_col.alias("concept_id"),
                        description_col.alias("description"),
                        pl.col("standard_concept").is_null().alias("is_non_standard"),
                    ]
                )
                .collect()
                .rows()
            )

            concept_id_to_code_map = {}
            non_standard_concepts = set()

            for code, concept_id, description, is_non_standard in processed_concepts:
                if code and concept_id is not None:
                    concept_id_to_code_map[concept_id] = code
                    if code not in description_map:
                        description_map[code] = description
                    if is_non_standard:
                        non_standard_concepts.add(concept_id)

            # Add OMOP concept_id to description map as OMOP_CONCEPT_ID/concept_id - > 	concept_name
            df = concept.select([concept_id_col, description_col]).collect()

            # Iterate over rows; each row is a tuple (concept_id, description)
            for concept_id, description in df.rows():
                description_map[f"OMOP_CONCEPT_ID/{concept_id}"] = description

            # Process CONCEPT_RELATIONSHIP.csv
            relationship_path = os.path.join(athena_path, "CONCEPT_RELATIONSHIP.csv")
            cleaned_relationship_path = preprocess_csv(relationship_path)

            relationship = pl.scan_csv(
                cleaned_relationship_path, separator="\t", infer_schema_length=0
            )

            relationship = relationship.filter(
                (pl.col("relationship_id") == "Maps to")
                & (pl.col("concept_id_1") != pl.col("concept_id_2"))
            )

            for concept_id_1, concept_id_2 in (
                relationship.select(
                    pl.col("concept_id_1").cast(pl.Int64),
                    pl.col("concept_id_2").cast(pl.Int64),
                )
                .collect()
                .rows()
            ):
                if concept_id_1 in non_standard_concepts:
                    if (
                        concept_id_1 in concept_id_to_code_map
                        and concept_id_2 in concept_id_to_code_map
                    ):
                        parents_map[concept_id_to_code_map[concept_id_1]].add(
                            concept_id_to_code_map[concept_id_2]
                        )

            # Process CONCEPT_ANCESTOR.csv
            ancestor_path = os.path.join(athena_path, "CONCEPT_ANCESTOR.csv")
            cleaned_ancestor_path = preprocess_csv(ancestor_path)

            ancestor = pl.scan_csv(
                cleaned_ancestor_path, separator="\t", infer_schema_length=0
            )

            ancestor = ancestor.filter(pl.col("min_levels_of_separation") == "1")

            for concept_id, parent_concept_id in (
                ancestor.select(
                    pl.col("descendant_concept_id").cast(pl.Int64),
                    pl.col("ancestor_concept_id").cast(pl.Int64),
                )
                .collect()
                .rows()
            ):
                if (
                    concept_id in concept_id_to_code_map
                    and parent_concept_id in concept_id_to_code_map
                ):
                    parents_map[concept_id_to_code_map[concept_id]].add(
                        concept_id_to_code_map[parent_concept_id]
                    )

            # Optional code metadata
            for code, code_info in code_metadata.items():
                if code_info.get("description"):
                    description_map[code] = code_info["description"]
                if code_info.get("parent_codes"):
                    parents_map[code] = set(code_info["parent_codes"])

            return cls(description_map, parents_map)

        except Exception as e:
            raise RuntimeError(f"Error processing Athena files: {e}")

    def get_description(self, code: str) -> Optional[str]:
        """Get description for a code."""
        return self.description_map.get(code)

    def get_children(self, code: str) -> Set[str]:
        """Get immediate children of a code."""
        return self.children_map.get(code, set())

    def get_parents(self, code: str) -> Set[str]:
        """Get immediate parents of a code."""
        return self.parents_map.get(code, set())
