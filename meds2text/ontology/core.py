import marisa_trie
import os
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional


class OntologyDescriptionLookupTable:
    """Lightweight Code to Description Lookup Table."""

    def __init__(self, descriptions: marisa_trie.BytesTrie = None):
        self.descriptions = descriptions

    def save(self, file_path: str):
        outfpath = os.path.join(file_path, "descriptions.trie")
        self.descriptions.save(outfpath)

    def load(self, file_path: str):
        trie = marisa_trie.BytesTrie()
        self.descriptions = trie.load(os.path.join(file_path, "descriptions.trie"))

    @classmethod
    def load_from_parquet(
        cls,
        file_path: str,
        code_column: str = "code",
        description_column: str = "description",
    ):
        description_table = pq.read_table(
            os.path.join(file_path, "descriptions.parquet")
        )
        codes = description_table.column(code_column).to_pylist()
        descriptions = description_table.column(description_column).to_pylist()

        trie = marisa_trie.BytesTrie(
            (code, description.encode("utf-8"))
            for code, description in zip(codes, descriptions)
        )

        return cls(trie)

    def get_description(self, code: str) -> Optional[str]:
        if code in self.descriptions:
            return self.descriptions[code][0].decode("utf-8")
        return None