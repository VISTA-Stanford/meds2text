import os
import collections
import zipfile
import io
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import networkx as nx
from typing import Dict, Optional, Iterable, Set, Type, Any, Union


# Utility to preprocess Athena CSV files
def preprocess_csv(file_path):
    # Stub for preprocessing, replace with actual logic if needed.
    return file_path


class AthenaFileReader:
    """Helper class to read files from either a zip archive or directory."""

    def __init__(self, athena_path: str):
        self.athena_path = athena_path
        self.is_zip = athena_path.lower().endswith(".zip")
        self.zip_file = None
        self.parent_dir = None

        if self.is_zip:
            self.zip_file = zipfile.ZipFile(athena_path, "r")
            self._detect_parent_directory()
        else:
            self._detect_parent_directory()

    def _detect_parent_directory(self):
        """Detect if files are in a parent directory and find the correct path."""
        required_files = [
            "CONCEPT.csv",
            "CONCEPT_RELATIONSHIP.csv",
            "CONCEPT_ANCESTOR.csv",
        ]

        if self.is_zip:
            # Check zip contents for parent directory
            zip_contents = self.zip_file.namelist()

            # Look for files directly in root or in a parent directory
            for file_path in zip_contents:
                filename = os.path.basename(file_path)
                if filename in required_files:
                    # Found a required file, check if it's in a parent directory
                    dir_path = os.path.dirname(file_path)
                    if dir_path and not self.parent_dir:
                        # Check if all required files are in this directory
                        all_files_present = all(
                            os.path.join(dir_path, req_file) in zip_contents
                            for req_file in required_files
                        )
                        if all_files_present:
                            self.parent_dir = dir_path
                            break
        else:
            # Check directory structure
            if os.path.isdir(self.athena_path):
                # Check if files are directly in the directory
                files_in_root = [
                    f for f in os.listdir(self.athena_path) if f in required_files
                ]

                if len(files_in_root) == len(required_files):
                    # Files are directly in the root directory
                    self.parent_dir = ""
                else:
                    # Look for a subdirectory containing all required files
                    for item in os.listdir(self.athena_path):
                        item_path = os.path.join(self.athena_path, item)
                        if os.path.isdir(item_path):
                            subdir_files = [
                                f for f in os.listdir(item_path) if f in required_files
                            ]
                            if len(subdir_files) == len(required_files):
                                self.parent_dir = item
                                break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.zip_file:
            self.zip_file.close()

    def read_csv(self, filename: str) -> pl.LazyFrame:
        """Read a CSV file from either zip archive or directory."""
        if self.is_zip:
            # Read from zip archive
            file_path = filename
            if self.parent_dir:
                file_path = os.path.join(self.parent_dir, filename)

            try:
                with self.zip_file.open(file_path) as file:
                    # Read the content as bytes and decode to string
                    content = file.read().decode("utf-8")
                    # Create a StringIO object for Polars to read from
                    return pl.scan_csv(
                        io.StringIO(content),
                        separator="\t",
                        infer_schema_length=0,
                        quote_char=None,
                    )
            except KeyError:
                raise FileNotFoundError(f"File {file_path} not found in zip archive")
        else:
            # Read from directory
            if self.parent_dir:
                file_path = os.path.join(self.athena_path, self.parent_dir, filename)
            else:
                file_path = os.path.join(self.athena_path, filename)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")

            cleaned_path = preprocess_csv(file_path)
            return pl.scan_csv(
                cleaned_path,
                separator="\t",
                infer_schema_length=0,
                quote_char=None,
            )


def hierarchy_pos(G, root, levels=None, width=2.0, height=1.0):
    """If there is a cycle that is reachable from root, then this will see infinite recursion.
    G: the graph
    root: the root node
    levels: a dictionary
            key: level number (starting from 0)
            value: number of nodes in this level
    width: horizontal space allocated for drawing
    height: vertical space allocated for drawing"""
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level"""
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        # Calculate minimum spacing based on node diameter
        # Assuming node_size=1000 in visualization, diameter is roughly sqrt(1000) ≈ 32
        # Convert to normalized coordinates (assuming width=2.0)
        min_spacing = 0.1  # Minimum spacing between node centers

        if levels[currentLevel][TOTAL] == 1:
            # Single node at this level - center it
            pos[node] = (width / 2, vert_loc)
        else:
            # Multiple nodes - distribute with minimum spacing
            total_width_needed = (levels[currentLevel][TOTAL] - 1) * min_spacing
            if total_width_needed > width:
                # Need to scale down spacing
                actual_spacing = width / (levels[currentLevel][TOTAL] - 1)
            else:
                actual_spacing = min_spacing

            # Calculate position
            left_margin = (width - total_width_needed) / 2
            pos[node] = (
                left_margin + levels[currentLevel][CURRENT] * actual_spacing,
                vert_loc,
            )

        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(
                    pos, neighbor, currentLevel + 1, node, vert_loc + vert_gap
                )
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})


class AthenaOntology:
    """
    Full OMOP Ontology with parent-child relationships.
    TODO: Optimize for speed and memory usage.
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

    def __len__(self) -> int:
        """Return the number of concepts in the ontology."""
        return len(self.description_map)

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
        cls,
        athena_path: str,
        code_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        ignore_invalid: bool = True,
    ) -> "AthenaOntology":
        """
        Load ontology from Athena snapshot and code metadata.

        Args:
            athena_path: Path to Athena snapshot directory or zip archive (.zip file)
            code_metadata: Optional dictionary mapping codes to metadata including parent codes
            ignore_invalid: If True, skip concepts where invalid_reason is not empty
        """
        print("Load from Athena Vocabulary Snapshot...")
        description_map: Dict[str, str] = {}
        parents_map: Dict[str, Set[str]] = collections.defaultdict(set)
        code_metadata = code_metadata or {}

        try:
            with AthenaFileReader(athena_path) as reader:
                concept_file = reader.read_csv("CONCEPT.csv")

                # Filter out invalid concepts if requested
                if ignore_invalid:
                    concept_file = concept_file.filter(
                        (pl.col("invalid_reason").is_null())
                        | (pl.col("invalid_reason") == "")
                    )

                code_col = pl.col("vocabulary_id") + "/" + pl.col("concept_code")
                description_col = pl.col("concept_name")
                concept_id_col = pl.col("concept_id").cast(pl.Int64)

                processed_concepts = (
                    concept_file.select(
                        [
                            code_col.alias("code"),
                            concept_id_col.alias("concept_id"),
                            description_col.alias("description"),
                            pl.col("standard_concept")
                            .is_null()
                            .alias("is_non_standard"),
                        ]
                    )
                    .collect()
                    .rows()
                )

                concept_id_to_code_map = {}
                non_standard_concepts = set()

                for (
                    code,
                    concept_id,
                    description,
                    is_non_standard,
                ) in processed_concepts:
                    if code and concept_id is not None:
                        concept_id_to_code_map[concept_id] = code
                        if code not in description_map:
                            description_map[code] = description
                        if is_non_standard:
                            non_standard_concepts.add(concept_id)

                # Add OMOP concept_id to description map as OMOP_CONCEPT_ID/concept_id -> concept_name
                df = concept_file.select([concept_id_col, description_col]).collect()

                # Iterate over rows; each row is a tuple (concept_id, description)
                for concept_id, description in df.rows():
                    description_map[f"OMOP_CONCEPT_ID/{concept_id}"] = description

                # Process CONCEPT_RELATIONSHIP.csv
                relationship_file = reader.read_csv("CONCEPT_RELATIONSHIP.csv")
                relationship_file = relationship_file.filter(
                    (pl.col("relationship_id") == "Maps to")
                    & (pl.col("concept_id_1") != pl.col("concept_id_2"))
                )

                for concept_id_1, concept_id_2 in (
                    relationship_file.select(
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
                ancestor_file = reader.read_csv("CONCEPT_ANCESTOR.csv")
                ancestor_file = ancestor_file.filter(
                    pl.col("min_levels_of_separation") == "1"
                )

                for concept_id, parent_concept_id in (
                    ancestor_file.select(
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

    def get_path_to_root(self, code: str, max_depth: int = 100) -> Optional[list[str]]:
        """
        Find the path from a code to the root node.

        Args:
            code: The starting code
            max_depth: Maximum depth to search to prevent infinite loops

        Returns:
            List of codes representing the path from code to root, or None if no path exists
            The path is ordered from the starting code to the root (e.g., [code, parent1, parent2, ..., root])
        """
        if code not in self.description_map:
            return None

        # Use depth-first search to find path to root
        visited = set()
        path = []

        def dfs(current_code: str, depth: int) -> bool:
            if depth > max_depth:
                return False

            if current_code in visited:
                return False

            visited.add(current_code)
            path.append(current_code)

            # Check if we've reached a root (no parents)
            parents = self.get_parents(current_code)
            if not parents:
                return True  # Found root

            # Try each parent
            for parent in parents:
                if dfs(parent, depth + 1):
                    return True

            # If no parent leads to root, backtrack
            path.pop()
            return False

        if dfs(code, 0):
            return path
        else:
            return None

    def get_all_paths_to_root(self, code: str, max_depth: int = 100) -> list[list[str]]:
        """
        Find all possible paths from a code to root nodes.

        Args:
            code: The starting code
            max_depth: Maximum depth to search to prevent infinite loops

        Returns:
            List of paths, where each path is a list of codes from code to a root
        """
        if code not in self.description_map:
            return []

        paths = []
        visited = set()

        def dfs(current_code: str, current_path: list[str], depth: int):
            if depth > max_depth:
                return

            if current_code in visited:
                return

            visited.add(current_code)
            current_path.append(current_code)

            # Check if we've reached a root (no parents)
            parents = self.get_parents(current_code)
            if not parents:
                paths.append(current_path.copy())  # Found a root
            else:
                # Try each parent
                for parent in parents:
                    dfs(parent, current_path, depth + 1)

            # Backtrack
            current_path.pop()
            visited.remove(current_code)

        dfs(code, [], 0)
        return paths

    def get_root_nodes(self) -> Set[str]:
        """
        Get all root nodes in the ontology (nodes with no parents).

        Returns:
            Set of codes that have no parents
        """
        all_codes = set(self.description_map.keys())
        codes_with_parents = set(self.parents_map.keys())
        return all_codes - codes_with_parents

    def get_subgraph_to_roots(self, code: str, max_depth: int = 100) -> nx.DiGraph:
        """
        Create a NetworkX directed graph representing the subgraph from a code to all reachable root nodes.

        Args:
            code: The starting code
            max_depth: Maximum depth to search to prevent infinite loops

        Returns:
            NetworkX DiGraph containing the subgraph with all paths to root nodes
        """
        if code not in self.description_map:
            raise ValueError(f"Code {code} not found in ontology")

        # Create directed graph
        G = nx.DiGraph()

        # Track visited nodes and their depths
        visited = set()
        node_depths = {}

        def add_node_to_graph(current_code: str, depth: int):
            """Recursively add nodes and edges to the graph."""
            if depth > max_depth or current_code in visited:
                return

            visited.add(current_code)
            node_depths[current_code] = depth

            # Add node with metadata
            G.add_node(
                current_code,
                description=self.description_map.get(current_code, ""),
                depth=depth,
                is_root=len(self.get_parents(current_code)) == 0,
            )

            # Add edges to parents
            for parent in self.get_parents(current_code):
                if parent in self.description_map:  # Ensure parent exists
                    G.add_edge(current_code, parent)
                    add_node_to_graph(parent, depth + 1)

        # Start building the graph from the given code
        add_node_to_graph(code, 0)

        # Add node attributes for visualization
        nx.set_node_attributes(G, "starting_node", False)
        G.nodes[code]["starting_node"] = True

        return G

    def visualize_subgraph(
        self,
        code: str,
        max_depth: int = 100,
        layout: str = "hierarchical",
        figsize: tuple = (12, 8),
        show_descriptions: bool = True,
        save_pdf: bool = False,
        pdf_filename: str = None,
    ) -> None:
        """
        Visualize the subgraph from a code to root nodes using matplotlib.

        Args:
            code: The starting code
            max_depth: Maximum depth to search
            layout: Layout algorithm ('hierarchical', 'spring', 'circular')
            figsize: Figure size (width, height)
            show_descriptions: Whether to show descriptions in node labels
            save_pdf: If True, save the plot as a PDF in the current directory
            pdf_filename: Optional filename for the PDF (default: auto-generated)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. Install with: pip install matplotlib"
            )

        G = self.get_subgraph_to_roots(code, max_depth)

        if len(G.nodes()) == 0:
            print(f"No subgraph found for code {code}")
            return

        if layout == "hierarchical":
            fig, ax = self._plot_hierarchical_levels(
                G, code, figsize=figsize, show_descriptions=show_descriptions
            )
            if save_pdf:
                if pdf_filename is None:
                    pdf_filename = f"subgraph_{code.replace('/', '_')}.pdf"
                fig.savefig(pdf_filename, bbox_inches="tight")
                print(f"Saved hierarchical plot to {pdf_filename}")
            plt.show()
            plt.close(fig)
            return
        elif layout == "spring":
            pos = nx.spring_layout(G, k=None, iterations=100, weight=0.5)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        # Prepare node labels
        if show_descriptions:
            labels = {
                node: (
                    f"{node}\n{G.nodes[node]['description'][:30]}..."
                    if len(G.nodes[node]["description"]) > 30
                    else f"{node}\n{G.nodes[node]['description']}"
                )
                for node in G.nodes()
            }
        else:
            labels = {node: node for node in G.nodes()}
        # Color nodes based on their role
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get("starting_node", False):
                node_colors.append("red")  # Starting node
            elif G.nodes[node].get("is_root", False):
                node_colors.append("green")  # Root nodes
            else:
                node_colors.append("lightblue")  # Intermediate nodes
        plt.figure(figsize=figsize, constrained_layout=True)
        nx.draw(
            G,
            pos,
            node_color=node_colors,
            node_size=1000,
            font_size=6,
            font_weight="normal",
            labels=labels,
            arrows=True,
            edge_color="gray",
            width=1.5,
            with_labels=True,
        )
        legend_elements = [
            mpatches.Patch(color="red", label="Starting Node"),
            mpatches.Patch(color="green", label="Root Nodes"),
            mpatches.Patch(color="lightblue", label="Intermediate Nodes"),
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        plt.gca().invert_yaxis()
        plt.title(
            f"Subgraph from {code} to Root Nodes\n"
            f"Total nodes: {len(G.nodes())}, Total edges: {len(G.edges())}"
        )
        if save_pdf:
            if pdf_filename is None:
                pdf_filename = f"subgraph_{code.replace('/', '_')}.pdf"
            plt.savefig(pdf_filename, bbox_inches="tight")
            print(f"Saved plot to {pdf_filename}")
        plt.show()
        plt.close()

    def _plot_hierarchical_levels(
        self,
        G,
        start_code,
        figsize=(16, 8),
        node_size=1000,
        font_size=8,
        show_descriptions=True,
    ):
        """
        Plot a layered hierarchical graph with levels assigned by distance from the start_code.
        The start_code is at the bottom, roots at the top.
        Returns (fig, ax) for further handling (show/save) by the caller.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        # Assign levels: BFS from start_code, each parent is one level higher
        levels = {start_code: 0}
        queue = [(start_code, 0)]
        while queue:
            node, level = queue.pop(0)
            for parent in G.successors(node):  # edges go from child to parent
                if parent not in levels or levels[parent] < level + 1:
                    levels[parent] = level + 1
                    queue.append((parent, level + 1))
        # Invert levels so roots are at the top
        max_level = max(levels.values())
        for node in levels:
            levels[node] = max_level - levels[node]

        # Group nodes by level
        level_nodes = {}
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)
        # Assign positions
        pos = {}
        for lvl, nodes in level_nodes.items():
            n = len(nodes)
            for i, node in enumerate(sorted(nodes)):
                # Spread nodes equally along x-axis
                pos[node] = (i - (n - 1) / 2, lvl)
        # Prepare node labels
        if show_descriptions:
            labels = {
                node: (
                    f"{node}\n{G.nodes[node].get('description','')[:30]}..."
                    if len(G.nodes[node].get("description", "")) > 30
                    else f"{node}\n{G.nodes[node].get('description','')}"
                )
                for node in G.nodes()
            }
        else:
            labels = {node: node for node in G.nodes()}
        # Color nodes based on their role
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get("starting_node", False):
                node_colors.append("red")  # Starting node
            elif G.nodes[node].get("is_root", False):
                node_colors.append("green")  # Root nodes
            else:
                node_colors.append("skyblue")  # Intermediate nodes
        # Plot
        nx.draw(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=node_size,
            font_size=font_size,
            node_color=node_colors,
            edge_color="gray",
            font_weight="normal",
            arrows=True,
            linewidths=1,
            labels=labels,
        )
        ax.invert_yaxis()  # So roots are at the top
        ax.set_title(f"Hierarchical plot from {start_code} to roots")
        return fig, ax

    def get_subgraph_statistics(
        self, code: str, max_depth: int = 100
    ) -> Dict[str, Any]:
        """
        Get statistics about the subgraph from a code to root nodes.

        Args:
            code: The starting code
            max_depth: Maximum depth to search

        Returns:
            Dictionary containing subgraph statistics
        """
        G = self.get_subgraph_to_roots(code, max_depth)

        if len(G.nodes()) == 0:
            return {"error": f"No subgraph found for code {code}"}

        # Count different types of nodes
        starting_nodes = [
            n for n in G.nodes() if G.nodes[n].get("starting_node", False)
        ]
        root_nodes = [n for n in G.nodes() if G.nodes[n].get("is_root", False)]
        intermediate_nodes = [
            n
            for n in G.nodes()
            if not G.nodes[n].get("starting_node", False)
            and not G.nodes[n].get("is_root", False)
        ]

        # Calculate path statistics
        all_paths = self.get_all_paths_to_root(code, max_depth)
        path_lengths = [len(path) for path in all_paths]

        return {
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "starting_nodes": len(starting_nodes),
            "root_nodes": len(root_nodes),
            "intermediate_nodes": len(intermediate_nodes),
            "max_depth": max(G.nodes[n]["depth"] for n in G.nodes()),
            "num_paths_to_root": len(all_paths),
            "shortest_path_length": min(path_lengths) if path_lengths else 0,
            "longest_path_length": max(path_lengths) if path_lengths else 0,
            "average_path_length": (
                sum(path_lengths) / len(path_lengths) if path_lengths else 0
            ),
            "root_node_codes": root_nodes,
            "starting_node_code": starting_nodes[0] if starting_nodes else None,
        }


if __name__ == "__main__":

    ontology = AthenaOntology.load_from_parquet("data/athena_omop_ontologies")
    print("Total concepts:", len(ontology))

    code = "SNOMED/363358000"  # Malignant tumor of lung
    print("Description:", ontology.get_description(code))
    print("Parents:", ontology.get_parents(code))
    print("Children:", ontology.get_children(code))
    print("Path to root:", ontology.get_path_to_root(code))
    print("All paths to root:")
    for path in ontology.get_all_paths_to_root(code):
        for pcode in path:
            print(f" {pcode} -> {ontology.get_description(pcode)}")
        print("-" * 10)

    # print("Root nodes:", len(ontology.get_root_nodes()))
    print("Subgraph to root nodes:", len(ontology.get_subgraph_to_roots(code)))
    # ontology.visualize_subgraph(code, max_depth=100, layout="spring")
    ontology.visualize_subgraph(code, max_depth=100, layout="hierarchical")

    print("Subgraph statistics:", ontology.get_subgraph_statistics(code))
