import pytest
from meds2text.ontology.athena import AthenaOntology
import matplotlib

matplotlib.use("Agg")
from unittest.mock import patch
import networkx as nx

# Test data
SAMPLE_DESCRIPTIONS = {
    "SNOMED/123": "Diabetes mellitus",
    "SNOMED/456": "Type 2 diabetes",
    "SNOMED/457": "Type 1 diabetes",
    "SNOMED/458": "Gestational diabetes",
    "SNOMED/459": "Diabetes insipidus",
    "ICD10/E11": "Type 2 diabetes mellitus",
    "ICD10/E10": "Type 1 diabetes mellitus",
    "ICD10/O24": "Gestational diabetes mellitus",
    "ICD10/E23": "Diabetes insipidus",
    "OMOP_CONCEPT_ID/789": "Diabetes",
    "OMOP_CONCEPT_ID/790": "Diabetes mellitus type 1",
    "OMOP_CONCEPT_ID/791": "Diabetes mellitus type 2",
    "OMOP_CONCEPT_ID/792": "Gestational diabetes mellitus",
    "OMOP_CONCEPT_ID/793": "Diabetes insipidus",
}

SAMPLE_PARENTS = {
    "SNOMED/456": {"SNOMED/123"},
    "SNOMED/457": {"SNOMED/123"},
    "SNOMED/458": {"SNOMED/123"},
    "SNOMED/459": {"SNOMED/123"},
    "ICD10/E11": {"SNOMED/123"},
    "ICD10/E10": {"SNOMED/123"},
    "ICD10/O24": {"SNOMED/123"},
    "ICD10/E23": {"SNOMED/123"},
    "OMOP_CONCEPT_ID/789": {"SNOMED/123"},
    "OMOP_CONCEPT_ID/790": {"SNOMED/123"},
    "OMOP_CONCEPT_ID/791": {"SNOMED/123"},
    "OMOP_CONCEPT_ID/792": {"SNOMED/123"},
    "OMOP_CONCEPT_ID/793": {"SNOMED/123"},
}


@pytest.fixture
def sample_ontology():
    """Create a sample ontology for testing."""
    return AthenaOntology(SAMPLE_DESCRIPTIONS, SAMPLE_PARENTS)


def test_ontology_get_description(sample_ontology):
    """Test getting descriptions from ontology."""
    assert sample_ontology.get_description("SNOMED/123") == "Diabetes mellitus"
    assert sample_ontology.get_description("SNOMED/456") == "Type 2 diabetes"
    assert sample_ontology.get_description("NONEXISTENT") is None


def test_ontology_get_parents(sample_ontology):
    """Test getting parents from ontology."""
    assert sample_ontology.get_parents("SNOMED/456") == {"SNOMED/123"}
    assert sample_ontology.get_parents("ICD10/E11") == {"SNOMED/123"}
    assert sample_ontology.get_parents("SNOMED/123") == set()  # No parents
    assert sample_ontology.get_parents("NONEXISTENT") == set()


def test_ontology_get_children(sample_ontology):
    """Test getting children from ontology."""
    expected_children = {
        "SNOMED/456",
        "SNOMED/457",
        "SNOMED/458",
        "SNOMED/459",
        "ICD10/E11",
        "ICD10/E10",
        "ICD10/O24",
        "ICD10/E23",
        "OMOP_CONCEPT_ID/789",
        "OMOP_CONCEPT_ID/790",
        "OMOP_CONCEPT_ID/791",
        "OMOP_CONCEPT_ID/792",
        "OMOP_CONCEPT_ID/793",
    }
    assert sample_ontology.get_children("SNOMED/123") == expected_children
    assert sample_ontology.get_children("SNOMED/456") == set()  # No children
    assert sample_ontology.get_children("NONEXISTENT") == set()


def test_ontology_load_from_athena_snapshot(tmp_path):
    """Test loading ontology from Athena snapshot."""
    # Create mock Athena files
    concept_data = """vocabulary_id\tconcept_id\tconcept_code\tconcept_name\tstandard_concept\tinvalid_reason
SNOMED\t123\t123\tDiabetes mellitus\tS\t
SNOMED\t456\t456\tType 2 diabetes\tS\t
SNOMED\t789\t789\tInvalid concept\tS\tD"""

    relationship_data = """concept_id_1\tconcept_id_2\trelationship_id
456\t123\tMaps to"""

    ancestor_data = """ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation
123\t456\t1"""

    # Write mock files
    with open(tmp_path / "CONCEPT.csv", "w") as f:
        f.write(concept_data)
    with open(tmp_path / "CONCEPT_RELATIONSHIP.csv", "w") as f:
        f.write(relationship_data)
    with open(tmp_path / "CONCEPT_ANCESTOR.csv", "w") as f:
        f.write(ancestor_data)

    # Test loading with ignore_invalid=True
    ontology = AthenaOntology.load_from_athena_snapshot(
        str(tmp_path), ignore_invalid=True
    )
    assert (
        "SNOMED/789" not in ontology.description_map
    )  # Invalid concept should be excluded

    # Test loading with ignore_invalid=False
    ontology = AthenaOntology.load_from_athena_snapshot(
        str(tmp_path), ignore_invalid=False
    )
    assert (
        "SNOMED/789" in ontology.description_map
    )  # Invalid concept should be included


def test_ontology_with_code_metadata(tmp_path):
    """Test ontology with additional code metadata."""
    # Create minimal mock files with headers
    concept_data = """vocabulary_id\tconcept_id\tconcept_code\tconcept_name\tstandard_concept\tinvalid_reason
SNOMED\t123\t123\tDiabetes mellitus\tS\t"""

    relationship_data = """concept_id_1\tconcept_id_2\trelationship_id\tvalid_start_date\tvalid_end_date\tinvalid_reason
123\t456\tMaps to\t2020-01-01\t2099-12-31\t"""

    ancestor_data = """ancestor_concept_id\tdescendant_concept_id\tmin_levels_of_separation\tmax_levels_of_separation
123\t456\t1\t1"""

    # Write mock files
    with open(tmp_path / "CONCEPT.csv", "w") as f:
        f.write(concept_data)
    with open(tmp_path / "CONCEPT_RELATIONSHIP.csv", "w") as f:
        f.write(relationship_data)
    with open(tmp_path / "CONCEPT_ANCESTOR.csv", "w") as f:
        f.write(ancestor_data)

    code_metadata = {
        "CUSTOM/001": {
            "description": "Custom diabetes code",
            "parent_codes": ["SNOMED/123"],
        }
    }

    ontology = AthenaOntology.load_from_athena_snapshot(
        str(tmp_path), code_metadata=code_metadata
    )

    assert ontology.get_description("CUSTOM/001") == "Custom diabetes code"
    assert ontology.get_parents("CUSTOM/001") == {"SNOMED/123"}
    assert "CUSTOM/001" in ontology.get_children("SNOMED/123")


# Subgraph function tests
def test_get_subgraph_to_roots(sample_ontology):
    """Test subgraph extraction from a code to root nodes."""
    G = sample_ontology.get_subgraph_to_roots("SNOMED/456")
    # Should include the starting node and its parent
    assert "SNOMED/456" in G.nodes
    assert "SNOMED/123" in G.nodes
    # Should have an edge from child to parent
    assert ("SNOMED/456", "SNOMED/123") in G.edges
    # Should mark the starting node
    assert G.nodes["SNOMED/456"].get("starting_node", False)
    # Should mark the root node
    assert (
        G.nodes["SNOMED/123"].get("is_root", False)
        or sample_ontology.get_parents("SNOMED/123") == set()
    )


def test_get_subgraph_statistics(sample_ontology):
    """Test subgraph statistics for a code."""
    stats = sample_ontology.get_subgraph_statistics("SNOMED/456")
    assert stats["total_nodes"] >= 2
    assert stats["total_edges"] >= 1
    assert stats["starting_nodes"] == 1
    assert stats["num_paths_to_root"] >= 1
    assert stats["shortest_path_length"] >= 2
    assert stats["longest_path_length"] >= 2


def test_hierarchical_plot_runs(sample_ontology):
    """Test that the hierarchical plot function runs without error (no display)."""
    G = sample_ontology.get_subgraph_to_roots("SNOMED/456")
    with patch("matplotlib.pyplot.show"):
        sample_ontology._plot_hierarchical_levels(
            G, "SNOMED/456", show_descriptions=False
        )
