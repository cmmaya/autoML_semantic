# automl_lib/knowledge/ontology.py
"""
Defines the foundational ontology (schema) for the Knowledge Graph.

This includes the types of nodes (e.g., Component, DataType) and edges
(e.g., Relationship) that can exist in the graph.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List

# --- Node Definitions ---

@dataclass
class Node:
    """Abstract base class for all nodes in the Knowledge Graph."""
    uid: str  # Unique identifier, e.g., "sklearn.ensemble.RandomForestClassifier"
    name: str # Human-readable name, e.g., "RandomForestClassifier"

@dataclass
class DataTypeNode(Node):
    """Represents a type of data, e.g., Tabular, Image."""
    description: str

@dataclass
class ComponentNode(Node):
    """Represents an ML component, like a model or transformer."""
    component_type: str # e.g., "Classifier", "Transformer"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

# --- Edge Definition ---

@dataclass
class Relationship:
    """Represents a directed edge between two nodes in the Knowledge Graph."""
    source_uid: str
    target_uid: str
    label: str # e.g., "is_a", "accepts_input", "produces_output"
    properties: Dict[str, Any] = field(default_factory=dict)

# --- Self-contained Test Block ---
if __name__ == '__main__':
    print("\n--- Running Test for ontology.py ---")

    # 1. Create a data type node
    tabular_data = DataTypeNode(
        uid="type.tabular",
        name="Tabular",
        description="Data organized in rows and columns."
    )
    assert tabular_data.uid == "type.tabular"
    print(f"✅ Created DataTypeNode: {tabular_data.name}")

    # 2. Create a component node
    rf_classifier = ComponentNode(
        uid="sklearn.ensemble.RandomForestClassifier",
        name="RandomForestClassifier",
        component_type="Classifier",
        hyperparameters={'n_estimators': 100, 'max_depth': None}
    )
    assert rf_classifier.component_type == "Classifier"
    print(f"✅ Created ComponentNode: {rf_classifier.name}")

    # 3. Create a relationship edge
    is_a_relationship = Relationship(
        source_uid=rf_classifier.uid,
        target_uid="concept.classifier", # Points to an abstract concept node
        label="is_a"
    )
    assert is_a_relationship.label == "is_a"
    print(f"✅ Created Relationship: {is_a_relationship.source_uid} -> {is_a_relationship.target_uid}")

    print("\n--- All ontology.py tests passed! ---")