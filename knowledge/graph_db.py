# automl_lib/knowledge/graph_db.py
"""
An in-memory graph database for storing and querying ML knowledge.

This implementation uses the `networkx` library to represent the graph and
provides an interface for adding components and finding compatible alternatives.
"""
import logging
import networkx as nx
from typing import List, Optional

from .ontology import Node, ComponentNode, Relationship

class KnowledgeGraph:
    """A graph-based store for ML component information and relationships."""

    def __init__(self):
        self.graph = nx.DiGraph()
        logging.info("KnowledgeGraph initialized with an in-memory MultiDiGraph.")

    def add_node(self, node: Node):
        """Adds a node to the graph with its attributes."""
        if self.graph.has_node(node.uid):
            logging.warning(f"Node '{node.uid}' already exists. Updating attributes.")
        self.graph.add_node(node.uid, **node.__dict__)

    def add_relationship(self, relationship: Relationship):
        """Adds a directed edge between two nodes."""
        self.graph.add_edge(
            relationship.source_uid,
            relationship.target_uid,
            label=relationship.label,
            **relationship.properties
        )

    def find_compatible_components(self, component_uid: str, relationship_label: str = "is_a") -> List[ComponentNode]:
        """
        Finds all components that share a common target via a specific relationship.

        This is the core method for finding valid substitutes. For example, it can find
        all components that are also a "Classifier".

        Args:
            component_uid: The UID of the source component.
            relationship_label: The type of relationship to follow (defaults to "is_a").

        Returns:
            A list of compatible ComponentNode objects (excluding the source component).
        """
        compatible_components = []
        if not self.graph.has_node(component_uid):
            return []

        # Find the common target(s) of the source component (e.g., the "Classifier" concept)
        targets = [
            target for _, target, data in self.graph.out_edges(component_uid, data=True)
            if data.get("label") == relationship_label
        ]

        if not targets:
            return []

        # Find all other nodes that also have the same relationship to the same target(s)
        for target in targets:
            for source, _, data in self.graph.in_edges(target, data=True):
                if data.get("label") == relationship_label and source != component_uid:
                    node_data = self.graph.nodes[source]
                    # Ensure it's a ComponentNode before adding
                    if 'component_type' in node_data:
                         compatible_components.append(ComponentNode(**node_data))
        
        # Remove duplicates
        return list({c.uid: c for c in compatible_components}.values())

# --- Self-contained Test Block ---
if __name__ == '__main__':
    from .ontology import ComponentNode
    
    print("\n--- Running Test for graph_db.py ---")

    kg = KnowledgeGraph()
    
    # 1. Define and add some nodes (concrete components and abstract concepts)
    rf_node = ComponentNode(uid="sklearn.ensemble.RandomForestClassifier", name="RF", component_type="Classifier")
    xgb_node = ComponentNode(uid="xgboost.XGBClassifier", name="XGB", component_type="Classifier")
    scaler_node = ComponentNode(uid="sklearn.preprocessing.StandardScaler", name="Scaler", component_type="Transformer")
    
    # Abstract concept nodes (don't need to be ComponentNode)
    classifier_concept = Node(uid="concept.classifier", name="Classifier Concept")
    transformer_concept = Node(uid="concept.transformer", name="Transformer Concept")
    
    kg.add_node(rf_node)
    kg.add_node(xgb_node)
    kg.add_node(scaler_node)
    kg.add_node(classifier_concept)
    kg.add_node(transformer_concept)
    print("✅ Added nodes to the graph.")

    # 2. Define and add relationships
    kg.add_relationship(Relationship(source_uid=rf_node.uid, target_uid=classifier_concept.uid, label="is_a"))
    kg.add_relationship(Relationship(source_uid=xgb_node.uid, target_uid=classifier_concept.uid, label="is_a"))
    kg.add_relationship(Relationship(source_uid=scaler_node.uid, target_uid=transformer_concept.uid, label="is_a"))
    print("✅ Added relationships to the graph.")

    # 3. Test the core "find_compatible_components" method
    print("\n--- Testing compatibility search ---")
    
    # Find all components that are also a "Classifier" like RandomForest
    compatible = kg.find_compatible_components(rf_node.uid)
    
    assert len(compatible) == 1, f"Expected 1 compatible component, but found {len(compatible)}"
    assert compatible[0].uid == xgb_node.uid, "Did not find the correct compatible component."
    print(f"✅ Found compatible components for '{rf_node.name}': {[c.name for c in compatible]}")

    # Test with a component that has no compatible alternatives in our graph
    compatible_for_scaler = kg.find_compatible_components(scaler_node.uid)
    assert len(compatible_for_scaler) == 0
    print(f"✅ Correctly found 0 compatible components for '{scaler_node.name}'")

    print("\n--- All graph_db.py tests passed! ---")