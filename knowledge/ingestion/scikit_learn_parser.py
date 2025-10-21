# automl_lib/knowledge/ingestion/scikit_learn_parser.py
"""
A concrete parser for the Scikit-learn library.

This parser programmatically inspects sklearn submodules to discover
classifiers, regressors, transformers, and their hyperparameters.
"""
import inspect
import pkgutil
import logging
from typing import Iterator, Tuple, Any, Dict

import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

from .base_parser import BaseParser
from knowledge.ontology import Node, ComponentNode, Relationship, DataTypeNode

class ScikitLearnParser(BaseParser):
    """Inspects the Scikit-learn library to build a knowledge graph."""

    def __init__(self):
        # Define the abstract concepts and data types relevant to sklearn
        self.tabular_data = DataTypeNode(uid="type.tabular", name="Tabular", description="Row/column data.")
        self.classifier_concept = Node(uid="concept.classifier", name="Classifier Concept")
        self.regressor_concept = Node(uid="concept.regressor", name="Regressor Concept")
        self.transformer_concept = Node(uid="concept.transformer", name="Transformer Concept")
        self.concepts = [self.classifier_concept, self.regressor_concept, self.transformer_concept]

    def _get_component_type(self, cls: type) -> str:
        """Determines the component type based on its parent mixins."""
        if issubclass(cls, ClassifierMixin):
            return "Classifier"
        if issubclass(cls, RegressorMixin):
            return "Regressor"
        if issubclass(cls, TransformerMixin):
            return "Transformer"
        return "Estimator" # Generic fallback

    def _get_hyperparameters(self, cls: type) -> Dict[str, Any]:
        """Extracts __init__ parameters as default hyperparameters."""
        try:
            sig = inspect.signature(cls.__init__)
            hparams = {}
            for param in sig.parameters.values():
                if param.name not in ['self', 'args', 'kwargs']:
                    hparams[param.name] = param.default if param.default is not inspect.Parameter.empty else None
            return hparams
        except (ValueError, TypeError):
            return {} # Handles objects with no standard __init__ (e.g., some functions)

    def parse(self) -> Iterator[Tuple[Node, list[Relationship]]]:
        """Walks through sklearn packages and yields components."""
        logging.info("Starting Scikit-learn library parsing...")

        # Yield the foundational data types and concepts first
        yield self.tabular_data, []
        for concept in self.concepts:
            yield concept, []

        # Walk all submodules of the sklearn package
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=sklearn.__path__, prefix=sklearn.__name__ + '.'):
            try:
                module = __import__(modname, fromlist="dummy")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # We only care about classes that are estimators and defined in sklearn
                    if issubclass(obj, BaseEstimator) and obj.__module__ == modname:
                        comp_type = self._get_component_type(obj)
                        uid = f"{modname}.{name}"

                        node = ComponentNode(
                            uid=uid,
                            name=name,
                            component_type=comp_type,
                            hyperparameters=self._get_hyperparameters(obj)
                        )
                        
                        relationships = []
                        # Relate component to its abstract concept
                        if comp_type == "Classifier":
                            relationships.append(Relationship(uid, self.classifier_concept.uid, "is_a"))
                        elif comp_type == "Regressor":
                            relationships.append(Relationship(uid, self.regressor_concept.uid, "is_a"))
                        elif comp_type == "Transformer":
                             relationships.append(Relationship(uid, self.transformer_concept.uid, "is_a"))
                        
                        # All sklearn components accept and produce tabular data
                        relationships.append(Relationship(uid, self.tabular_data.uid, "accepts_input"))
                        relationships.append(Relationship(uid, self.tabular_data.uid, "produces_output"))

                        yield node, relationships

            except Exception as e:
                # Some modules may fail to import, which is fine
                logging.debug(f"Could not import or inspect module {modname}: {e}")
        
        logging.info("Finished Scikit-learn library parsing.")