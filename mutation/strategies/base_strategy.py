# automl_lib/mutation/strategies/base_strategy.py
"""
Defines the abstract base class for all mutation strategies.
"""

import ast
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

KnowledgeGraph = Any

class BaseStrategy(ABC):
    """
    The interface for a mutation strategy.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def mutate(
        self,
        source_ast: ast.AST,
        hparams: Dict[str, Any],
        knowledge_graph: Optional[KnowledgeGraph] = None
    ) -> tuple[ast.AST, Dict[str, Any]]:
        """
        Applies a transformation and returns the new state.

        A strategy can modify the AST, the hyperparameters, or both.

        Args:
            source_ast: The original Abstract Syntax Tree of the script.
            hparams: The dictionary of hyperparameters to be mutated.
            knowledge_graph: The knowledge graph for intelligent mutations.

        Returns:
            A tuple containing the (potentially modified) AST and the
            (potentially modified) hyperparameter dictionary.
        """
        pass

    def __repr__(self) -> str:
        return f"<MutationStrategy: {self.name}>"