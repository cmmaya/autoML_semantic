# automl_lib/mutation/engine.py
"""
The Mutation Engine manages and applies various mutation strategies.
"""
import ast
import random
import logging
from typing import List, Dict, Any, Optional

from .strategies.base_strategy import BaseStrategy, KnowledgeGraph


class MutationEngine:
    """
    Manages a collection of mutation strategies and applies them to generate new candidates.
    """
    def __init__(self, strategies: List[BaseStrategy]):
        if not strategies:
            raise ValueError("MutationEngine requires at least one strategy.")
        self.strategies = strategies
        logging.info(f"MutationEngine initialized with {len(self.strategies)} strategies.")

    def mutate(
        self,
        source_ast: ast.AST,
        hparams: Dict[str, Any],
        knowledge_graph: Optional[KnowledgeGraph] = None
    ) -> tuple[ast.AST, Dict[str, Any]]:
        """
        Selects and applies a mutation strategy.

        In the future, this selection will be guided by a multi-armed bandit.
        For now, it selects a strategy at random.
        """
        strategy_to_use = random.choice(self.strategies)
        logging.debug(f"Selected mutation strategy: {strategy_to_use.name}")
        
        return strategy_to_use.mutate(source_ast, hparams, knowledge_graph)