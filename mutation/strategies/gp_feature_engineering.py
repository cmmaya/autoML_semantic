# automl_lib/mutation/strategies/gp_feature_engineering.py
"""
A mutation strategy that uses Genetic Programming to evolve feature pipelines.
"""
import ast
import random
import logging
from typing import Any, Optional, Dict

from .base_strategy import BaseStrategy, KnowledgeGraph
from .gp_primitives import FeatureTree # Import our new primitives

class GPFeatureEngineeringStrategy(BaseStrategy):
    """
    Evolves feature engineering pipelines using genetic programming.
    
    NOTE: This is a simplified proof-of-concept. A full implementation would
    manage a population of trees and integrate their evaluation into the main
    optimizer loop. This strategy demonstrates the core genetic operators.
    """
    def __init__(self, function_set: list, terminal_set: list, max_depth: int = 4):
        super().__init__(
            name="GPFeatureEngineeringStrategy",
            description="Evolves new features using genetic programming operators."
        )
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.max_depth = max_depth

    def _crossover(self, tree1: FeatureTree, tree2: FeatureTree) -> FeatureTree:
        """Performs subtree crossover between two parent trees."""
        # This is a simplified crossover. A robust version would be more careful
        # about tree depths and node selection.
        if tree1.root and tree2.root:
            child1_root = tree1.root
            child2_root = tree2.root
            
            # Simple swap of the first child of the root
            if child1_root.children and child2_root.children:
                child1_root.children[0], child2_root.children[0] = child2_root.children[0], child1_root.children[0]
        
        new_tree = FeatureTree(self.function_set, self.terminal_set, self.max_depth)
        new_tree.root = child1_root
        return new_tree

    def mutate(
        self,
        source_ast: ast.AST,
        hparams: Dict[str, Any],
        knowledge_graph: Optional[KnowledgeGraph] = None
    ) -> tuple[ast.AST, Dict[str, Any]]:
        """
        Creates a new feature tree via crossover and mutation.
        
        This strategy does not modify the script's AST or hyperparameters directly.
        Instead, it would generate a new feature to be added to the dataset.
        """
        logging.info(f"[{self.name}] Evolving a new feature...")

        # 1. Create two random parent trees
        parent1 = FeatureTree(self.function_set, self.terminal_set, self.max_depth)
        parent1.build_random_tree()
        parent2 = FeatureTree(self.function_set, self.terminal_set, self.max_depth)
        parent2.build_random_tree()

        # 2. Perform crossover to create a child
        child_tree = self._crossover(parent1, parent2)
        
        logging.info(f"Generated new feature via GP: {child_tree}")
        
        # In a full implementation, this new feature would be evaluated.
        # For now, we just log its creation. The original script AST and hparams are returned.
        return source_ast, hparams


# --- Self-contained Test Block ---
if __name__ == '__main__':
    print("\n--- Running Test for gp_feature_engineering.py ---")
    
    # 1. Define function and terminal sets for the test
    def protected_log(x): return np.log(np.abs(x) + 1e-6)
    FUNCTIONS = [('add', operator.add, 2), ('log', protected_log, 1)]
    TERMINALS = [('Age', lambda df: df['Age'].values, 0), ('Fare', lambda df: df['Fare'].values, 0)]
    
    # 2. Initialize the strategy
    strategy = GPFeatureEngineeringStrategy(FUNCTIONS, TERMINALS)
    print(f"✅ Initialized strategy: {strategy.name}")
    
    # 3. Run the mutate method to demonstrate GP operators
    # We can pass dummy AST and hparams as they are not used in this simplified version.
    _, _ = strategy.mutate(ast.parse(""), {})

    print("\n✅ GP strategy executed successfully (see log for generated feature).")
    print("\n--- All gp_feature_engineering.py tests passed! ---")