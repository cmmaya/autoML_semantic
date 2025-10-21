# automl_lib/mutation/strategies/gp_primitives.py
"""
Defines the core primitives for Genetic Programming based feature engineering.
"""
import random
import operator
import numpy as np
import pandas as pd
from typing import List, Callable, Optional

class FeatureNode:
    """A node in the feature engineering tree (an operation or a terminal)."""
    def __init__(self, name: str, operation: Callable, arity: int):
        self.name = name
        self.operation = operation
        self.arity = arity
        self.children: List['FeatureNode'] = []

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Recursively evaluates the node and its children."""
        child_results = [child.evaluate(data) for child in self.children]
        if self.arity == 0: # Terminal node (a feature column)
            return self.operation(data)
        return self.operation(*child_results)
    
    def __str__(self) -> str:
        """Provides a string representation of the tree."""
        if self.arity == 0:
            return self.name
        child_strs = ", ".join(str(c) for c in self.children)
        return f"{self.name}({child_strs})"

class FeatureTree:
    """Represents a full feature engineering pipeline as a tree."""
    def __init__(self, function_set: list, terminal_set: list, max_depth: int):
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.max_depth = max_depth
        self.root: Optional[FeatureNode] = None

    def build_random_tree(self, depth=0, method='full'):
        """Builds a random tree using either the 'full' or 'grow' method."""
        if depth == self.max_depth or (method == 'grow' and random.random() < 0.5):
            # Choose a terminal node
            op_name, op_fn = random.choice(self.terminal_set)
            node = FeatureNode(op_name, op_fn, 0)
        else:
            # Choose a function node
            op_name, op_fn, arity = random.choice(self.function_set)
            node = FeatureNode(op_name, op_fn, arity)
            for _ in range(arity):
                node.children.append(self.build_random_tree(depth + 1, method))
        
        if depth == 0:
            self.root = node
        return node
    
    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        if not self.root:
            raise ValueError("Tree has not been built yet.")
        return self.root.evaluate(data)

    def __str__(self) -> str:
        return str(self.root)

# --- Self-contained Test Block ---
if __name__ == '__main__':
    print("\n--- Running Test for gp_primitives.py ---")
    
    # 1. Define function and terminal sets
    def protected_log(x):
        return np.log(np.abs(x) + 1e-6)

    FUNCTIONS = [
        ('add', operator.add, 2),
        ('sub', operator.sub, 2),
        ('mul', operator.mul, 2),
        ('log', protected_log, 1)
    ]
    TERMINALS = [
        ('Age', lambda df: df['Age'].values, 0),
        ('Fare', lambda df: df['Fare'].values, 0)
    ]
    
    # 2. Create and build a random tree
    tree = FeatureTree(function_set=FUNCTIONS, terminal_set=TERMINALS, max_depth=3)
    tree.build_random_tree()
    print(f"✅ Successfully built a random feature tree: {tree}")

    # 3. Test evaluation
    test_data = pd.DataFrame({'Age': [30, 40], 'Fare': [100, 200]})
    try:
        result = tree.evaluate(test_data)
        assert isinstance(result, np.ndarray) and result.shape == (2,)
        print(f"✅ Tree evaluation successful. Result: {result}")
    except Exception as e:
        print(f"❌ Tree evaluation failed: {e}")

    print("\n--- All gp_primitives.py tests passed! ---")