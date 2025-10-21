# automl_lib/mutation/strategies/hyperparameters.py
"""
A concrete mutation strategy for randomly perturbing hyperparameters.
"""
import ast
import copy
import random
import logging
from typing import Any, Optional, Dict

from .base_strategy import BaseStrategy, KnowledgeGraph


class RandomHyperparameterMutation(BaseStrategy):
    """
    A simple mutation strategy that randomly perturbs one hyperparameter.
    """
    def __init__(self):
        super().__init__(
            name="RandomHyperparameterMutation",
            description="Randomly perturbs a single numerical or boolean hyperparameter."
        )

    def mutate(
        self,
        source_ast: ast.AST,
        hparams: Dict[str, Any],
        knowledge_graph: Optional[KnowledgeGraph] = None
    ) -> tuple[ast.AST, Dict[str, Any]]:
        """
        Applies a random perturbation to one hyperparameter.

        This strategy does not modify the AST.
        """
        mutated_hparams = copy.deepcopy(hparams)
        if not mutated_hparams:
            return source_ast, mutated_hparams # Return unchanged

        param_to_mutate = random.choice(list(mutated_hparams.keys()))
        current_value = mutated_hparams[param_to_mutate]

        if isinstance(current_value, (int, float)):
            noise = random.uniform(-0.2, 0.2) * current_value + random.uniform(-1e-4, 1e-4)
            new_value = current_value + noise
            if isinstance(current_value, int):
                # Clamp to a minimum of 1 if it's an integer parameter
                mutated_hparams[param_to_mutate] = max(1, int(new_value))
            else:
                mutated_hparams[param_to_mutate] = new_value

        elif isinstance(current_value, bool):
            mutated_hparams[param_to_mutate] = not current_value

        logging.debug(
            f"[{self.name}] Mutated '{param_to_mutate}': "
            f"{current_value} -> {mutated_hparams[param_to_mutate]}"
        )
        
        # The AST is not changed by this strategy, so we return it as is.
        return source_ast, mutated_hparams