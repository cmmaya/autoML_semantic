# automl_lib/orchestration/optimizer.py
"""
The main orchestrator for the AutoML process.

This module contains the Optimizer class, which manages the entire evolutionary
optimization loop, from analysis and execution to intelligent strategy selection.
"""

import ast
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any

# Import the components we've already built
from analysis.analyzer import Analyzer, AnalysisResult
from execution.sandbox import SandboxExecutor, ExecutionResult
from mutation.strategies.hyperparameters import RandomHyperparameterMutation
from mutation.strategies.structure import StructuralSwapStrategy # <-- 1. IMPORT
from orchestration.bandit import UCB1Bandit

# --- Configuration ---

@dataclass
class OptimizerConfig:
    """Configuration for the optimization process."""
    num_generations: int = 10
    population_size: int = 5

# --- Main Optimizer Class ---

class Optimizer:
    """
    Manages the end-to-end optimization of a user-provided script.
    """
    def __init__(
        self,
        script_path: Union[str, Path],
        data_path: Union[str, Path],
        config: OptimizerConfig,
        metric_to_optimize: str,
        optimization_goal: str = 'maximize'
    ):
        """
        Initializes the Optimizer.

        Args:
            script_path: Path to the user's script.
            data_path: Path to the dataset.
            config: Configuration object for the optimization run.
            metric_to_optimize: The key in the returned metrics dict to optimize.
            optimization_goal: 'maximize' or 'minimize'.
        """
        self.script_path = Path(script_path)
        self.data_path = Path(data_path)
        self.config = config
        self.metric_to_optimize = metric_to_optimize
        self.optimization_goal = optimization_goal
        
        # The optimizer owns its tools
        self.analyzer = Analyzer(self.script_path)
        self.executor = SandboxExecutor(timeout_seconds=60)
        
        # Instantiate available mutation strategies
        hparam_strategy = RandomHyperparameterMutation()
        structural_strategy = StructuralSwapStrategy() # <-- 2. INSTANTIATE

        
        self.strategies = [hparam_strategy, structural_strategy] # <-- 3. ADD TO LIST
        self.bandit = UCB1Bandit(strategies=self.strategies)
        
        # State tracking
        self.population: List[ExecutionResult] = []
        self.best_result: Optional[ExecutionResult] = None
        
        logging.info(f"Optimizer initialized for script '{self.script_path.name}'.")
        logging.info(f"Goal: {self.optimization_goal} '{self.metric_to_optimize}'.")

    def _update_best_result(self, result: ExecutionResult):
        """Compares a new result to the current best and updates if necessary."""
        if result.status != 'SUCCESS' or result.metrics is None:
            return

        current_best_score = -float('inf') if self.optimization_goal == 'maximize' else float('inf')
        if self.best_result and self.best_result.metrics:
            current_best_score = self.best_result.metrics.get(self.metric_to_optimize, current_best_score)

        new_score = result.metrics.get(self.metric_to_optimize)
        if new_score is None:
            logging.warning(f"Metric '{self.metric_to_optimize}' not found in result: {result.metrics}")
            return

        is_better = (self.optimization_goal == 'maximize' and new_score > current_best_score) or \
                    (self.optimization_goal == 'minimize' and new_score < current_best_score)
        
        if is_better or self.best_result is None:
            self.best_result = result
            logging.info(f"New best score found: {new_score:.4f}")

    def run(self) -> Optional[ExecutionResult]:
        """
        Executes the full optimization process.
        """
        logging.info("--- Starting Optimization Run ---")
        
        try:
            analysis_result = self.analyzer.analyze()
            initial_hparams = analysis_result.hyperparameter_space
            class_name = analysis_result.optimizable_class_name
            source_ast = ast.parse(self.script_path.read_text())
        except Exception as e:
            logging.error(f"Failed during initial analysis: {e}")
            return None

        logging.info("--- Generation 0: Evaluating baseline ---")
        baseline_result = self.executor.run(
            script_path=self.script_path,
            class_name=class_name,
            data_path=self.data_path,
            hparams=initial_hparams
        )
        self.population.append(baseline_result)
        self._update_best_result(baseline_result)
        
        for gen in range(1, self.config.num_generations + 1):
            logging.info(f"--- Generation {gen}/{self.config.num_generations} ---")
            
            # --- FIX STARTS HERE ---
            # Initialize parent_score with a worst-case default value first.
            parent_score = -float('inf') if self.optimization_goal == 'maximize' else float('inf')
            parent = self.best_result
            
            if not parent or not parent.metrics:
                logging.warning("No successful parent to mutate from. Using baseline.")
                parent_hparams = initial_hparams
                # parent_score is already set to the worst-case default
            else:
                parent_hparams = parent.hparams_used
                # Update parent_score with the actual score from the best result
                parent_score = parent.metrics.get(self.metric_to_optimize, parent_score)
            # --- FIX ENDS HERE ---

            for i in range(self.config.population_size):
                strategy_to_use = self.bandit.select_arm()
                _ , mutated_hparams = strategy_to_use.mutate(source_ast, parent_hparams)
                
                logging.info(f"Evaluating candidate {i+1}/{self.config.population_size} using '{strategy_to_use.name}'...")
                
                result = self.executor.run(
                    script_path=self.script_path,
                    class_name=class_name,
                    data_path=self.data_path,
                    hparams=mutated_hparams
                )
                
                reward = 0.0
                if result.status == 'SUCCESS' and result.metrics:
                    new_score = result.metrics.get(self.metric_to_optimize)
                    if new_score is not None:
                        reward = new_score - parent_score if self.optimization_goal == 'maximize' else parent_score - new_score
                    self.population.append(result)
                    self._update_best_result(result)
                else:
                    reward = -1.0
                
                self.bandit.update(strategy_to_use.name, reward)

        logging.info("--- Optimization Run Finished ---")
        if self.best_result and self.best_result.metrics:
            score = self.best_result.metrics.get(self.metric_to_optimize, 'N/A')
            logging.info(f"Best score found: {score:.4f}")
            logging.info(f"Best hyperparameters: {self.best_result.hparams_used}")
        else:
            logging.warning("No successful configurations were found.")
            
        return self.best_result

# --- The self-contained test block remains the same ---
if __name__ == '__main__':
    import tempfile
    import pandas as pd
    from examples.contract import Optimizable

    print("\n--- Running Integration Test for optimizer.py ---")

    test_script_code = """
from examples.contract import Optimizable
class SimpleOptimizerTest(Optimizable):
    DEFAULT_HPARAMS = {'x': 1.0, 'y': 5.0}
    def run(self, hparams):
        x = hparams.get('x', 0)
        y = hparams.get('y', 0)
        score = -((x - 3) ** 2) - ((y - 10) ** 2)
        return {'score': score, 'other_metric': x + y}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        lib_dir = temp_dir / "automl_lib"
        (lib_dir / "examples").mkdir(parents=True, exist_ok=True)
        (lib_dir / "__init__.py").touch()
        (lib_dir / "examples" / "__init__.py").touch()
        
        (lib_dir / "examples" / "contract.py").write_text("""
from abc import ABC, abstractmethod
from typing import Dict, Any
class Optimizable(ABC):
    def __init__(self, data): self.data = data
    @abstractmethod
    def run(self, hparams: Dict[str, Any]) -> Dict[str, float]: pass
""")

        script_path = temp_dir / "test_script.py"
        script_path.write_text(test_script_code)

        dummy_df = pd.DataFrame({'a': [1], 'b': [2]})
        dummy_data_path = temp_dir / "data.csv"
        dummy_df.to_csv(dummy_data_path, index=False)

        print("Created temporary project files.")

        config = OptimizerConfig(num_generations=20, population_size=4)
        optimizer = Optimizer(
            script_path=script_path,
            data_path=dummy_data_path,
            config=config,
            metric_to_optimize='score',
            optimization_goal='maximize'
        )

        baseline_score = -29.0
        final_result = optimizer.run()

        assert final_result is not None, "Optimizer did not produce a result."
        assert final_result.status == 'SUCCESS', "Final result was not a success."
        assert final_result.metrics is not None, "Final result has no metrics."
        
        final_score = final_result.metrics['score']
        print(f"\nBaseline Score: {baseline_score:.4f}")
        print(f"Final Best Score: {final_score:.4f}")

        assert final_score > baseline_score, "Optimizer failed to improve upon the baseline score."

        print("\n✅ Optimizer successfully improved the score.")

    print("\n--- All optimizer.py tests passed! ---")