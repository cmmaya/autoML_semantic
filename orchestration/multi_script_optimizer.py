# automl_lib/orchestration/multi_script_optimizer.py
import ast
import logging
import pickle
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any

from rich.console import Console

from analysis.analyzer import Analyzer
from execution.sandbox import SandboxExecutor, ExecutionResult
from ensembling.builder import GreedyEnsembleBuilder
from orchestration.bandit import UCB1Bandit
from knowledge.graph_db import KnowledgeGraph

@dataclass
class OptimizerConfig:
    num_generations: int = 5
    population_size: int = 4

class _SingleRunOptimizer:
    def __init__(
        self,
        script_path: Path,
        data_path: Path,
        config: OptimizerConfig,
        metric_to_optimize: str,
        optimization_goal: str,
        strategies: List,
        knowledge_graph: Optional[KnowledgeGraph] = None
    ):
        self.script_path = script_path
        self.data_path = data_path
        self.config = config
        self.metric_to_optimize = metric_to_optimize
        self.optimization_goal = optimization_goal
        self.analyzer = Analyzer(self.script_path)
        self.executor = SandboxExecutor(timeout_seconds=60)
        self.bandit = UCB1Bandit(strategies=strategies)
        self.knowledge_graph = knowledge_graph
        self.population: List[ExecutionResult] = []
        self.best_result: Optional[ExecutionResult] = None
        self.code_archive: Dict[str, str] = {}
        self.console = Console()

    def _update_best_result(self, result: ExecutionResult):
        if result.status != 'SUCCESS' or result.metrics is None: return
        current_best_score = -float('inf') if self.optimization_goal == 'maximize' else float('inf')
        if self.best_result and self.best_result.metrics:
            current_best_score = self.best_result.metrics.get(self.metric_to_optimize, current_best_score)
        new_score = result.metrics.get(self.metric_to_optimize)
        if new_score is None: return
        is_better = (self.optimization_goal == 'maximize' and new_score > current_best_score) or \
                    (self.optimization_goal == 'minimize' and new_score < current_best_score)
        if is_better or self.best_result is None:
            self.best_result = result
            logging.info(f"[{self.script_path.name}] New best score: {new_score:.4f}")

    def run(self) -> List[ExecutionResult]:
        logging.info(f"--- Starting optimization for island: {self.script_path.name} ---")
        try:
            analysis_result = self.analyzer.analyze()
            initial_hparams = analysis_result.hyperparameter_space
            class_name = analysis_result.optimizable_class_name
            source_code = self.script_path.read_text()
            source_ast = ast.parse(source_code)
        except Exception as e:
            logging.error(f"Failed initial analysis for {self.script_path.name}: {e}")
            return []

        baseline_result = self.executor.run(self.script_path, class_name, self.data_path, initial_hparams)
        if baseline_result.status == 'SUCCESS':
            run_id = f"gen_0_baseline_{self.script_path.stem}"
            self.code_archive[run_id] = source_code
            baseline_result.hparams_used['run_id'] = run_id
            self.population.append(baseline_result)
        self._update_best_result(baseline_result)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            for gen in range(1, self.config.num_generations + 1):
                logging.info(f"[{self.script_path.name}] Generation {gen}/{self.config.num_generations}")
                parent_score = -float('inf') if self.optimization_goal == 'maximize' else float('inf')
                parent = self.best_result
                if not parent or not parent.metrics:
                    parent_hparams = initial_hparams
                    parent_ast = source_ast
                else:
                    parent_hparams = parent.hparams_used
                    parent_run_id = parent.hparams_used.get('run_id')
                    parent_code = self.code_archive.get(parent_run_id, source_code)
                    parent_ast = ast.parse(parent_code)
                    parent_score = parent.metrics.get(self.metric_to_optimize, parent_score)

                for i in range(self.config.population_size):
                    strategy_to_use = self.bandit.select_arm()
                    self.console.log(f"Gen {gen} | Cand. {i+1} | Island '{self.script_path.name}' | Arm: [bold cyan]{strategy_to_use.name}[/bold cyan]")
                    clean_parent_hparams = {k: v for k, v in parent_hparams.items() if k != 'run_id'}
                    mutated_ast, mutated_hparams = strategy_to_use.mutate(parent_ast, clean_parent_hparams, self.knowledge_graph)
                    new_code = ast.unparse(mutated_ast)
                    run_id = f"gen_{gen}_cand_{i}_{self.script_path.stem}"
                    temp_script_path = temp_dir / f"{run_id}.py"
                    temp_script_path.write_text(new_code)
                    result = self.executor.run(temp_script_path, class_name, self.data_path, mutated_hparams)
                    if result.status == 'SUCCESS':
                        self.code_archive[run_id] = new_code
                        result.hparams_used['run_id'] = run_id
                    reward = -1.0
                    if result.status == 'SUCCESS' and result.metrics:
                        new_score = result.metrics.get(self.metric_to_optimize)
                        if new_score is not None:
                            reward = new_score - parent_score if self.optimization_goal == 'maximize' else parent_score - new_score
                        self.population.append(result)
                        self._update_best_result(result)
                    self.bandit.update(strategy_to_use.name, reward)
        
        logging.info(f"--- Finished optimization for island: {self.script_path.name} ---")
        return [res for res in self.population if res.status == 'SUCCESS']

class MultiScriptOptimizer:
    def __init__(
        self,
        script_paths: List[Union[str, Path]],
        data_path: Union[str, Path],
        config: OptimizerConfig,
        metric_to_optimize: str,
        optimization_goal: str,
        strategies: List,
        knowledge_graph_path: Optional[Union[str, Path]] = None
    ):
        self.script_paths = [Path(p) for p in script_paths]
        self.data_path = data_path
        self.config = config
        self.metric_to_optimize = metric_to_optimize
        self.optimization_goal = optimization_goal
        self.strategies = strategies
        self.knowledge_graph = None
        if knowledge_graph_path and Path(knowledge_graph_path).exists():
            try:
                with open(knowledge_graph_path, "rb") as f:
                    graph_data = pickle.load(f)
                self.knowledge_graph = KnowledgeGraph()
                self.knowledge_graph.graph = graph_data
                logging.info(f"Successfully loaded knowledge graph from {knowledge_graph_path}")
            except Exception as e:
                logging.error(f"Failed to load knowledge graph: {e}")
        logging.info(f"MultiScriptOptimizer initialized for {len(self.script_paths)} islands.")

    def run(self) -> Dict[str, Any]:
        all_results: List[ExecutionResult] = []
        all_code_archives: Dict[str, str] = {}
        for script_path in self.script_paths:
            island_optimizer = _SingleRunOptimizer(
                script_path, self.data_path, self.config,
                self.metric_to_optimize, self.optimization_goal, 
                self.strategies, self.knowledge_graph
            )
            island_results = island_optimizer.run()
            all_results.extend(island_results)
            all_code_archives.update(island_optimizer.code_archive)

        best_single_model = None
        if all_results:
            key_fn = lambda r: r.metrics.get(self.metric_to_optimize, -float('inf'))
            if self.optimization_goal == 'minimize':
                key_fn = lambda r: r.metrics.get(self.metric_to_optimize, float('inf'))
            best_single_model = max(all_results, key=key_fn) if self.optimization_goal == 'maximize' else min(all_results, key=key_fn)

        logging.info("--- Building Final Ensemble from all islands ---")
        builder = GreedyEnsembleBuilder(
            task_type='regression',
            metric_to_optimize=self.metric_to_optimize,
            optimization_goal=self.optimization_goal
        )
        final_ensemble_hparams = builder.build(all_results, None, None)

        best_model_code = None
        if best_single_model and best_single_model.hparams_used.get('run_id'):
            best_model_code = all_code_archives.get(best_single_model.hparams_used['run_id'])

        final_output = {
            "best_single_model_result": best_single_model,
            "best_single_model_code": best_model_code,
            "final_ensemble_hparams": final_ensemble_hparams,
        }
        logging.info("--- Multi-Script Optimization Finished ---")
        return final_output