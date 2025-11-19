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
from mutation.strategies.structure import _ForceSingleThreadTransformer, _ForceDefaultHparamsTransformer

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
        if result.status != 'SUCCESS' or result.metrics is None:
            return
        current_best_score = -float('inf') if self.optimization_goal == 'maximize' else float('inf')
        if self.best_result and self.best_result.metrics:
            current_best_score = self.best_result.metrics.get(self.metric_to_optimize, current_best_score)
        new_score = result.metrics.get(self.metric_to_optimize)
        if new_score is None:
            return
        is_better = (
            (self.optimization_goal == 'maximize' and new_score > current_best_score)
            or (self.optimization_goal == 'minimize' and new_score < current_best_score)
        )
        if is_better or self.best_result is None:
            self.best_result = result
            logging.info(f"[{self.script_path.name}] New best score: {new_score:.4f}")

    def _patch_auc_block(self, source: str) -> str:
        try:
            lines = source.splitlines()
            patched = []
            for line in lines:
                if "preds_proba = pipeline.predict_proba(" in line:
                    patched.append(
                        "        if hasattr(pipeline, 'predict_proba'):\n"
                        "            preds_proba = pipeline.predict_proba(X_val)[:, 1]\n"
                        "        elif hasattr(pipeline, 'decision_function'):\n"
                        "            preds_proba = pipeline.decision_function(X_val)\n"
                        "        else:\n"
                        "            preds_proba = pipeline.predict(X_val).astype(float)\n"
                        "        auc = roc_auc_score(y_val, preds_proba)\n"
                    )
                else:
                    patched.append(line)
            return "\n".join(patched)
        except Exception:
            return source

    def run(self) -> List[ExecutionResult]:
        logging.info(f"--- Starting optimization for island: {self.script_path.name} ---")

        # =====================================================
        # READ ORIGINAL SOURCE AND PARSE AST
        # =====================================================
        try:
            original_source = self.script_path.read_text()
            source_ast = ast.parse(original_source)
        except Exception as e:
            logging.error(f"Failed to read/parse original script: {e}")
            return []

        # =====================================================
        # APPLY BASELINE PATCHES BEFORE ANALYZER RUNS
        # =====================================================
        try:
            # 1) patch GridSearchCV(... n_jobs=-1)
            source_ast = _ForceSingleThreadTransformer().visit(source_ast)

            # 2) patch DEFAULT_HPARAMS['n_jobs'] = 1
            source_ast = _ForceDefaultHparamsTransformer().visit(source_ast)

            ast.fix_missing_locations(source_ast)
            patched_source_code = ast.unparse(source_ast)
        except Exception as e:
            logging.error(f"Failed to apply baseline AST patches: {e}")
            patched_source_code = original_source  # fallback

        # =====================================================
        # ANALYZE THE *PATCHED* SOURCE (critical fix)
        # =====================================================
        try:
            analysis_result = self.analyzer.analyze_source(patched_source_code)
            initial_hparams = analysis_result.hyperparameter_space
            class_name = analysis_result.optimizable_class_name
        except Exception as e:
            logging.error(f"Analyzer failed on patched source for {self.script_path.name}: {e}")
            return []

        # =====================================================
        # WRITE PATCHED BASELINE TO TEMP FILE AND RUN IT
        # =====================================================
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            patched_baseline_path = Path(f.name)
            patched_baseline_path.write_text(patched_source_code, encoding="utf-8")

        baseline_result = self.executor.run(
            patched_baseline_path, class_name, self.data_path, initial_hparams
        )

        if baseline_result.status == 'SUCCESS':
            run_id = f"gen_0_baseline_{self.script_path.stem}"
            self.code_archive[run_id] = patched_source_code
            baseline_result.hparams_used['run_id'] = run_id
            self.population.append(baseline_result)

        self._update_best_result(baseline_result)

        # =====================================================
        # BEGIN GENETIC EVOLUTION
        # =====================================================
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            for gen in range(1, self.config.num_generations + 1):
                logging.info(f"[{self.script_path.name}] Generation {gen}/{self.config.num_generations}")

                parent_score = -float('inf') if self.optimization_goal == 'maximize' else float('inf')
                parent = self.best_result

                if not parent or not parent.metrics:
                    parent_hparams = initial_hparams
                    parent_ast = source_ast  # patched baseline AST
                else:
                    parent_hparams = parent.hparams_used
                    parent_run_id = parent.hparams_used.get('run_id')
                    parent_code = self.code_archive.get(parent_run_id, patched_source_code)
                    parent_ast = ast.parse(parent_code)
                    parent_score = parent.metrics.get(self.metric_to_optimize, parent_score)

                # =====================================================
                # MUTATION LOOP
                # =====================================================
                for i in range(self.config.population_size):
                    strategy_to_use = self.bandit.select_arm()
                    self.console.log(
                        f"Gen {gen} | Cand. {i+1} | Island '{self.script_path.name}' | "
                        f"Arm: [bold cyan]{strategy_to_use.name}[/bold cyan]"
                    )

                    clean_parent_hparams = {k: v for k, v in parent_hparams.items() if k != 'run_id'}

                    # --- MUTATE AST ---
                    mutated_ast, mutated_hparams = strategy_to_use.mutate(
                        parent_ast, clean_parent_hparams, self.knowledge_graph
                    )

                    # =====================================================
                    # APPLY THE SAME PATCHES TO EVERY MUTATION
                    # =====================================================
                    try:
                        mutated_ast = _ForceSingleThreadTransformer().visit(mutated_ast)
                        mutated_ast = _ForceDefaultHparamsTransformer().visit(mutated_ast)
                        ast.fix_missing_locations(mutated_ast)
                    except Exception as e:
                        logging.error(f"Failed applying AST patches to mutation: {e}")
                        continue

                    new_code = ast.unparse(mutated_ast)

                    # AUC fallback patch
                    if "predict_proba(" in new_code and "roc_auc_score" in new_code:
                        new_code = self._patch_auc_block(new_code)

                    run_id = f"gen_{gen}_cand_{i}_{self.script_path.stem}"
                    temp_script_path = temp_dir / f"{run_id}.py"
                    temp_script_path.write_text(new_code, encoding="utf-8")

                    result = self.executor.run(
                        temp_script_path, class_name, self.data_path, mutated_hparams
                    )

                    if result.status == 'SUCCESS':
                        self.code_archive[run_id] = new_code
                        result.hparams_used['run_id'] = run_id

                    reward = -1.0
                    if result.status == 'SUCCESS' and result.metrics:
                        new_score = result.metrics.get(self.metric_to_optimize)
                        if new_score is not None:
                            reward = (new_score - parent_score
                                    if self.optimization_goal == 'maximize'
                                    else parent_score - new_score)
                        self.population.append(result)
                        self._update_best_result(result)

                    self.bandit.update(strategy_to_use.name, reward)

        logging.info(f"--- Finished optimization for island: {self.script_path.name} ---")

        return {
            "baseline": baseline_result,
            "mutations": [res for res in self.population if res.status == 'SUCCESS']
        }

class MultiScriptOptimizer:
    def __init__(
        self,
        script_paths: List[Union[str, Path]],
        data_path: Union[str, Path],
        config: OptimizerConfig,
        metric_to_optimize: str,
        optimization_goal: str,
        strategies: List,
        knowledge_graph_path: Optional[Union[str, Path]] = None,
        retriever_script: Optional[Path] = None
    ):
        # Retriever-related state
        self.retriever_script = retriever_script
        self.retriever_result = None

        # Base config
        self.script_paths = [Path(p) for p in script_paths]
        self.data_path = data_path
        self.config = config
        self.metric_to_optimize = metric_to_optimize
        self.optimization_goal = optimization_goal
        self.strategies = strategies

        # Knowledge Graph
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

    # ------------------------------------------------------------------
    #  MAIN MULTI-SCRIPT EVOLUTION LOOP
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:

        # Accumulators
        all_mutation_results: List[ExecutionResult] = []
        baselines: List[ExecutionResult] = []
        code_archive: Dict[str, str] = {}

        # --------------------------------------------------------------
        # 1) RUN EACH SCRIPT ISLAND (original, retriever, mutations)
        # --------------------------------------------------------------
        for script_path in self.script_paths:

            island_optimizer = _SingleRunOptimizer(
                script_path,
                self.data_path,
                self.config,
                self.metric_to_optimize,
                self.optimization_goal,
                self.strategies,
                self.knowledge_graph
            )

            result_dict = island_optimizer.run()

            # Individual baseline
            baseline = result_dict.get("baseline")
            if baseline:
                baselines.append(baseline)

            # Retriever baseline if matching script
            if self.retriever_script and script_path == self.retriever_script:
                self.retriever_result = baseline

            # All mutation outputs from this island
            island_mutations = result_dict.get("mutations", [])
            all_mutation_results.extend(island_mutations)

            # Store this island's produced scripts
            code_archive.update(island_optimizer.code_archive)

        # --------------------------------------------------------------
        # 2) COMPILE ALL CANDIDATES
        # --------------------------------------------------------------
        candidates = []

        # Original baselines
        for b in baselines:
            if b and b.status == "SUCCESS":
                candidates.append(("baseline", b))

        # Retriever baseline (if exists)
        if self.retriever_result and self.retriever_result.status == "SUCCESS":
            candidates.append(("retriever", self.retriever_result))

        # Mutations
        for m in all_mutation_results:
            candidates.append(("mutation", m))

        # If nothing succeeded
        if not candidates:
            logging.error("No successful models from baselines or mutations.")
            return {
                "best_type": None,
                "best_model": None,
                "best_script_code": None,
                "all_mutations": [],
                "baselines": baselines,
                "retriever_baseline": self.retriever_result,
                "final_ensemble_hparams": {}
            }

        # --------------------------------------------------------------
        # 3) SELECT BEST MODEL ACCORDING TO METRIC & GOAL
        # --------------------------------------------------------------
        if self.optimization_goal == "maximize":
            candidates.sort(
                key=lambda x: x[1].metrics.get(self.metric_to_optimize, -float("inf")),
                reverse=True
            )
        else:
            candidates.sort(
                key=lambda x: x[1].metrics.get(self.metric_to_optimize, float("inf"))
            )

        best_type, best_model = candidates[0]
        best_model_code = None

        if best_model and best_model.hparams_used.get("run_id"):
            best_model_code = code_archive.get(best_model.hparams_used["run_id"])

        # --------------------------------------------------------------
        # 4) OPTIONAL FINAL ENSEMBLING
        # --------------------------------------------------------------
        logging.info("--- Building Final Ensemble from all islands ---")
        builder = GreedyEnsembleBuilder(
            task_type="regression",
            metric_to_optimize=self.metric_to_optimize,
            optimization_goal=self.optimization_goal
        )

        final_ensemble_hparams = builder.build(
            all_mutation_results, None, None
        )

        # --------------------------------------------------------------
        # 5) RETURN EVERYTHING USEFUL
        # --------------------------------------------------------------
        logging.info("--- Multi-Script Optimization Finished ---")

        return {
            "best_type": best_type,
            "best_model": best_model,
            "best_script_code": best_model_code,
            "all_mutations": all_mutation_results,
            "baselines": baselines,
            "retriever_baseline": self.retriever_result,
            "final_ensemble_hparams": final_ensemble_hparams,
        }
