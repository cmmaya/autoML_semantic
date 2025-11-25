# automl_lib/mutation/strategies/structure.py
"""
A concrete mutation strategy for structurally modifying a script's AST.
"""
import ast
import random
import logging
import importlib
import inspect
from typing import Any, Optional, Dict, List, Tuple

from .base_strategy import BaseStrategy
from knowledge.graph_db import KnowledgeGraph, ComponentNode
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from utils.model_llm_filter import llm_filter_models


# =============================================================================
#  AST HELPERS
# =============================================================================

class _ModelSwapTransformer(ast.NodeTransformer):
    """Finds and replaces a model-instantiation call with a new model."""
    def __init__(self, model_to_swap: str, new_model: ComponentNode):
        self.model_to_swap = model_to_swap
        self.new_model = new_model
        self.successful_swap = False
        self.new_import_needed: Optional[Tuple[str, str, Optional[str]]] = None

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Detect assignment like: model = RandomForestClassifier()
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == self.model_to_swap:
                logging.info(
                    f"Found model instantiation for '{node.value.func.id}'. "
                    f"Swapping with '{self.new_model.name}'."
                )

                new_model_uid_parts = self.new_model.uid.split(".")
                class_name = self.new_model.name

                # 1) UID like: xgboost.XGBClassifier
                if len(new_model_uid_parts) == 2:
                    module_name = new_model_uid_parts[0]
                    node.value.func = ast.Attribute(
                        value=ast.Name(id=module_name, ctx=ast.Load()),
                        attr=class_name,
                        ctx=ast.Load()
                    )
                    self.new_import_needed = (module_name, class_name, None)

                # 2) sklearn deep imports
                elif new_model_uid_parts[0] == "sklearn" and len(new_model_uid_parts) > 2:
                    node.value.func = ast.Name(id=class_name, ctx=ast.Load())
                    module_path = ".".join(new_model_uid_parts[:-1])
                    self.new_import_needed = (module_path, class_name, None)

                # 3) Other multi-level modules
                elif len(new_model_uid_parts) > 2:
                    node.value.func = ast.Name(id=class_name, ctx=ast.Load())
                    module_path = ".".join(new_model_uid_parts[:-1])
                    self.new_import_needed = (module_path, class_name, None)

                # 4) Fallback
                else:
                    node.value.func = ast.Name(id=class_name, ctx=ast.Load())
                    module_path = ".".join(new_model_uid_parts[:-1]) if len(new_model_uid_parts) > 1 else ""
                    self.new_import_needed = (module_path, class_name, None)

                # Remove old incompatible keyword args
                node.value.keywords = []
                self.successful_swap = True

        return self.generic_visit(node)


def _load_class_from_uid(uid: str):
    mod = ".".join(uid.split(".")[:-1])
    cls = uid.split(".")[-1]
    module = importlib.import_module(mod)
    return getattr(module, cls)


def _is_instantiable_estimator(uid: str, require_proba: bool = True) -> bool:
    """True iff uid is a concrete sklearn estimator supporting proba/decision_function."""
    try:
        cls = _load_class_from_uid(uid)

        if cls.__name__.startswith("_"):
            return False
        if inspect.isabstract(cls):
            return False

        if not issubclass(cls, BaseEstimator):
            return False
        if not (issubclass(cls, ClassifierMixin) or issubclass(cls, RegressorMixin)):
            return False

        sig = inspect.signature(cls.__init__)
        required = [
            p for n, p in sig.parameters.items()
            if n != "self"
            and p.default is inspect._empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        if required:
            return False

        if require_proba:
            has_proba = any("predict_proba" in dir(c) for c in cls.mro())
            has_decision = any("decision_function" in dir(c) for c in cls.mro())
            if not (has_proba or has_decision):
                return False

        return True

    except Exception:
        return False


# =============================================================================
#  NEW — Force any GridSearchCV / RandomizedSearchCV to use n_jobs=1
# =============================================================================
class _ForceDefaultHparamsTransformer(ast.NodeTransformer):
    """
    Rewrites DEFAULT_HPARAMS['n_jobs'] and GridSearchCV n_jobs
    inside annotated or non-annotated assignments.
    """

    def visit_Assign(self, node: ast.Assign):
        # Standard assignment: DEFAULT_HPARAMS = {...}
        if (isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DEFAULT_HPARAMS"
            and isinstance(node.value, ast.Dict)):
            return self._rewrite_dict(node)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Annotated assignment: DEFAULT_HPARAMS: Type = {...}
        if (isinstance(node.target, ast.Name)
            and node.target.id == "DEFAULT_HPARAMS"
            and isinstance(node.value, ast.Dict)):
            return self._rewrite_dict(node)
        return self.generic_visit(node)

    def _rewrite_dict(self, node):
        """Force DEFAULT_HPARAMS['n_jobs'] = 1."""
        new_keys = []
        new_vals = []

        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(k, ast.Constant) and k.value == "n_jobs":
                new_keys.append(k)
                new_vals.append(ast.Constant(value=1))
            else:
                new_keys.append(k)
                new_vals.append(v)

        node.value = ast.Dict(keys=new_keys, values=new_vals)
        return node

class _ForceSingleThreadTransformer(ast.NodeTransformer):
    TARGETS = {"GridSearchCV", "RandomizedSearchCV"}

    def _func_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""

    def visit_Call(self, node: ast.Call):
        func_name = self._func_name(node.func)

        if func_name in self.TARGETS:
            new_keywords = []
            found = False

            for kw in node.keywords:
                if kw.arg == "n_jobs":
                    found = True
                    new_keywords.append(
                        ast.keyword(arg="n_jobs", value=ast.Constant(value=1))
                    )
                else:
                    new_keywords.append(kw)

            if not found:
                new_keywords.append(
                    ast.keyword(arg="n_jobs", value=ast.Constant(value=1))
                )

            node.keywords = new_keywords

        return self.generic_visit(node)

# =============================================================================
#  MAIN STRATEGY
# =============================================================================

class StructuralSwapStrategy(BaseStrategy):
    def __init__(
        self,
        use_llm_filter: bool = False,
        dataset_summary: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            name="StructuralSwapStrategy",
            description="Swaps a model with a compatible alternative."
        )
        self.use_llm_filter = use_llm_filter
        self.dataset_summary = dataset_summary

        # NEW: LLM cache to avoid repeated calls
        self._cached_llm_filtered_models = None
        self.failed_model_uids = set()

    def _add_import(self, tree: ast.AST, new_model_uid: str) -> ast.AST:
        """Adds correct import for the new model."""
        parts = new_model_uid.split('.')

        if len(parts) == 2:
            new_import = ast.Import(names=[ast.alias(name=parts[0])])
        else:
            module_path = ".".join(parts[:-1])
            class_name = parts[-1]
            new_import = ast.ImportFrom(
                module=module_path,
                names=[ast.alias(name=class_name)],
                level=0
            )

        # Avoid duplicates
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.dump(node) == ast.dump(new_import):
                return tree

        tree.body.insert(0, new_import)
        ast.fix_missing_locations(tree)
        return tree

    # ----------------------------------------------------------------------
    # MUTATE
    # ----------------------------------------------------------------------
    def mutate(
        self,
        source_ast: ast.AST,
        hparams: Dict[str, Any],
        knowledge_graph: Optional[KnowledgeGraph] = None
    ) -> Tuple[ast.AST, Dict[str, Any]]:

        if not knowledge_graph:
            logging.warning(f"[{self.name}] cannot run without a knowledge graph.")
            return source_ast, hparams

        # 1) Collect predictors from KG
        predictors = {
            data["name"]: uid
            for uid, data in knowledge_graph.graph.nodes(data=True)
            if data.get("component_type") in ["Classifier", "Regressor"]
        }
        if not predictors:
            return source_ast, hparams

        # 2) Find model used in script
        current_name = None
        current_uid = None
        for node in ast.walk(source_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in predictors:
                    current_name = node.func.id
                    current_uid = predictors[current_name]
                    break

        if not current_uid:
            return source_ast, hparams

        # 3) Candidate replacements
        candidates = knowledge_graph.find_compatible_components(current_uid)
        if not candidates:
            return source_ast, hparams

        safe = [
            m for m in candidates
            if _is_instantiable_estimator(m.uid, require_proba=True)
            and m.uid not in self.failed_model_uids      # skip models that failed earlier
        ]

        if not safe:
            return source_ast, hparams

        # Optional LLM filtering, cached to avoid repeated calls
        if self.use_llm_filter and self.dataset_summary:
            try:
                if self._cached_llm_filtered_models is None:
                    # First time → call LLM
                    logging.info(f"[{self.name}] Calling LLM model filter.")
                    safe_llm = llm_filter_models(
                        models=safe,
                        dataset_summary=self.dataset_summary,
                        top_k=5
                    )
                    if safe_llm:
                        self._cached_llm_filtered_models = safe_llm
                    else:
                        self._cached_llm_filtered_models = safe  # fallback

                # Reuse cached list
                safe = [
                    m for m in safe
                    if m in self._cached_llm_filtered_models
                ]

                # If empty (rare), fallback to unfiltered list
                if not safe:
                    safe = self._cached_llm_filtered_models

            except Exception as e:
                logging.error(f"[{self.name}] Error in llm_filter_models: {e}")

        # 4) Swap
        new_model = random.choice(safe)

        swapper = _ModelSwapTransformer(current_name, new_model)
        new_ast = swapper.visit(source_ast)
        ast.fix_missing_locations(new_ast)

        if swapper.successful_swap:
            new_ast = self._add_import(new_ast, new_model.uid)

            # 5) Rebuild hparams
            allowed = set(new_model.hyperparameters.keys())
            new_hparams = {}

            prefix = ""
            for k in hparams:
                if "__" in k:
                    prefix = k.split("__", 1)[0] + "__"
                    break
            if not prefix:
                prefix = "model__"

            for k, v in hparams.items():
                base = k.split("__")[-1]
                if base in allowed:
                    new_hparams[f"{prefix}{base}"] = v

            for pname, pval in new_model.hyperparameters.items():
                full = f"{prefix}{pname}"
                new_hparams.setdefault(full, pval)

            hparams = new_hparams

            # 6) Remove old param_grid
            class _Remove(ast.NodeTransformer):
                def visit_Assign(self, node: ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "param_grid" for t in node.targets):
                        return ast.Assign(
                            targets=[ast.Name(id="param_grid", ctx=ast.Store())],
                            value=ast.Dict(keys=[], values=[])
                        )
                    return self.generic_visit(node)

            new_ast = _Remove().visit(new_ast)
            ast.fix_missing_locations(new_ast)

            # 7) Insert new param_grid
            keys = []
            vals = []
            for pname, pvalues in new_model.hyperparameters.items():
                keys.append(ast.Constant(value=f"{prefix}{pname}"))
                vals.append(ast.Constant(value=pvalues))

            pg_node = ast.Assign(
                targets=[ast.Name(id="param_grid", ctx=ast.Store())],
                value=ast.Dict(keys=keys, values=vals)
            )
            new_ast.body.insert(1, pg_node)
            ast.fix_missing_locations(new_ast)

        # ------------------------------------------------------------------
        # 8) FINAL FIX — Enforce single-thread GridSearchCV everywhere
        # ------------------------------------------------------------------
        new_ast = _ForceSingleThreadTransformer().visit(new_ast)
        new_ast = _ForceDefaultHparamsTransformer().visit(new_ast)
        ast.fix_missing_locations(new_ast)

        return new_ast, hparams
