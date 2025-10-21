# automl_lib/mutation/strategies/structure.py
"""
A concrete mutation strategy for structurally modifying a script's AST.
"""
import ast
import random
import logging
from typing import Any, Optional, Dict, List, Tuple

from .base_strategy import BaseStrategy
from knowledge.graph_db import KnowledgeGraph, ComponentNode

class _ModelSwapTransformer(ast.NodeTransformer):
    """An AST transformer that finds and replaces a model instantiation call."""
    def __init__(self, model_to_swap: str, new_model: ComponentNode):
        self.model_to_swap = model_to_swap
        self.new_model = new_model
        self.successful_swap = False
        self.new_import_needed: Optional[Tuple[str, str, Optional[str]]] = None # (module, name, alias)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # We're looking for an assignment, e.g., `model = RandomForestClassifier()`
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == self.model_to_swap:
                logging.info(f"Found model instantiation for '{node.value.func.id}'. Swapping with '{self.new_model.name}'.")
                
                new_model_uid_parts = self.new_model.uid.split('.')
                class_name = self.new_model.name
                
                # Logic to handle different import styles
                if len(new_model_uid_parts) > 1 and new_model_uid_parts[0] != class_name:
                    # e.g., import xgboost as xgb -> xgb.XGBClassifier
                    module_name = new_model_uid_parts[0]
                    alias = module_name # Simple alias for now
                    node.value.func = ast.Attribute(value=ast.Name(id=alias, ctx=ast.Load()), attr=class_name, ctx=ast.Load())
                    self.new_import_needed = (module_name, class_name, alias)
                else: 
                    # e.g., from sklearn.ensemble import RandomForestClassifier
                    node.value.func.id = class_name
                    module_path = ".".join(new_model_uid_parts[:-1])
                    self.new_import_needed = (module_path, class_name, None)

                node.value.keywords = [] # Discard old keywords
                self.successful_swap = True
        return self.generic_visit(node)

class StructuralSwapStrategy(BaseStrategy):
    """Swaps a model in the script with a compatible alternative from the Knowledge Graph."""
    def __init__(self):
        super().__init__(
            name="StructuralSwapStrategy",
            description="Swaps a model with a compatible alternative (e.g., RandomForest -> XGBoost)."
        )

    def _add_import(self, tree: ast.AST, new_model_uid: str) -> ast.AST:
        """Adds the correct import statement for the new model's UID."""
        
        uid_parts = new_model_uid.split('.')
        
        # Handle cases like 'xgboost.XGBClassifier' -> import xgboost
        if len(uid_parts) == 2:
            module_name = uid_parts[0]
            new_import = ast.Import(names=[ast.alias(name=module_name)])
        # Handle cases like 'sklearn.naive_bayes.MultinomialNB'
        else:
            module_path = ".".join(uid_parts[:-1])
            class_name = uid_parts[-1]
            new_import = ast.ImportFrom(module=module_path, names=[ast.alias(name=class_name)], level=0)
        
        # Check if a similar import already exists to avoid duplicates
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.dump(node) == ast.dump(new_import):
                return tree # Import already exists
        
        tree.body.insert(0, new_import)
        ast.fix_missing_locations(tree)
        return tree

    def mutate(
        self,
        source_ast: ast.AST,
        hparams: Dict[str, Any],
        knowledge_graph: Optional[KnowledgeGraph] = None
    ) -> tuple[ast.AST, Dict[str, Any]]:
        if not knowledge_graph:
            logging.warning(f"[{self.name}] cannot run without a knowledge graph. Returning original.")
            return source_ast, hparams
        
        predictors = {
            data['name']: uid for uid, data in knowledge_graph.graph.nodes(data=True)
            if data.get('component_type') in ['Classifier', 'Regressor']
        }
        if not predictors:
            logging.warning(f"[{self.name}] No Classifiers or Regressors found in the Knowledge Graph.")
            return source_ast, hparams

        current_model_name = None
        current_model_uid = None
        for node in ast.walk(source_ast):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in predictors:
                current_model_name = node.func.id
                current_model_uid = predictors[current_model_name]
                break
        
        if not current_model_uid or not current_model_name:
            logging.warning(f"[{self.name}] could not find a known predictor model to swap in the script.")
            return source_ast, hparams

        compatible_models = knowledge_graph.find_compatible_components(current_model_uid)
        if not compatible_models:
            logging.warning(f"[{self.name}] found no compatible models for '{current_model_uid}'.")
            return source_ast, hparams

        new_model = random.choice(compatible_models)
        transformer = _ModelSwapTransformer(current_model_name, new_model)
        new_ast = transformer.visit(source_ast)

        if transformer.successful_swap:
            new_ast = self._add_import(new_ast, new_model.uid)

        return new_ast, hparams

# --- Self-contained Test Block ---
if __name__ == '__main__':
    from knowledge.graph_db import KnowledgeGraph
    from knowledge.ontology import Node, ComponentNode, Relationship

    print("\n--- Running Test for structure.py ---")

    # 1. Create a dummy Knowledge Graph
    kg = KnowledgeGraph()
    rf_node = ComponentNode(uid="sklearn.ensemble.RandomForestClassifier", name="RandomForestClassifier", component_type="Classifier")
    xgb_node = ComponentNode(uid="xgboost.XGBClassifier", name="XGBClassifier", component_type="Classifier")
    nb_node = ComponentNode(uid="sklearn.naive_bayes.MultinomialNB", name="MultinomialNB", component_type="Classifier")
    classifier_concept = Node(uid="concept.classifier", name="Classifier Concept")
    
    kg.add_node(rf_node); kg.add_node(xgb_node); kg.add_node(nb_node); kg.add_node(classifier_concept)
    kg.add_relationship(Relationship(source_uid=rf_node.uid, target_uid=classifier_concept.uid, label="is_a"))
    kg.add_relationship(Relationship(source_uid=xgb_node.uid, target_uid=classifier_concept.uid, label="is_a"))
    kg.add_relationship(Relationship(source_uid=nb_node.uid, target_uid=classifier_concept.uid, label="is_a"))
    
    # 2. Define a simple source script
    source_code = """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
"""
    source_ast = ast.parse(source_code)
    
    # 3. Initialize and run the strategy
    strategy = StructuralSwapStrategy()
    new_ast, _ = strategy.mutate(source_ast, {}, kg)
    
    # 4. Unparse the new AST back to code and verify
    new_code = ast.unparse(new_ast)
    
    print("\nOriginal Code:\n---")
    print(source_code.strip())
    print("\nMutated Code:\n---")
    print(new_code.strip())
    
    is_xgb = "XGBClassifier" in new_code and "import xgboost" in new_code
    is_nb = "MultinomialNB" in new_code and "from sklearn.naive_bayes import MultinomialNB" in new_code
    
    assert is_xgb or is_nb, "Swap to a compatible model failed."
    assert "RandomForestClassifier" not in new_code, "Old model was not removed."
    
    print("\n✅ Structural swap was successful.")
    print("\n--- All structure.py tests passed! ---")