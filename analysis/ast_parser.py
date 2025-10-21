# automl_lib/analysis/ast_parser.py
"""
Low-level AST parsing utilities to extract optimizable components from a script.

This module provides a parser that can walk a script's Abstract Syntax Tree
to find classes that adhere to the Optimizable contract and extract their
default hyperparameter configurations.
"""

import ast
import logging
from typing import Dict, Any, Optional, Tuple

class ScriptParser:
    """
    Parses a Python script's source code to find key optimizable elements.
    """

    def __init__(self, source_code: str):
        """
        Initializes the parser with the script's source code.

        Args:
            source_code (str): The string content of the Python script.

        Raises:
            SyntaxError: If the source code is not valid Python.
        """
        self.source_code = source_code
        try:
            self.tree = ast.parse(self.source_code)
            logging.info("Successfully parsed source code into AST.")
        except SyntaxError as e:
            logging.error(f"Failed to parse source code: {e}")
            raise

    def find_optimizable_class(self) -> Optional[Tuple[str, ast.ClassDef]]:
        """
        Finds the first class in the AST that inherits from 'Optimizable'.

        Returns:
            A tuple containing the class name and its AST node, or None if not found.
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'Optimizable':
                        logging.info(f"Found Optimizable class: '{node.name}'")
                        return node.name, node
        logging.warning("No class inheriting from 'Optimizable' found in the script.")
        return None

    def extract_default_hyperparameters(self, class_node: ast.ClassDef) -> Optional[Dict[str, Any]]:
        """
        Extracts a `DEFAULT_HPARAMS` dictionary from a class definition node.

        This method looks for a class-level variable assignment, supporting both
        simple (`VAR = value`) and annotated (`VAR: type = value`) styles.

        It uses `ast.literal_eval` for safe evaluation of the dictionary.

        Args:
            class_node (ast.ClassDef): The AST node of the class to inspect.

        Returns:
            The hyperparameter dictionary, or None if not found or invalid.
        """
        for node in class_node.body:
            # --- FIX STARTS HERE ---
            # Check for both annotated and simple assignments.
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_name = None
                # Handle simple assignment: VAR = value
                if isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        target_name = node.targets[0].id
                # Handle annotated assignment: VAR: type = value
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        target_name = node.target.id

                if target_name == 'DEFAULT_HPARAMS':
                    if node.value is None: # Handle case like `VAR: type` with no value
                        logging.warning("`DEFAULT_HPARAMS` is annotated but not assigned a value.")
                        return None
                    try:
                        hparams = ast.literal_eval(node.value)
                        if not isinstance(hparams, dict):
                             logging.warning("`DEFAULT_HPARAMS` is not a valid dictionary.")
                             return None
                        logging.info(f"Successfully extracted DEFAULT_HPARAMS: {hparams}")
                        return hparams
                    except ValueError:
                        logging.error("Could not safely evaluate `DEFAULT_HPARAMS`.")
                        return None
            # --- FIX ENDS HERE ---
        logging.warning("No `DEFAULT_HPARAMS` class attribute found.")
        return None


# --- Self-contained Test Block ---
if __name__ == '__main__':
    print("--- Running Test for ast_parser.py ---")

    # 1. Define a sample script that adheres to our contract.
    # This test is now fully self-contained and does not require external files.
    sample_script_code = """
from abc import ABC

# Define the contract locally for this test
class Optimizable(ABC):
    pass

class MyRFEnhancedClassifier(Optimizable):
    # The parser should find this annotated assignment
    DEFAULT_HPARAMS: dict = {
        'n_estimators': 150,
        'max_depth': 20,
        'criterion': 'gini',
        'bootstrap': True
    }

    def run(self, hparams):
        return {'accuracy': 0.9}

# Another class to ensure we find the right one
class NotOptimizable:
    pass
"""
    # 2. Test successful parsing
    try:
        parser = ScriptParser(sample_script_code)
        print("✅ Parser initialized successfully.")
    except SyntaxError:
        print("❌ Test failed: Parser initialization raised SyntaxError.")
        exit()

    # 3. Test finding the optimizable class
    class_info = parser.find_optimizable_class()
    assert class_info is not None, "❌ Test failed: Did not find the Optimizable class."
    class_name, class_node = class_info
    assert class_name == "MyRFEnhancedClassifier", f"❌ Test failed: Found wrong class name '{class_name}'."
    print(f"✅ Found correct optimizable class: '{class_name}'")

    # 4. Test extracting hyperparameters
    hparams = parser.extract_default_hyperparameters(class_node)
    assert hparams is not None, "❌ Test failed: Did not extract hyperparameters."
    assert hparams['n_estimators'] == 150, "❌ Test failed: Incorrect hyperparameter value."
    print(f"✅ Extracted hyperparameters successfully: {hparams}")

    print("\n--- All ast_parser.py tests passed! ---")