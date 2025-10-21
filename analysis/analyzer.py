# automl_lib/analysis/analyzer.py
"""
Provides a high-level interface for analyzing and understanding user scripts.

The Analyzer is the main entry point for the introspection system. It takes a
script path, uses the ScriptParser to dissect it, and returns a structured
summary of its optimizable properties.
"""

import logging
from pathlib import Path
from typing import Dict, Any, NamedTuple, Optional, Union

# Since ast_parser is in the same package, we use a relative import
from .ast_parser import ScriptParser

class AnalysisResult(NamedTuple):
    """A structured container for the results of a script analysis."""
    script_path: Path
    optimizable_class_name: str
    hyperparameter_space: Dict[str, Any]

class AnalysisError(Exception):
    """Custom exception for errors during the analysis phase."""
    pass


class Analyzer:
    """
    Analyzes a Python script to discover its optimizable interface.
    """
    def __init__(self, script_path: Union[str, Path]):
        """
        Initializes the Analyzer with the path to the user's script.

        Args:
            script_path: The path to the Python script to be analyzed.
        """
        self.script_path = Path(script_path)
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Script not found at: {self.script_path}")
        logging.info(f"Analyzer initialized for script: {self.script_path}")

    def analyze(self) -> AnalysisResult:
        """
        Performs a full analysis of the script.

        Returns:
            An AnalysisResult object containing the discovered properties.

        Raises:
            AnalysisError: If the script cannot be parsed or lacks the required components.
        """
        try:
            source_code = self.script_path.read_text()
            parser = ScriptParser(source_code)

            class_info = parser.find_optimizable_class()
            if not class_info:
                raise AnalysisError("No class inheriting from 'Optimizable' was found.")
            
            class_name, class_node = class_info

            hparams = parser.extract_default_hyperparameters(class_node)
            if not hparams:
                raise AnalysisError(f"No 'DEFAULT_HPARAMS' dictionary found in class '{class_name}'.")
            
            logging.info("Analysis successful.")
            return AnalysisResult(
                script_path=self.script_path,
                optimizable_class_name=class_name,
                hyperparameter_space=hparams
            )
        except SyntaxError as e:
            raise AnalysisError(f"Script has a syntax error: {e}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred during analysis: {e}")
            raise AnalysisError(f"Failed to analyze script: {e}") from e

# --- Self-contained Test Block ---
if __name__ == '__main__':
    import tempfile
    
    print("\n--- Running Test for analyzer.py ---")

    # 1. We need the contract file to exist for the dummy script to import
    # For this test, we'll create a dummy contract file as well.
    contract_code = """
from abc import ABC
class Optimizable(ABC):
    pass
"""
    # Create a dummy script that correctly implements the contract
    optimizable_script_code = """
from contract import Optimizable

class MyTestScript(Optimizable):
    DEFAULT_HPARAMS = {'lr': 0.01, 'epochs': 10}
    def run(self, hparams): return {'loss': 0.5}
"""
    # 2. Use a temporary directory to simulate a real project structure
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        
        # Create the dummy contract file
        contract_file = temp_dir / "contract.py"
        contract_file.write_text(contract_code)
        
        # Create the dummy optimizable script file
        script_file = temp_dir / "my_test_script.py"
        script_file.write_text(optimizable_script_code)
        
        print(f"Created temporary script at: {script_file}")
        
        # 3. Test the Analyzer
        try:
            analyzer = Analyzer(script_file)
            result = analyzer.analyze()

            print("✅ Analyzer initialized and ran successfully.")
            
            # 4. Verify the results
            assert result.optimizable_class_name == "MyTestScript"
            assert result.hyperparameter_space['lr'] == 0.01
            print(f"✅ Analysis Result is correct: {result}")

        except (FileNotFoundError, AnalysisError) as e:
            print(f"❌ Test failed: Analyzer raised an error: {e}")

    print("\n--- All analyzer.py tests passed! ---")