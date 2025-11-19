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
        Initialize the Analyzer with the path to the user's script.

        Parameters
        ----------
        script_path : str or Path
            Path to the Python script that will be analyzed.

        Raises
        ------
        FileNotFoundError
            If the script_path does not point to a valid file.
        """
        self.script_path = Path(script_path)
        if not self.script_path.is_file():
            raise FileNotFoundError(f"Script not found at: {self.script_path}")

        logging.info(f"Analyzer initialized for script: {self.script_path}")

    # ----------------------------------------------------------------------
    # New method: analyze the script directly from a source string.
    # This is required to analyze AST-patched code before writing it to disk.
    # ----------------------------------------------------------------------
    def analyze_source(self, source: str) -> AnalysisResult:
        """
        Analyze a script provided as a source string. This allows the optimizer
        to analyze scripts that have been modified in memory (via AST patches).

        Parameters
        ----------
        source : str
            Full Python script as text.

        Returns
        -------
        AnalysisResult
            Results containing extracted class name and hyperparameters.

        Raises
        ------
        AnalysisError
            If the script fails to parse or does not contain a valid Optimizable class.
        """
        try:
            parser = ScriptParser(source)
            return self._analyze_parsed(parser, source)
        except SyntaxError as e:
            raise AnalysisError(f"Patched script contains a syntax error: {e}") from e
        except Exception as e:
            logging.error(f"Unexpected error analyzing patched script: {e}")
            raise AnalysisError(f"Failed to analyze patched script: {e}") from e

    # ----------------------------------------------------------------------
    # Existing method: analyze script from disk.
    # This version remains unchanged for backwards compatibility.
    # ----------------------------------------------------------------------
    def analyze(self) -> AnalysisResult:
        """
        Analyze the script stored on disk at `self.script_path`.

        Returns
        -------
        AnalysisResult
            Results containing discovered class name and hyperparameters.

        Raises
        ------
        AnalysisError
            If the script cannot be parsed or does not contain the required components.
        """
        try:
            source_code = self.script_path.read_text()
            parser = ScriptParser(source_code)
            return self._analyze_parsed(parser, source_code)

        except SyntaxError as e:
            raise AnalysisError(f"Script has a syntax error: {e}") from e
        except Exception as e:
            logging.error(f"Unexpected error during analysis: {e}")
            raise AnalysisError(f"Failed to analyze script: {e}") from e

    # ----------------------------------------------------------------------
    # Internal shared logic used by both analyze() and analyze_source().
    # Performs extraction of the optimizable class and its hyperparameters.
    # ----------------------------------------------------------------------
    def _analyze_parsed(self, parser: "ScriptParser", source: str) -> AnalysisResult:
        """
        Internal helper that extracts:
        - the Optimizable class definition
        - DEFAULT_HPARAMS dictionary

        Parameters
        ----------
        parser : ScriptParser
            ScriptParser instance initialized with script contents.

        source : str
            The raw script text (unused here but kept for debugging consistency).

        Returns
        -------
        AnalysisResult
            Parsed analysis results.

        Raises
        ------
        AnalysisError
            If the script does not contain a valid Optimizable class definition
            or does not define DEFAULT_HPARAMS.
        """
        class_info = parser.find_optimizable_class()
        if not class_info:
            raise AnalysisError("No class inheriting from 'Optimizable' was found in the script.")

        class_name, class_node = class_info

        hparams = parser.extract_default_hyperparameters(class_node)
        if not hparams:
            raise AnalysisError(f"'DEFAULT_HPARAMS' dictionary not found in class '{class_name}'.")

        logging.info("Analysis completed successfully.")

        return AnalysisResult(
            script_path=self.script_path,
            optimizable_class_name=class_name,
            hyperparameter_space=hparams
        )


# ----------------------------------------------------------------------
# Self-contained test block
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import tempfile

    print("\n--- Analyzer Test ---")

    contract_code = """
from abc import ABC
class Optimizable(ABC):
    pass
"""

    optimizable_script_code = """
from contract import Optimizable

class MyTestScript(Optimizable):
    DEFAULT_HPARAMS = {'lr': 0.01, 'epochs': 10}
    def run(self, hparams):
        return {'loss': 0.5}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        contract_file = temp_dir / "contract.py"
        contract_file.write_text(contract_code)

        script_file = temp_dir / "my_test_script.py"
        script_file.write_text(optimizable_script_code)

        print(f"Created temporary script at: {script_file}")

        try:
            analyzer = Analyzer(script_file)
            result = analyzer.analyze()

            print("Analyzer initialized and executed successfully.")

            assert result.optimizable_class_name == "MyTestScript"
            assert result.hyperparameter_space['lr'] == 0.01
            print(f"Analysis result valid: {result}")

        except (FileNotFoundError, AnalysisError) as e:
            print(f"Test failed: Analyzer raised an error: {e}")

    print("\n--- Analyzer tests completed ---")
