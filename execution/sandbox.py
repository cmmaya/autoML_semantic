# automl_lib/execution/sandbox.py
"""
Provides a sandboxed environment for executing user-provided scripts.

This module uses Python's multiprocessing to run user code in a separate,
isolated process. This prevents crashes, hangs, or dependency conflicts in the
user's script from affecting the main optimizer process.
"""
import logging
import time
import importlib.util
import sys # <-- Import sys
from pathlib import Path
from multiprocessing import Process, Queue
from dataclasses import dataclass, field
from typing import Dict, Any, Union, Optional

# Assuming these are defined in our project
from examples.contract import Optimizable, Hyperparameters, Metrics, DataFrame

# --- Data Structures for Communication ---

@dataclass
class ExecutionResult:
    """A structured container for the results of a sandboxed run."""
    status: str  # e.g., 'SUCCESS', 'FAILURE', 'TIMEOUT'
    metrics: Optional[Metrics] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    hparams_used: Hyperparameters = field(default_factory=dict)


def _bootstrap_script(
    script_path: Path,
    class_name: str,
    data_path: Path,
    hparams: Hyperparameters,
    result_queue: Queue
):
    """
    The target function that runs in the separate process.

    This function dynamically imports the user's script, instantiates their
    Optimizable class, and executes the `run` method.
    """
    try:
        # --- FIX STARTS HERE ---
        # Add the script's directory to the Python path.
        # This is crucial for the new process to find local module imports.
        sys.path.insert(0, str(script_path.parent))
        # --- FIX ENDS HERE ---

        # Dynamically import the user's module
        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        user_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_module)

        # Find and instantiate the user's class
        OptimizableClass = getattr(user_module, class_name)
        
        # A real implementation would load data properly (e.g., pd.read_csv)
        # For this example, we'll pass a dummy dataframe.
        import pandas as pd
        dummy_data = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame({'col1': [1, 2]})

        instance: Optimizable = OptimizableClass(data=dummy_data)
        
        # Run the main logic and put the result in the queue
        metrics = instance.run(hparams)
        result_queue.put(metrics)

    except Exception as e:
        # If anything goes wrong, put the exception in the queue
        result_queue.put(e)


class SandboxExecutor:
    """
    Executes an optimizable script in a sandboxed subprocess.
    """
    def __init__(self, timeout_seconds: int = 300):
        """
        Args:
            timeout_seconds (int): The maximum time allowed for a script to run.
        """
        self.timeout_seconds = timeout_seconds
        logging.info(f"SandboxExecutor initialized with a timeout of {timeout_seconds} seconds.")

    def run(
        self,
        script_path: Union[str, Path],
        class_name: str,
        data_path: Union[str, Path],
        hparams: Hyperparameters
    ) -> ExecutionResult:
        """
        Executes the specified script with the given hyperparameters.

        Args:
            script_path: Path to the user's script.
            class_name: The name of the Optimizable class to run.
            data_path: Path to the dataset.
            hparams: The hyperparameter dictionary for this run.
        
        Returns:
            An ExecutionResult object summarizing the outcome.
        """
        script_path = Path(script_path)
        data_path = Path(data_path)
        result_queue = Queue()
        
        process = Process(
            target=_bootstrap_script,
            args=(script_path, class_name, data_path, hparams, result_queue)
        )

        start_time = time.time()
        process.start()
        process.join(timeout=self.timeout_seconds)
        duration = time.time() - start_time

        if process.is_alive():
            process.terminate()  # Forcefully stop the process
            process.join()
            logging.warning(f"Process for {script_path.name} exceeded timeout and was terminated.")
            return ExecutionResult(
                status='TIMEOUT',
                duration_seconds=duration,
                error=f"Execution timed out after {self.timeout_seconds} seconds.",
                hparams_used=hparams
            )

        if result_queue.empty():
            error_msg = "Process terminated unexpectedly with no result."
            logging.error(error_msg)
            return ExecutionResult(status='FAILURE', duration_seconds=duration, error=error_msg, hparams_used=hparams)
        
        result = result_queue.get()

        if isinstance(result, Exception):
            logging.error(f"Script execution failed with an exception: {result}")
            return ExecutionResult(status='FAILURE', duration_seconds=duration, error=str(result), hparams_used=hparams)
        
        logging.info(f"Script execution successful. Metrics: {result}")
        return ExecutionResult(status='SUCCESS', metrics=result, duration_seconds=duration, hparams_used=hparams)


# --- Self-contained Test Block ---
if __name__ == '__main__':
    import tempfile
    import pandas as pd
    
    print("\n--- Running Test for sandbox.py ---")

    contract_code = """
from abc import ABC, abstractmethod
from typing import Dict, Any
class Optimizable(ABC):
    def __init__(self, data): self.data = data
    @abstractmethod
    def run(self, hparams: Dict[str, Any]) -> Dict[str, float]: pass
"""
    
    # Script 1: A successful run
    success_script_code = """
from contract import Optimizable
import time
class SuccessScript(Optimizable):
    def run(self, hparams):
        time.sleep(0.1)
        return {'accuracy': 0.95, 'loss': hparams.get('lr', 0.1)}
"""
    # Script 2: A script that will raise an error
    error_script_code = """
from contract import Optimizable
class ErrorScript(Optimizable):
    def run(self, hparams):
        # This will raise a ZeroDivisionError
        x = 1 / 0
        return {'accuracy': 0.5}
"""
    # Script 3: A script that will run too long
    timeout_script_code = """
from contract import Optimizable
import time
class TimeoutScript(Optimizable):
    def run(self, hparams):
        time.sleep(5)
        return {'accuracy': 0.99}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        
        # Create dummy files
        (temp_dir / "contract.py").write_text(contract_code)
        (temp_dir / "success.py").write_text(success_script_code)
        (temp_dir / "error.py").write_text(error_script_code)
        (temp_dir / "timeout.py").write_text(timeout_script_code)
        
        dummy_df = pd.DataFrame({'a': [1], 'b': [2]})
        dummy_data_path = temp_dir / "data.csv"
        dummy_df.to_csv(dummy_data_path, index=False)

        # 1. Test the SUCCESS case
        print("\n--- Testing SUCCESS case ---")
        executor_success = SandboxExecutor(timeout_seconds=5)
        result_success = executor_success.run(
            script_path=temp_dir / "success.py",
            class_name="SuccessScript",
            data_path=dummy_data_path,
            hparams={'lr': 0.05}
        )
        assert result_success.status == 'SUCCESS'
        assert result_success.metrics['accuracy'] == 0.95
        assert result_success.metrics['loss'] == 0.05
        print(f"✅ SUCCESS case passed: {result_success}")

        # 2. Test the FAILURE case
        print("\n--- Testing FAILURE case ---")
        executor_fail = SandboxExecutor(timeout_seconds=5)
        result_fail = executor_fail.run(
            script_path=temp_dir / "error.py",
            class_name="ErrorScript",
            data_path=dummy_data_path,
            hparams={}
        )
        assert result_fail.status == 'FAILURE'
        assert 'division by zero' in result_fail.error
        print(f"✅ FAILURE case passed: {result_fail}")

        # 3. Test the TIMEOUT case
        print("\n--- Testing TIMEOUT case ---")
        executor_timeout = SandboxExecutor(timeout_seconds=1) # Short timeout
        result_timeout = executor_timeout.run(
            script_path=temp_dir / "timeout.py",
            class_name="TimeoutScript",
            data_path=dummy_data_path,
            hparams={}
        )
        assert result_timeout.status == 'TIMEOUT'
        assert 'timed out' in result_timeout.error
        print(f"✅ TIMEOUT case passed: {result_timeout}")

    print("\n--- All sandbox.py tests passed! ---")