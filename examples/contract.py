# examples/contract.py
"""
Defines the core contract for an optimizable script.

Any user-provided script that needs to be optimized by the system must
implement the `Optimizable` abstract base class. This ensures a consistent
interface for the Orchestrator to call.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import pandas as pd

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Type alias for clarity
DataFrame = Union[pd.DataFrame, Any]
Hyperparameters = Dict[str, Any]
Metrics = Dict[str, float]


class Optimizable(ABC):
    """
    An abstract base class representing an optimizable machine learning script.

    To make a script compatible with the optimizer, create a class that inherits
    from this one and implement the `run` method. The optimizer will instantiate
    this class and call `run` within a sandboxed environment.
    """

    def __init__(self, data: DataFrame):
        """
        Initializes the script with the necessary data.

        Args:
            data (DataFrame): The full dataset provided by the user. The `run`
                              method will be responsible for splitting this data.
        """
        self.data = data
        logging.info(f"{self.__class__.__name__} initialized with dataset of shape {getattr(data, 'shape', 'N/A')}.")

    @abstractmethod
    def run(self, hparams: Hyperparameters) -> Metrics:
        """
        Executes the main training and evaluation logic of the script.

        This method should contain the end-to-end ML pipeline:
        1. Data splitting (train/validation).
        2. Preprocessing.
        3. Model instantiation and training.
        4. Evaluation.

        Args:
            hparams (Hyperparameters): A dictionary of hyperparameters to be used
                                      for this specific run. The keys and value
                                      types are discovered by the Analyzer.

        Returns:
            Metrics: A dictionary of performance metrics, e.g.,
                     {'validation_accuracy': 0.95, 'training_time_seconds': 120.5}.
                     The optimizer will use these metrics to guide the search.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Optimizable Script>"