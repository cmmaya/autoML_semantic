# automl_lib/ensembling/builder.py
"""
Provides tools for building a weighted ensemble from a population of trained models.
"""
import logging
import uuid
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, mean_squared_error

from execution.sandbox import ExecutionResult

# A placeholder for a generic trained model object
TrainedModel = Any

class GreedyEnsembleBuilder:
    """
    Builds an ensemble using a greedy forward selection algorithm.
    """
    def __init__(
        self,
        task_type: str,
        metric_to_optimize: str,
        optimization_goal: str = 'maximize',
        ensemble_size: int = 5
    ):
        self.task_type = task_type
        self.metric = self._get_metric_fn(metric_to_optimize)
        self.goal = optimization_goal
        self.ensemble_size = ensemble_size
        logging.info(f"GreedyEnsembleBuilder initialized for '{task_type}' task.")

    def _get_metric_fn(self, metric_name: str):
        """Maps a metric name to a callable scoring function."""
        # This would be more extensive in a real system
        if self.task_type == 'classification': return roc_auc_score
        return mean_squared_error

    def build(
        self,
        population: List[ExecutionResult],
        validation_data: pd.DataFrame,
        validation_target: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Constructs the ensemble from a population of successful script runs.

        Returns:
            A list of dictionaries, each representing a selected model.
        """
        if not population:
            logging.warning("Cannot build ensemble from an empty population.")
            return []

        # --- FIX STARTS HERE ---
        # Don't assume 'x' exists. Create a unique ID for each result.
        candidate_models: Dict[str, ExecutionResult] = {
            f"run_{uuid.uuid4().hex[:8]}": res for res in population
        }
        
        # In a real system, you would load saved models and get their real predictions.
        # For this example, we generate dummy predictions based on the model's score.
        candidate_preds: Dict[str, np.ndarray] = {
             uid: np.full(10, res.metrics.get(self.metric, 0.0))
             for uid, res in candidate_models.items()
        }
        # --- FIX ENDS HERE ---

        ensemble_uids = []
        ensemble_preds = None
        best_ensemble_score = -float('inf') if self.goal == 'maximize' else float('inf')

        for i in range(min(self.ensemble_size, len(candidate_models))):
            best_candidate_for_round = None
            best_score_for_round = best_ensemble_score
            best_candidate_preds_for_round = None

            for uid, preds in candidate_preds.items():
                if uid in ensemble_uids: continue

                if ensemble_preds is None:
                    current_preds = preds
                else:
                    current_preds = (ensemble_preds * len(ensemble_uids) + preds) / (len(ensemble_uids) + 1)
                
                # Use a dummy metric (mean of predictions) for this example
                score = np.mean(current_preds)

                is_better = (self.goal == 'maximize' and score > best_score_for_round) or \
                            (self.goal == 'minimize' and score < best_score_for_round)

                if is_better:
                    best_score_for_round = score
                    best_candidate_for_round = uid
                    best_candidate_preds_for_round = preds
            
            if best_candidate_for_round:
                logging.info(f"Adding model to ensemble (size {i+1}): {best_candidate_for_round} (Score: {best_score_for_round:.4f})")
                ensemble_uids.append(best_candidate_for_round)
                if ensemble_preds is None:
                    ensemble_preds = best_candidate_preds_for_round
                else:
                    ensemble_preds = (ensemble_preds * (len(ensemble_uids) - 1) + best_candidate_preds_for_round) / len(ensemble_uids)
                best_ensemble_score = best_score_for_round
            else:
                break

        logging.info(f"Finished building ensemble of size {len(ensemble_uids)}.")
        return [candidate_models[uid].hparams_used for uid in ensemble_uids]

# --- Self-contained Test Block ---
if __name__ == '__main__':
    print("\n--- Running Test for builder.py ---")
    pop = [
        ExecutionResult(status='SUCCESS', metrics={'score': -2.0}, hparams_used={'model': 'A'}),
        ExecutionResult(status='SUCCESS', metrics={'score': -0.25}, hparams_used={'model': 'B'}),
        ExecutionResult(status='SUCCESS', metrics={'score': 0.0}, hparams_used={'model': 'C'}), # Best
        ExecutionResult(status='SUCCESS', metrics={'score': -0.25}, hparams_used={'model': 'D'}),
        ExecutionResult(status='SUCCESS', metrics={'score': -25.0}, hparams_used={'model': 'E'}), # Diverse but bad
    ]
    builder = GreedyEnsembleBuilder('regression', 'score', 'maximize')
    final_ensemble = builder.build(pop, pd.DataFrame(), pd.Series())

    assert len(final_ensemble) > 0
    assert final_ensemble[0]['model'] == 'C'
    print(f"✅ Ensemble built successfully. Selected models: {final_ensemble}")
    
    print("\n--- All builder.py tests passed! ---")