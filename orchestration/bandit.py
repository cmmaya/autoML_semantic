# automl_lib/orchestration/bandit.py
"""
Implements a multi-armed bandit for intelligent strategy selection.

This module provides a UCB1 (Upper Confidence Bound) bandit that balances
exploration and exploitation to learn which mutation strategies are most
effective over time.
"""

import logging
import math
from typing import List, Dict

from mutation.strategies.base_strategy import BaseStrategy

class UCB1Bandit:
    """
    An implementation of the UCB1 multi-armed bandit algorithm.
    """
    def __init__(self, strategies: List[BaseStrategy], exploration_factor: float = 2.0):
        """
        Initializes the bandit with a list of strategies (arms).

        Args:
            strategies: The list of BaseStrategy objects to choose from.
            exploration_factor: A constant that balances exploration and exploitation.
                                Higher values encourage exploring less-tried arms.
        """
        if not strategies:
            raise ValueError("Bandit must be initialized with at least one strategy.")
        
        self.strategies = {s.name: s for s in strategies}
        self.exploration_factor = exploration_factor
        
        # Internal state to track performance of each arm
        self.pull_counts: Dict[str, int] = {name: 0 for name in self.strategies}
        self.rewards: Dict[str, float] = {name: 0.0 for name in self.strategies}
        self.total_pulls: int = 0
        
        logging.info(f"UCB1Bandit initialized with {len(strategies)} arms.")

    def select_arm(self) -> BaseStrategy:
        """
        Selects the best strategy (arm) to pull based on the UCB1 formula.

        The formula is: average_reward + exploration_term
        This ensures all arms are tried at least once before exploitation begins.

        Returns:
            The selected BaseStrategy object.
        """
        # First, ensure every arm is tried at least once (cold start)
        for name in self.strategies:
            if self.pull_counts[name] == 0:
                logging.debug(f"Bandit cold start: selecting untried arm '{name}'.")
                return self.strategies[name]

        # Calculate UCB1 scores for all arms
        ucb_scores: Dict[str, float] = {}
        for name, strategy in self.strategies.items():
            avg_reward = self.rewards[name] / self.pull_counts[name]
            exploration_term = self.exploration_factor * math.sqrt(
                math.log(self.total_pulls) / self.pull_counts[name]
            )
            ucb_scores[name] = avg_reward + exploration_term
        
        best_arm_name = max(ucb_scores, key=ucb_scores.get)
        logging.debug(f"Bandit scores: {ucb_scores}. Selected arm: '{best_arm_name}'.")
        return self.strategies[best_arm_name]

    def update(self, strategy_name: str, reward: float):
        """
        Updates the state of an arm after it has been pulled.

        Args:
            strategy_name: The name of the strategy that was used.
            reward: The performance improvement observed (can be positive or negative).
        """
        if strategy_name not in self.strategies:
            logging.warning(f"Attempted to update a non-existent arm: '{strategy_name}'.")
            return
            
        self.pull_counts[strategy_name] += 1
        self.rewards[strategy_name] += reward
        self.total_pulls += 1
        logging.debug(f"Bandit updated arm '{strategy_name}' with reward {reward:.4f}.")

# --- Self-contained Test Block ---
if __name__ == '__main__':
    from mutation.strategies.hyperparameters import RandomHyperparameterMutation
    
    print("\n--- Running Test for bandit.py ---")

    # 1. Setup strategies (arms)
    strategy1 = RandomHyperparameterMutation()
    strategy1.name = "MutatorA" # Give them unique names for clarity
    strategy2 = RandomHyperparameterMutation()
    strategy2.name = "MutatorB"
    
    bandit = UCB1Bandit(strategies=[strategy1, strategy2])
    
    # 2. Test cold start
    print("\n--- Testing cold start ---")
    arm1 = bandit.select_arm()
    assert arm1.name == "MutatorA"
    print(f"✅ Selected '{arm1.name}' on first pull.")
    bandit.update(arm1.name, reward=0.5)
    
    arm2 = bandit.select_arm()
    assert arm2.name == "MutatorB"
    print(f"✅ Selected '{arm2.name}' on second pull.")
    bandit.update(arm2.name, reward=0.1) # MutatorB is worse

    # 3. Test exploitation
    print("\n--- Testing exploitation ---")
    # After both have been pulled, the bandit should exploit the better one (A)
    selections = [bandit.select_arm().name for _ in range(10)]
    count_A = selections.count("MutatorA")
    count_B = selections.count("MutatorB")
    print(f"Selections over 10 rounds: A={count_A}, B={count_B}")
    assert count_A > count_B
    print("✅ Bandit correctly exploited the higher-reward arm.")
    
    print("\n--- All bandit.py tests passed! ---")