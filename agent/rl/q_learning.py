from __future__ import annotations
"""
Q-Learning RL Agent for experimental comparison against LinUCB Contextual Bandit.

Implements Linear Value Function Approximation with Experience Replay / Q-updates:
  Q(s, a) = w_a^T s
  w_a ← w_a + α [r + γ max_{a' ∈ Allowed} Q(s', a') - Q(s, a)] s
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Set
import numpy as np

from backend.models.enums import ActionType
from agent.bandit.features import extract_feature_vector, FEATURE_NAMES

logger = logging.getLogger(__name__)

ALL_ACTIONS = [
    ActionType.WAIT,
    ActionType.RETRY,
    ActionType.PAYMENT_LINK,
    ActionType.SEND_REMINDER,
    ActionType.REQUEST_ALTERNATE_METHOD,
    ActionType.ESCALATE_TO_HUMAN,
    ActionType.STOP_RECOVERY,
]


class LinearQLearningAgent:
    """
    Linear Q-Learning agent with safety-bounded action selection.
    """

    def __init__(
        self,
        n_features: int = 16,
        learning_rate: float = 0.01,
        gamma: float = 0.90,
        epsilon: float = 0.10,
    ):
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.version = "qlearning_v1"

        # Weights per action: shape (n_features,)
        self.weights: Dict[str, np.ndarray] = {
            act.value: np.zeros(n_features, dtype=np.float64)
            for act in ALL_ACTIONS
        }

    def q_value(self, state: np.ndarray, action: ActionType) -> float:
        return float(state @ self.weights[action.value])

    def predict(
        self,
        state: np.ndarray,
        allowed_actions: Optional[Set[ActionType]] = None,
        explore: bool = False,
    ) -> ActionType:
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = set(ALL_ACTIONS)

        allowed_list = [a for a in ALL_ACTIONS if a in allowed_actions]

        if explore and np.random.rand() < self.epsilon:
            return np.random.choice(allowed_list)

        q_vals = [(act, self.q_value(state, act)) for act in allowed_list]
        best_action, _ = max(q_vals, key=lambda item: item[1])
        return best_action

    def update(
        self,
        state: np.ndarray,
        action: ActionType,
        reward: float,
        next_state: Optional[np.ndarray],
        next_allowed_actions: Optional[Set[ActionType]],
        done: bool,
    ):
        """Bellman error linear Q-update."""
        current_q = self.q_value(state, action)
        if done or next_state is None:
            target = reward
        else:
            if next_allowed_actions is None or len(next_allowed_actions) == 0:
                next_allowed_actions = set(ALL_ACTIONS)
            next_qs = [
                self.q_value(next_state, a)
                for a in ALL_ACTIONS
                if a in next_allowed_actions
            ]
            max_next_q = max(next_qs) if next_qs else 0.0
            target = reward + self.gamma * max_next_q

        error = target - current_q
        self.weights[action.value] += self.lr * error * state.astype(np.float64)
