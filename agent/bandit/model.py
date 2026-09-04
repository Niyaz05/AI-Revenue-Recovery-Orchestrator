from __future__ import annotations
"""
LinUCB Contextual Bandit Model for Revenue Recovery.

Learns disjoint linear models per recovery action arm:
  p(a | x) = x^T θ_a + α * sqrt(x^T A_a^{-1} x)

Strictly respects Level 1 Safety policy by restricting choice to allowed_actions.
"""

from dataclasses import dataclass
import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple
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


@dataclass
class BanditDecision:
    selected_action: ActionType
    score: float
    confidence: float
    all_scores: Dict[str, float]
    allowed_actions: List[str]
    features: Dict[str, float]
    model_version: str = "linucb_v1"

    def to_dict(self) -> dict:
        return {
            "selected_action": self.selected_action.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "allowed_actions": self.allowed_actions,
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "model_version": self.model_version,
        }


class LinUCBBandit:
    """
    Disjoint Linear Upper Confidence Bound Contextual Bandit.
    """

    def __init__(self, n_features: int = 16, alpha: float = 1.0, l2_reg: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.l2_reg = l2_reg
        self.version = "linucb_v1"

        # Initialize A_a = I_d, b_a = 0_d for each action
        self.A: Dict[str, np.ndarray] = {
            act.value: np.identity(n_features, dtype=np.float64) * l2_reg
            for act in ALL_ACTIONS
        }
        self.b: Dict[str, np.ndarray] = {
            act.value: np.zeros(n_features, dtype=np.float64)
            for act in ALL_ACTIONS
        }
        self.n_updates: Dict[str, int] = {act.value: 0 for act in ALL_ACTIONS}

    def predict(
        self,
        x: np.ndarray,
        allowed_actions: Optional[Set[ActionType]] = None,
    ) -> BanditDecision:
        """
        Evaluate LinUCB score for all actions, but pick highest only from allowed_actions.
        """
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = set(ALL_ACTIONS)

        scores: Dict[str, float] = {}
        ucb_bonuses: Dict[str, float] = {}

        for act in ALL_ACTIONS:
            act_val = act.value
            A_inv = np.linalg.inv(self.A[act_val])
            theta_a = A_inv @ self.b[act_val]
            expected_payoff = float(x @ theta_a)
            variance = float(x @ A_inv @ x)
            ucb_bonus = self.alpha * np.sqrt(max(1e-8, variance))

            scores[act_val] = expected_payoff + ucb_bonus
            ucb_bonuses[act_val] = ucb_bonus

        # Filter only to allowed actions from Level 1 Hard Safety
        allowed_candidates = [
            (act, scores[act.value])
            for act in ALL_ACTIONS
            if act in allowed_actions
        ]

        if not allowed_candidates:
            selected_action = ActionType.STOP_RECOVERY
            best_score = scores.get(ActionType.STOP_RECOVERY.value, 0.0)
        else:
            # Pick argmax
            best_act, best_score = max(allowed_candidates, key=lambda item: item[1])
            selected_action = best_act

        # Compute pseudo-confidence: softmax-like sharpness or inverse uncertainty
        conf = float(1.0 / (1.0 + ucb_bonuses.get(selected_action.value, 1.0)))

        features_dict = {
            FEATURE_NAMES[i]: float(x[i])
            for i in range(min(len(FEATURE_NAMES), len(x)))
        }

        return BanditDecision(
            selected_action=selected_action,
            score=best_score,
            confidence=conf,
            all_scores=scores,
            allowed_actions=[a.value for a in allowed_actions],
            features=features_dict,
            model_version=self.version,
        )

    def update(self, action: ActionType, x: np.ndarray, reward: float):
        """
        Online / offline batch update: A_a += x x^T, b_a += r x
        """
        act_val = action.value
        if act_val not in self.A:
            return

        x_col = x.reshape(-1, 1).astype(np.float64)
        self.A[act_val] += x_col @ x_col.T
        self.b[act_val] += reward * x.astype(np.float64)
        self.n_updates[act_val] += 1

    def save(self, filepath: str):
        """Save model parameters to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "alpha": self.alpha,
            "l2_reg": self.l2_reg,
            "n_features": self.n_features,
            "version": self.version,
            "n_updates": self.n_updates,
        }
        for act_val, mat in self.A.items():
            data[f"A_{act_val}"] = mat
        for act_val, vec in self.b.items():
            data[f"b_{act_val}"] = vec

        np.savez_compressed(filepath, **data)
        logger.info(f"Saved LinUCB model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "LinUCBBandit":
        """Load saved LinUCB model parameters."""
        data = np.load(filepath, allow_pickle=True)
        n_feat = int(data["n_features"])
        alpha = float(data["alpha"])
        l2_reg = float(data["l2_reg"])
        model = cls(n_features=n_feat, alpha=alpha, l2_reg=l2_reg)
        model.version = str(data["version"])
        if "n_updates" in data:
            model.n_updates = data["n_updates"].item()

        for act in ALL_ACTIONS:
            act_val = act.value
            k_A = f"A_{act_val}"
            k_b = f"b_{act_val}"
            if k_A in data:
                model.A[act_val] = data[k_A]
            if k_b in data:
                model.b[act_val] = data[k_b]

        logger.info(f"Loaded LinUCB model from {filepath}")
        return model
