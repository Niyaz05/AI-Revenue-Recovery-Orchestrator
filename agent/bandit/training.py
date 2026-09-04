from __future__ import annotations
"""
LinUCB Training and Hyperparameter Tuning on Synthetic Historical Data.

Trains disjoint ridge models per arm using off-policy updates, and tunes
the exploration parameter α ∈ [0.1, 0.5, 1.0, 2.0, 5.0] on validation data.
"""

import logging
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from agent.bandit.features import extract_feature_vector
from agent.bandit.model import LinUCBBandit
from backend.models.enums import ActionType

logger = logging.getLogger(__name__)


def train_linucb_from_dataframe(
    df: pd.DataFrame,
    alpha: float = 1.0,
    l2_reg: float = 1.0,
) -> LinUCBBandit:
    """
    Train LinUCB disjoint linear parameters from logged historical batch data.
    """
    model = LinUCBBandit(n_features=16, alpha=alpha, l2_reg=l2_reg)

    for _, row in df.iterrows():
        x = extract_feature_vector(
            failure_reason=str(row["failure_reason"]),
            amount=float(row["amount"]),
            attempt_count=int(row["attempt_count"]),
            hours_since_first_failure=float(row["hours_since_first_failure"]),
            customer_ltv=float(row["customer_ltv"]),
            customer_failure_rate=float(row["customer_failure_rate"]),
            subscription_paid_count=int(row["subscription_paid_count"]),
            is_weekend=bool(row["is_weekend"]),
        )

        try:
            act_enum = ActionType(row["action_taken"])
            reward = float(row["reward"])
            model.update(action=act_enum, x=x, reward=reward)
        except ValueError:
            continue

    return model


def evaluate_bandit_on_dataframe(model: LinUCBBandit, df: pd.DataFrame) -> dict:
    """
    Off-policy evaluation using Direct Method / Reward Estimation on dataframe.
    """
    predicted_rewards = []
    actions_distribution = {a.value: 0 for a in ActionType}

    for _, row in df.iterrows():
        x = extract_feature_vector(
            failure_reason=str(row["failure_reason"]),
            amount=float(row["amount"]),
            attempt_count=int(row["attempt_count"]),
            hours_since_first_failure=float(row["hours_since_first_failure"]),
            customer_ltv=float(row["customer_ltv"]),
            customer_failure_rate=float(row["customer_failure_rate"]),
            subscription_paid_count=int(row["subscription_paid_count"]),
            is_weekend=bool(row["is_weekend"]),
        )

        decision = model.predict(x)
        actions_distribution[decision.selected_action.value] += 1
        predicted_rewards.append(decision.score)

    return {
        "mean_predicted_payoff": float(np.mean(predicted_rewards)),
        "std_predicted_payoff": float(np.std(predicted_rewards)),
        "actions_distribution": actions_distribution,
    }


def tune_bandit_alpha(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    alpha_candidates: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
) -> Tuple[LinUCBBandit, float, dict]:
    """
    Grid search over α on validation dataset.
    """
    best_alpha = 1.0
    best_payoff = -float("inf")
    best_model = None
    tuning_history = {}

    for alpha in alpha_candidates:
        model = train_linucb_from_dataframe(train_df, alpha=alpha)
        eval_res = evaluate_bandit_on_dataframe(model, val_df)
        payoff = eval_res["mean_predicted_payoff"]
        tuning_history[str(alpha)] = eval_res

        if payoff > best_payoff:
            best_payoff = payoff
            best_alpha = alpha
            best_model = model

    print(f"Alpha tuning complete. Best α={best_alpha} (Validation Payoff: {best_payoff:.3f})")
    return best_model, best_alpha, tuning_history
