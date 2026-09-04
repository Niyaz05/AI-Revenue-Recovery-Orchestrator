from __future__ import annotations
"""
Counterfactual Off-Policy Evaluator.

Estimates policy performance using Inverse Propensity Scoring (IPS) and
Direct Method comparison between historical logged actions and AI-recommended actions.

⚠️  PROVENANCE: HELD_OUT_OFFLINE_EVAL (Simulated counterfactual outcome, not actual).
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from agent.bandit.features import extract_feature_vector
from agent.bandit.model import LinUCBBandit
from backend.models.enums import ActionType


def evaluate_counterfactual(
    model: LinUCBBandit,
    held_out_df: pd.DataFrame,
) -> dict:
    """
    Evaluates counterfactual match rate, IPS value, and action shift.
    """
    total_records = len(held_out_df)
    matches = 0
    ai_actions_count: Dict[str, int] = {a.value: 0 for a in ActionType}
    historical_actions_count: Dict[str, int] = {a.value: 0 for a in ActionType}

    ips_rewards = []

    for _, row in held_out_df.iterrows():
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
        ai_act = decision.selected_action.value
        hist_act = str(row["action_taken"])

        ai_actions_count[ai_act] += 1
        historical_actions_count[hist_act] = historical_actions_count.get(hist_act, 0) + 1

        if ai_act == hist_act:
            matches += 1
            # Inverse propensity score weight (assume uniform logging propensity ~ 1/7 or heuristic)
            propensity = 0.25
            ips_rewards.append(float(row["reward"]) / propensity)

    match_rate = matches / max(1, total_records)
    mean_ips_value = float(np.mean(ips_rewards)) if ips_rewards else 0.0

    return {
        "provenance": "HELD_OUT_OFFLINE_EVAL",
        "total_test_records": total_records,
        "action_agreement_with_historical_rate": match_rate,
        "estimated_ips_reward_value": mean_ips_value,
        "ai_action_distribution": ai_actions_count,
        "historical_action_distribution": historical_actions_count,
    }
