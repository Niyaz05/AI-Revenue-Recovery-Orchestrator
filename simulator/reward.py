from __future__ import annotations
"""
Simulation Reward Function and Sensitivity Analysis.

Formal Objective Function:
  Reward = recovered_revenue
           - intervention_cost
           - unnecessary_retry_penalty
           - customer_friction_penalty
           - policy_violation_penalty

Policy violation penalty is set to a severe negative value (-100,000), ensuring
that violating merchant safety rules is strictly dominated by safe fallback actions.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class RewardWeights:
    recovered_revenue_weight: float = 1.0
    intervention_cost_weight: float = 1.0
    retry_penalty_weight: float = 50.0       # ₹50 per unnecessary retry (gateway fee + friction)
    friction_penalty_weight: float = 100.0   # ₹100 per customer-facing friction unit
    policy_violation_penalty: float = 100000.0  # Massive penalty ensuring zero-violation optimality
    # Expected future-LTV lost to over-contacting. Modeled as a probability of
    # reduced future renewal, scaled by the customer's LTV, that grows with
    # contacts beyond the first — and accelerates past 3 (spam threshold).
    # This is what makes "fewer touches for the same recovery" show up as a
    # genuine revenue advantage rather than only an efficiency footnote.
    churn_risk_per_contact: float = 0.03     # +3% non-renewal risk per contact beyond the 1st
    churn_risk_spam_multiplier: float = 2.0  # risk accelerates beyond 3 contacts


# Default direct execution costs per action type (in INR)
ACTION_COSTS: Dict[str, float] = {
    "WAIT": 0.0,
    "RETRY": 5.0,              # Razorpay gateway / bank processing attempt cost
    "PAYMENT_LINK": 15.0,      # Payment link generation + SMS/Email notification delivery cost
    "SEND_REMINDER": 10.0,     # Multichannel reminder dispatch
    "REQUEST_ALTERNATE_METHOD": 20.0, # Specialized checkout reconfiguration + SMS
    "ESCALATE_TO_HUMAN": 150.0,# Merchant support agent labor cost per ticket
    "STOP_RECOVERY": 0.0,
}

# Friction impact per action type (scale 0..3)
ACTION_FRICTION: Dict[str, float] = {
    "WAIT": 0.0,
    "RETRY": 0.2,             # Silent retry, low direct user annoyance
    "PAYMENT_LINK": 1.0,     # Customer receives recovery link
    "SEND_REMINDER": 1.2,    # Notification ping
    "REQUEST_ALTERNATE_METHOD": 1.5, # Asks customer to re-enter billing credentials
    "ESCALATE_TO_HUMAN": 0.8, # White-glove outreach (high touch, moderate friction)
    "STOP_RECOVERY": 0.0,
}


def compute_episode_reward(
    recovered_amount: float,
    action_costs: List[float],
    contacts_count: int,
    resolved: bool,
    weights: RewardWeights = RewardWeights(),
    customer_ltv: float | None = None,
) -> float:
    """
    Terminal (episode-level) reward for the sequential RL policy.

    Concentrates the real signal at episode termination:
        terminal_reward = recovered_amount_if_resolved
                          - sum(per-action gateway/SMS costs across the episode)
                          - fatigue_penalty(cumulative_contacts)
                          - churn_risk_penalty(cumulative_contacts, customer_ltv)

    Intermediate timesteps should carry near-zero reward (the agent should be
    motivated to resolve the episode, not to farm step-level bonuses).

    `contacts_count` is the number of customer-facing touches over the whole
    episode; fatigue grows super-linearly past 3 contacts (spam guardrail).

    `customer_ltv` grounds the churn-risk term in an actual expected-value
    cost: each contact beyond the first carries a modeled probability of
    reducing future renewal, and that probability is priced against the
    customer's lifetime value — not a flat per-contact number. This is what
    lets a policy that resolves the same case with fewer touches show a real
    revenue advantage over one that resolves it with more, instead of the two
    looking identical because the direct dispatch costs are too small to
    matter next to the recovered amount.
    """
    revenue_component = weights.recovered_revenue_weight * (
        recovered_amount if resolved else 0.0
    )
    total_cost = weights.intervention_cost_weight * float(sum(action_costs))

    # Fatigue penalty: grows quadratically beyond 3 customer contacts.
    # This makes hammering a customer across a whole episode strictly dominated.
    over_contacts = max(0, contacts_count - 3)
    fatigue_penalty = weights.friction_penalty_weight * (contacts_count * 0.2 + 0.5 * over_contacts ** 2)

    # Churn-risk penalty: expected future LTV lost to over-contacting.
    ltv_basis = customer_ltv if customer_ltv is not None else recovered_amount
    over_first = max(0, contacts_count - 1)
    churn_risk_prob = weights.churn_risk_per_contact * over_first
    churn_risk_prob += weights.churn_risk_per_contact * weights.churn_risk_spam_multiplier * (over_contacts ** 2)
    churn_risk_penalty = churn_risk_prob * ltv_basis

    return float(revenue_component - total_cost - fatigue_penalty - churn_risk_penalty)


def compute_reward(
    recovered_revenue: float,
    action_type: str,
    is_unnecessary_retry: bool = False,
    is_policy_violation: bool = False,
    excessive_intervention_count: int = 0,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """
    Computes the scalar net business reward for an intervention.
    """
    if is_policy_violation:
        return -weights.policy_violation_penalty

    revenue_component = weights.recovered_revenue_weight * recovered_revenue
    direct_cost = weights.intervention_cost_weight * ACTION_COSTS.get(action_type, 0.0)

    retry_penalty = (
        weights.retry_penalty_weight if (action_type == "RETRY" and is_unnecessary_retry) else 0.0
    )

    base_friction = ACTION_FRICTION.get(action_type, 0.0)
    # Extra fatigue penalty for spamming customer (>3 interventions)
    fatigue_multiplier = 1.0 + (0.5 * max(0, excessive_intervention_count - 3))
    friction_penalty = weights.friction_penalty_weight * base_friction * fatigue_multiplier

    net_reward = revenue_component - direct_cost - retry_penalty - friction_penalty
    return float(net_reward)


def run_reward_sensitivity_analysis() -> Dict[str, List[Dict[str, float]]]:
    """
    Sensitivity analysis across varying cost, penalty, and friction weight configurations.
    Returns sensitivity metrics for reporting and validation.
    """
    configs = [
        ("Base Case", RewardWeights()),
        ("High Friction Aversion", RewardWeights(friction_penalty_weight=250.0)),
        ("High Gateway Retry Cost", RewardWeights(retry_penalty_weight=150.0, intervention_cost_weight=2.0)),
        ("Aggressive Recovery Mode", RewardWeights(friction_penalty_weight=30.0, retry_penalty_weight=20.0)),
    ]

    scenarios = [
        {"name": "Successful Payment Link (₹5,000)", "amount": 5000.0, "action": "PAYMENT_LINK", "rec": 5000.0, "unrec_retry": False, "viol": False, "interv": 1},
        {"name": "Failed Silent Retry (₹5,000)", "amount": 5000.0, "action": "RETRY", "rec": 0.0, "unrec_retry": True, "viol": False, "interv": 1},
        {"name": "Spam Reminder (4th attempt)", "amount": 5000.0, "action": "SEND_REMINDER", "rec": 0.0, "unrec_retry": False, "viol": False, "interv": 4},
        {"name": "High-Touch Human Escalation (₹50,000)", "amount": 50000.0, "action": "ESCALATE_TO_HUMAN", "rec": 50000.0, "unrec_retry": False, "viol": False, "interv": 1},
        {"name": "Safety Policy Violation", "amount": 5000.0, "action": "RETRY", "rec": 5000.0, "unrec_retry": False, "viol": True, "interv": 1},
    ]

    results = {}
    for cfg_name, cfg in configs:
        cfg_results = []
        for sc in scenarios:
            r = compute_reward(
                recovered_revenue=sc["rec"],
                action_type=sc["action"],
                is_unnecessary_retry=sc["unrec_retry"],
                is_policy_violation=sc["viol"],
                excessive_intervention_count=sc["interv"],
                weights=cfg,
            )
            cfg_results.append({"scenario": sc["name"], "reward": round(r, 2)})
        results[cfg_name] = cfg_results

    return results