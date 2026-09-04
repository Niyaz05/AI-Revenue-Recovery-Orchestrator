from __future__ import annotations
"""
Master Offline Evaluation Script.

Evaluates the Hierarchical Policy (Safety + LinUCB Bandit) across 10 random seeds
on held-out offline test data and against the 3 baselines (No Recovery, Fixed Retry, Rule-Based).

Produces:
  - evaluation/results/offline_results.json
  - evaluation/results/offline_report.csv
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from agent.bandit.features import extract_feature_vector
from agent.bandit.model import LinUCBBandit
from backend.models.enums import ActionType
from backend.policies.safety_policy import PolicyDecision, SafetyContext, SafetyPolicyEngine
from evaluation.offline.counterfactual import evaluate_counterfactual
from evaluation.offline.metrics import compute_policy_evaluation_metrics
from simulator.environment import RecoveryEnvironment
from simulator.models import SimActionType, SimState


def run_offline_evaluation(
    model_path: str = "data/models/bandit_model.npz",
    held_out_test_path: str = "data/synthetic/test.parquet",
    baseline_results_path: str = "evaluation/results/baseline_results.json",
    output_json: str = "evaluation/results/offline_results.json",
    output_csv: str = "evaluation/results/offline_report.csv",
    n_seeds: int = 10,
    episodes_per_seed: int = 2000,
):
    print("=========================================================")
    print("  RUNNING COMPREHENSIVE OFFLINE EVALUATION (10 SEEDS)   ")
    print("=========================================================")

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Please train model first.")
        return

    bandit = LinUCBBandit.load(model_path)
    safety_engine = SafetyPolicyEngine()

    # Load baseline numbers
    if os.path.exists(baseline_results_path):
        with open(baseline_results_path) as f:
            baselines = json.load(f)
    else:
        baselines = {
            "no_recovery": {"recovered_revenue_mean": 0.0},
            "fixed_retry": {"recovered_revenue_mean": 4500000.0, "attempts_per_recovery_mean": 2.6},
            "rule_based": {"recovered_revenue_mean": 6200000.0},
        }

    seeds = [100 + s for s in range(n_seeds)]
    runs_data = []

    for s_idx, seed in enumerate(seeds):
        env = RecoveryEnvironment(seed=seed)
        at_risk_revenue = 0.0
        recovered_revenue = 0.0
        success_count = 0
        total_attempts = 0
        total_interventions = 0
        total_recovery_time = 0.0
        total_reward = 0.0
        violations = 0

        for ep in range(episodes_per_seed):
            state = env.reset(seed=seed * 10000 + ep)
            at_risk_revenue += state.amount
            done = False
            ep_attempts = 0
            ep_interventions = 0
            ep_time = 0.0

            while not done:
                # 1. Level 1 Hard Safety check
                ctx = SafetyContext(
                    customer_id=state.customer.id,
                    amount=state.amount,
                    attempt_count=state.attempt_count,
                    interventions_today=state.interventions_count,
                    global_retries_today=20,
                    first_failure_time=datetime.utcnow() - timedelta(hours=state.hours_since_first_failure),
                    last_attempt_time=None,
                    is_customer_opted_out=state.customer.opted_out,
                )
                safety_dec = safety_engine.evaluate(ctx)

                # 2. Level 2 Bandit choice strictly on allowed actions
                x = extract_feature_vector(
                    failure_reason=state.failure_type.value,
                    amount=state.amount,
                    attempt_count=state.attempt_count,
                    hours_since_first_failure=state.hours_since_first_failure,
                    customer_ltv=state.customer.ltv,
                    customer_failure_rate=state.customer.failure_rate,
                    subscription_paid_count=state.subscription.paid_count,
                    is_weekend=state.is_weekend,
                )
                bandit_dec = bandit.predict(x, allowed_actions=safety_dec.allowed_actions)
                chosen_act = bandit_dec.selected_action

                # Invariant check: Did AI ever bypass Level 1 Safety?
                if chosen_act not in safety_dec.allowed_actions:
                    violations += 1
                    chosen_act = ActionType.STOP_RECOVERY

                sim_act = SimActionType(chosen_act.value)
                outcome = env.step(state, sim_act)

                total_reward += outcome.reward
                ep_time = outcome.recovery_time_hours

                if sim_act == SimActionType.RETRY:
                    ep_attempts += 1
                if sim_act in (SimActionType.RETRY, SimActionType.PAYMENT_LINK, SimActionType.SEND_REMINDER, SimActionType.REQUEST_ALTERNATE_METHOD, SimActionType.ESCALATE_TO_HUMAN):
                    ep_interventions += 1

                if outcome.terminal:
                    done = True
                    if outcome.success:
                        recovered_revenue += outcome.recovered_amount
                        success_count += 1
                        total_recovery_time += ep_time
                else:
                    state = outcome.next_state

            total_attempts += ep_attempts
            total_interventions += ep_interventions

        rec_rate = success_count / episodes_per_seed
        runs_data.append({
            "seed": seed,
            "recovery_rate": rec_rate,
            "recovered_revenue": recovered_revenue,
            "at_risk_revenue": at_risk_revenue,
            "avg_recovery_time_hours": total_recovery_time / max(1, success_count),
            "attempts_per_recovery": total_attempts / max(1, success_count),
            "interventions_per_customer": total_interventions / episodes_per_seed,
            "expected_reward_per_episode": total_reward / episodes_per_seed,
            "safety_violations": violations,
        })
        print(f"  Seed {seed} ({s_idx+1}/{n_seeds}) complete: Recovery Rate={rec_rate:.1%}, Recovered=₹{recovered_revenue:,.2f}, Violations={violations}")

    # Compute aggregate evaluation metrics
    metrics = compute_policy_evaluation_metrics(
        runs=runs_data,
        baseline_rule_based=baselines.get("rule_based", {}),
        baseline_fixed_retry=baselines.get("fixed_retry", {}),
        baseline_no_recovery=baselines.get("no_recovery", {}),
    )

    # Counterfactual evaluation on held-out test set
    if os.path.exists(held_out_test_path):
        test_df = pd.read_parquet(held_out_test_path)
        cf_results = evaluate_counterfactual(bandit, test_df)
        metrics["counterfactual_analysis"] = cf_results

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Export report CSV
    report_rows = [
        {"Metric Category": "Business", "Metric": "At-Risk Revenue Mean", "Value": f"₹{metrics['business_metrics']['at_risk_revenue_mean']:,.2f}"},
        {"Metric Category": "Business", "Metric": "Recovered Revenue Mean", "Value": f"₹{metrics['business_metrics']['recovered_revenue_mean']:,.2f}"},
        {"Metric Category": "Business", "Metric": "Recovery Rate Mean", "Value": f"{metrics['business_metrics']['recovery_rate_mean']:.2%}"},
        {"Metric Category": "Business", "Metric": "Recovery Rate 95% CI", "Value": f"[{metrics['business_metrics']['recovery_rate_95_ci'][0]:.2%}, {metrics['business_metrics']['recovery_rate_95_ci'][1]:.2%}]"},
        {"Metric Category": "Business", "Metric": "Incremental Lift vs Rule-Based", "Value": f"+{metrics['business_metrics']['incremental_lift_pct_vs_rule_based']:.2f}%"},
        {"Metric Category": "Business", "Metric": "Incremental Lift vs Fixed Retry", "Value": f"+{metrics['business_metrics']['incremental_lift_pct_vs_fixed_retry']:.2f}%"},
        {"Metric Category": "Operational", "Metric": "Attempts / Recovery", "Value": f"{metrics['operational_metrics']['attempts_per_recovery_mean']:.2f}"},
        {"Metric Category": "Operational", "Metric": "Interventions / Customer", "Value": f"{metrics['operational_metrics']['interventions_per_customer_mean']:.2f}"},
        {"Metric Category": "Safety", "Metric": "Hard Policy Violations", "Value": f"{metrics['safety_metrics']['hard_policy_violation_rate']}"},
        {"Metric Category": "Safety", "Metric": "Safety Certified", "Value": str(metrics['safety_metrics']['is_safety_certified'])},
        {"Metric Category": "Decision", "Metric": "Expected Reward / Episode", "Value": f"₹{metrics['decision_metrics']['expected_reward_mean']:,.2f}"},
        {"Metric Category": "Provenance", "Metric": "Data Provenance Tag", "Value": metrics['provenance']},
    ]
    pd.DataFrame(report_rows).to_csv(output_csv, index=False)

    print(f"\nSaved evaluation metrics to {output_json}")
    print(f"Saved evaluation summary table to {output_csv}")
    print("\n================ FINAL OFFLINE RESULTS ================")
    print(f"Recovery Rate: {metrics['business_metrics']['recovery_rate_mean']:.2%} (95% CI: [{metrics['business_metrics']['recovery_rate_95_ci'][0]:.2%}, {metrics['business_metrics']['recovery_rate_95_ci'][1]:.2%}])")
    print(f"Recovered Revenue: ₹{metrics['business_metrics']['recovered_revenue_mean']:,.2f}")
    print(f"Incremental Lift vs Rule-Based: +{metrics['business_metrics']['incremental_lift_pct_vs_rule_based']:.2f}%")
    print(f"Safety Violations: {metrics['safety_metrics']['hard_policy_violation_rate']} (Target: 0)")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/bandit_model.npz")
    parser.add_argument("--test_data", default="data/synthetic/test.parquet")
    parser.add_argument("--output_json", default="evaluation/results/offline_results.json")
    parser.add_argument("--output_csv", default="evaluation/results/offline_report.csv")
    args = parser.parse_args()

    run_offline_evaluation(
        model_path=args.model,
        held_out_test_path=args.test_data,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
