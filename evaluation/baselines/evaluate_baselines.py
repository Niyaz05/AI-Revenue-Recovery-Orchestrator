from __future__ import annotations
"""
Baseline Evaluation Suite.

Evaluates all 3 baselines (No Recovery, Fixed Retry, Rule-Based) across
multiple seeds on the simulation sandbox environment.
"""

import argparse
import json
import os
from typing import Dict, List
import numpy as np

from evaluation.baselines.fixed_retry import FixedRetryBaseline
from evaluation.baselines.no_recovery import NoRecoveryBaseline
from evaluation.baselines.rule_based import RuleBasedBaseline
from simulator.environment import RecoveryEnvironment
from simulator.models import SimActionType, SimState


def evaluate_policy_on_sim(
    policy_obj,
    n_episodes: int = 2000,
    seeds: List[int] = [42, 43, 44, 45, 46],
) -> dict:
    all_runs_metrics = []

    for seed in seeds:
        env = RecoveryEnvironment(seed=seed)
        
        at_risk_revenue = 0.0
        recovered_revenue = 0.0
        success_count = 0
        total_attempts = 0
        total_interventions = 0
        total_recovery_time = 0.0
        total_reward = 0.0

        for ep in range(n_episodes):
            state = env.reset(seed=seed * 10000 + ep)
            initial_amount = state.amount
            at_risk_revenue += initial_amount

            done = False
            ep_reward = 0.0
            ep_time = 0.0
            ep_attempts = 0
            ep_interventions = 0

            while not done:
                action = policy_obj.predict(state)
                outcome = env.step(state, action)
                ep_reward += outcome.reward
                ep_time = outcome.recovery_time_hours
                if action == SimActionType.RETRY:
                    ep_attempts += 1
                if action in (SimActionType.RETRY, SimActionType.PAYMENT_LINK, SimActionType.SEND_REMINDER, SimActionType.REQUEST_ALTERNATE_METHOD, SimActionType.ESCALATE_TO_HUMAN):
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
            total_reward += ep_reward

        recovery_rate = success_count / n_episodes
        avg_recovery_time = total_recovery_time / max(1, success_count)
        attempts_per_rec = total_attempts / max(1, success_count)
        interventions_per_cust = total_interventions / n_episodes
        mean_reward = total_reward / n_episodes

        all_runs_metrics.append({
            "recovery_rate": recovery_rate,
            "recovered_revenue": recovered_revenue,
            "at_risk_revenue": at_risk_revenue,
            "avg_recovery_time_hours": avg_recovery_time,
            "attempts_per_recovery": attempts_per_rec,
            "interventions_per_customer": interventions_per_cust,
            "expected_reward_per_episode": mean_reward,
        })

    # Compute mean and standard deviation across seeds
    summary = {
        "policy_name": policy_obj.name,
        "n_episodes_per_seed": n_episodes,
        "n_seeds": len(seeds),
        "recovery_rate_mean": float(np.mean([m["recovery_rate"] for m in all_runs_metrics])),
        "recovery_rate_std": float(np.std([m["recovery_rate"] for m in all_runs_metrics])),
        "recovered_revenue_mean": float(np.mean([m["recovered_revenue"] for m in all_runs_metrics])),
        "at_risk_revenue_mean": float(np.mean([m["at_risk_revenue"] for m in all_runs_metrics])),
        "avg_recovery_time_hours_mean": float(np.mean([m["avg_recovery_time_hours"] for m in all_runs_metrics])),
        "attempts_per_recovery_mean": float(np.mean([m["attempts_per_recovery"] for m in all_runs_metrics])),
        "interventions_per_customer_mean": float(np.mean([m["interventions_per_customer"] for m in all_runs_metrics])),
        "expected_reward_mean": float(np.mean([m["expected_reward_per_episode"] for m in all_runs_metrics])),
        "provenance": "SIMULATED",
    }
    return summary


def run_baseline_evaluation(output_file: str = "evaluation/results/baseline_results.json"):
    print("Evaluating Baseline 1: No Recovery...")
    res_no = evaluate_policy_on_sim(NoRecoveryBaseline())

    print("Evaluating Baseline 2: Fixed Retry Schedule...")
    res_fixed = evaluate_policy_on_sim(FixedRetryBaseline())

    print("Evaluating Baseline 3: Rule-Based Expert...")
    res_rule = evaluate_policy_on_sim(RuleBasedBaseline())

    results = {
        "no_recovery": res_no,
        "fixed_retry": res_fixed,
        "rule_based": res_rule,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved baseline evaluation results to {output_file}")
    print("\n--- Summary of Baselines ---")
    for k, v in results.items():
        print(f"[{v['policy_name']}]")
        print(f"  Recovery Rate: {v['recovery_rate_mean']:.1%} ± {v['recovery_rate_std']:.1%}")
        print(f"  Recovered Revenue: ₹{v['recovered_revenue_mean']:,.2f}")
        print(f"  Interventions / Customer: {v['interventions_per_customer_mean']:.2f}")
        print(f"  Expected Reward: ₹{v['expected_reward_mean']:,.2f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/results/baseline_results.json")
    args = parser.parse_args()
    run_baseline_evaluation(args.output)
