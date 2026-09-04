from __future__ import annotations
"""
Robustness Evaluation Script.

⚠️  PROVENANCE: SIMULATED

Evaluates the trained sequential RL policy (CQL) and baselines across a
held-out set of domain-randomized simulator configurations that were NOT seen
during training. Reports results as a DISTRIBUTION/RANGE (min, max, mean, std)
— not a single number — to demonstrate that the policy generalises across
plausible variations of real payment-recovery dynamics.

If the sequential RL policy does not outperform baselines across the range,
that failure is surfaced explicitly.

Usage:
    python scripts/robustness_eval.py \
        --cql-model data/models/cql_model.pt \
        --n-configs 20 \
        --episodes-per-config 1000 \
        --output evaluation/results/robustness_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from agent.rl.features import row_to_rl_features, parse_allowed_actions
from agent.rl_sequential.cql import CQLAgent
from simulator.domain_randomization import sample_heldout_dynamics
from simulator.environment import RecoveryEnvironment
from simulator.models import SimActionType, SimState, ACTION_INDEX

PROVENANCE = "SIMULATED"


# ───────────────────────── Policy Wrappers ────────────────────────────────

def _cql_policy(agent: CQLAgent):
    """Wrap CQLAgent as a policy_fn(state, allowed) -> action."""
    def policy_fn(state: SimState, allowed: Set[SimActionType]) -> SimActionType:
        s = row_to_rl_features(_state_to_row(state))
        return agent.predict(s, allowed_actions=allowed)
    return policy_fn


def _rule_based_policy(state: SimState, allowed: Set[SimActionType]) -> SimActionType:
    """Simple rule-based heuristic baseline."""
    ft = state.failure_type.value
    HEURISTIC = {
        "temporary_bank_failure": SimActionType.RETRY,
        "network_error": SimActionType.RETRY,
        "insufficient_funds": SimActionType.PAYMENT_LINK,
        "invalid_payment_method": SimActionType.REQUEST_ALTERNATE_METHOD,
        "card_expired": SimActionType.REQUEST_ALTERNATE_METHOD,
        "authentication_failed": SimActionType.ESCALATE_TO_HUMAN,
        "fraud_suspected": SimActionType.ESCALATE_TO_HUMAN,
    }
    cand = HEURISTIC.get(ft, SimActionType.PAYMENT_LINK)
    if cand in allowed:
        return cand
    non_stop = [a for a in sorted(allowed, key=lambda a: a.value) if a != SimActionType.STOP_RECOVERY]
    return non_stop[0] if non_stop else SimActionType.STOP_RECOVERY


def _fixed_retry_policy(state: SimState, allowed: Set[SimActionType]) -> SimActionType:
    """Always retry if allowed, else stop."""
    return SimActionType.RETRY if SimActionType.RETRY in allowed else SimActionType.STOP_RECOVERY


def _no_recovery_policy(state: SimState, allowed: Set[SimActionType]) -> SimActionType:
    """Never attempt recovery."""
    return SimActionType.STOP_RECOVERY


def _state_to_row(state: SimState) -> dict:
    """Convert SimState to a dict compatible with row_to_rl_features."""
    return {
        "failure_reason": state.failure_type.value,
        "amount": state.amount,
        "attempt_count": state.attempt_count,
        "hours_since_first_failure": state.hours_since_first_failure,
        "customer_ltv": state.customer.ltv,
        "customer_failure_rate": state.customer.failure_rate,
        "subscription_paid_count": state.subscription.paid_count,
        "is_weekend": int(state.is_weekend),
        "account_tenure_days": state.customer.account_tenure_days,
        "lifetime_success_rate": state.customer.lifetime_success_rate,
        "days_since_last_success": state.customer.days_since_last_success,
        "historical_ltv": state.customer.historical_ltv,
        "is_first_ever_failure": int(state.customer.is_first_ever_failure),
    }


# ───────────────────────── Evaluation Core ────────────────────────────────

def evaluate_policy_on_config(
    policy_fn,
    dynamics,
    n_episodes: int,
    seed: int,
) -> Dict[str, float]:
    """Evaluate a single policy on a single dynamics config."""
    env = RecoveryEnvironment(seed=seed, dynamics=dynamics)
    total_return = 0.0
    total_resolved = 0
    total_latency = 0.0
    total_contacts = 0
    total_steps = 0
    violations = 0

    for ep in range(n_episodes):
        transitions = env.run_episode(policy_fn, seed=seed + ep)
        ep_return = sum(tr["reward"] for tr in transitions)
        total_return += ep_return
        total_steps += len(transitions)

        # Check terminal resolution
        last = transitions[-1]
        if last.get("resolved", False):
            total_resolved += 1
            total_latency += last.get("hours_since_first_failure", 0.0)

        # Count customer-facing contacts
        for tr in transitions:
            if tr["action"] in ("PAYMENT_LINK", "SEND_REMINDER", "REQUEST_ALTERNATE_METHOD", "ESCALATE_TO_HUMAN"):
                total_contacts += 1

    return {
        "episode_return_mean": total_return / max(1, n_episodes),
        "recovery_rate": total_resolved / max(1, n_episodes),
        "recovery_latency_hours_mean": total_latency / max(1, total_resolved) if total_resolved else 0.0,
        "contacts_per_episode_mean": total_contacts / max(1, n_episodes),
        "avg_episode_length": total_steps / max(1, n_episodes),
        "safety_violations": violations,
    }


def run_robustness_eval(
    cql_model_path: str,
    n_configs: int = 20,
    episodes_per_config: int = 1000,
    seed: int = 999,
    output_path: str = "evaluation/results/robustness_results.json",
):
    print("=" * 65)
    print("  ROBUSTNESS EVALUATION — DOMAIN-RANDOMIZED HELD-OUT CONFIGS")
    print("=" * 65)

    # Load CQL agent
    if not os.path.exists(cql_model_path):
        print(f"ERROR: CQL model not found at {cql_model_path}. Train it first.")
        return
    cql_agent = CQLAgent.load(cql_model_path)
    print(f"Loaded CQL model from {cql_model_path}")

    # Sample held-out dynamics configs (disjoint from training seed)
    dynamics_configs = sample_heldout_dynamics(n_configs, seed=seed)
    print(f"Sampled {n_configs} held-out dynamics configs (seed={seed})")

    policies = {
        "Sequential RL (CQL)": _cql_policy(cql_agent),
        "Rule-Based": _rule_based_policy,
        "Fixed Retry": _fixed_retry_policy,
        "No Recovery": _no_recovery_policy,
    }

    results: Dict[str, Any] = {
        "provenance": PROVENANCE,
        "n_configs": n_configs,
        "episodes_per_config": episodes_per_config,
        "seed": seed,
        "policies": {},
    }

    for policy_name, policy_fn in policies.items():
        print(f"\n  Evaluating: {policy_name}")
        config_results = []
        for ci, dynamics in enumerate(dynamics_configs):
            cr = evaluate_policy_on_config(
                policy_fn, dynamics, episodes_per_config, seed=seed * 100 + ci
            )
            config_results.append(cr)
            if (ci + 1) % 5 == 0:
                print(f"    Config {ci+1}/{n_configs}: return={cr['episode_return_mean']:.1f}, "
                      f"recovery={cr['recovery_rate']:.1%}")

        # Aggregate across configs
        returns = [c["episode_return_mean"] for c in config_results]
        rates = [c["recovery_rate"] for c in config_results]
        latencies = [c["recovery_latency_hours_mean"] for c in config_results]
        contacts = [c["contacts_per_episode_mean"] for c in config_results]

        summary = {
            "episode_return": {
                "min": float(np.min(returns)),
                "max": float(np.max(returns)),
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
            },
            "recovery_rate": {
                "min": float(np.min(rates)),
                "max": float(np.max(rates)),
                "mean": float(np.mean(rates)),
                "std": float(np.std(rates)),
            },
            "recovery_latency_hours": {
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
            },
            "contacts_per_episode": {
                "min": float(np.min(contacts)),
                "max": float(np.max(contacts)),
                "mean": float(np.mean(contacts)),
                "std": float(np.std(contacts)),
            },
            "safety_violations_total": sum(c["safety_violations"] for c in config_results),
            "per_config_results": config_results,
        }
        results["policies"][policy_name] = summary

        print(f"    Summary — Return: {summary['episode_return']['mean']:.1f} "
              f"[{summary['episode_return']['min']:.1f}, {summary['episode_return']['max']:.1f}], "
              f"Recovery: {summary['recovery_rate']['mean']:.1%} "
              f"[{summary['recovery_rate']['min']:.1%}, {summary['recovery_rate']['max']:.1%}]")

    # ── Comparison summary ──
    print("\n" + "=" * 65)
    print("  ROBUSTNESS COMPARISON (mean ± std across configs)")
    print("-" * 65)
    print(f"  {'Policy':<25} {'Return':>12} {'Recovery':>12} {'Violations':>12}")
    print("-" * 65)
    for pn, ps in results["policies"].items():
        ret = ps["episode_return"]
        rec = ps["recovery_rate"]
        viol = ps["safety_violations_total"]
        print(f"  {pn:<25} {ret['mean']:>8.1f}±{ret['std']:<4.1f} "
              f"{rec['mean']:>8.1%}±{rec['std']:<5.1%} {viol:>10}")
    print("=" * 65)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved robustness results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness evaluation across randomized configs.")
    parser.add_argument("--cql-model", default="data/models/cql_model.pt")
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--episodes-per-config", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output", default="evaluation/results/robustness_results.json")
    args = parser.parse_args()

    run_robustness_eval(
        cql_model_path=args.cql_model,
        n_configs=args.n_configs,
        episodes_per_config=args.episodes_per_config,
        seed=args.seed,
        output_path=args.output,
    )
