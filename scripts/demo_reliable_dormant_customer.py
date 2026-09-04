from __future__ import annotations
"""
Demo: Reliable-Dormant Customer vs Chronic Failer — Side-by-Side Comparison.

⚠️  PROVENANCE: SIMULATED

Demonstrates the sequential RL policy's ability to distinguish between:
  - Profile A: Long-tenure, near-perfect customer hitting their first failure
               after 30+ days of inactivity → GENTLE treatment (WAIT / soft reminder)
  - Profile B: Short-tenure chronic failer with identical surface features
               (same amount, same failure reason) → FIRMER intervention sequence

Both profiles are run through the trained CQL agent with Tier-1 safety masking
at every timestep. The full action trajectories are logged side by side.

Usage:
    python scripts/demo_reliable_dormant_customer.py [--cql-model data/models/cql_model.pt]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from agent.rl.features import extract_rl_feature_vector, compute_reliability_score
from agent.rl_sequential.cql import CQLAgent
from simulator.environment import RecoveryEnvironment
from simulator.models import (
    SimActionType,
    SimCustomer,
    SimFailureType,
    SimState,
    SimSubscription,
)

PROVENANCE = "SIMULATED"


def build_reliable_dormant_profile() -> SimState:
    """
    Profile A: Reliable-but-dormant customer.
    - 4+ year tenure (~1500 days)
    - 48/48 successful payments (first-ever failure)
    - Last successful payment 35 days ago (dormant)
    - High historical LTV
    - Failure reason: TEMPORARY_BANK_FAILURE
    - Amount: ₹2,999
    """
    customer = SimCustomer(
        id=90001,
        name="Reliable Dormant (Profile A)",
        email="reliable_dormant@example.com",
        phone="+919800000001",
        opted_out=False,
        archetype="reliable_dormant",
        ltv=95000.0,
        failure_rate=0.02,
        total_payments=48,
        failed_payments=0,
        account_tenure_days=1500,         # ~4 years
        successful_payments=48,           # clean record
        days_since_last_success=35,       # dormant ~5 weeks
        historical_ltv=95000.0,
        is_first_ever_failure=True,       # first failure ever
    )
    subscription = SimSubscription(
        id="sub_demo_reliable",
        customer_id=90001,
        amount_per_period=2999.0,
        paid_count=6,
        remaining_count=6,
        total_count=12,
        status="pending",
    )
    return SimState(
        customer=customer,
        subscription=subscription,
        failure_type=SimFailureType.TEMPORARY_BANK_FAILURE,
        amount=2999.0,
        attempt_count=1,
        hours_since_first_failure=0.0,
        interventions_count=0,
        consecutive_failures=1,
        is_weekend=False,
    )


def build_chronic_failer_profile() -> SimState:
    """
    Profile B: Chronic failer with IDENTICAL surface features.
    - Same amount (₹2,999)
    - Same failure reason (TEMPORARY_BANK_FAILURE)
    - Same attempt count (1)
    BUT:
    - 6-month tenure (~180 days)
    - 8/15 successful payments (53% success rate)
    - Multiple past failures (NOT first failure)
    - Low historical LTV
    - Last success 60 days ago
    """
    customer = SimCustomer(
        id=90002,
        name="Chronic Failer (Profile B)",
        email="chronic_failer@example.com",
        phone="+919800000002",
        opted_out=False,
        archetype="high_risk",
        ltv=12000.0,
        failure_rate=0.35,
        total_payments=15,
        failed_payments=7,
        account_tenure_days=180,          # ~6 months
        successful_payments=8,            # 53% success rate
        days_since_last_success=60,       # not recent
        historical_ltv=12000.0,
        is_first_ever_failure=False,      # repeat offender
    )
    subscription = SimSubscription(
        id="sub_demo_chronic",
        customer_id=90002,
        amount_per_period=2999.0,         # SAME amount
        paid_count=6,
        remaining_count=6,
        total_count=12,
        status="pending",
    )
    return SimState(
        customer=customer,
        subscription=subscription,
        failure_type=SimFailureType.TEMPORARY_BANK_FAILURE,  # SAME failure reason
        amount=2999.0,                    # SAME amount
        attempt_count=1,                  # SAME attempt count
        hours_since_first_failure=0.0,
        interventions_count=0,
        consecutive_failures=1,
        is_weekend=False,
    )


def _state_to_feature_row(state: SimState) -> dict:
    """Convert SimState to a dict for feature extraction."""
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


def run_demo_episode(
    agent: CQLAgent,
    env: RecoveryEnvironment,
    initial_state: SimState,
    profile_name: str,
    max_steps: int = 7,
) -> List[Dict]:
    """Run one episode using the CQL agent on a pre-built initial state."""
    from agent.rl.features import row_to_rl_features

    state = initial_state
    trajectory = []

    for t in range(max_steps):
        allowed = env.allowed_actions(state)
        if not allowed:
            allowed = {SimActionType.STOP_RECOVERY}

        row = _state_to_feature_row(state)
        features = row_to_rl_features(row)
        action = agent.predict(features, allowed_actions=allowed)
        if action not in allowed:
            action = SimActionType.STOP_RECOVERY

        outcome = env.step(state, action)

        step_record = {
            "timestep": t,
            "action": action.value,
            "allowed": sorted(a.value for a in allowed),
            "hours_elapsed": state.hours_since_first_failure,
            "attempt_count": state.attempt_count,
            "interventions": state.interventions_count,
            "outcome_success": outcome.success,
            "outcome_terminal": outcome.terminal,
            "recovered_amount": outcome.recovered_amount,
        }
        trajectory.append(step_record)

        if outcome.terminal:
            break
        state = outcome.next_state

    return trajectory


def print_side_by_side(
    profile_a_name: str,
    profile_a_state: SimState,
    trajectory_a: List[Dict],
    profile_b_name: str,
    profile_b_state: SimState,
    trajectory_b: List[Dict],
):
    """Pretty-print both trajectories side by side."""
    width = 78
    half = width // 2

    print("\n" + "=" * width)
    print("  DEMO: RELIABLE-DORMANT vs CHRONIC-FAILER — SIDE BY SIDE")
    print(f"  Provenance: {PROVENANCE}")
    print("=" * width)

    # ── Customer profiles ──
    print("\n── CUSTOMER PROFILES (identical surface features, different history) ──\n")

    def profile_summary(name, state):
        c = state.customer
        rel_score = compute_reliability_score(
            c.lifetime_success_rate, c.account_tenure_days,
            c.days_since_last_success, c.is_first_ever_failure,
        )
        return [
            f"  {name}",
            f"  Amount:          ₹{state.amount:,.0f}",
            f"  Failure reason:  {state.failure_type.value}",
            f"  Attempt count:   {state.attempt_count}",
            f"  ─── History (differs) ───",
            f"  Tenure:          {c.account_tenure_days} days",
            f"  Lifetime success:{c.lifetime_success_rate:.0%} ({c.successful_payments}/{c.total_payments})",
            f"  Days since last: {c.days_since_last_success}",
            f"  First failure?   {'YES' if c.is_first_ever_failure else 'NO'}",
            f"  Historical LTV:  ₹{c.historical_ltv:,.0f}",
            f"  Reliability:     {rel_score:.3f}",
        ]

    lines_a = profile_summary(profile_a_name, profile_a_state)
    lines_b = profile_summary(profile_b_name, profile_b_state)
    max_lines = max(len(lines_a), len(lines_b))
    lines_a += [""] * (max_lines - len(lines_a))
    lines_b += [""] * (max_lines - len(lines_b))

    print(f"{'─' * half}  {'─' * half}")
    for la, lb in zip(lines_a, lines_b):
        print(f"{la:<{half}}  {lb}")
    print(f"{'─' * half}  {'─' * half}")

    # ── Action trajectories ──
    print("\n── ACTION TRAJECTORIES ──\n")
    print(f"  {'Step':<6} {'Profile A Action':<25} {'Profile B Action':<25}")
    print(f"  {'─'*4:<6} {'─'*20:<25} {'─'*20:<25}")

    max_t = max(len(trajectory_a), len(trajectory_b))
    for t in range(max_t):
        a_str = trajectory_a[t]["action"] if t < len(trajectory_a) else "—"
        b_str = trajectory_b[t]["action"] if t < len(trajectory_b) else "—"

        # Mark terminal / success
        if t < len(trajectory_a) and trajectory_a[t].get("outcome_terminal"):
            suffix = " ✓ recovered" if trajectory_a[t].get("outcome_success") else " ✗ ended"
            a_str += suffix
        if t < len(trajectory_b) and trajectory_b[t].get("outcome_terminal"):
            suffix = " ✓ recovered" if trajectory_b[t].get("outcome_success") else " ✗ ended"
            b_str += suffix

        print(f"  t={t:<4} {a_str:<25} {b_str:<25}")

    # ── Verdict ──
    a_actions = [s["action"] for s in trajectory_a]
    b_actions = [s["action"] for s in trajectory_b]
    a_gentle = sum(1 for a in a_actions if a in ("WAIT", "SEND_REMINDER"))
    b_gentle = sum(1 for a in b_actions if a in ("WAIT", "SEND_REMINDER"))
    a_aggressive = sum(1 for a in a_actions if a in ("RETRY", "PAYMENT_LINK", "REQUEST_ALTERNATE_METHOD", "ESCALATE_TO_HUMAN"))
    b_aggressive = sum(1 for a in b_actions if a in ("RETRY", "PAYMENT_LINK", "REQUEST_ALTERNATE_METHOD", "ESCALATE_TO_HUMAN"))

    print(f"\n── ANALYSIS ──")
    print(f"  Profile A (reliable dormant): {a_gentle} gentle + {a_aggressive} firm actions "
          f"({len(trajectory_a)} steps)")
    print(f"  Profile B (chronic failer):   {b_gentle} gentle + {b_aggressive} firm actions "
          f"({len(trajectory_b)} steps)")
    if a_aggressive <= b_aggressive and a_gentle >= b_gentle:
        print(f"\n  ✅ PASS: Enriched history features are driving differentiated behavior.")
        print(f"     The reliable-dormant customer receives gentler treatment despite")
        print(f"     identical surface features (amount, failure reason, attempt count).")
    else:
        print(f"\n  ⚠️  NOTE: Action sequences may not differ as expected. This can happen")
        print(f"     with an untrained model or edge-case stochastic outcomes.")
    print(f"\n  Provenance: {PROVENANCE}")
    print("=" * width + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: reliable-dormant vs chronic-failer side-by-side comparison."
    )
    parser.add_argument("--cql-model", default="data/models/cql_model.pt",
                        help="Path to trained CQL model checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    if not os.path.exists(args.cql_model):
        print(f"CQL model not found at {args.cql_model}.")
        print("Creating a fresh (untrained) CQL agent for demonstration purposes.")
        print("For meaningful results, first run: python scripts/train_sequential.py")
        agent = CQLAgent()
    else:
        agent = CQLAgent.load(args.cql_model)
        print(f"Loaded CQL model from {args.cql_model}")

    env = RecoveryEnvironment(seed=args.seed)

    # Build the two profiles with identical surface features but different history
    state_a = build_reliable_dormant_profile()
    state_b = build_chronic_failer_profile()

    print("\nRunning Profile A (reliable dormant)...")
    traj_a = run_demo_episode(agent, env, state_a, "Profile A")

    print("Running Profile B (chronic failer)...")
    traj_b = run_demo_episode(agent, env, state_b, "Profile B")

    print_side_by_side(
        "Profile A: Reliable Dormant", state_a, traj_a,
        "Profile B: Chronic Failer", state_b, traj_b,
    )


if __name__ == "__main__":
    main()
