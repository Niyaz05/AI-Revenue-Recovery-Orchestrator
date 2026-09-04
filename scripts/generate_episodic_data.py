from __future__ import annotations
"""
Episodic Training Data Generator for the sequential RL policy.

⚠️  SIMULATION ASSUMPTION SET — provenance: SYNTHETIC_TRAINING_DATA.

Generates FULL multi-day recovery episodes (not single events). Each episode is
a sequence of (state, action, reward, next_state) transitions from first payment
failure until resolution / opt-out / window expiry.

The logging policy is ε-varied (mixture of random + heuristic) so that the same
customer-state is observed under MULTIPLE different actions across the dataset.
This action coverage is *required* for offline RL — without it, the algorithm
cannot learn reliable value comparisons between actions.

Episodes are optionally generated across domain-randomized simulator configs so
the policy learns dynamics-invariant behavior.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulator.environment import RecoveryEnvironment
from simulator.models import SimActionType
from simulator.domain_randomization import sample_training_dynamics

PROVENANCE = "SYNTHETIC_TRAINING_DATA"

HEURISTIC = {
    "temporary_bank_failure": SimActionType.RETRY,
    "network_error": SimActionType.RETRY,
    "insufficient_funds": SimActionType.PAYMENT_LINK,
    "invalid_payment_method": SimActionType.REQUEST_ALTERNATE_METHOD,
    "card_expired": SimActionType.REQUEST_ALTERNATE_METHOD,
    "authentication_failed": SimActionType.ESCALATE_TO_HUMAN,
    "fraud_suspected": SimActionType.ESCALATE_TO_HUMAN,
}


def make_logging_policy(rng: np.random.Generator, epsilon: float):
    """ε-varied logging policy: random w.p. epsilon, else a reasonable heuristic."""

    def policy(state, allowed):
        allowed_list = sorted(allowed, key=lambda a: a.value)
        if rng.random() < epsilon:
            return allowed_list[rng.integers(0, len(allowed_list))]
        cand = HEURISTIC.get(state.failure_type.value, SimActionType.PAYMENT_LINK)
        if cand in allowed:
            return cand
        # fall back to any non-STOP allowed action if possible
        non_stop = [a for a in allowed_list if a != SimActionType.STOP_RECOVERY]
        return non_stop[0] if non_stop else SimActionType.STOP_RECOVERY

    return policy


def generate_episodes(
    n_episodes: int,
    seed: int = 42,
    epsilon: float = 0.5,
    use_domain_randomization: bool = True,
    n_random_configs: int = 8,
    max_steps: int = 7,
) -> pd.DataFrame:
    print(f"Generating {n_episodes:,} full episodes (seed={seed}, epsilon={epsilon})...")

    dynamics_pool = (
        sample_training_dynamics(n_random_configs, seed=seed)
        if use_domain_randomization
        else [None]
    )

    rng = np.random.default_rng(seed)
    env = RecoveryEnvironment(seed=seed)
    rows: List[dict] = []

    for ep in range(n_episodes):
        if (ep + 1) % 10000 == 0:
            print(f"  {ep + 1:,} / {n_episodes:,} episodes...")

        # Rotate through randomized dynamics configs for robustness.
        env.dynamics = (
            dynamics_pool[ep % len(dynamics_pool)]
            if dynamics_pool[0] is not None
            else env.dynamics
        )

        policy = make_logging_policy(rng, epsilon)
        transitions = env.run_episode(policy, seed=seed + ep, max_steps=max_steps)

        ep_return = sum(tr["reward"] for tr in transitions)
        for t, tr in enumerate(transitions):
            tr["episode_id"] = f"ep_{ep:07d}"
            tr["timestep"] = t
            tr["episode_length"] = len(transitions)
            tr["episode_return"] = ep_return
            tr["provenance"] = PROVENANCE
            rows.append(tr)

    df = pd.DataFrame(rows)
    n_eps = df["episode_id"].nunique()
    resolved_rate = df.groupby("episode_id")["resolved"].max().mean()
    print(
        f"Generated {n_eps:,} episodes ({len(df):,} transitions). "
        f"Mean length={df.groupby('episode_id').size().mean():.2f} steps, "
        f"resolution rate={resolved_rate:.1%}."
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate episodic RL training data.")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of FULL episodes")
    parser.add_argument("--output", default="data/episodic/episodes.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--random-configs", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = generate_episodes(
        n_episodes=args.episodes,
        seed=args.seed,
        epsilon=args.epsilon,
        use_domain_randomization=not args.no_domain_randomization,
        n_random_configs=args.random_configs,
    )
    df.to_parquet(args.output, index=False)
    print(f"Saved episodic dataset to {args.output}")
