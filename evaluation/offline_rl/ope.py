from __future__ import annotations
"""
Doubly-Robust (DR) and Importance-Sampling (IS) Off-Policy Evaluation
for the sequential RL policy.

⚠️  PROVENANCE: HELD_OUT_OFFLINE_EVAL

Off-policy evaluation estimates what the NEW target policy's return *would have
been* on held-out logged episodes, without ever executing it live. This is a
mandatory gating step: the sequential RL policy must not reach `action_executor`
or `orchestrator.py` until its OPE-estimated return has been logged and reviewed.

Two estimators are provided:

1. **Importance Sampling (IS)** — the simpler fallback.  Weight each logged
   episode's return by the product of per-step importance ratios
   π_target(a|s) / π_behavior(a|s), clipped to avoid degenerate variance.

2. **Doubly-Robust (DR)** — the primary estimator.  Combines an IS correction
   with a direct-method (DM) baseline reward model as a control variate:

       V_DR  =  V_DM  +  IS_correction(reward − reward_hat)

   DR is unbiased if *either* the DM reward model *or* the IS weights are
   correct, and has lower variance than pure IS when both are approximate.
   The DM reward model is a simple MLP fitted on training-set transitions.

References:
  - Dudík et al., "Doubly Robust Policy Evaluation and Learning" (ICML 2011)
  - Thomas & Brunskill, "Data-Efficient Off-Policy Evaluation" (ICML 2016)
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from agent.rl.features import row_to_rl_features, parse_allowed_actions, N_RL_FEATURES
from agent.rl_sequential.cql import CQLAgent
from simulator.models import SimActionType, ACTION_INDEX, INDEX_ACTION

logger = logging.getLogger(__name__)

N_ACTIONS = len(SimActionType)
PROVENANCE = "HELD_OUT_OFFLINE_EVAL"


# ───────────────────────── Direct-Method reward model ──────────────────────

class _RewardModel(nn.Module):
    """Simple MLP that predicts expected reward given (state, action)."""

    def __init__(self, state_dim: int = N_RL_FEATURES, n_actions: int = N_ACTIONS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


def _fit_reward_model(
    train_df: pd.DataFrame,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> _RewardModel:
    """Fit a direct-method reward model on training transitions."""
    states, action_onehots, rewards = [], [], []
    for _, row in train_df.iterrows():
        s = row_to_rl_features(row)
        a_oh = np.zeros(N_ACTIONS, dtype=np.float32)
        a_oh[ACTION_INDEX[SimActionType(row["action"])]] = 1.0
        states.append(s)
        action_onehots.append(a_oh)
        rewards.append(float(row["reward"]))

    S = torch.tensor(np.array(states), dtype=torch.float32)
    A = torch.tensor(np.array(action_onehots), dtype=torch.float32)
    R = torch.tensor(np.array(rewards), dtype=torch.float32)

    model = _RewardModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = S.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = model(S[idx], A[idx])
            loss = nn.functional.mse_loss(pred, R[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        logger.debug(f"DM reward model epoch {epoch+1}/{epochs}: loss={epoch_loss/nb:.4f}")

    return model


# ───────────────────────── OPE Estimators ─────────────────────────────────

class ImportanceSamplingOPE:
    """
    Per-step Importance Sampling (IS) estimator for episodic off-policy evaluation.

    For each logged episode, computes the cumulative importance weight
    ρ = Π_t  π_target(a_t|s_t) / π_behavior(a_t|s_t)  and scales the episode
    return accordingly. Weights are clipped to [1/clip, clip] per step to
    reduce variance.
    """

    def __init__(self, target_policy: CQLAgent, clip: float = 10.0):
        self.target = target_policy
        self.clip = clip

    def _target_prob(self, state: np.ndarray, action: SimActionType, allowed: Set[SimActionType]) -> float:
        """
        Compute the target policy's probability of choosing `action` given
        the safety-masked softmax over Q-values.
        """
        q = self.target.q_values(state).astype(np.float64).copy()
        allowed_idx = [ACTION_INDEX[a] for a in allowed]
        mask = np.full(N_ACTIONS, -np.inf)
        mask[allowed_idx] = 0.0
        masked_q = q + mask

        # Softmax over allowed actions to get a probability distribution
        finite_mask = np.isfinite(masked_q)
        if not finite_mask.any():
            return 1.0 / max(1, len(allowed))
        exp_q = np.zeros(N_ACTIONS)
        exp_q[finite_mask] = np.exp(masked_q[finite_mask] - masked_q[finite_mask].max())
        probs = exp_q / exp_q.sum()

        a_idx = ACTION_INDEX[action]
        return max(probs[a_idx], 1e-8)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Compute IS-estimated return over held-out episodes."""
        episode_returns = []

        for ep_id, ep in test_df.groupby("episode_id", sort=False):
            ep = ep.sort_values("timestep")
            rho = 1.0  # cumulative importance weight
            ep_return = 0.0

            for _, row in ep.iterrows():
                state = row_to_rl_features(row)
                action = SimActionType(row["action"])
                allowed = set(parse_allowed_actions(row["allowed_actions"]))
                behavior_prob = max(float(row["behavior_prob"]), 1e-8)

                target_prob = self._target_prob(state, action, allowed)
                ratio = np.clip(target_prob / behavior_prob, 1.0 / self.clip, self.clip)
                rho *= ratio
                ep_return += float(row["reward"])

            episode_returns.append(rho * ep_return)

        mean_return = float(np.mean(episode_returns))
        ci_lo, ci_hi = _bootstrap_ci(episode_returns)

        return {
            "estimator": "importance_sampling",
            "provenance": PROVENANCE,
            "estimated_return_mean": mean_return,
            "estimated_return_95ci_lo": ci_lo,
            "estimated_return_95ci_hi": ci_hi,
            "n_episodes": len(episode_returns),
        }


class DoublyRobustOPE:
    """
    Doubly-Robust (DR) off-policy estimator.

    Combines a direct-method (DM) reward model with importance-sampling
    corrections:

        V_DR = (1/N) Σ_episodes [ Σ_t  ρ_{0:t-1} * (
            r_t - Q_hat(s_t, a_t)
        ) + V_hat(s_0) ]

    For simplicity in the episodic setting with terminal-only reward, we use a
    per-episode formulation:

        V_DR = V_DM  +  (1/N) Σ  ρ * (G_episode - G_hat_episode)

    where G_hat_episode is the DM model's predicted episode return and ρ is
    the cumulative IS weight.
    """

    def __init__(
        self,
        target_policy: CQLAgent,
        reward_model: _RewardModel,
        clip: float = 10.0,
    ):
        self.target = target_policy
        self.reward_model = reward_model
        self.clip = clip
        self._is_ope = ImportanceSamplingOPE(target_policy, clip)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Compute DR-estimated return over held-out episodes."""
        dr_values = []

        for ep_id, ep in test_df.groupby("episode_id", sort=False):
            ep = ep.sort_values("timestep")
            rho = 1.0
            ep_return = 0.0
            dm_return = 0.0

            for _, row in ep.iterrows():
                state = row_to_rl_features(row)
                action = SimActionType(row["action"])
                allowed = set(parse_allowed_actions(row["allowed_actions"]))
                behavior_prob = max(float(row["behavior_prob"]), 1e-8)

                target_prob = self._is_ope._target_prob(state, action, allowed)
                ratio = np.clip(target_prob / behavior_prob, 1.0 / self.clip, self.clip)
                rho *= ratio

                ep_return += float(row["reward"])

                # DM predicted reward for this (state, action)
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    a_oh = torch.zeros(1, N_ACTIONS)
                    a_oh[0, ACTION_INDEX[action]] = 1.0
                    dm_return += float(self.reward_model(s_t, a_oh).item())

            # DR = DM + IS_correction
            dr_value = dm_return + rho * (ep_return - dm_return)
            dr_values.append(dr_value)

        mean_return = float(np.mean(dr_values))
        ci_lo, ci_hi = _bootstrap_ci(dr_values)

        return {
            "estimator": "doubly_robust",
            "provenance": PROVENANCE,
            "estimated_return_mean": mean_return,
            "estimated_return_95ci_lo": ci_lo,
            "estimated_return_95ci_hi": ci_hi,
            "n_episodes": len(dr_values),
        }


# ───────────────────────── Full OPE Pipeline ──────────────────────────────

def run_ope(
    target_policy: CQLAgent,
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    output_path: str = "evaluation/results/ope_results.json",
) -> Dict:
    """
    Run both IS and DR off-policy evaluation and save results.

    If `train_df` is provided, fits a DM reward model for the DR estimator.
    Otherwise, falls back to IS-only.
    """
    results: Dict = {"provenance": PROVENANCE}

    # IS estimator (always available)
    is_ope = ImportanceSamplingOPE(target_policy)
    results["importance_sampling"] = is_ope.evaluate(test_df)
    logger.info(
        f"IS estimated return: {results['importance_sampling']['estimated_return_mean']:.2f} "
        f"(95% CI: [{results['importance_sampling']['estimated_return_95ci_lo']:.2f}, "
        f"{results['importance_sampling']['estimated_return_95ci_hi']:.2f}])"
    )

    # DR estimator (requires training data for the DM model)
    if train_df is not None and len(train_df) > 0:
        logger.info("Fitting direct-method reward model for DR estimator...")
        reward_model = _fit_reward_model(train_df)
        dr_ope = DoublyRobustOPE(target_policy, reward_model)
        results["doubly_robust"] = dr_ope.evaluate(test_df)
        logger.info(
            f"DR estimated return: {results['doubly_robust']['estimated_return_mean']:.2f} "
            f"(95% CI: [{results['doubly_robust']['estimated_return_95ci_lo']:.2f}, "
            f"{results['doubly_robust']['estimated_return_95ci_hi']:.2f}])"
        )
    else:
        logger.warning("No training data provided — skipping DR estimator, IS-only results.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"OPE results saved to {output_path}")

    return results


# ───────────────────────── Deployment Gate ─────────────────────────────────

def check_ope_gate(
    ope_results_path: str = "evaluation/results/ope_results.json",
    min_return_threshold: float = 0.0,
) -> bool:
    """
    Deployment gate: returns True only if OPE results exist and the estimated
    return exceeds the minimum threshold. Called by the orchestrator before
    allowing an RL policy to execute live actions.
    """
    if not os.path.exists(ope_results_path):
        logger.warning(
            f"OPE gate FAILED: results file not found at {ope_results_path}. "
            "Sequential RL policy deployment blocked."
        )
        return False

    with open(ope_results_path) as f:
        results = json.load(f)

    # Prefer DR estimate; fall back to IS
    estimator = results.get("doubly_robust", results.get("importance_sampling"))
    if estimator is None:
        logger.warning("OPE gate FAILED: no estimator results found in OPE results file.")
        return False

    estimated_return = estimator.get("estimated_return_mean", float("-inf"))
    if estimated_return < min_return_threshold:
        logger.warning(
            f"OPE gate FAILED: estimated return {estimated_return:.2f} < "
            f"threshold {min_return_threshold:.2f}. "
            "Sequential RL policy deployment blocked."
        )
        return False

    logger.info(
        f"OPE gate PASSED: estimated return {estimated_return:.2f} >= "
        f"threshold {min_return_threshold:.2f}. Sequential RL policy cleared for deployment."
    )
    return True


# ───────────────────────── Helpers ────────────────────────────────────────

def _bootstrap_ci(
    data: List[float], n_bootstraps: int = 2000, ci: float = 0.95
) -> Tuple[float, float]:
    """Non-parametric bootstrap 95% confidence interval."""
    if len(data) == 0:
        return 0.0, 0.0
    arr = np.array(data)
    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_bootstraps)]
    alpha_lo = ((1.0 - ci) / 2.0) * 100
    alpha_hi = (ci + (1.0 - ci) / 2.0) * 100
    return float(np.percentile(boot_means, alpha_lo)), float(np.percentile(boot_means, alpha_hi))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Off-Policy Evaluation (DR and IS) on sequential RL policy.")
    parser.add_argument("--model", default="data/models/cql_model.pt", help="Path to trained CQL model checkpoint")
    parser.add_argument("--test", default="data/episodic/episodes_test.parquet", help="Path to held-out test episodes")
    parser.add_argument("--train", default="data/episodic/episodes_train.parquet", help="Path to training episodes (for DR reward model)")
    parser.add_argument("--output", default="evaluation/results/ope_results.json", help="Path to save OPE results JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if not os.path.exists(args.model):
        print(f"Error: model file not found at {args.model}")
        sys.exit(1)
    if not os.path.exists(args.test):
        print(f"Error: test data not found at {args.test}")
        sys.exit(1)

    target_agent = CQLAgent.load(args.model)
    test_df = pd.read_parquet(args.test)
    train_df = pd.read_parquet(args.train) if os.path.exists(args.train) else None

    print(f"Loaded target CQL agent from {args.model}")
    print(f"Loaded {test_df['episode_id'].nunique():,} test episodes ({len(test_df):,} transitions)")
    if train_df is not None:
        print(f"Loaded {train_df['episode_id'].nunique():,} train episodes for DR reward model")

    results = run_ope(
        target_policy=target_agent,
        test_df=test_df,
        train_df=train_df,
        output_path=args.output,
    )
    print("\nOPE Evaluation Complete:")
    print(json.dumps(results, indent=2))

