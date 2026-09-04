from __future__ import annotations
"""
Domain Randomization for robustness validation of the sequential RL policy.

Perturbs the simulator's hidden parameters (base failure rates by category,
fatigue decay speed, self-cure probability, per-action success curves) within
defined reasonable ranges, so the policy is trained / evaluated across a
*distribution* of worlds rather than a single fixed simulator configuration.

A policy that only works on the exact training configuration is overfit to the
simulator's assumptions; one that still wins across randomized configurations is
far more likely to transfer to real payment-recovery dynamics.
"""

import logging
from typing import List, Optional

import numpy as np

from simulator.environment import SimDynamics

logger = logging.getLogger(__name__)


def sample_dynamics(rng: np.random.Generator) -> SimDynamics:
    """
    Sample a perturbed SimDynamics config within reasonable bounds.

    Each parameter is jittered around its default by a multiplicative or
    additive factor, clipped to plausible ranges so the world stays coherent
    (e.g. success probabilities stay in [0,1], fatigue stays positive).
    """
    def j(v: float, lo: float, hi: float, rel: float = 0.2) -> float:
        """Multiplicative jitter by ±rel, clipped to [lo, hi]."""
        return float(np.clip(v * rng.uniform(1 - rel, 1 + rel), lo, hi))

    # Perturb the failure-type distribution: jitter each cumulative threshold,
    # keep them sorted and within (0, 1).
    base_fd = [0.40, 0.70, 0.85, 0.92, 0.97]
    fd = sorted(float(np.clip(t + rng.uniform(-0.05, 0.05), 0.05, 0.99)) for t in base_fd)

    return SimDynamics(
        fatigue_coef=j(0.08, 0.02, 0.20),
        self_cure_base=j(0.45, 0.15, 0.80),
        fail_dist=fd,
        high_value_escalate_success=j(0.75, 0.55, 0.90),
        temp_retry_after_cd_base=j(0.78, 0.50, 0.95),
        temp_retry_after_cd_decay=j(0.15, 0.05, 0.30),
        temp_retry_immediate=j(0.25, 0.10, 0.45),
        temp_link=j(0.50, 0.30, 0.75),
        temp_escalate=j(0.60, 0.40, 0.85),
        insuff_retry_after48_base=j(0.32, 0.15, 0.50),
        insuff_retry_after48_decay=j(0.10, 0.03, 0.25),
        insuff_retry_immediate=j(0.12, 0.05, 0.30),
        insuff_link=j(0.68, 0.45, 0.85),
        insuff_reminder=j(0.35, 0.15, 0.60),
        insuff_escalate=j(0.70, 0.50, 0.90),
        invalid_alt_method=j(0.72, 0.50, 0.90),
        invalid_link=j(0.58, 0.35, 0.80),
        invalid_escalate=j(0.65, 0.45, 0.85),
        fraud_retry=j(0.02, 0.0, 0.10),
        fraud_escalate=j(0.55, 0.35, 0.80),
        fraud_link=j(0.40, 0.20, 0.65),
        base_p=j(0.20, 0.05, 0.40),
    )


def sample_training_dynamics(
    n_configs: int, seed: int = 42
) -> List[SimDynamics]:
    """Sample `n_configs` randomized configs for robustness-aware training."""
    rng = np.random.default_rng(seed)
    return [sample_dynamics(rng) for _ in range(n_configs)]


def sample_heldout_dynamics(
    n_configs: int, seed: int = 999
) -> List[SimDynamics]:
    """
    Sample randomized configs NOT seen during training (disjoint seed) for the
    held-out robustness evaluation in `scripts/robustness_eval.py`.
    """
    rng = np.random.default_rng(seed)
    return [sample_dynamics(rng) for _ in range(n_configs)]
