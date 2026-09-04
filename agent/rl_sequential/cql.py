from __future__ import annotations
"""
Conservative Q-Learning (CQL) agent — primary offline sequential RL algorithm.

Why offline RL? We must learn from *logged* recovery episodes (no live
exploration against real customers). Plain Q-learning on logged data tends to
overestimate Q-values for state-action pairs that were rarely (or never) taken
by the logging policy, and then exploit that overestimation — dangerous when the
"actions" are customer-facing interventions.

CQL fixes this by adding a conservative penalty to the standard Bellman loss:

    L = L_bellman  +  alpha * ( logsumexp_a Q(s, a)  -  Q(s, a_data) )

The logsumexp term pushes DOWN the Q-value of *all* actions, while the
-Q(s, a_data) term pushes UP only the actions actually seen in the data. The net
effect: unseen/rare state-action pairs get systematically lower (pessimistic)
Q-values, so the learned policy prefers well-supported actions.

Architecture is deliberately small and explainable for a demo:
  Q(s) -> a 22-dim state -> MLP(128, 128) -> 7 action values.

The Tier-1 safety mask is applied BEFORE argmax at inference: forbidden actions
are set to -inf so the network can never select them, even internally.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from simulator.models import SimActionType, ACTION_INDEX, INDEX_ACTION
from agent.rl.features import N_RL_FEATURES

logger = logging.getLogger(__name__)

N_ACTIONS = len(SimActionType)


class QNetwork(nn.Module):
    """Small MLP Q-network: state -> Q-value per action."""

    def __init__(self, state_dim: int = N_RL_FEATURES, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CQLAgent:
    def __init__(
        self,
        state_dim: int = N_RL_FEATURES,
        n_actions: int = N_ACTIONS,
        hidden: int = 128,
        gamma: float = 0.95,
        lr: float = 1e-3,
        cql_alpha: float = 1.0,
        tau: float = 0.005,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.q_target = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.version = "cql_v1"

    # ─────────────────────────── inference ──────────────────────────────

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            return self.q(s).cpu().numpy()

    def predict(
        self,
        state: np.ndarray,
        allowed_actions: Optional[Set[SimActionType]] = None,
    ) -> SimActionType:
        """
        Safety-masked action selection. The mask is applied BEFORE argmax so a
        forbidden action is never selected — even internally.
        """
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = set(SimActionType)

        q = self.q_values(state).astype(np.float64).copy()
        allowed_idx = [ACTION_INDEX[a] for a in allowed_actions]
        mask = np.full(self.n_actions, -np.inf)
        mask[allowed_idx] = 0.0
        masked_q = q + mask
        best_idx = int(np.argmax(masked_q))
        return INDEX_ACTION[best_idx]

    # ─────────────────────────── training ───────────────────────────────

    def train_on_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        next_allowed_masks: torch.Tensor,
    ) -> Dict[str, float]:
        """One CQL gradient step on a batch of transitions."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_allowed_masks = next_allowed_masks.to(self.device)  # 1.0 = allowed

        # ── Bellman target (double Q-learning, masked next-action max) ──
        with torch.no_grad():
            next_q_online = self.q(next_states)
            next_q_online = next_q_online.masked_fill(next_allowed_masks < 0.5, -1e9)
            next_action = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.q_target(next_states).gather(1, next_action).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * next_q_target

        # ── Standard Bellman (TD) loss ──
        q_all = self.q(states)
        q_data = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        bellman_loss = nn.functional.mse_loss(q_data, target)

        # ── CQL conservative penalty ──
        # logsumexp over all actions pushes down unseen state-action values;
        # subtracting Q(s, a_data) pushes up the actions actually observed.
        logsumexp_q = torch.logsumexp(q_all, dim=1)
        cql_loss = (logsumexp_q - q_data).mean()

        loss = bellman_loss + self.cql_alpha * cql_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft target-network update
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {
            "loss": float(loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "cql_loss": float(cql_loss.item()),
        }

    # ─────────────────────────── persistence ────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.q.state_dict(),
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "version": self.version,
                "cql_alpha": self.cql_alpha,
                "gamma": self.gamma,
            },
            path,
        )
        logger.info(f"Saved CQL model to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "CQLAgent":
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(
            state_dim=ckpt["state_dim"],
            n_actions=ckpt["n_actions"],
            cql_alpha=ckpt.get("cql_alpha", 1.0),
            gamma=ckpt.get("gamma", 0.95),
            device=device,
        )
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.q_target.load_state_dict(ckpt["state_dict"])
        agent.version = ckpt.get("version", "cql_v1")
        logger.info(f"Loaded CQL model from {path}")
        return agent
