from __future__ import annotations
"""
Decision Transformer (secondary / comparison sequential RL approach).

Frames offline RL as return-conditioned sequence modeling: instead of learning
value functions, we feed the recent trajectory as a sequence of
(return-to-go, state, action) tokens and train the model to predict the action
that historically followed. At inference we condition on a TARGET return (how
much net ₹ we want to recover this episode) and let the model propose actions
that, historically, led to that return.

This is a deliberately small, explainable implementation:
  - 3 token types embedded into a shared d_model
  - a single causal self-attention block (2 layers, 2 heads)
  - an action head producing a logit per action

Safety: at inference the Tier-1 mask is applied to the action logits BEFORE
argmax, so a forbidden action is never selected — even internally.
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


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = self.causal_mask[:T, :T]
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_heads, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    Minimal Decision Transformer.

    Sequence layout per timestep t: [R_t, s_t, a_t] (return-to-go, state, action),
    so positions 3t+1 hold the state tokens from which we predict action a_t.
    """

    def __init__(
        self,
        state_dim: int = N_RL_FEATURES,
        n_actions: int = N_ACTIONS,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        max_timesteps: int = 8,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.d_model = d_model
        self.max_timesteps = max_timesteps
        block_size = 3 * max_timesteps  # 3 tokens per timestep

        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Embedding(n_actions, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_pos = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(*[_Block(d_model, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, n_actions)

    def forward(
        self,
        states: torch.Tensor,      # (B, T, state_dim)
        actions: torch.Tensor,     # (B, T)
        rtgs: torch.Tensor,        # (B, T, 1)
    ) -> torch.Tensor:
        B, T, _ = states.shape
        s_tok = self.embed_state(states)                 # (B, T, d)
        a_tok = self.embed_action(actions)               # (B, T, d)
        r_tok = self.embed_rtg(rtgs)                     # (B, T, d)

        # Interleave [R, s, a] -> (B, 3T, d)
        tok = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(B, 3 * T, self.d_model)
        pos = torch.arange(3 * T, device=states.device).unsqueeze(0)
        tok = tok + self.embed_pos(pos)

        h = self.ln_f(self.blocks(tok))                  # (B, 3T, d)

        # State tokens sit at positions 3t+1 -> gather those for action prediction.
        state_positions = h[:, 1::3, :]                  # (B, T, d)
        logits = self.action_head(state_positions)       # (B, T, n_actions)
        return logits


class DecisionTransformerAgent:
    def __init__(
        self,
        state_dim: int = N_RL_FEATURES,
        n_actions: int = N_ACTIONS,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        max_timesteps: int = 8,
        lr: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DecisionTransformer(
            state_dim, n_actions, d_model, n_heads, n_layers, max_timesteps
        ).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.max_timesteps = max_timesteps
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.version = "dt_v1"

    def train_on_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
    ) -> Dict[str, float]:
        states = states.to(self.device)
        actions = actions.to(self.device)
        rtgs = rtgs.to(self.device)

        logits = self.model(states, actions, rtgs)       # (B, T, n_actions)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.n_actions), actions.reshape(-1)
        )
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.item())}

    def predict(
        self,
        state_history: np.ndarray,     # (T, state_dim)
        action_history: np.ndarray,    # (T,) action indices so far (last is placeholder)
        rtg_history: np.ndarray,       # (T,) return-to-go
        allowed_actions: Optional[Set[SimActionType]] = None,
    ) -> SimActionType:
        if allowed_actions is None or len(allowed_actions) == 0:
            allowed_actions = set(SimActionType)

        s = torch.as_tensor(state_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(action_history, dtype=torch.long, device=self.device).unsqueeze(0)
        r = torch.as_tensor(rtg_history, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            logits = self.model(s, a, r)[0, -1].cpu().numpy().astype(np.float64)

        # Safety mask BEFORE argmax.
        mask = np.full(self.n_actions, -np.inf)
        mask[[ACTION_INDEX[a] for a in allowed_actions]] = 0.0
        best_idx = int(np.argmax(logits + mask))
        return INDEX_ACTION[best_idx]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "max_timesteps": self.max_timesteps,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "DecisionTransformerAgent":
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(
            state_dim=ckpt["state_dim"],
            n_actions=ckpt["n_actions"],
            max_timesteps=ckpt.get("max_timesteps", 8),
            device=device,
        )
        agent.model.load_state_dict(ckpt["state_dict"])
        return agent
