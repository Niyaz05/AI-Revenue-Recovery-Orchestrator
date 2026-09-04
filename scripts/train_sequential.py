from __future__ import annotations
"""
Train the sequential offline-RL policy (Tier 2 upgrade).

Trains:
  - CQL (primary): conservative Q-learning over logged episodes.
  - Decision Transformer (secondary/comparison).

Data provenance: SYNTHETIC_TRAINING_DATA (episodic). Both agents consume the
22-dim enriched state vector and are safety-masked at inference.
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
import torch

from agent.rl.features import row_to_rl_features, parse_allowed_actions
from agent.rl_sequential.cql import CQLAgent, N_ACTIONS
from agent.rl_sequential.decision_transformer import DecisionTransformerAgent
from simulator.models import SimActionType, ACTION_INDEX


# ─────────────────────────── data preparation ───────────────────────────


def build_cql_transitions(df: pd.DataFrame):
    """
    Convert episodic transitions into (s, a, r, s', done, next_allowed_mask).
    Next-state is the following timestep in the same episode; terminal steps
    have done=1 and an all-zero next mask (unused by the target).
    """
    states, actions, rewards, next_states, dones, next_masks = [], [], [], [], [], []

    for _, ep in df.groupby("episode_id", sort=False):
        ep = ep.sort_values("timestep")
        rows = list(ep.itertuples(index=False))
        vecs = [row_to_rl_features(r._asdict() if hasattr(r, "_asdict") else r) for r in rows]
        for t, row in enumerate(rows):
            s = vecs[t]
            a = ACTION_INDEX[SimActionType(row.action)]
            r = float(row.reward)
            done = bool(row.done)
            if not done and t + 1 < len(rows):
                ns = vecs[t + 1]
                nm = np.zeros(N_ACTIONS, dtype=np.float32)
                for act in parse_allowed_actions(rows[t + 1].allowed_actions):
                    nm[ACTION_INDEX[act]] = 1.0
            else:
                ns = np.zeros_like(s)
                nm = np.zeros(N_ACTIONS, dtype=np.float32)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(float(done))
            next_masks.append(nm)

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(np.array(actions), dtype=torch.long),
        torch.tensor(np.array(rewards), dtype=torch.float32),
        torch.tensor(np.array(next_states), dtype=torch.float32),
        torch.tensor(np.array(dones), dtype=torch.float32),
        torch.tensor(np.array(next_masks), dtype=torch.float32),
    )


def build_dt_sequences(df: pd.DataFrame, max_timesteps: int):
    """
    Build (states, actions, returns-to-go) sequences per episode, padded to
    max_timesteps. RTG_t = sum of rewards from t to end (terminal-only reward
    here, so RTG equals the episode return for all steps).
    """
    seq_states, seq_actions, seq_rtgs = [], [], []
    state_dim = None

    for _, ep in df.groupby("episode_id", sort=False):
        ep = ep.sort_values("timestep")
        rows = list(ep.itertuples(index=False))
        T = len(rows)
        if T == 0:
            continue
        vecs = np.stack([row_to_rl_features(r._asdict()) for r in rows])
        acts = np.array([ACTION_INDEX[SimActionType(r.action)] for r in rows], dtype=np.int64)
        rews = np.array([float(r.reward) for r in rows], dtype=np.float32)
        rtg = np.cumsum(rews[::-1])[::-1].astype(np.float32)  # returns-to-go

        # pad to max_timesteps
        pad = max_timesteps - T
        if pad < 0:
            vecs, acts, rtg = vecs[:max_timesteps], acts[:max_timesteps], rtg[:max_timesteps]
            pad = 0
        if pad > 0:
            vecs = np.vstack([vecs, np.zeros((pad, vecs.shape[1]), dtype=np.float32)])
            acts = np.concatenate([acts, np.zeros(pad, dtype=np.int64)])
            rtg = np.concatenate([rtg, np.zeros(pad, dtype=np.float32)])

        seq_states.append(vecs)
        seq_actions.append(acts)
        seq_rtgs.append(rtg)
        state_dim = vecs.shape[1]

    return (
        torch.tensor(np.array(seq_states), dtype=torch.float32),
        torch.tensor(np.array(seq_actions), dtype=torch.long),
        torch.tensor(np.array(seq_rtgs), dtype=torch.float32).unsqueeze(-1),
        state_dim,
    )


# ─────────────────────────── training loops ───────────────────────────


def train_cql(train_df, val_df, epochs, batch_size, output):
    agent = CQLAgent(cql_alpha=args.cql_alpha)
    print("Building CQL transition tensors...")
    data = build_cql_transitions(train_df)
    n = data[0].shape[0]
    print(f"Training CQL on {n:,} transitions for {epochs} epochs (batch={batch_size})...")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_loss = ep_bell = ep_cql = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = [t[idx] for t in data]
            metrics = agent.train_on_batch(*batch)
            ep_loss += metrics["loss"]
            ep_bell += metrics["bellman_loss"]
            ep_cql += metrics["cql_loss"]
            nb += 1
        print(
            f"  Epoch {epoch + 1}/{epochs}: loss={ep_loss/nb:.4f} "
            f"bellman={ep_bell/nb:.4f} cql={ep_cql/nb:.4f}"
        )

    agent.save(output)
    return agent


def train_dt(train_df, epochs, batch_size, output, max_timesteps=8):
    print("Building Decision Transformer sequences...")
    states, actions, rtgs, state_dim = build_dt_sequences(train_df, max_timesteps)
    n = states.shape[0]
    agent = DecisionTransformerAgent(max_timesteps=max_timesteps)
    print(f"Training Decision Transformer on {n:,} episodes for {epochs} epochs...")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            metrics = agent.train_on_batch(states[idx], actions[idx], rtgs[idx])
            ep_loss += metrics["loss"]
            nb += 1
        print(f"  Epoch {epoch + 1}/{epochs}: ce_loss={ep_loss/nb:.4f}")

    agent.save(output)
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sequential RL policies.")
    parser.add_argument("--model", choices=["cql", "dt", "both"], default="cql")
    parser.add_argument("--train", default="data/episodic/episodes_train.parquet")
    parser.add_argument("--val", default="data/episodic/episodes_val.parquet")
    parser.add_argument("--output_cql", default="data/models/cql_model.pt")
    parser.add_argument("--output_dt", default="data/models/dt_model.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--cql_alpha", type=float, default=1.0,
        help="CQL conservative penalty weight. Lower values (e.g. 0.3-0.5) let the "
             "learned policy diverge further from the logged behavior policy; too "
             "high pulls it back toward whatever generated the logs (a likely cause "
             "if CQL's return converges close to the rule-based baseline it was "
             "partly logged from). Sweep and re-run the OPE gate before trusting a "
             "lower value.",
    )
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    val_df = pd.read_parquet(args.val) if os.path.exists(args.val) else None
    print(f"Loaded {train_df['episode_id'].nunique():,} training episodes.")

    if args.model in ("cql", "both"):
        train_cql(train_df, val_df, args.epochs, args.batch_size, args.output_cql)
    if args.model in ("dt", "both"):
        train_dt(train_df, args.epochs, args.batch_size, args.output_dt)

    print("Sequential RL training complete.")