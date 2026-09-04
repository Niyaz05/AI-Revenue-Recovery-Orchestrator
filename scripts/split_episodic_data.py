from __future__ import annotations
"""
Episodic Dataset Splitter.

Splits the episodic RL dataset into 70% train / 15% validation / 15% held-out,
at the EPISODE level — a single episode's timesteps are NEVER split across sets
(doing so would leak trajectory information and inflate evaluation).
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def split_episodic_dataset(input_file: str, output_dir: str, seed: int = 42):
    print(f"Loading episodic data from {input_file}...")
    df = pd.read_parquet(input_file)

    episode_ids = df["episode_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    n = len(episode_ids)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ids = set(episode_ids[:n_train])
    val_ids = set(episode_ids[n_train : n_train + n_val])
    test_ids = set(episode_ids[n_train + n_val :])

    train_df = df[df["episode_id"].isin(train_ids)].copy()
    val_df = df[df["episode_id"].isin(val_ids)].copy()
    test_df = df[df["episode_id"].isin(test_ids)].copy()
    test_df["provenance"] = "HELD_OUT_OFFLINE_EVAL"

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "episodes_train.parquet")
    val_path = os.path.join(output_dir, "episodes_val.parquet")
    test_path = os.path.join(output_dir, "episodes_test.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print("Episodic split complete (episode-level, no leakage):")
    print(f"  Train:    {len(train_ids):,} episodes / {len(train_df):,} transitions -> {train_path}")
    print(f"  Val:      {len(val_ids):,} episodes / {len(val_df):,} transitions -> {val_path}")
    print(f"  Held-out: {len(test_ids):,} episodes / {len(test_df):,} transitions -> {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/episodic/episodes.parquet")
    parser.add_argument("--output_dir", default="data/episodic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_episodic_dataset(args.input, args.output_dir, seed=args.seed)
