from __future__ import annotations
"""
Dataset Splitter.

Splits synthetic data into:
- 70% Train (for model fitting)
- 15% Validation (for alpha tuning & hyperparameter selection)
- 15% Held-Out Offline Evaluation (NEVER used for training or tuning)
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(input_file: str, output_dir: str, seed: int = 42):
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    n_total = len(df)
    print(f"Total dataset size: {n_total:,} records")

    # Stratified split on (failure_reason, recovered)
    stratify_key = df["failure_reason"].astype(str) + "_" + df["recovered"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=stratify_key
    )

    stratify_temp = temp_df["failure_reason"].astype(str) + "_" + temp_df["recovered"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=stratify_temp
    )

    # Set provenance for held out test set
    test_df = test_df.copy()
    test_df["provenance"] = "HELD_OUT_OFFLINE_EVAL"

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Split complete:")
    print(f"  Train:    {len(train_df):,} records ({len(train_df)/n_total:.1%}) -> {train_path}")
    print(f"  Val:      {len(val_df):,} records ({len(val_df)/n_total:.1%}) -> {val_path}")
    print(f"  Held-out: {len(test_df):,} records ({len(test_df)/n_total:.1%}) -> {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/historical_records.parquet")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.input, args.output_dir, seed=args.seed)
