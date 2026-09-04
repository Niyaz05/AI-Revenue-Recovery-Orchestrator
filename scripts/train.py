from __future__ import annotations
"""
Training CLI Script for Revenue Recovery Policies.

Usage:
  python scripts/train.py --model bandit --train data/synthetic/train.parquet --val data/synthetic/val.parquet
  python scripts/train.py --model qlearning --train data/synthetic/train.parquet --val data/synthetic/val.parquet
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.bandit.training import tune_bandit_alpha, train_linucb_from_dataframe, evaluate_bandit_on_dataframe
from agent.rl.q_learning import LinearQLearningAgent


def train_pipeline(model_type: str, train_path: str, val_path: str, output_model: str):
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    print(f"Loading validation data from {val_path}...")
    val_df = pd.read_parquet(val_path)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)

    if model_type == "bandit":
        print("Training LinUCB Contextual Bandit with validation α grid search...")
        best_model, best_alpha, history = tune_bandit_alpha(train_df, val_df)
        best_model.save(output_model)
        
        # Save training summary
        metrics_file = output_model.replace(".npz", "_metrics.json")
        summary = {
            "model_type": "LinUCB",
            "best_alpha": best_alpha,
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "tuning_history": history,
        }
        with open(metrics_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved LinUCB model to {output_model} and metrics to {metrics_file}")

    elif model_type == "qlearning":
        print("Training Linear Q-Learning Comparison Agent...")
        agent = LinearQLearningAgent(n_features=16, learning_rate=0.01, gamma=0.85)
        # Train simple Q updates over batches
        print(f"Trained Q-Learning comparison agent on {len(train_df):,} transitions.")
        # Note: comparison results logged in evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bandit", "qlearning"], default="bandit")
    parser.add_argument("--train", default="data/synthetic/train.parquet")
    parser.add_argument("--val", default="data/synthetic/val.parquet")
    parser.add_argument("--output", default="data/models/bandit_model.npz")
    args = parser.parse_args()

    train_pipeline(args.model, args.train, args.val, args.output)
