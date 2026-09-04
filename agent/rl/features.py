from __future__ import annotations
"""
Enriched feature extraction for the *sequential* RL policy.

Builds on the bandit's 16-dim single-decision feature vector
(`agent/bandit/features.extract_feature_vector`) and appends 6 long-horizon,
history-derived features. These are computed from the customer's *entire*
relationship — independent of the current 7-day recovery episode — which is what
fixes the "reliable dormant customer" blind spot: a 4-year customer with a 98%
lifetime success rate who fails once after a dormant month is surfaced as a
very different state from a chronic failer with identical surface features
(same amount, same failure reason).

Resulting vector: 22-dim (16 base + 6 history).
"""

import numpy as np

from agent.bandit.features import extract_feature_vector, FEATURE_NAMES

# Names of the 6 appended history features (indices 16..21).
HISTORY_FEATURE_NAMES = [
    "account_tenure_days_norm",   # log1p(tenure_days) / log1p(3650)  → [0,1]
    "lifetime_success_rate",      # successful / total charges         → [0,1]
    "days_since_last_success_norm",  # min(days,180)/180                → [0,1]
    "historical_ltv_norm",        # log1p(historical_ltv) / 12         → [0,1]
    "is_first_ever_failure",      # boolean 0/1
    "reliability_score",          # composite reputation metric         → [0,1]
]

RL_FEATURE_NAMES = FEATURE_NAMES + HISTORY_FEATURE_NAMES
N_RL_FEATURES = 22


def compute_reliability_score(
    lifetime_success_rate: float,
    account_tenure_days: int,
    days_since_last_success: int,
    is_first_ever_failure: bool,
) -> float:
    """
    Rolling customer reputation metric in [0, 1].

    Weighted composite (weights chosen to be explainable, sum to 1.0):

        reliability = 0.45 * lifetime_success_rate          # proven payment history
                    + 0.25 * tenure_score                   # loyalty / account age
                    + 0.20 * recency_score                  # how recently they last paid
                    + 0.10 * is_first_ever_failure          # clean record → benefit of doubt

    where:
        tenure_score  = min(1, account_tenure_days / 365)      (saturates at 1 year)
        recency_score = 1 - min(1, days_since_last_success / 90)  (recent success → 1)

    A reliable dormant customer scores ≈ 0.95; a chronic failer ≈ 0.35.
    This separation is what lets the agent pick a gentle touch (WAIT / a single
    reminder) for the former and a firmer sequence for the latter.
    """
    tenure_score = min(1.0, max(0.0, account_tenure_days) / 365.0)
    recency_score = 1.0 - min(1.0, max(0.0, days_since_last_success) / 90.0)
    score = (
        0.45 * float(lifetime_success_rate)
        + 0.25 * tenure_score
        + 0.20 * recency_score
        + 0.10 * (1.0 if is_first_ever_failure else 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def extract_rl_feature_vector(
    # ── base (bandit) context ──
    failure_reason: str,
    amount: float,
    attempt_count: int,
    hours_since_first_failure: float,
    customer_ltv: float,
    customer_failure_rate: float,
    subscription_paid_count: int,
    is_weekend: bool = False,
    # ── long-horizon history (independent of current episode) ──
    account_tenure_days: int = 365,
    lifetime_success_rate: float = 0.9,
    days_since_last_success: int = 30,
    historical_ltv: float = 50000.0,
    is_first_ever_failure: bool = False,
) -> np.ndarray:
    """Extract the 22-dim state vector for the sequential RL policy."""
    base = extract_feature_vector(
        failure_reason=failure_reason,
        amount=amount,
        attempt_count=attempt_count,
        hours_since_first_failure=hours_since_first_failure,
        customer_ltv=customer_ltv,
        customer_failure_rate=customer_failure_rate,
        subscription_paid_count=subscription_paid_count,
        is_weekend=is_weekend,
    )

    hist = np.zeros(6, dtype=np.float32)
    hist[0] = float(np.log1p(max(0.0, account_tenure_days)) / np.log1p(3650.0))
    hist[1] = float(np.clip(lifetime_success_rate, 0.0, 1.0))
    hist[2] = float(min(180.0, max(0.0, days_since_last_success)) / 180.0)
    hist[3] = float(np.log1p(max(0.0, historical_ltv)) / 12.0)
    hist[4] = 1.0 if is_first_ever_failure else 0.0
    hist[5] = compute_reliability_score(
        lifetime_success_rate=lifetime_success_rate,
        account_tenure_days=account_tenure_days,
        days_since_last_success=days_since_last_success,
        is_first_ever_failure=is_first_ever_failure,
    )

    return np.concatenate([base, hist]).astype(np.float32)


def row_to_rl_features(row) -> np.ndarray:
    """
    Build the 22-dim RL feature vector from an episodic-transition record
    (a pandas row / dict produced by `RecoveryEnvironment.run_episode`).
    """
    return extract_rl_feature_vector(
        failure_reason=str(row["failure_reason"]),
        amount=float(row["amount"]),
        attempt_count=int(row["attempt_count"]),
        hours_since_first_failure=float(row["hours_since_first_failure"]),
        customer_ltv=float(row["customer_ltv"]),
        customer_failure_rate=float(row["customer_failure_rate"]),
        subscription_paid_count=int(row["subscription_paid_count"]),
        is_weekend=bool(int(row["is_weekend"])),
        account_tenure_days=int(row["account_tenure_days"]),
        lifetime_success_rate=float(row["lifetime_success_rate"]),
        days_since_last_success=int(row["days_since_last_success"]),
        historical_ltv=float(row["historical_ltv"]),
        is_first_ever_failure=bool(int(row["is_first_ever_failure"])),
    )


def parse_allowed_actions(allowed_str) -> list:
    """Parse the comma-separated allowed_actions string back into SimActionType list."""
    from simulator.models import SimActionType

    if isinstance(allowed_str, (list, tuple, set)):
        return [SimActionType(a) for a in allowed_str]
    return [SimActionType(a) for a in str(allowed_str).split(",") if a]
