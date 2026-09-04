from __future__ import annotations
"""
Feature extraction and vectorization for Contextual Bandit and RL models.
Extracts a normalized 16-dimensional feature vector from internal state.
"""

from typing import Any, Dict, List
import numpy as np
from backend.models.enums import FailureReason

FAILURE_REASONS_ORDER = [
    FailureReason.TEMPORARY_BANK_FAILURE.value,
    FailureReason.INSUFFICIENT_FUNDS.value,
    FailureReason.INVALID_PAYMENT_METHOD.value,
    FailureReason.NETWORK_ERROR.value,
    FailureReason.FRAUD_SUSPECTED.value,
    FailureReason.BANK_DECLINED.value,
    FailureReason.CARD_EXPIRED.value,
    FailureReason.AUTHENTICATION_FAILED.value,
]

FEATURE_NAMES = [
    # 8 one-hot failure reasons
    "is_temp_bank_failure",
    "is_insufficient_funds",
    "is_invalid_payment_method",
    "is_network_error",
    "is_fraud_suspected",
    "is_bank_declined",
    "is_card_expired",
    "is_auth_failed",
    # Continuous / normalized context features
    "amount_log_norm",         # log(amount + 1) / 10
    "attempt_count_norm",      # attempt_count / 5.0
    "hours_since_failure_norm",# hours / 168.0 (7 days max)
    "customer_ltv_norm",       # log(ltv + 1) / 12
    "customer_failure_rate",   # [0, 1]
    "subscription_age_norm",   # paid_count / 12.0
    "is_weekend",              # binary 0 or 1
    "bias",                    # constant 1.0
]


def extract_feature_vector(
    failure_reason: str,
    amount: float,
    attempt_count: int,
    hours_since_first_failure: float,
    customer_ltv: float,
    customer_failure_rate: float,
    subscription_paid_count: int,
    is_weekend: bool = False,
) -> np.ndarray:
    """
    Extract a normalized 16-dimensional vector for bandit / ML policy evaluation.
    """
    vec = np.zeros(16, dtype=np.float32)

    # 1. One-hot failure reason (indices 0..7)
    reason_clean = failure_reason.lower() if failure_reason else ""
    for idx, r in enumerate(FAILURE_REASONS_ORDER):
        if r in reason_clean:
            vec[idx] = 1.0
            break
    else:
        # Unknown/fallback
        if "insufficient" in reason_clean:
            vec[1] = 1.0
        elif "invalid" in reason_clean:
            vec[2] = 1.0
        else:
            vec[0] = 1.0

    # 2. Normalized continuous signals (indices 8..15)
    vec[8] = float(np.log1p(max(0.0, amount)) / 10.0)
    vec[9] = float(min(5.0, attempt_count) / 5.0)
    vec[10] = float(min(168.0, hours_since_first_failure) / 168.0)
    vec[11] = float(np.log1p(max(0.0, customer_ltv)) / 12.0)
    vec[12] = float(np.clip(customer_failure_rate, 0.0, 1.0))
    vec[13] = float(min(12.0, subscription_paid_count) / 12.0)
    vec[14] = 1.0 if is_weekend else 0.0
    vec[15] = 1.0  # Bias term

    return vec
