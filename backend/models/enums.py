from __future__ import annotations
"""
Shared enums used across models.

These are our *internal* domain enums — not 1:1 mirrors of Razorpay's API values.
Razorpay field values are mapped to these in field_mapper.py.
"""

import enum


class SubscriptionStatus(str, enum.Enum):
    CREATED = "created"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    PENDING = "pending"
    HALTED = "halted"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    EXPIRED = "expired"
    PAUSED = "paused"


class PaymentStatus(str, enum.Enum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"


class ActionType(str, enum.Enum):
    WAIT = "WAIT"
    RETRY = "RETRY"
    PAYMENT_LINK = "PAYMENT_LINK"
    SEND_REMINDER = "SEND_REMINDER"
    REQUEST_ALTERNATE_METHOD = "REQUEST_ALTERNATE_METHOD"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    STOP_RECOVERY = "STOP_RECOVERY"


class ActionStatus(str, enum.Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class OutcomeType(str, enum.Enum):
    RECOVERED = "recovered"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    ESCALATED = "escalated"


class Provenance(str, enum.Enum):
    """
    Every number in the system is labeled with exactly one provenance tag.
    This is a non-negotiable design principle — see architecture.md.
    """
    SIMULATED = "SIMULATED"
    SYNTHETIC_TRAINING_DATA = "SYNTHETIC_TRAINING_DATA"
    HELD_OUT_OFFLINE_EVAL = "HELD_OUT_OFFLINE_EVAL"
    RAZORPAY_TEST_MODE = "RAZORPAY_TEST_MODE"


class FailureReason(str, enum.Enum):
    """Internal classification of payment failure reasons."""
    TEMPORARY_BANK_FAILURE = "temporary_bank_failure"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_PAYMENT_METHOD = "invalid_payment_method"
    NETWORK_ERROR = "network_error"
    FRAUD_SUSPECTED = "fraud_suspected"
    BANK_DECLINED = "bank_declined"
    CARD_EXPIRED = "card_expired"
    AUTHENTICATION_FAILED = "authentication_failed"
    UNKNOWN = "unknown"
