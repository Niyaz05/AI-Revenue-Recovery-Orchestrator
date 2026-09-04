from __future__ import annotations
"""
Field Mapper and State Machine Unit Tests.

Tests:
1. Field mapper translation accuracy
2. State machine subscription transitions and at-risk state detection
"""

import pytest
from backend.models.enums import FailureReason, PaymentStatus, SubscriptionStatus
from backend.services.field_mapper import (
    classify_failure_reason,
    map_payment_fields,
    map_subscription_fields,
    paise_to_rupees,
)
from backend.services.state_machine import is_at_risk_state, is_valid_subscription_transition


def test_field_mapper_translations():
    rz_pay = {
        "id": "pay_123",
        "amount": 499900,  # 4999 rupees in paise
        "currency": "INR",
        "status": "failed",
        "error_code": "BAD_REQUEST_ERROR",
        "error_reason": "insufficient_balance",
        "error_description": "Card declined due to low balance",
        "method": "card",
    }
    mapped = map_payment_fields(rz_pay)
    assert mapped["amount"] == 4999.0
    assert mapped["failure_reason"] == FailureReason.INSUFFICIENT_FUNDS
    assert mapped["status"] == PaymentStatus.FAILED

    rz_sub = {
        "id": "sub_123",
        "plan_id": "plan_abc",
        "status": "pending",
        "paid_count": 2,
        "remaining_count": 4,
        "total_count": 6,
    }
    mapped_sub = map_subscription_fields(rz_sub)
    assert mapped_sub["status"] == SubscriptionStatus.PENDING
    assert mapped_sub["paid_count"] == 2


def test_state_machine_transitions():
    assert is_valid_subscription_transition(SubscriptionStatus.AUTHENTICATED, SubscriptionStatus.ACTIVE) is True
    assert is_valid_subscription_transition(SubscriptionStatus.ACTIVE, SubscriptionStatus.PENDING) is True
    assert is_valid_subscription_transition(SubscriptionStatus.CANCELLED, SubscriptionStatus.ACTIVE) is False
    assert is_at_risk_state(SubscriptionStatus.PENDING) is True
    assert is_at_risk_state(SubscriptionStatus.HALTED) is True
    assert is_at_risk_state(SubscriptionStatus.ACTIVE) is False
