from __future__ import annotations
"""
Critical Automated Safety Policy Unit Tests.

PROVES THE NON-NEGOTIABLE SAFETY INVARIANT:
The AI / learning model or LLM CAN NEVER bypass merchant-defined hard safety limits.
"""

from datetime import datetime, timedelta
import pytest

from backend.models.enums import ActionType
from backend.models.merchant_policy import MerchantPolicy
from backend.policies.safety_policy import PolicyDecision, SafetyContext, SafetyPolicyEngine
from backend.policies.strategy_policy import StrategyPolicyEngine


@pytest.fixture
def standard_policy():
    return MerchantPolicy(
        max_retry_attempts=3,
        retry_cooldown_hours=4.0,
        recovery_window_hours=168.0,
        max_auto_recovery_amount=50000.0,
        global_daily_retry_budget=100,
        human_approval_threshold_amount=100000.0,
        daily_intervention_cap_per_customer=3,
        opted_out_customer_ids="[999]",
    )


@pytest.fixture
def safety_engine(standard_policy):
    return SafetyPolicyEngine(merchant_policy=standard_policy)


def test_rule_opted_out_customer_blocks_all_actions(safety_engine):
    """Proves that opted-out customers receive ONLY STOP_RECOVERY."""
    ctx = SafetyContext(
        customer_id=999,  # In opted out list
        amount=1000.0,
        attempt_count=1,
        interventions_today=0,
        global_retries_today=0,
        first_failure_time=datetime.utcnow(),
        last_attempt_time=None,
        is_customer_opted_out=True,
    )
    decision = safety_engine.evaluate(ctx)

    assert decision.must_stop is True
    assert decision.allowed_actions == {ActionType.STOP_RECOVERY}
    assert ActionType.RETRY in decision.blocked_actions
    assert ActionType.PAYMENT_LINK in decision.blocked_actions


def test_rule_retry_limit_blocks_retry(safety_engine):
    """Proves that attempt_count >= max_retry_attempts blocks RETRY."""
    ctx = SafetyContext(
        customer_id=101,
        amount=2500.0,
        attempt_count=3,  # Max is 3
        interventions_today=1,
        global_retries_today=10,
        first_failure_time=datetime.utcnow() - timedelta(hours=10),
        last_attempt_time=datetime.utcnow() - timedelta(hours=5),
        is_customer_opted_out=False,
    )
    decision = safety_engine.evaluate(ctx)

    assert ActionType.RETRY not in decision.allowed_actions
    assert ActionType.RETRY in decision.blocked_actions


def test_rule_cooldown_blocks_premature_retry(safety_engine):
    """Proves that retry within cooldown period (< 4h) is strictly blocked."""
    ctx = SafetyContext(
        customer_id=102,
        amount=2500.0,
        attempt_count=1,
        interventions_today=0,
        global_retries_today=5,
        first_failure_time=datetime.utcnow() - timedelta(hours=2),
        last_attempt_time=datetime.utcnow() - timedelta(hours=1),  # Only 1h ago (< 4h)
        is_customer_opted_out=False,
    )
    decision = safety_engine.evaluate(ctx)

    assert ActionType.RETRY not in decision.allowed_actions
    assert "Retry cooldown active" in " ".join(decision.reasons)


def test_rule_recovery_window_expired_terminates_recovery(safety_engine):
    """Proves that failure older than recovery window (> 168h) forces STOP_RECOVERY."""
    ctx = SafetyContext(
        customer_id=103,
        amount=5000.0,
        attempt_count=1,
        interventions_today=0,
        global_retries_today=0,
        first_failure_time=datetime.utcnow() - timedelta(hours=180),  # > 168h
        last_attempt_time=None,
        is_customer_opted_out=False,
    )
    decision = safety_engine.evaluate(ctx)

    assert decision.must_stop is True
    assert decision.allowed_actions == {ActionType.STOP_RECOVERY}


def test_rule_high_value_forces_human_escalation(safety_engine):
    """Proves that transactions > ₹1,00,000 MUST be escalated to humans."""
    ctx = SafetyContext(
        customer_id=104,
        amount=150000.0,  # ₹1.5 Lakhs > ₹1.0 Lakh limit
        attempt_count=1,
        interventions_today=0,
        global_retries_today=0,
        first_failure_time=datetime.utcnow(),
        last_attempt_time=None,
        is_customer_opted_out=False,
    )
    decision = safety_engine.evaluate(ctx)

    assert decision.must_escalate is True
    assert ActionType.RETRY not in decision.allowed_actions
    assert ActionType.PAYMENT_LINK not in decision.allowed_actions
    assert ActionType.ESCALATE_TO_HUMAN in decision.allowed_actions


def test_rule_global_retry_budget_exhaustion(safety_engine):
    """Proves that merchant-wide daily retry budget exhaustion blocks further retries."""
    ctx = SafetyContext(
        customer_id=105,
        amount=1000.0,
        attempt_count=1,
        interventions_today=0,
        global_retries_today=100,  # Budget is 100
        first_failure_time=datetime.utcnow() - timedelta(hours=10),
        last_attempt_time=datetime.utcnow() - timedelta(hours=6),
        is_customer_opted_out=False,
    )
    decision = safety_engine.evaluate(ctx)

    assert ActionType.RETRY not in decision.allowed_actions
    assert ActionType.PAYMENT_LINK in decision.allowed_actions


def test_ai_cannot_bypass_safety_layer(safety_engine):
    """
    CRITICAL INVARIANT TEST:
    Even if the ML strategy model strongly favors RETRY with high score,
    the Level 1 Safety layer strictly removes RETRY and prevents its selection.
    """
    # Create safety context where retry is blocked (max attempts reached)
    ctx = SafetyContext(
        customer_id=106,
        amount=2500.0,
        attempt_count=3,  # Blocked
        interventions_today=0,
        global_retries_today=0,
        first_failure_time=datetime.utcnow() - timedelta(hours=10),
        last_attempt_time=datetime.utcnow() - timedelta(hours=6),
        is_customer_opted_out=False,
    )
    safety_decision = safety_engine.evaluate(ctx)

    strategy_engine = StrategyPolicyEngine()
    chosen_action, meta = strategy_engine.select_action(
        failure_reason="temporary_bank_failure",  # Model would normally want retry
        amount=2500.0,
        attempt_count=3,
        hours_since_first_failure=10.0,
        customer_ltv=50000.0,
        customer_failure_rate=0.05,
        subscription_paid_count=4,
        safety_decision=safety_decision,
    )

    # Invariant: AI could NOT choose RETRY
    assert chosen_action != ActionType.RETRY
    assert chosen_action in safety_decision.allowed_actions
