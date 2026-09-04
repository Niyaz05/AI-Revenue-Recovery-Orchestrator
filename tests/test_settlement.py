from __future__ import annotations
"""
Settlement Service Tests — proves the recovery loop closes automatically.

Covers:
  (a) a valid `payment_link.paid` payload settles the action and sets recovered_amount
  (b) a duplicate / redelivered event does NOT double-settle (idempotency)
  (c) an unknown recovery_action_id is handled gracefully (no crash)
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models.base import Base
from backend.models.customer import Customer
from backend.models.enums import ActionStatus, ActionType, OutcomeType, Provenance, SubscriptionStatus
from backend.models.merchant_policy import MerchantPolicy
from backend.models.payment_attempt import PaymentAttempt
from backend.models.recovery_action import RecoveryAction
from backend.models.recovery_outcome import RecoveryOutcome
from backend.models.subscription import Subscription
from backend.models.webhook_event import WebhookEvent
from backend.services.settlement import SettlementService


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def recovery_action(db):
    """Create a customer, subscription, payment attempt, and an EXECUTING recovery action."""
    customer = Customer(
        razorpay_customer_id="cust_settle_01",
        name="Settle Test",
        email="settle@example.com",
        phone="+919800000000",
        lifetime_value=60000.0,
        total_payments=4,
        failed_payments=1,
        recovered_payments=0,
    )
    db.add(customer)
    db.flush()

    sub = Subscription(
        razorpay_subscription_id="sub_settle_01",
        customer_id=customer.id,
        plan_name="Pro Plan",
        amount_per_period=14999.0,
        status=SubscriptionStatus.PENDING,
        paid_count=1,
        remaining_count=11,
        total_count=12,
    )
    db.add(sub)
    db.flush()

    pay = PaymentAttempt(
        razorpay_payment_id="pay_settle_01",
        subscription_id=sub.id,
        customer_id=customer.id,
        amount=14999.0,
        status="failed",
        failure_reason="insufficient_funds",
        attempt_number=1,
    )
    db.add(pay)
    db.flush()

    action = RecoveryAction(
        payment_attempt_id=pay.id,
        subscription_id=sub.id,
        customer_id=customer.id,
        action_type=ActionType.PAYMENT_LINK,
        status=ActionStatus.EXECUTING,
        razorpay_payment_link_id="plink_settle_01",
        idempotency_key="rec_settle_01",
        provenance=Provenance.RAZORPAY_TEST_MODE,
    )
    db.add(action)
    db.flush()
    return action


def _paid_payload(recovery_action_id, event_id="evt_settle_01", amount_paise=1499900):
    return {
        "id": event_id,
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": "plink_settle_01",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "paid",
                    "notes": {"recovery_action_id": str(recovery_action_id)},
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_settle_paid_01",
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "captured",
                    "method": "upi",
                }
            },
        },
    }


def test_valid_payload_settles_action(db, recovery_action):
    service = SettlementService(db=db)
    result = service.settle_paid_payment_link(_paid_payload(recovery_action.id))

    assert result["status"] == "settled"
    assert result["action_id"] == recovery_action.id
    assert result["recovered_amount"] == pytest.approx(14999.0)

    # Action flipped to COMPLETED with recovered amount
    db.refresh(recovery_action)
    assert recovery_action.status == ActionStatus.COMPLETED
    assert recovery_action.recovered_amount == pytest.approx(14999.0)
    assert recovery_action.completed_at is not None

    # RecoveryOutcome created
    outcome = db.query(RecoveryOutcome).filter(
        RecoveryOutcome.recovery_action_id == recovery_action.id
    ).first()
    assert outcome is not None
    assert outcome.outcome_type == OutcomeType.RECOVERED
    assert outcome.recovered_amount == pytest.approx(14999.0)

    # Customer recovered_payments bumped
    customer = db.query(Customer).filter(Customer.id == recovery_action.customer_id).first()
    assert customer.recovered_payments == 1

    # Audit log written
    from backend.models.audit_log import AuditLog
    audit = db.query(AuditLog).filter(AuditLog.event_type == "recovery.settled").first()
    assert audit is not None
    assert audit.recovered_amount == pytest.approx(14999.0)

    # Webhook event recorded as processed (idempotency)
    evt = db.query(WebhookEvent).filter(WebhookEvent.razorpay_event_id == "evt_settle_01").first()
    assert evt is not None
    assert evt.processed is True


def test_duplicate_event_does_not_double_settle(db, recovery_action):
    service = SettlementService(db=db)
    first = service.settle_paid_payment_link(_paid_payload(recovery_action.id))
    assert first["status"] == "settled"

    # Redeliver the SAME event id -> must be skipped, no double count
    second = service.settle_paid_payment_link(_paid_payload(recovery_action.id))
    assert second["status"] == "skipped"

    # Exactly one RecoveryOutcome, exactly one recovered amount
    outcomes = db.query(RecoveryOutcome).filter(
        RecoveryOutcome.recovery_action_id == recovery_action.id
    ).all()
    assert len(outcomes) == 1

    customer = db.query(Customer).filter(Customer.id == recovery_action.customer_id).first()
    assert customer.recovered_payments == 1  # not 2


def test_unknown_recovery_action_handled_gracefully(db):
    service = SettlementService(db=db)
    result = service.settle_paid_payment_link(_paid_payload(999999, event_id="evt_unknown_01"))

    assert result["status"] == "skipped"
    assert "no matching recovery action" in result["reason"]
    # No crash, no outcome rows created
    assert db.query(RecoveryOutcome).count() == 0


def test_fallback_matches_by_payment_link_id(db, recovery_action):
    """If notes.recovery_action_id is missing, fall back to payment_link.id match."""
    service = SettlementService(db=db)
    payload = _paid_payload(None, event_id="evt_fallback_01")
    # remove the notes recovery_action_id entirely
    payload["payload"]["payment_link"]["entity"]["notes"] = {}
    result = service.settle_paid_payment_link(payload)

    assert result["status"] == "settled"
    assert result["action_id"] == recovery_action.id
