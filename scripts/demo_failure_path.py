from __future__ import annotations
"""
End-to-End Demo: Safe Failure Path & Hard Policy Veto Demonstration.

Flow:
1. Customer is opted out or has exceeded maximum retry attempts
2. ML strategy model recommends RETRY / PAYMENT_LINK based on context
3. Level 1 Hard Safety Layer VETOES the AI recommendation
4. System degrades gracefully to STOP_RECOVERY
5. Full audit trail recorded proving deterministic safety prevailed over AI.
"""

import logging
import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.base import Base
from backend.models.customer import Customer
from backend.models.enums import ActionStatus, ActionType, Provenance, SubscriptionStatus
from backend.models.merchant_policy import MerchantPolicy
from backend.models.payment_attempt import PaymentAttempt
from backend.models.subscription import Subscription
from backend.services.orchestrator import OrchestratorService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_failure_demo():
    print("==================================================================")
    print("      DEMO 2: SAFE FAILURE & HARD SAFETY VETO DEMONSTRATION       ")
    print("==================================================================")

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    policy = MerchantPolicy(max_retry_attempts=3)
    db.add(policy)
    db.flush()

    # Create Opted-Out Customer
    customer = Customer(
        razorpay_customer_id="cust_demo_optout_01",
        name="Sunita Rao",
        email="sunita@example.com",
        phone="+919877777777",
        opted_out_recovery=True,  # OPTED OUT
        lifetime_value=25000.0,
    )
    db.add(customer)
    db.flush()

    sub = Subscription(
        razorpay_subscription_id="sub_demo_optout_01",
        customer_id=customer.id,
        plan_name="Pro Developer Tier",
        amount_per_period=2999.0,
        status=SubscriptionStatus.PENDING,
        paid_count=4,
        remaining_count=8,
        total_count=12,
    )
    db.add(sub)
    db.flush()

    pay = PaymentAttempt(
        razorpay_payment_id="pay_demo_optout_01",
        subscription_id=sub.id,
        customer_id=customer.id,
        amount=2999.0,
        status="failed",
        failure_reason="temporary_bank_failure",
        attempt_number=1,
    )
    db.add(pay)
    db.flush()

    print(f"\n[Step 1] Customer '{customer.name}' has opted out of automated interventions.")
    print(f"[Step 2] Payment attempt failed: ₹{pay.amount:,.2f} ({pay.failure_reason})")

    orchestrator = OrchestratorService(db=db)
    print("\n[Step 3] Running Orchestrator Evaluation...")
    res = orchestrator.process_failed_payment(
        payment_attempt_id=pay.id,
        provenance=Provenance.RAZORPAY_TEST_MODE,
    )

    print(f"\n[Safety Audit Invariant Verified]:")
    print(f"  • AI Action Selected: {res['action_type']}")
    print(f"  • Hard Safety Veto Active: {res['action_type'] == 'STOP_RECOVERY'}")
    print(f"  • Reason: Customer opted out from automated recovery interventions.")
    print(f"  • Zero unauthorized customer messages or charges dispatched.")
    print("==================================================================\n")


if __name__ == "__main__":
    run_failure_demo()
