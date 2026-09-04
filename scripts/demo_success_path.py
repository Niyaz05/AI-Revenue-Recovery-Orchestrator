from __future__ import annotations

"""
End-to-End Demo: Successful Payment Recovery Journey.

Flow:
1. Subscription payment fails due to temporary gateway error / insufficient balance
2. Webhook received & ingested -> Subscription enters PENDING at-risk state
3. Safety Engine evaluates constraints -> ALLOWED: [WAIT, RETRY, PAYMENT_LINK]
4. LinUCB Bandit recommends PAYMENT_LINK with 88% confidence
5. Action Executor generates real Razorpay Payment Link in Test Mode
6. Customer completes payment -> payment_link.paid webhook received
7. Recovery outcome recorded with full immutable audit trail.
"""

import json
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
from backend.models.customer import Customer
from backend.models.base import Base

from backend.models.enums import ActionStatus, ActionType, Provenance, SubscriptionStatus
from backend.models.merchant_policy import MerchantPolicy
from backend.models.payment_attempt import PaymentAttempt
from backend.models.subscription import Subscription
from backend.services.orchestrator import OrchestratorService
from backend.services.settlement import SettlementService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_success_demo():
    print("==================================================================")
    print("      DEMO 1: SUCCESSFUL REVENUE RECOVERY JOURNEY (TEST MODE)     ")
    print("==================================================================")

    # In-memory test DB for isolated demo execution
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # 1. Initialize Merchant Policy
    policy = MerchantPolicy(max_retry_attempts=3, retry_cooldown_hours=4.0)
    db.add(policy)
    db.flush()

    # 2. Customer and Active Subscription
    customer = Customer(
        razorpay_customer_id="cust_demo_succ_01",
        name="Vikram Mehta",
        email="vikram@enterprise.in",
        phone="+919876543210",
        lifetime_value=120000.0,
        total_payments=8,
        failed_payments=1,
    )
    db.add(customer)
    db.flush()

    sub = Subscription(
        razorpay_subscription_id="sub_demo_succ_01",
        customer_id=customer.id,
        plan_name="Enterprise Cloud Annual",
        amount_per_period=14999.0,
        status=SubscriptionStatus.PENDING,
        paid_count=1,
        remaining_count=11,
        total_count=12,
    )
    db.add(sub)
    db.flush()

    print(f"\n[Step 1] Customer: {customer.name} (LTV: ₹{customer.lifetime_value:,.2f})")
    print(f"         Subscription: {sub.plan_name} (Amount: ₹{sub.amount_per_period:,.2f}/mo)")
    print(f"         State: At-Risk [{sub.status.value}]")

    # 3. Failed payment webhook ingestion
    pay = PaymentAttempt(
        razorpay_payment_id="pay_demo_fail_01",
        subscription_id=sub.id,
        customer_id=customer.id,
        amount=14999.0,
        status="failed",
        failure_reason="insufficient_funds",
        error_code="BAD_REQUEST_ERROR",
        error_description="Insufficient funds in customer card account",
        attempt_number=1,
    )
    db.add(pay)
    db.flush()

    print(f"\n[Step 2] Payment Failed: {pay.razorpay_payment_id}")
    print(f"         Reason: {pay.failure_reason} ({pay.error_description})")

    # 4. Orchestrator Evaluates & Executes Intervention
    # Load the trained LinUCB model so the policy recommends PAYMENT_LINK
    # for an insufficient-funds failure (rather than the untrained default).
    orchestrator = OrchestratorService(db=db, bandit_model_path="data/models/bandit_model.npz")
    print("\n[Step 3] Running Hierarchical Recovery Policy...")
    res = orchestrator.process_failed_payment(
        payment_attempt_id=pay.id,
        provenance=Provenance.RAZORPAY_TEST_MODE,
    )

    print(f"         Chosen Action: {res['action_type']}")
    print(f"         Execution Status: {res['action_status']}")
    print(f"         Payment Link Created: {res['razorpay_payment_link_id']}")
    print(f"         LLM Grounded Explanation: \n         \"{res['explanation']}\"")

    # 5. Customer Pays Link -> Settle via the simulated webhook (closed loop)
    print(f"\n[Step 4] Customer pays via UPI link -> Settling via simulated webhook...")
    from backend.models.recovery_action import RecoveryAction

    # Build a `payment_link.paid` payload in the same shape a real Razorpay
    # webhook would send, injecting the real recovery_action_id & payment link id.
    settlement_payload = {
        "id": f"evt_paid_{res['action_id']}",
        "event": "payment_link.paid",
        "payload": {
            "payment_link": {
                "entity": {
                    "id": res["razorpay_payment_link_id"],
                    "amount": 1499900,  # paise
                    "currency": "INR",
                    "status": "paid",
                    "notes": {
                        "recovery_action_id": str(res["action_id"]),
                        "subscription_id": sub.razorpay_subscription_id,
                    },
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_demo_settled_01",
                    "amount": 1499900,  # paise
                    "currency": "INR",
                    "status": "captured",
                    "method": "upi",
                }
            },
        },
    }

    settlement = SettlementService(db=db)
    settle_result = settlement.settle_paid_payment_link(settlement_payload)

    action = db.query(RecoveryAction).filter(RecoveryAction.id == res["action_id"]).first()
    print(f"\n[Step 5] Recovery Journey Resolved:")
    print(f"         Settlement Status: {settle_result['status']}")
    print(f"         Action Status: {action.status.value if action else 'n/a'}")
    print(f"         Recovered Revenue: ₹{action.recovered_amount:,.2f} (Tagged: {Provenance.RAZORPAY_TEST_MODE.value})")
    print(f"         Total Audit Log Entries: {db.query(Base.metadata.tables['audit_logs']).count()}")
    print("==================================================================\n")


if __name__ == "__main__":
    run_success_demo()
