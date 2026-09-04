from __future__ import annotations
"""
Main Orchestrator Service.

Coordinates:
- Webhook Ingestion & State Machine updates
- Recovery Evaluation & Hard Safety verification
- Razorpay Test Mode execution
- Decision Explanation generation
- Recovery Outcome recording
"""

import json
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from backend.models.audit_log import AuditLog
from backend.models.customer import Customer
from backend.models.enums import ActionStatus, ActionType, Provenance, SubscriptionStatus
from backend.models.payment_attempt import PaymentAttempt
from backend.models.recovery_action import RecoveryAction
from backend.models.subscription import Subscription
from backend.services.explainer import DecisionExplainer
from backend.services.recovery_evaluator import RecoveryEvaluator
from razorpay.client import RazorpayClient

from evaluation.offline_rl.ope import check_ope_gate

logger = logging.getLogger(__name__)


class OrchestratorService:
    REQUIRE_OPE_GATE: bool = True

    def __init__(
        self,
        db: Session,
        bandit_model_path: Optional[str] = None,
        rl_model_path: Optional[str] = None,
        require_ope_gate: Optional[bool] = None,
        ope_results_path: str = "evaluation/results/ope_results.json",
        min_return_threshold: float = 0.0,
    ):
        self.db = db
        self.client = RazorpayClient()
        self.require_ope_gate = self.REQUIRE_OPE_GATE if require_ope_gate is None else require_ope_gate
        self.ope_results_path = ope_results_path
        self.min_return_threshold = min_return_threshold
        self.rl_model_path = rl_model_path
        self.rl_cleared = self.check_rl_deployment_clearance()

        self.evaluator = RecoveryEvaluator(
            db=db,
            bandit_model_path=bandit_model_path,
            razorpay_client=self.client,
        )
        self.explainer = DecisionExplainer()

    def check_rl_deployment_clearance(self) -> bool:
        """
        Verifies the OPE deployment gate. Returns True if sequential RL policy
        is approved for production / test mode action execution.
        """
        if not self.require_ope_gate:
            logger.info("OPE deployment gate is bypassed (require_ope_gate=False).")
            return True

        cleared = check_ope_gate(
            ope_results_path=self.ope_results_path,
            min_return_threshold=self.min_return_threshold,
        )
        if not cleared:
            logger.warning(
                "OPE deployment gate blocked sequential RL execution: "
                f"results file missing or return below threshold {self.min_return_threshold}. "
                "Falling back to LinUCB contextual bandit / rule-based fallback."
            )
        return cleared


    def process_failed_payment(
        self,
        payment_attempt_id: int,
        provenance: Provenance = Provenance.RAZORPAY_TEST_MODE,
    ) -> Dict[str, Any]:
        """
        Orchestrates recovery evaluation and execution for a failed payment.
        """
        action = self.evaluator.evaluate_at_risk_payment(
            payment_attempt_id=payment_attempt_id,
            provenance=provenance,
        )
        if not action:
            return {"status": "error", "message": "Evaluation failed or payment not found"}

        # Generate LLM grounded explanation
        payment = self.db.query(PaymentAttempt).filter(PaymentAttempt.id == payment_attempt_id).first()
        customer = self.db.query(Customer).filter(Customer.id == action.customer_id).first()
        ai_meta = action.get_ai_recommendation()
        policy_dec = action.get_policy_decision()

        explanation = self.explainer.explain_recovery_decision(
            customer_name=customer.name if customer else "Customer",
            amount=payment.amount if payment else 0.0,
            failure_reason=payment.failure_reason if payment else "unknown",
            chosen_action=action.action_type.value,
            score=ai_meta.get("score", 0.0),
            confidence=ai_meta.get("confidence", 1.0),
            features=ai_meta.get("features", {}),
            safety_reasons=policy_dec.get("reasons", []),
        )

        act_type_str = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        act_status_str = action.status.value if hasattr(action.status, "value") else str(action.status)
        prov_str = action.provenance.value if hasattr(action.provenance, "value") else str(action.provenance)

        return {
            "status": "success",
            "action_id": action.id,
            "action_type": act_type_str,
            "action_status": act_status_str,
            "blocked_by_policy": action.blocked_by_policy,
            "razorpay_payment_link_id": action.razorpay_payment_link_id,
            "explanation": explanation,
            "provenance": prov_str,
        }

    def trigger_test_mode_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Creates a test customer, subscription, and failure attempt in SQLite,
        then evaluates and executes recovery against Razorpay Test Mode.
        """
        import uuid
        uid = uuid.uuid4().hex[:8]

        # Create test customer
        cust = Customer(
            razorpay_customer_id=f"cust_{scenario_name[:6]}_{uid}",
            name=f"Test Customer ({scenario_name})",
            email=f"{scenario_name}_{uid}@example.com",
            phone="+919876543210",
            lifetime_value=45000.0,
            total_payments=5,
            failed_payments=1,
        )
        self.db.add(cust)
        self.db.flush()

        amount = 75000.0 if scenario_name == "high_value_customer" else 2499.0

        sub = Subscription(
            razorpay_subscription_id=f"sub_{scenario_name[:6]}_{uid}",
            customer_id=cust.id,
            razorpay_plan_id="plan_test_standard",
            plan_name="Standard SaaS Pro Plan",
            amount_per_period=amount,
            status=SubscriptionStatus.PENDING,
            paid_count=3,
            remaining_count=9,
            total_count=12,
        )
        self.db.add(sub)
        self.db.flush()

        reason_map = {
            "temporary_bank_failure": "temporary_bank_failure",
            "insufficient_funds": "insufficient_funds",
            "invalid_payment_method": "invalid_payment_method",
            "high_value_customer": "authentication_failed",
            "opted_out_customer": "temporary_bank_failure",
        }

        if scenario_name == "opted_out_customer":
            cust.opted_out_recovery = True

        pay = PaymentAttempt(
            razorpay_payment_id=f"pay_{scenario_name[:6]}_{uid}",
            subscription_id=sub.id,
            customer_id=cust.id,
            amount=amount,
            status="failed",
            failure_reason=reason_map.get(scenario_name, "unknown"),
            error_code="PAYMENT_FAILED",
            error_description=f"Simulated test mode failure for {scenario_name}",
            attempt_number=1,
        )
        self.db.add(pay)
        self.db.flush()

        res = self.process_failed_payment(
            payment_attempt_id=pay.id,
            provenance=Provenance.RAZORPAY_TEST_MODE,
        )
        return res
