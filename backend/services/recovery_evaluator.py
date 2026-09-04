from __future__ import annotations
"""
Recovery Evaluator Service.

Coordinates the end-to-end recovery evaluation pipeline:
1. Context gathering (Customer, Subscription, History, Policies)
2. Level 1 Hard Safety Policy (Deterministic rules & veto)
3. Level 2 Strategy Selection (Contextual Bandit LinUCB / Rule fallback)
4. Level 3 Action Execution (Razorpay Payment Links / Schedule)
5. Immutable Audit Trail logging with strict provenance labeling.
"""

from datetime import datetime, timedelta
import json
import logging
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from backend.models.audit_log import AuditLog
from backend.models.customer import Customer
from backend.models.enums import (
    ActionStatus,
    ActionType,
    FailureReason,
    Provenance,
    SubscriptionStatus,
)
from backend.models.merchant_policy import MerchantPolicy
from backend.models.payment_attempt import PaymentAttempt
from backend.models.recovery_action import RecoveryAction
from backend.models.subscription import Subscription
from backend.policies.action_executor import ActionExecutor, ExecutionResult
from backend.policies.safety_policy import PolicyDecision, SafetyContext, SafetyPolicyEngine
from backend.policies.strategy_policy import StrategyPolicyEngine
from razorpay.client import RazorpayClient

logger = logging.getLogger(__name__)


class RecoveryEvaluator:
    def __init__(
        self,
        db: Session,
        bandit_model_path: Optional[str] = None,
        razorpay_client: Optional[RazorpayClient] = None,
    ):
        self.db = db
        self.safety_engine = SafetyPolicyEngine()
        self.strategy_engine = StrategyPolicyEngine(bandit_model_path=bandit_model_path)
        self.executor = ActionExecutor(razorpay_client=razorpay_client)

    def evaluate_at_risk_payment(
        self,
        payment_attempt_id: int,
        provenance: Provenance = Provenance.RAZORPAY_TEST_MODE,
    ) -> Optional[RecoveryAction]:
        """
        Main entrypoint when a payment fails or subscription degrades.
        """
        payment = self.db.query(PaymentAttempt).filter(PaymentAttempt.id == payment_attempt_id).first()
        if not payment:
            logger.error(f"PaymentAttempt {payment_attempt_id} not found.")
            return None

        customer = self.db.query(Customer).filter(Customer.id == payment.customer_id).first()
        subscription = (
            self.db.query(Subscription).filter(Subscription.id == payment.subscription_id).first()
            if payment.subscription_id
            else None
        )

        # Load active merchant policy
        policy = (
            self.db.query(MerchantPolicy)
            .filter(MerchantPolicy.is_active == True)
            .order_by(MerchantPolicy.id.desc())
            .first()
        )
        if not policy:
            policy = MerchantPolicy()
            self.db.add(policy)
            self.db.flush()

        self.safety_engine = SafetyPolicyEngine(merchant_policy=policy)

        # 1. Build Safety Context
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Count interventions today for this customer
        interventions_today = (
            self.db.query(RecoveryAction)
            .filter(
                RecoveryAction.customer_id == customer.id,
                RecoveryAction.created_at >= today_start,
                RecoveryAction.status != ActionStatus.BLOCKED,
            )
            .count()
        )

        # Count global retries today
        global_retries_today = (
            self.db.query(RecoveryAction)
            .filter(
                RecoveryAction.action_type == ActionType.RETRY,
                RecoveryAction.created_at >= today_start,
                RecoveryAction.status != ActionStatus.BLOCKED,
            )
            .count()
        )

        # Find first failure time and last attempt time
        first_failure = payment.created_at
        last_attempt = payment.created_at
        if subscription:
            earliest_fail = (
                self.db.query(PaymentAttempt)
                .filter(
                    PaymentAttempt.subscription_id == subscription.id,
                    PaymentAttempt.status == "failed",
                )
                .order_by(PaymentAttempt.created_at.asc())
                .first()
            )
            if earliest_fail:
                first_failure = earliest_fail.created_at

        safety_ctx = SafetyContext(
            customer_id=customer.id,
            amount=payment.amount,
            attempt_count=payment.attempt_number,
            interventions_today=interventions_today,
            global_retries_today=global_retries_today,
            first_failure_time=first_failure,
            last_attempt_time=last_attempt,
            is_customer_opted_out=customer.opted_out_recovery,
            current_time=datetime.utcnow(),
        )

        # 2. Evaluate Level 1 Safety Layer (Deterministic Veto)
        safety_decision: PolicyDecision = self.safety_engine.evaluate(safety_ctx)

        # 3. Evaluate Level 2 Strategy (LinUCB / Rule fallback strictly on allowed actions)
        hours_since_first = max(0.0, (datetime.utcnow() - first_failure).total_seconds() / 3600.0)
        paid_count = subscription.paid_count if subscription else 0

        chosen_action, strategy_meta = self.strategy_engine.select_action(
            failure_reason=payment.failure_reason or "unknown",
            amount=payment.amount,
            attempt_count=payment.attempt_number,
            hours_since_first_failure=hours_since_first,
            customer_ltv=customer.lifetime_value,
            customer_failure_rate=customer.failure_rate,
            subscription_paid_count=paid_count,
            safety_decision=safety_decision,
            is_weekend=datetime.utcnow().weekday() >= 5,
        )

        # Ensure safety layer hard veto is respected
        is_blocked = chosen_action not in safety_decision.allowed_actions
        block_reason = safety_decision.blocked_actions.get(chosen_action, "") if is_blocked else ""

        if is_blocked:
            logger.warning(f"AI chosen action {chosen_action} was BLOCKED by safety layer! Vetoing to STOP_RECOVERY.")
            chosen_action = ActionType.STOP_RECOVERY

        # 4. Create RecoveryAction Record (Idempotent)
        sub_id_str = subscription.razorpay_subscription_id if subscription else f"sub_local_{payment.id}"
        idempotency_key = f"rec_{payment.id}_{chosen_action.value}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        action_record = RecoveryAction(
            payment_attempt_id=payment.id,
            subscription_id=subscription.id if subscription else None,
            customer_id=customer.id,
            action_type=chosen_action,
            status=ActionStatus.PENDING,
            blocked_by_policy=is_blocked,
            block_reason=block_reason,
            ai_recommendation=json.dumps(strategy_meta),
            policy_decision=json.dumps(safety_decision.to_dict()),
            idempotency_key=idempotency_key,
            model_version=strategy_meta.get("model_version", "v1"),
            policy_version=safety_decision.policy_version,
            provenance=provenance,
            created_at=datetime.utcnow(),
        )
        self.db.add(action_record)
        self.db.flush()

        # 5. Execute Level 3 Action
        exec_result: ExecutionResult = self.executor.execute(
            action_type=chosen_action,
            subscription_id=sub_id_str,
            customer_id=customer.id,
            customer_name=customer.name,
            customer_email=customer.email,
            customer_phone=customer.phone,
            amount=payment.amount,
            recovery_action_id=str(action_record.id),
            provenance=provenance,
        )

        # Update action record with execution results
        action_record.status = exec_result.status
        action_record.executed_at = datetime.utcnow()
        action_record.result = json.dumps(exec_result.details)
        action_record.razorpay_payment_link_id = exec_result.razorpay_payment_link_id
        if exec_result.status == ActionStatus.COMPLETED and exec_result.recovered_amount > 0:
            action_record.completed_at = datetime.utcnow()
            action_record.recovered_amount = exec_result.recovered_amount

        # 6. Write Immutable Audit Log
        pay_status_str = payment.status.value if hasattr(payment.status, "value") else str(payment.status)
        sub_status_str = (
            (subscription.status.value if hasattr(subscription.status, "value") else str(subscription.status))
            if subscription
            else "none"
        )
        action_status_str = action_record.status.value if hasattr(action_record.status, "value") else str(action_record.status)
        chosen_action_str = chosen_action.value if hasattr(chosen_action, "value") else str(chosen_action)

        audit = AuditLog(
            event_type="recovery.decision_executed",
            customer_id=customer.id,
            payment_id=payment.id,
            subscription_id=subscription.id if subscription else None,
            state_before=json.dumps({"payment_status": pay_status_str, "subscription_status": sub_status_str}),
            state_after=json.dumps({"action_status": action_status_str}),
            ai_recommendation=json.dumps(strategy_meta),
            policy_decision=json.dumps(safety_decision.to_dict()),
            executed_action=chosen_action_str,
            result=json.dumps(exec_result.details),
            recovered_amount=exec_result.recovered_amount,
            reason="; ".join(strategy_meta.get("reasons", [])),
            model_version=strategy_meta.get("model_version", "v1"),
            policy_version=safety_decision.policy_version,
            provenance=provenance,
        )
        self.db.add(audit)
        self.db.commit()

        logger.info(f"Recovery evaluation complete: action={chosen_action_str} status={action_status_str}")
        return action_record
