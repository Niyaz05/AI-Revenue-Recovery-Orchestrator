from __future__ import annotations
"""
Settlement Service — closes the recovery loop.

Given a `payment_link.paid` payload (in the same shape Razorpay sends, or a local
fixture), this service correlates the settled payment back to the originating
RecoveryAction and records a completed, auditable recovery outcome.

This is intentionally credential-free: the exact same logic runs whether the
payload arrives from a real Razorpay webhook or from the local simulated
`POST /api/webhooks/payment-link-paid` endpoint. A real deployment would simply
point the Razorpay webhook at that endpoint.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.models.audit_log import AuditLog
from backend.models.customer import Customer
from backend.models.enums import ActionStatus, OutcomeType, Provenance
from backend.models.payment_attempt import PaymentAttempt
from backend.models.recovery_action import RecoveryAction
from backend.models.recovery_outcome import RecoveryOutcome
from backend.models.webhook_event import WebhookEvent
from backend.services.field_mapper import paise_to_rupees

logger = logging.getLogger(__name__)


class SettlementService:
    """
    Settles a paid recovery payment link and records the outcome/audit trail.
    """

    def __init__(self, db: Session):
        self.db = db

    def settle_paid_payment_link(self, payload: dict) -> dict:
        """
        Process a `payment_link.paid` payload.

        Returns a structured result dict describing what happened:
          - status: "settled" | "skipped" | "error"
          - reason: human-readable explanation
          - action_id / recovered_amount when settled
        """
        event_id = self._extract_event_id(payload)
        event_type = self._extract_event_type(payload)

        # ── Idempotency: skip events already processed ──────────────────────
        if event_id:
            existing = (
                self.db.query(WebhookEvent)
                .filter(WebhookEvent.razorpay_event_id == event_id)
                .first()
            )
            if existing and existing.processed:
                logger.info(f"Skipping already-processed event {event_id}")
                return {
                    "status": "skipped",
                    "reason": "event already processed",
                    "event_id": event_id,
                }

        try:
            result = self._settle(payload, event_id=event_id, event_type=event_type)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Settlement failed: {exc}", exc_info=True)
            self._record_webhook_event(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                processed=False,
                error=str(exc),
            )
            self.db.commit()
            return {"status": "error", "reason": str(exc)}

        self.db.commit()
        return result

    # ─────────────────────────── internal helpers ───────────────────────────

    def _settle(self, payload: dict, event_id: Optional[str], event_type: Optional[str]) -> dict:
        plink_entity = self._payment_link_entity(payload)
        pay_entity = self._payment_entity(payload)

        # 1. Resolve the RecoveryAction
        recovery_action = self._resolve_recovery_action(plink_entity, pay_entity)

        if recovery_action is None:
            reason = "no matching recovery action found (unknown recovery_action_id / payment link)"
            self._record_webhook_event(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                processed=True,
                error="",
            )
            logger.warning(reason)
            return {"status": "skipped", "reason": reason}

        # 2. Idempotent: already settled → skip without double counting
        if recovery_action.status == ActionStatus.COMPLETED:
            reason = f"recovery action {recovery_action.id} already completed"
            self._record_webhook_event(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                processed=True,
                error="",
            )
            logger.info(reason)
            return {"status": "skipped", "reason": reason, "action_id": recovery_action.id}

        # 3. Determine recovered amount (paise → rupees)
        recovered_amount = paise_to_rupees(pay_entity.get("amount")) if pay_entity else 0.0
        if recovered_amount <= 0.0:
            recovered_amount = recovery_action.razorpay_payment_link_id and plink_entity.get("amount")
            recovered_amount = paise_to_rupees(recovered_amount) if recovered_amount else 0.0
        if recovered_amount <= 0.0:
            # Fall back to the amount we attempted to recover
            recovered_amount = self._fallback_amount(recovery_action)

        # 4. Flip the action to COMPLETED
        recovery_action.status = ActionStatus.COMPLETED
        recovery_action.completed_at = datetime.utcnow()
        recovery_action.recovered_amount = recovered_amount
        recovery_action.result = json.dumps(
            {
                "settled_via": "payment_link.paid",
                "payment_id": pay_entity.get("id") if pay_entity else None,
                "payment_link_id": plink_entity.get("id") if plink_entity else None,
                "method": pay_entity.get("method") if pay_entity else None,
            }
        )

        # 5. Create a RecoveryOutcome row
        outcome = RecoveryOutcome(
            recovery_action_id=recovery_action.id,
            subscription_id=recovery_action.subscription_id,
            outcome_type=OutcomeType.RECOVERED,
            recovered_amount=recovered_amount,
            recovery_time_seconds=self._recovery_time_seconds(recovery_action),
            total_attempts=recovery_action.payment_attempt.attempt_number
            if recovery_action.payment_attempt
            else 0,
            total_interventions=1,
            reward=recovered_amount,
            provenance=recovery_action.provenance,
        )
        self.db.add(outcome)

        # 6. Bump customer recovered payments
        customer = self.db.query(Customer).filter(Customer.id == recovery_action.customer_id).first()
        if customer:
            customer.recovered_payments = (customer.recovered_payments or 0) + 1

        # 7. Write an immutable audit log
        audit = AuditLog(
            event_type="recovery.settled",
            customer_id=recovery_action.customer_id,
            payment_id=recovery_action.payment_attempt_id,
            subscription_id=recovery_action.subscription_id,
            state_before=json.dumps({"action_status": "executing"}),
            state_after=json.dumps({"action_status": "completed", "recovered_amount": recovered_amount}),
            executed_action=recovery_action.action_type.value,
            result=json.dumps({"payment_id": pay_entity.get("id") if pay_entity else None}),
            recovered_amount=recovered_amount,
            reason=f"payment_link.paid settled recovery action {recovery_action.id}",
            model_version=recovery_action.model_version,
            policy_version=recovery_action.policy_version,
            provenance=recovery_action.provenance,
        )
        self.db.add(audit)

        # 8. Record the webhook event as processed (idempotency)
        self._record_webhook_event(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
            processed=True,
            error="",
        )

        logger.info(
            f"Settled recovery action {recovery_action.id}: ₹{recovered_amount:,.2f} "
            f"(event {event_id})"
        )
        return {
            "status": "settled",
            "action_id": recovery_action.id,
            "recovered_amount": recovered_amount,
            "event_id": event_id,
            "reason": "payment link paid",
        }

    def _resolve_recovery_action(self, plink_entity: dict, pay_entity: dict) -> Optional[RecoveryAction]:
        """Resolve the RecoveryAction from notes.recovery_action_id or the payment link id."""
        notes = (plink_entity or {}).get("notes") or {}
        recovery_action_id = notes.get("recovery_action_id")

        if recovery_action_id is not None:
            try:
                action = (
                    self.db.query(RecoveryAction)
                    .filter(RecoveryAction.id == int(recovery_action_id))
                    .first()
                )
                if action:
                    return action
            except (TypeError, ValueError):
                logger.warning(f"Non-integer recovery_action_id in notes: {recovery_action_id!r}")

        # Fallback: match by razorpay payment link id
        plink_id = (plink_entity or {}).get("id")
        if plink_id:
            action = (
                self.db.query(RecoveryAction)
                .filter(RecoveryAction.razorpay_payment_link_id == plink_id)
                .first()
            )
            if action:
                return action

        return None

    def _fallback_amount(self, recovery_action: RecoveryAction) -> float:
        """Derive an amount from the linked payment attempt if available."""
        if recovery_action.payment_attempt:
            return recovery_action.payment_attempt.amount or 0.0
        return 0.0

    def _recovery_time_seconds(self, recovery_action: RecoveryAction) -> int:
        if recovery_action.created_at:
            delta = (datetime.utcnow() - recovery_action.created_at).total_seconds()
            return int(max(0, delta))
        return 0

    def _record_webhook_event(
        self,
        event_id: Optional[str],
        event_type: Optional[str],
        payload: dict,
        processed: bool,
        error: str,
    ) -> None:
        """Persist the raw event for idempotency / audit / replay."""
        if not event_id:
            return
        existing = (
            self.db.query(WebhookEvent)
            .filter(WebhookEvent.razorpay_event_id == event_id)
            .first()
        )
        if existing:
            existing.processed = processed
            existing.processing_error = error
            existing.processed_at = datetime.utcnow()
            return
        self.db.add(
            WebhookEvent(
                razorpay_event_id=event_id,
                event_type=event_type or "payment_link.paid",
                raw_payload=json.dumps(payload),
                signature_valid=False,  # simulated path, no HMAC
                processed=processed,
                processing_error=error,
                idempotency_key=f"sim_{event_id}",
            )
        )

    # ─────────────────────────── payload accessors ──────────────────────────

    @staticmethod
    def _extract_event_id(payload: dict) -> Optional[str]:
        return payload.get("id") or payload.get("razorpay_event_id")

    @staticmethod
    def _extract_event_type(payload: dict) -> Optional[str]:
        return payload.get("event")

    @staticmethod
    def _payment_link_entity(payload: dict) -> dict:
        plink = payload.get("payload", {}).get("payment_link", {})
        entity = plink.get("entity", {}) if isinstance(plink, dict) else {}
        return entity if isinstance(entity, dict) else {}

    @staticmethod
    def _payment_entity(payload: dict) -> dict:
        pay = payload.get("payload", {}).get("payment", {})
        entity = pay.get("entity", {}) if isinstance(pay, dict) else {}
        return entity if isinstance(entity, dict) else {}
