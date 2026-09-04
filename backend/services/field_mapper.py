from __future__ import annotations
"""
Field Mapper — translates Razorpay webhook/API fields to internal model fields.

This is the single point of translation between Razorpay's schema and ours.
We intentionally DO NOT mirror Razorpay's internal DB — we map their fields
into our own domain schema at ingestion time.

Mapping Table:
───────────────────────────────────────────────────────────────────────────
Razorpay Field                → Internal Field
───────────────────────────────────────────────────────────────────────────
subscription.id               → Subscription.razorpay_subscription_id
subscription.plan_id          → Subscription.razorpay_plan_id
subscription.customer_id      → Customer.razorpay_customer_id
subscription.status           → Subscription.status (mapped via enum)
subscription.current_start    → Subscription.current_period_start (epoch→dt)
subscription.current_end      → Subscription.current_period_end (epoch→dt)
subscription.paid_count       → Subscription.paid_count
subscription.remaining_count  → Subscription.remaining_count
subscription.total_count      → Subscription.total_count
subscription.charge_at        → Subscription.charge_at (epoch→datetime)
payment.id                    → PaymentAttempt.razorpay_payment_id
payment.amount                → PaymentAttempt.amount (paise→rupees)
payment.status                → PaymentAttempt.status (mapped via enum)
payment.method                → PaymentAttempt.payment_method
payment.error_code            → PaymentAttempt.error_code
payment.error_description     → PaymentAttempt.error_description
payment.error_reason          → PaymentAttempt.failure_reason (classified)
payment.error_source          → PaymentAttempt.error_source
payment_link.id               → RecoveryAction.razorpay_payment_link_id
───────────────────────────────────────────────────────────────────────────
"""

import logging
from datetime import datetime
from typing import Any

from backend.models.enums import (
    FailureReason,
    PaymentStatus,
    SubscriptionStatus,
)

logger = logging.getLogger(__name__)

# ─────────────────────── Razorpay error_reason → FailureReason ───────────────

_ERROR_REASON_MAP: dict[str, FailureReason] = {
    # Razorpay error reasons → our internal classification
    "insufficient_balance": FailureReason.INSUFFICIENT_FUNDS,
    "insufficient_funds": FailureReason.INSUFFICIENT_FUNDS,
    "bank_declined": FailureReason.BANK_DECLINED,
    "network_error": FailureReason.NETWORK_ERROR,
    "gateway_error": FailureReason.TEMPORARY_BANK_FAILURE,
    "server_error": FailureReason.TEMPORARY_BANK_FAILURE,
    "payment_processing_failed": FailureReason.TEMPORARY_BANK_FAILURE,
    "invalid_card": FailureReason.INVALID_PAYMENT_METHOD,
    "card_expired": FailureReason.CARD_EXPIRED,
    "invalid_vpa": FailureReason.INVALID_PAYMENT_METHOD,
    "authentication_failed": FailureReason.AUTHENTICATION_FAILED,
    "suspected_fraud": FailureReason.FRAUD_SUSPECTED,
}


def classify_failure_reason(
    error_reason: str | None,
    error_code: str | None = None,
    error_description: str | None = None,
) -> FailureReason:
    """Map Razorpay error fields to our internal FailureReason enum."""
    if not error_reason:
        # Try to infer from description
        desc = (error_description or "").lower()
        if "insufficient" in desc or "balance" in desc:
            return FailureReason.INSUFFICIENT_FUNDS
        if "expired" in desc:
            return FailureReason.CARD_EXPIRED
        if "invalid" in desc:
            return FailureReason.INVALID_PAYMENT_METHOD
        return FailureReason.UNKNOWN

    return _ERROR_REASON_MAP.get(error_reason.lower(), FailureReason.UNKNOWN)


# ─────────────────────── Epoch → datetime ────────────────────────────────────

def epoch_to_datetime(epoch: int | None) -> datetime | None:
    """Convert Unix epoch timestamp to datetime. Returns None if input is None/0."""
    if not epoch:
        return None
    return datetime.utcfromtimestamp(epoch)


# ─────────────────────── Paise → Rupees ──────────────────────────────────────

def paise_to_rupees(paise: int | None) -> float:
    """Convert Razorpay amount (paise) to rupees."""
    if paise is None:
        return 0.0
    return paise / 100.0


# ─────────────────────── Status mapping ──────────────────────────────────────

_SUBSCRIPTION_STATUS_MAP: dict[str, SubscriptionStatus] = {
    "created": SubscriptionStatus.CREATED,
    "authenticated": SubscriptionStatus.AUTHENTICATED,
    "active": SubscriptionStatus.ACTIVE,
    "pending": SubscriptionStatus.PENDING,
    "halted": SubscriptionStatus.HALTED,
    "cancelled": SubscriptionStatus.CANCELLED,
    "completed": SubscriptionStatus.COMPLETED,
    "expired": SubscriptionStatus.EXPIRED,
    "paused": SubscriptionStatus.PAUSED,
}

_PAYMENT_STATUS_MAP: dict[str, PaymentStatus] = {
    "created": PaymentStatus.CREATED,
    "authorized": PaymentStatus.AUTHORIZED,
    "captured": PaymentStatus.CAPTURED,
    "failed": PaymentStatus.FAILED,
    "refunded": PaymentStatus.REFUNDED,
}


def map_subscription_status(rz_status: str) -> SubscriptionStatus:
    return _SUBSCRIPTION_STATUS_MAP.get(rz_status.lower(), SubscriptionStatus.CREATED)


def map_payment_status(rz_status: str) -> PaymentStatus:
    return _PAYMENT_STATUS_MAP.get(rz_status.lower(), PaymentStatus.CREATED)


# ─────────────────────── Full entity mappers ─────────────────────────────────

def map_subscription_fields(rz_sub: dict[str, Any]) -> dict[str, Any]:
    """
    Map a Razorpay subscription entity dict to our internal field dict.
    Returns a dict suitable for creating/updating a Subscription model instance.
    """
    return {
        "razorpay_subscription_id": rz_sub.get("id"),
        "razorpay_plan_id": rz_sub.get("plan_id", ""),
        "status": map_subscription_status(rz_sub.get("status", "created")),
        "paid_count": rz_sub.get("paid_count", 0),
        "remaining_count": rz_sub.get("remaining_count", 0),
        "total_count": rz_sub.get("total_count", 0),
        "current_period_start": epoch_to_datetime(rz_sub.get("current_start")),
        "current_period_end": epoch_to_datetime(rz_sub.get("current_end")),
        "charge_at": epoch_to_datetime(rz_sub.get("charge_at")),
        "ended_at": epoch_to_datetime(rz_sub.get("ended_at")),
    }


def map_payment_fields(rz_pay: dict[str, Any]) -> dict[str, Any]:
    """
    Map a Razorpay payment entity dict to our internal field dict.
    Returns a dict suitable for creating/updating a PaymentAttempt model instance.
    """
    return {
        "razorpay_payment_id": rz_pay.get("id"),
        "amount": paise_to_rupees(rz_pay.get("amount")),
        "currency": rz_pay.get("currency", "INR"),
        "status": map_payment_status(rz_pay.get("status", "created")),
        "payment_method": rz_pay.get("method", ""),
        "error_code": rz_pay.get("error_code", "") or "",
        "error_description": rz_pay.get("error_description", "") or "",
        "error_source": rz_pay.get("error_source", "") or "",
        "failure_reason": classify_failure_reason(
            rz_pay.get("error_reason"),
            rz_pay.get("error_code"),
            rz_pay.get("error_description"),
        ),
    }


def extract_customer_id_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract Razorpay customer_id from various webhook payload shapes."""
    # Try subscription entity
    sub = payload.get("subscription", {}).get("entity", {})
    if sub.get("customer_id"):
        return sub["customer_id"]

    # Try payment entity
    pay = payload.get("payment", {}).get("entity", {})
    if pay.get("customer_id"):
        return pay["customer_id"]

    # Try payment_link entity
    plink = payload.get("payment_link", {}).get("entity", {})
    customer = plink.get("customer", {})
    if isinstance(customer, dict) and customer.get("id"):
        return customer["id"]

    return None


def extract_subscription_id_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract Razorpay subscription_id from various webhook payload shapes."""
    sub = payload.get("subscription", {}).get("entity", {})
    if sub.get("id"):
        return sub["id"]

    # Check payment notes for subscription_id
    pay = payload.get("payment", {}).get("entity", {})
    notes = pay.get("notes", {})
    if isinstance(notes, dict) and notes.get("subscription_id"):
        return notes["subscription_id"]

    return None
