from __future__ import annotations
"""
Razorpay Payment Links service wrapper for recovery execution in Test Mode.
"""

import logging
from typing import Any, Optional
from razorpay.client import RazorpayClient, RazorpayClientError

logger = logging.getLogger(__name__)


class PaymentLinkService:
    def __init__(self, client: Optional[RazorpayClient] = None):
        self.client = client or RazorpayClient()

    def create_recovery_link(
        self,
        amount_in_rupees: float,
        customer_name: str,
        customer_email: str,
        customer_phone: str,
        subscription_id: str,
        recovery_action_id: str,
        description: str = "",
        expire_in_hours: int = 72,
        notify_sms: bool = True,
        notify_email: bool = True,
    ) -> dict:
        """
        Create a real Razorpay Test Mode payment link specifically for revenue recovery.
        """
        import time

        amount_paise = int(amount_in_rupees * 100)
        expire_by = int(time.time()) + (expire_in_hours * 3600)

        notes = {
            "subscription_id": subscription_id,
            "recovery_action_id": str(recovery_action_id),
            "provenance": "RAZORPAY_TEST_MODE",
            "purpose": "revenue_recovery",
        }

        customer = {
            "name": customer_name or "Valued Customer",
            "email": customer_email or "billing@example.com",
            "contact": customer_phone or "+919876543210",
        }

        notify = {
            "sms": notify_sms,
            "email": notify_email,
        }

        desc = description or f"Payment Recovery for Subscription {subscription_id}"

        return self.client.create_payment_link(
            amount=amount_paise,
            currency="INR",
            description=desc,
            customer=customer,
            notify=notify,
            reminder_enable=True,
            expire_by=expire_by,
            notes=notes,
        )

    def get_payment_link(self, link_id: str) -> dict:
        return self.client.fetch_payment_link(link_id)

    def cancel_payment_link(self, link_id: str) -> dict:
        return self.client.cancel_payment_link(link_id)

    def resend_notification(self, link_id: str, medium: str = "email") -> dict:
        return self.client.resend_payment_link_notification(link_id, medium=medium)
