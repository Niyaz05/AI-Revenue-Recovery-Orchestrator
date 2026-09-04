from __future__ import annotations
"""
Razorpay Subscriptions service wrapper for lifecycle management in Test Mode.
"""

import logging
from typing import Any, Optional
from razorpay.client import RazorpayClient, RazorpayClientError

logger = logging.getLogger(__name__)


class SubscriptionService:
    def __init__(self, client: Optional[RazorpayClient] = None):
        self.client = client or RazorpayClient()

    def create_monthly_plan(self, name: str, amount_in_rupees: float, description: str = "") -> dict:
        """Create a monthly subscription plan. Amount converted to paise."""
        amount_paise = int(amount_in_rupees * 100)
        return self.client.create_plan(
            period="monthly",
            interval=1,
            name=name,
            amount=amount_paise,
            currency="INR",
            description=description or f"Plan: {name}",
        )

    def create_subscription(
        self,
        plan_id: str,
        total_count: int = 12,
        customer_notify: int = 1,
        notes: Optional[dict] = None,
    ) -> dict:
        """Create a recurring subscription."""
        return self.client.create_subscription(
            plan_id=plan_id,
            total_count=total_count,
            quantity=1,
            customer_notify=customer_notify,
            notes=notes or {"system": "ai-revenue-recovery", "mode": "test"},
        )

    def get_subscription(self, subscription_id: str) -> dict:
        return self.client.fetch_subscription(subscription_id)

    def cancel_subscription(self, subscription_id: str, at_cycle_end: bool = False) -> dict:
        return self.client.cancel_subscription(subscription_id, cancel_at_cycle_end=at_cycle_end)

    def pause_subscription(self, subscription_id: str) -> dict:
        return self.client.pause_subscription(subscription_id)

    def resume_subscription(self, subscription_id: str) -> dict:
        return self.client.resume_subscription(subscription_id)
