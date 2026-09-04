from __future__ import annotations
"""
Razorpay API Client Wrapper (Test Mode).

Provides a typed, error-handled interface to Razorpay REST APIs.
All methods include retry logic, logging, and never expose secrets.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RazorpayClientError(Exception):
    """Raised when a Razorpay API call fails."""

    def __init__(self, status_code: int, error_code: str, description: str, raw: dict):
        self.status_code = status_code
        self.error_code = error_code
        self.description = description
        self.raw = raw
        super().__init__(f"Razorpay API error {status_code}: {error_code} — {description}")


class RazorpayClient:
    """
    Thin wrapper around Razorpay REST API v1.

    Usage:
        client = RazorpayClient()              # reads from env
        client = RazorpayClient(key_id, key_secret)  # explicit
    """

    BASE_URL = "https://api.razorpay.com/v1"

    def __init__(
        self,
        key_id: Optional[str] = None,
        key_secret: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "")
        self.key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "")
        self.timeout = timeout

        if not self.key_id or not self.key_secret:
            logger.warning("Razorpay credentials not configured — API calls will fail.")

        # Session with automatic retries on transient errors
        self._session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.auth = (self.key_id, self.key_secret)

    # ──────────────────────────────── HTTP helpers ────────────────────────────

    def _request(
        self, method: str, path: str, data: Optional[dict] = None, params: Optional[dict] = None
    ) -> dict:
        url = f"{self.BASE_URL}{path}"
        logger.info("Razorpay %s %s", method, path)
        start = time.monotonic()

        try:
            resp = self._session.request(
                method, url, json=data, params=params, timeout=self.timeout
            )
        except requests.RequestException as exc:
            logger.error("Razorpay request failed: %s", exc)
            raise RazorpayClientError(0, "REQUEST_FAILED", str(exc), {}) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("Razorpay %s %s → %d (%.0fms)", method, path, resp.status_code, elapsed_ms)

        if resp.status_code >= 400:
            body = resp.json() if resp.content else {}
            err = body.get("error", {})
            raise RazorpayClientError(
                status_code=resp.status_code,
                error_code=err.get("code", "UNKNOWN"),
                description=err.get("description", resp.text),
                raw=body,
            )

        return resp.json() if resp.content else {}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, data: Optional[dict] = None) -> dict:
        return self._request("POST", path, data=data)

    def _put(self, path: str, data: Optional[dict] = None) -> dict:
        return self._request("PUT", path, data=data)

    def _patch(self, path: str, data: Optional[dict] = None) -> dict:
        return self._request("PATCH", path, data=data)

    def _delete(self, path: str) -> dict:
        return self._request("DELETE", path)

    # ──────────────────────────────── Plans ───────────────────────────────────

    def create_plan(
        self,
        period: str,
        interval: int,
        name: str,
        amount: int,
        currency: str = "INR",
        description: str = "",
    ) -> dict:
        """Create a subscription plan. Amount in paise."""
        return self._post(
            "/plans",
            {
                "period": period,
                "interval": interval,
                "item": {
                    "name": name,
                    "amount": amount,
                    "currency": currency,
                    "description": description,
                },
            },
        )

    def fetch_plan(self, plan_id: str) -> dict:
        return self._get(f"/plans/{plan_id}")

    def fetch_all_plans(self, count: int = 10, skip: int = 0) -> dict:
        return self._get("/plans", params={"count": count, "skip": skip})

    # ─────────────────────────────── Subscriptions ───────────────────────────

    def create_subscription(
        self,
        plan_id: str,
        total_count: int,
        quantity: int = 1,
        customer_notify: int = 1,
        notes: Optional[dict] = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "plan_id": plan_id,
            "total_count": total_count,
            "quantity": quantity,
            "customer_notify": customer_notify,
        }
        if notes:
            payload["notes"] = notes
        return self._post("/subscriptions", payload)

    def fetch_subscription(self, subscription_id: str) -> dict:
        return self._get(f"/subscriptions/{subscription_id}")

    def fetch_all_subscriptions(self, count: int = 10, skip: int = 0) -> dict:
        return self._get("/subscriptions", params={"count": count, "skip": skip})

    def cancel_subscription(self, subscription_id: str, cancel_at_cycle_end: bool = False) -> dict:
        return self._post(
            f"/subscriptions/{subscription_id}/cancel",
            {"cancel_at_cycle_end": 1 if cancel_at_cycle_end else 0},
        )

    def pause_subscription(self, subscription_id: str) -> dict:
        return self._post(f"/subscriptions/{subscription_id}/pause", {"pause_initiated_by": "customer"})

    def resume_subscription(self, subscription_id: str) -> dict:
        return self._post(f"/subscriptions/{subscription_id}/resume", {"resume_at": "now"})

    # ──────────────────────────── Payment Links ──────────────────────────────

    def create_payment_link(
        self,
        amount: int,
        currency: str = "INR",
        description: str = "",
        customer: Optional[dict] = None,
        notify: Optional[dict] = None,
        reminder_enable: bool = True,
        callback_url: Optional[str] = None,
        callback_method: str = "get",
        expire_by: Optional[int] = None,
        notes: Optional[dict] = None,
    ) -> dict:
        """Create a payment link. Amount in paise."""
        payload: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "description": description,
            "reminder_enable": reminder_enable,
            "callback_method": callback_method,
        }
        if customer:
            payload["customer"] = customer
        if notify:
            payload["notify"] = notify
        if callback_url:
            payload["callback_url"] = callback_url
        if expire_by:
            payload["expire_by"] = expire_by
        if notes:
            payload["notes"] = notes
        return self._post("/payment_links", payload)

    def fetch_payment_link(self, link_id: str) -> dict:
        return self._get(f"/payment_links/{link_id}")

    def fetch_all_payment_links(self, count: int = 10, skip: int = 0) -> dict:
        return self._get("/payment_links", params={"count": count, "skip": skip})

    def update_payment_link(self, link_id: str, updates: dict) -> dict:
        return self._patch(f"/payment_links/{link_id}", updates)

    def cancel_payment_link(self, link_id: str) -> dict:
        return self._post(f"/payment_links/{link_id}/cancel")

    def resend_payment_link_notification(self, link_id: str, medium: str = "email") -> dict:
        """Resend notification for a payment link. medium: 'email' or 'sms'."""
        return self._post(f"/payment_links/{link_id}/notify_by/{medium}")

    # ──────────────────────────── Payments ────────────────────────────────────

    def fetch_payment(self, payment_id: str) -> dict:
        return self._get(f"/payments/{payment_id}")

    def fetch_all_payments(self, count: int = 10, skip: int = 0) -> dict:
        return self._get("/payments", params={"count": count, "skip": skip})

    # ──────────────────────────── Customers ───────────────────────────────────

    def create_customer(self, name: str, email: str, contact: str, notes: Optional[dict] = None) -> dict:
        payload: dict[str, Any] = {"name": name, "email": email, "contact": contact}
        if notes:
            payload["notes"] = notes
        return self._post("/customers", payload)

    def fetch_customer(self, customer_id: str) -> dict:
        return self._get(f"/customers/{customer_id}")
