from __future__ import annotations
"""
Decision Explainer & Customer Communication Assistant.

Provides clear, grounded natural language explanations derived directly from
numerical features and deterministic safety constraints — without any external LLM dependency.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DecisionExplainer:
    """
    Deterministic rule-grounded explainer for revenue recovery decisions.
    Synthesizes clear explanations from feature signals and safety decisions.
    """

    def explain_recovery_decision(
        self,
        customer_name: str,
        amount: float,
        failure_reason: str,
        chosen_action: str,
        score: float,
        confidence: float,
        features: Dict[str, float],
        safety_reasons: list[str],
    ) -> str:
        """
        Generates a clear, grounded natural language explanation of why an action was chosen.
        """
        feature_highlights = []
        if features.get("is_temp_bank_failure"):
            feature_highlights.append("Temporary banking gateway issue detected")
        if features.get("is_insufficient_funds"):
            feature_highlights.append("Insufficient account balance indicated")
        if features.get("is_card_expired") or features.get("is_invalid_payment_method"):
            feature_highlights.append("Payment method expired or invalidated")
        if features.get("amount_log_norm", 0) > 0.8:
            feature_highlights.append("High monetary value transaction")

        reasons_text = "; ".join(safety_reasons) if safety_reasons else "All safety constraints verified."
        signals_text = ", ".join(feature_highlights) if feature_highlights else "Standard recurring charge profile"

        explanation = (
            f"The AI Orchestrator recommended **{chosen_action}** for customer '{customer_name}' "
            f"(Amount: ₹{amount:,.2f}) with {confidence:.1%} confidence and expected payoff score of {score:.2f}. "
            f"Key factors considered: {signals_text}. "
            f"Safety verification: {reasons_text}."
        )
        return explanation

    def draft_customer_message(
        self,
        customer_name: str,
        amount: float,
        subscription_name: str,
        payment_link_url: Optional[str] = None,
        action_type: str = "PAYMENT_LINK",
    ) -> Dict[str, str]:
        """
        Drafts respectful, personalized email and SMS copy for payment recovery.
        """
        first_name = customer_name.split()[0] if customer_name else "Valued Customer"
        link_str = payment_link_url or "https://rzp.io/i/example"

        if action_type == "REQUEST_ALTERNATE_METHOD":
            email_subject = f"Action Required: Update billing details for {subscription_name}"
            email_body = (
                f"Hi {first_name},\n\n"
                f"We noticed that your payment method on file for {subscription_name} (₹{amount:,.2f}) could not be charged. "
                f"To prevent any disruption to your service, please take a moment to update your payment method or complete the payment here:\n\n"
                f"{link_str}\n\n"
                f"Thank you for being with us!\nCustomer Billing Team"
            )
            sms_body = f"Hi {first_name}, update your payment method for {subscription_name} (₹{amount:,.2f}) to keep your service active: {link_str}"
        else:
            email_subject = f"Friendly Reminder: Renewal for {subscription_name}"
            email_body = (
                f"Hi {first_name},\n\n"
                f"Your recent renewal charge of ₹{amount:,.2f} for {subscription_name} was unsuccessful due to a momentary processing error. "
                f"You can quickly settle your renewal via UPI, Card, or NetBanking using this secure link:\n\n"
                f"{link_str}\n\n"
                f"Warm regards,\nCustomer Billing Team"
            )
            sms_body = f"Hi {first_name}, settle your renewal for {subscription_name} (₹{amount:,.2f}) easily via Razorpay: {link_str}"

        return {
            "email_subject": email_subject,
            "email_body": email_body,
            "sms_body": sms_body,
        }


# Alias for backward compatibility
LLMDecisionExplainer = DecisionExplainer
