from __future__ import annotations
"""
Level 1: Deterministic Hard Safety Policy Layer.

This layer MUST gate every financial and customer-facing intervention.
The AI / ML model or LLM can NEVER bypass this layer under any circumstances.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
import logging

from backend.models.enums import ActionType
from backend.models.merchant_policy import MerchantPolicy

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    allowed_actions: Set[ActionType]
    blocked_actions: Dict[ActionType, str] = field(default_factory=dict)
    must_escalate: bool = False
    must_stop: bool = False
    reasons: List[str] = field(default_factory=list)
    policy_version: str = "v1"

    def to_dict(self) -> dict:
        return {
            "allowed_actions": [a.value for a in self.allowed_actions],
            "blocked_actions": {k.value: v for k, v in self.blocked_actions.items()},
            "must_escalate": self.must_escalate,
            "must_stop": self.must_stop,
            "reasons": self.reasons,
            "policy_version": self.policy_version,
        }


@dataclass
class SafetyContext:
    customer_id: int
    amount: float
    attempt_count: int
    interventions_today: int
    global_retries_today: int
    first_failure_time: datetime
    last_attempt_time: Optional[datetime]
    is_customer_opted_out: bool
    current_time: Optional[datetime] = None


class SafetyPolicyEngine:
    """
    Deterministic rules engine implementing hard safety constraints.
    Outputs the exact subset of ActionTypes that are safe and legally permitted.
    """

    def __init__(self, merchant_policy: Optional[MerchantPolicy] = None):
        self.policy = merchant_policy or MerchantPolicy()

    def evaluate(self, ctx: SafetyContext) -> PolicyDecision:
        current_time = ctx.current_time or datetime.utcnow()
        all_actions = set(ActionType)
        allowed: Set[ActionType] = set(all_actions)
        blocked: Dict[ActionType, str] = {}
        reasons: List[str] = []
        must_stop = False
        must_escalate = False

        # Rule 1: Customer Opt-Out
        opted_out_ids = self.policy.get_opted_out_ids()
        if ctx.is_customer_opted_out or ctx.customer_id in opted_out_ids:
            reasons.append("Customer is opted out from automated recovery interventions.")
            return PolicyDecision(
                allowed_actions={ActionType.STOP_RECOVERY},
                blocked_actions={a: "Customer opted out" for a in all_actions if a != ActionType.STOP_RECOVERY},
                must_stop=True,
                reasons=reasons,
                policy_version=self.policy.version,
            )

        # Rule 2: Recovery Window Expiration
        recovery_window = timedelta(hours=self.policy.recovery_window_hours)
        if (current_time - ctx.first_failure_time) > recovery_window:
            reasons.append(f"Recovery window expired ({self.policy.recovery_window_hours}h limit exceeded).")
            return PolicyDecision(
                allowed_actions={ActionType.STOP_RECOVERY},
                blocked_actions={a: "Recovery window expired" for a in all_actions if a != ActionType.STOP_RECOVERY},
                must_stop=True,
                reasons=reasons,
                policy_version=self.policy.version,
            )

        # Rule 3: High-Value Human Approval Threshold
        if ctx.amount > self.policy.human_approval_threshold_amount:
            reasons.append(
                f"Amount ₹{ctx.amount:,.2f} exceeds human approval threshold ₹{self.policy.human_approval_threshold_amount:,.2f}."
            )
            # Automated customer contact / retries blocked without human sign-off
            must_escalate = True
            return PolicyDecision(
                allowed_actions={ActionType.ESCALATE_TO_HUMAN, ActionType.WAIT, ActionType.STOP_RECOVERY},
                blocked_actions={
                    ActionType.RETRY: "Amount exceeds auto threshold, requires human approval",
                    ActionType.PAYMENT_LINK: "Amount exceeds auto threshold, requires human approval",
                    ActionType.SEND_REMINDER: "Amount exceeds auto threshold, requires human approval",
                    ActionType.REQUEST_ALTERNATE_METHOD: "Amount exceeds auto threshold, requires human approval",
                },
                must_escalate=True,
                reasons=reasons,
                policy_version=self.policy.version,
            )

        # Rule 4: Max Auto-Recovery Amount
        if ctx.amount > self.policy.max_auto_recovery_amount:
            allowed.discard(ActionType.RETRY)
            blocked[ActionType.RETRY] = f"Amount ₹{ctx.amount:,.2f} exceeds max auto-recovery retry amount."
            reasons.append("Auto retry disabled for amount above standard recovery cap.")

        # Rule 5: Retry Count Limit
        if ctx.attempt_count >= self.policy.max_retry_attempts:
            allowed.discard(ActionType.RETRY)
            blocked[ActionType.RETRY] = f"Max retry attempts reached ({ctx.attempt_count}/{self.policy.max_retry_attempts})."
            reasons.append("Max retry attempts reached.")

        # Rule 6: Retry Cooldown
        if ctx.last_attempt_time is not None:
            cooldown = timedelta(hours=self.policy.retry_cooldown_hours)
            if (current_time - ctx.last_attempt_time) < cooldown:
                allowed.discard(ActionType.RETRY)
                blocked[ActionType.RETRY] = f"Retry cooldown in effect ({self.policy.retry_cooldown_hours}h required)."
                reasons.append("Retry cooldown active.")

        # Rule 7: Global Daily Retry Budget
        if ctx.global_retries_today >= self.policy.global_daily_retry_budget:
            allowed.discard(ActionType.RETRY)
            blocked[ActionType.RETRY] = f"Global merchant daily retry budget exhausted ({self.policy.global_daily_retry_budget})."
            reasons.append("Merchant daily retry budget exhausted.")

        # Rule 8: Daily Intervention Cap Per Customer
        if ctx.interventions_today >= self.policy.daily_intervention_cap_per_customer:
            for action in [ActionType.RETRY, ActionType.PAYMENT_LINK, ActionType.SEND_REMINDER, ActionType.REQUEST_ALTERNATE_METHOD]:
                if action in allowed:
                    allowed.discard(action)
                    blocked[action] = f"Daily customer intervention cap ({self.policy.daily_intervention_cap_per_customer}) reached."
            reasons.append("Daily customer intervention cap reached.")

        # Always ensure STOP_RECOVERY and WAIT remain valid if other actions get pruned
        allowed.add(ActionType.STOP_RECOVERY)
        allowed.add(ActionType.WAIT)

        return PolicyDecision(
            allowed_actions=allowed,
            blocked_actions=blocked,
            must_escalate=must_escalate,
            must_stop=must_stop,
            reasons=reasons,
            policy_version=self.policy.version,
        )
