from __future__ import annotations
"""
State Machine for Subscription and Recovery Lifecycle.
Validates state transitions and detects when a subscription/payment enters an at-risk state.
"""

import logging
from typing import Dict, Set
from backend.models.enums import SubscriptionStatus, ActionStatus, OutcomeType

logger = logging.getLogger(__name__)

# Valid transitions for Subscription
SUBSCRIPTION_TRANSITIONS: Dict[SubscriptionStatus, Set[SubscriptionStatus]] = {
    SubscriptionStatus.CREATED: {SubscriptionStatus.AUTHENTICATED, SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.AUTHENTICATED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.PENDING, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.ACTIVE: {SubscriptionStatus.PENDING, SubscriptionStatus.PAUSED, SubscriptionStatus.CANCELLED, SubscriptionStatus.COMPLETED},
    SubscriptionStatus.PENDING: {SubscriptionStatus.ACTIVE, SubscriptionStatus.HALTED, SubscriptionStatus.CANCELLED, SubscriptionStatus.PENDING},
    SubscriptionStatus.HALTED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED, SubscriptionStatus.EXPIRED},
    SubscriptionStatus.PAUSED: {SubscriptionStatus.ACTIVE, SubscriptionStatus.CANCELLED},
    SubscriptionStatus.CANCELLED: set(),
    SubscriptionStatus.COMPLETED: set(),
    SubscriptionStatus.EXPIRED: set(),
}

# Valid transitions for RecoveryAction
ACTION_TRANSITIONS: Dict[ActionStatus, Set[ActionStatus]] = {
    ActionStatus.PENDING: {ActionStatus.EXECUTING, ActionStatus.BLOCKED, ActionStatus.CANCELLED},
    ActionStatus.EXECUTING: {ActionStatus.COMPLETED, ActionStatus.FAILED, ActionStatus.CANCELLED},
    ActionStatus.BLOCKED: set(),
    ActionStatus.COMPLETED: set(),
    ActionStatus.FAILED: {ActionStatus.PENDING}, # can retry with new action
    ActionStatus.CANCELLED: set(),
}


def is_valid_subscription_transition(current: SubscriptionStatus, next_state: SubscriptionStatus) -> bool:
    """Check if subscription state transition is mathematically valid in lifecycle."""
    if current == next_state:
        return True
    allowed = SUBSCRIPTION_TRANSITIONS.get(current, set())
    return next_state in allowed


def is_at_risk_state(status: SubscriptionStatus) -> bool:
    """True if subscription requires revenue recovery orchestration."""
    return status in (SubscriptionStatus.PENDING, SubscriptionStatus.HALTED)
