from __future__ import annotations
"""
Level 2: Strategy Policy Layer.

Coordinates strategy selection between the learned Contextual Bandit (LinUCB)
and deterministic rule-based fallback when model is unavailable or in fallback mode.
Enforces that only Level-1-allowed actions can ever be chosen.
"""

import logging
import os
from typing import Dict, Optional, Set, Tuple
import numpy as np

from backend.models.enums import ActionType, FailureReason
from backend.policies.safety_policy import PolicyDecision
from agent.bandit.features import extract_feature_vector
from agent.bandit.model import BanditDecision, LinUCBBandit

logger = logging.getLogger(__name__)


class StrategyPolicyEngine:
    def __init__(self, bandit_model_path: Optional[str] = None):
        self.bandit: Optional[LinUCBBandit] = None
        if bandit_model_path and os.path.exists(bandit_model_path):
            try:
                self.bandit = LinUCBBandit.load(bandit_model_path)
            except Exception as e:
                logger.warning(f"Failed to load bandit model from {bandit_model_path}: {e}")
        
        if self.bandit is None:
            # Initialize fresh default bandit
            self.bandit = LinUCBBandit()

    def select_action(
        self,
        failure_reason: str,
        amount: float,
        attempt_count: int,
        hours_since_first_failure: float,
        customer_ltv: float,
        customer_failure_rate: float,
        subscription_paid_count: int,
        safety_decision: PolicyDecision,
        is_weekend: bool = False,
        force_rule_based: bool = False,
    ) -> Tuple[ActionType, dict]:
        """
        Selects strategy strictly constrained by safety_decision.allowed_actions.
        Returns (chosen_action, metadata_dict).
        """
        allowed = safety_decision.allowed_actions

        # If safety says must stop or only one action allowed
        if len(allowed) == 1:
            chosen = next(iter(allowed))
            return chosen, {
                "strategy_mode": "safety_forced",
                "score": 0.0,
                "confidence": 1.0,
                "all_scores": {},
                "reasons": safety_decision.reasons,
            }

        # Rule-based fallback logic
        if force_rule_based or self.bandit is None:
            return self._rule_based_select(
                failure_reason=failure_reason,
                amount=amount,
                attempt_count=attempt_count,
                allowed=allowed,
            )

        # Bandit selection
        x = extract_feature_vector(
            failure_reason=failure_reason,
            amount=amount,
            attempt_count=attempt_count,
            hours_since_first_failure=hours_since_first_failure,
            customer_ltv=customer_ltv,
            customer_failure_rate=customer_failure_rate,
            subscription_paid_count=subscription_paid_count,
            is_weekend=is_weekend,
        )

        decision: BanditDecision = self.bandit.predict(x, allowed_actions=allowed)

        meta = {
            "strategy_mode": "bandit",
            "score": decision.score,
            "confidence": decision.confidence,
            "all_scores": decision.all_scores,
            "features": decision.features,
            "model_version": decision.model_version,
            "reasons": [
                f"LinUCB selected {decision.selected_action.value} (payoff score: {decision.score:.3f}, confidence: {decision.confidence:.1%})"
            ] + safety_decision.reasons,
        }

        return decision.selected_action, meta

    def _rule_based_select(
        self,
        failure_reason: str,
        amount: float,
        attempt_count: int,
        allowed: Set[ActionType],
    ) -> Tuple[ActionType, dict]:
        """Rule-based heuristic baseline / fallback."""
        reason_clean = failure_reason.lower() if failure_reason else ""

        if "temp" in reason_clean or "bank" in reason_clean:
            cand = ActionType.RETRY if ActionType.RETRY in allowed else ActionType.WAIT
        elif "insufficient" in reason_clean:
            cand = ActionType.PAYMENT_LINK if ActionType.PAYMENT_LINK in allowed else ActionType.WAIT
        elif "invalid" in reason_clean or "card_expired" in reason_clean:
            cand = ActionType.REQUEST_ALTERNATE_METHOD if ActionType.REQUEST_ALTERNATE_METHOD in allowed else ActionType.PAYMENT_LINK
        elif attempt_count >= 2:
            cand = ActionType.PAYMENT_LINK if ActionType.PAYMENT_LINK in allowed else ActionType.ESCALATE_TO_HUMAN
        elif amount > 50000.0:
            cand = ActionType.ESCALATE_TO_HUMAN if ActionType.ESCALATE_TO_HUMAN in allowed else ActionType.PAYMENT_LINK
        else:
            cand = ActionType.WAIT

        if cand not in allowed:
            cand = next(iter(allowed)) if allowed else ActionType.STOP_RECOVERY

        return cand, {
            "strategy_mode": "rule_based_fallback",
            "score": 0.5,
            "confidence": 0.8,
            "all_scores": {},
            "reasons": [f"Rule-based fallback mapped {failure_reason} -> {cand.value}"],
        }
