from __future__ import annotations
"""
Baseline 3: Rule-Based Expert Heuristic Policy.

Rules:
- Temporary bank failure -> WAIT 4h then RETRY (max 2 retries)
- Insufficient funds -> PAYMENT_LINK
- Repeated failure (>2 attempts) -> PAYMENT_LINK
- Invalid payment method / Card expired -> REQUEST_ALTERNATE_METHOD
- High-value customer (> ₹50,000) -> ESCALATE_TO_HUMAN
- Opted-out / Expired window -> STOP_RECOVERY
"""

from simulator.models import SimActionType, SimFailureType, SimState


class RuleBasedBaseline:
    name = "Rule-Based Expert Heuristic"

    def predict(self, state: SimState) -> SimActionType:
        if state.customer.opted_out or state.hours_since_first_failure >= 168.0:
            return SimActionType.STOP_RECOVERY

        if state.amount > 50000.0:
            return SimActionType.ESCALATE_TO_HUMAN

        ft = state.failure_type
        if ft in (SimFailureType.INVALID_PAYMENT_METHOD, SimFailureType.CARD_EXPIRED):
            return SimActionType.REQUEST_ALTERNATE_METHOD

        if ft == SimFailureType.INSUFFICIENT_FUNDS:
            if state.interventions_count == 0:
                return SimActionType.PAYMENT_LINK
            return SimActionType.SEND_REMINDER

        if ft in (SimFailureType.TEMPORARY_BANK_FAILURE, SimFailureType.NETWORK_ERROR):
            if state.attempt_count <= 2:
                if state.hours_since_first_failure < 4.0:
                    return SimActionType.WAIT
                return SimActionType.RETRY
            return SimActionType.PAYMENT_LINK

        if state.attempt_count >= 3:
            return SimActionType.PAYMENT_LINK

        return SimActionType.WAIT
