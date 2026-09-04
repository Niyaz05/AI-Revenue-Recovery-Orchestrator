from __future__ import annotations
"""
Baseline 2: Fixed Retry Schedule.

Policy:
  Retries up to 3 times on a fixed schedule (attempt 1, 2, 3 -> RETRY, then STOP_RECOVERY).

⚠️  NOTE: This represents a generic naive merchant cron retry loop.
    It is explicitly NOT a reproduction of Razorpay's internal smart retry system.
"""

from simulator.models import SimActionType, SimState


class FixedRetryBaseline:
    name = "Fixed Retry Schedule"

    def predict(self, state: SimState) -> SimActionType:
        if state.attempt_count <= 3:
            return SimActionType.RETRY
        return SimActionType.STOP_RECOVERY
