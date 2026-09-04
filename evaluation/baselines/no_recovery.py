from __future__ import annotations
"""
Baseline 1: No Recovery.

Policy: Every failed payment is immediately terminated (STOP_RECOVERY).
Zero intervention cost, zero recovered revenue. Serves as lower-bound baseline.
"""

from backend.models.enums import ActionType
from simulator.models import SimActionType, SimState


class NoRecoveryBaseline:
    name = "No Recovery (Lower Bound)"

    def predict(self, state: SimState) -> SimActionType:
        return SimActionType.STOP_RECOVERY
