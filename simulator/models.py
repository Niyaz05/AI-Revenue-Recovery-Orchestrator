from __future__ import annotations
"""
Simulation Data Models and Dataclasses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class SimFailureType(str, Enum):
    TEMPORARY_BANK_FAILURE = "temporary_bank_failure"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    INVALID_PAYMENT_METHOD = "invalid_payment_method"
    NETWORK_ERROR = "network_error"
    FRAUD_SUSPECTED = "fraud_suspected"
    BANK_DECLINED = "bank_declined"
    CARD_EXPIRED = "card_expired"
    AUTHENTICATION_FAILED = "authentication_failed"


class SimActionType(str, Enum):
    WAIT = "WAIT"
    RETRY = "RETRY"
    PAYMENT_LINK = "PAYMENT_LINK"
    SEND_REMINDER = "SEND_REMINDER"
    REQUEST_ALTERNATE_METHOD = "REQUEST_ALTERNATE_METHOD"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    STOP_RECOVERY = "STOP_RECOVERY"


# Ordered action index — shared by RL agents, OPE, and episodic data generation.
ACTION_INDEX: Dict[SimActionType, int] = {a: i for i, a in enumerate(SimActionType)}
INDEX_ACTION: Dict[int, SimActionType] = {i: a for a, i in ACTION_INDEX.items()}


@dataclass
class SimCustomer:
    id: int
    name: str
    email: str
    phone: str
    opted_out: bool = False
    archetype: str = "reliable"  # reliable / occasional / high_risk / reliable_dormant
    ltv: float = 50000.0
    failure_rate: float = 0.05
    total_payments: int = 10
    failed_payments: int = 1

    # ── Long-horizon history features (independent of the current episode) ──
    # These describe the customer's *entire* multi-year relationship, not just the
    # current failure. They are what let the agent distinguish a reliable-but-dormant
    # customer from a chronic failer who looks identical on surface features.
    account_tenure_days: int = 365      # days since customer signup
    successful_payments: int = 9        # total successful charges over lifetime
    days_since_last_success: int = 30   # recency of last successful payment
    historical_ltv: float = 50000.0     # lifetime value accumulated to date
    is_first_ever_failure: bool = False # clean multi-year record hitting first failure

    @property
    def lifetime_success_rate(self) -> float:
        """successful / total charges across the entire history."""
        if self.total_payments <= 0:
            return 0.0
        return float(self.successful_payments) / float(self.total_payments)


@dataclass
class SimSubscription:
    id: str
    customer_id: int
    amount_per_period: float
    paid_count: int = 3
    remaining_count: int = 9
    total_count: int = 12
    status: str = "pending"


@dataclass
class SimState:
    customer: SimCustomer
    subscription: SimSubscription
    failure_type: SimFailureType
    amount: float
    attempt_count: int = 1
    hours_since_first_failure: float = 0.0
    interventions_count: int = 0
    consecutive_failures: int = 1
    is_weekend: bool = False


@dataclass
class SimOutcome:
    success: bool
    recovered_amount: float
    recovery_time_hours: float
    action_cost: float
    friction: float
    next_attempt_count: int
    next_state: Optional[SimState]
    reward: float = 0.0
    terminal: bool = False
