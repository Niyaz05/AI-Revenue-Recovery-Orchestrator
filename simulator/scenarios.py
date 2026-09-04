from __future__ import annotations
"""
The 8 Canonical Test Scenarios required by the specification.
Used for offline policy verification, sandbox unit testing, and demonstration.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np

from simulator.models import (
    SimActionType,
    SimCustomer,
    SimFailureType,
    SimState,
    SimSubscription,
)


def scenario_temporary_bank_failure(seed: int = 101) -> SimState:
    """Scenario 1: Temporary bank downtime/gateway glitch."""
    return SimState(
        customer=SimCustomer(id=101, name="Acme Corp", email="billing@acme.com", phone="+919811111111", ltv=80000.0),
        subscription=SimSubscription(id="sub_scen_1", customer_id=101, amount_per_period=4999.0),
        failure_type=SimFailureType.TEMPORARY_BANK_FAILURE,
        amount=4999.0,
        attempt_count=1,
        hours_since_first_failure=0.0,
    )


def scenario_insufficient_funds(seed: int = 102) -> SimState:
    """Scenario 2: Insufficient funds on salary account before payday."""
    return SimState(
        customer=SimCustomer(id=102, name="Rahul Sharma", email="rahul@example.com", phone="+919822222222", ltv=30000.0),
        subscription=SimSubscription(id="sub_scen_2", customer_id=102, amount_per_period=2499.0),
        failure_type=SimFailureType.INSUFFICIENT_FUNDS,
        amount=2499.0,
        attempt_count=1,
        hours_since_first_failure=12.0,
    )


def scenario_repeated_failure(seed: int = 103) -> SimState:
    """Scenario 3: Repeated failures (3+ attempts) where retries are burned out."""
    return SimState(
        customer=SimCustomer(id=103, name="Design Studio", email="admin@studio.io", phone="+919833333333", ltv=65000.0),
        subscription=SimSubscription(id="sub_scen_3", customer_id=103, amount_per_period=9999.0),
        failure_type=SimFailureType.INSUFFICIENT_FUNDS,
        amount=9999.0,
        attempt_count=3,
        hours_since_first_failure=48.0,
        interventions_count=2,
    )


def scenario_invalid_payment_method(seed: int = 104) -> SimState:
    """Scenario 4: Expired credit card / invalid payment method."""
    return SimState(
        customer=SimCustomer(id=104, name="Priya Patel", email="priya@domain.com", phone="+919844444444", ltv=45000.0),
        subscription=SimSubscription(id="sub_scen_4", customer_id=104, amount_per_period=1999.0),
        failure_type=SimFailureType.CARD_EXPIRED,
        amount=1999.0,
        attempt_count=1,
        hours_since_first_failure=2.0,
    )


def scenario_high_value_customer(seed: int = 105) -> SimState:
    """Scenario 5: High-value enterprise customer (>₹50,000) justifying white-glove human escalation."""
    return SimState(
        customer=SimCustomer(id=105, name="Enterprise Global Ltd", email="finance@enterprise.com", phone="+919855555555", ltv=500000.0),
        subscription=SimSubscription(id="sub_scen_5", customer_id=105, amount_per_period=75000.0),
        failure_type=SimFailureType.AUTHENTICATION_FAILED,
        amount=75000.0,
        attempt_count=1,
        hours_since_first_failure=4.0,
    )


def scenario_opted_out_customer(seed: int = 106) -> SimState:
    """Scenario 6: Customer who opted out from recovery interventions."""
    return SimState(
        customer=SimCustomer(id=106, name="Opted Out User", email="optout@test.com", phone="+919866666666", opted_out=True, ltv=15000.0),
        subscription=SimSubscription(id="sub_scen_6", customer_id=106, amount_per_period=999.0),
        failure_type=SimFailureType.TEMPORARY_BANK_FAILURE,
        amount=999.0,
        attempt_count=1,
        hours_since_first_failure=0.0,
    )


def scenario_retry_budget_exhausted(seed: int = 107) -> SimState:
    """Scenario 7: Merchant global daily retry budget exhausted."""
    return SimState(
        customer=SimCustomer(id=107, name="Standard User", email="user7@example.com", phone="+919877777777", ltv=20000.0),
        subscription=SimSubscription(id="sub_scen_7", customer_id=107, amount_per_period=1499.0),
        failure_type=SimFailureType.TEMPORARY_BANK_FAILURE,
        amount=1499.0,
        attempt_count=1,
        hours_since_first_failure=1.0,
    )


def scenario_reliable_dormant_customer_single_failure(seed: int = 109) -> SimState:
    """Scenario 9: Reliable-but-dormant customer hitting their first-ever failure.

    Long tenure (~4 years), near-perfect lifetime success rate, last success a
    month ago (dormant), and this is their first failure ever. The enriched
    history features should make the agent pick a distinctly gentler sequence
    (e.g. WAIT / a single soft reminder) than for a chronic failer with the same
    surface features (amount, failure reason).
    """
    cust = SimCustomer(
        id=109,
        name="Long-Term Loyal Customer",
        email="loyal@example.com",
        phone="+919899999999",
        ltv=95000.0,
        archetype="reliable_dormant",
        total_payments=48,
        successful_payments=48,          # clean record (this is failure #1)
        failed_payments=0,
        account_tenure_days=1460,         # ~4 years
        days_since_last_success=32,       # dormant ~1 month
        historical_ltv=95000.0,
        is_first_ever_failure=True,
    )
    cust.failed_payments = 0
    return SimState(
        customer=cust,
        subscription=SimSubscription(id="sub_scen_9", customer_id=109, amount_per_period=2999.0),
        failure_type=SimFailureType.TEMPORARY_BANK_FAILURE,
        amount=2999.0,
        attempt_count=1,
        hours_since_first_failure=0.0,
    )


def scenario_recovery_window_expired(seed: int = 108) -> SimState:
    """Scenario 8: Recovery window exceeded (>168 hours / 7 days)."""
    return SimState(
        customer=SimCustomer(id=108, name="Dormant Account", email="dormant@example.com", phone="+919888888888", ltv=10000.0),
        subscription=SimSubscription(id="sub_scen_8", customer_id=108, amount_per_period=999.0),
        failure_type=SimFailureType.INSUFFICIENT_FUNDS,
        amount=999.0,
        attempt_count=2,
        hours_since_first_failure=175.0,  # > 168h limit
    )


ALL_SCENARIOS = {
    "temporary_bank_failure": scenario_temporary_bank_failure,
    "insufficient_funds": scenario_insufficient_funds,
    "repeated_failure": scenario_repeated_failure,
    "invalid_payment_method": scenario_invalid_payment_method,
    "high_value_customer": scenario_high_value_customer,
    "opted_out_customer": scenario_opted_out_customer,
    "retry_budget_exhausted": scenario_retry_budget_exhausted,
    "recovery_window_expired": scenario_recovery_window_expired,
    "reliable_dormant_customer_single_failure": scenario_reliable_dormant_customer_single_failure,
}
