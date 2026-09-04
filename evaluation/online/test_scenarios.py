from __future__ import annotations
"""
Online Evaluation Test Scenarios for Razorpay Test Mode.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class OnlineScenario:
    id: str
    name: str
    customer_name: str
    customer_email: str
    customer_phone: str
    amount: float
    failure_type: str
    expected_action: str


ONLINE_TEST_SCENARIOS = [
    OnlineScenario(
        id="online_01",
        name="Temporary Bank Gateway Failure",
        customer_name="Aarav Sharma",
        customer_email="aarav.sharma@test.com",
        customer_phone="+919811001100",
        amount=4999.0,
        failure_type="temporary_bank_failure",
        expected_action="WAIT",
    ),
    OnlineScenario(
        id="online_02",
        name="Insufficient Funds - Payment Link Recovery",
        customer_name="Neha Gupta",
        customer_email="neha.gupta@test.com",
        customer_phone="+919822002200",
        amount=2999.0,
        failure_type="insufficient_funds",
        expected_action="PAYMENT_LINK",
    ),
    OnlineScenario(
        id="online_03",
        name="Expired Card - Alternate Method Request",
        customer_name="Karan Verma",
        customer_email="karan.verma@test.com",
        customer_phone="+919833003300",
        amount=1499.0,
        failure_type="invalid_payment_method",
        expected_action="REQUEST_ALTERNATE_METHOD",
    ),
    OnlineScenario(
        id="online_04",
        name="Enterprise High Value Subscription Escalation",
        customer_name="Nexus Technologies",
        customer_email="billing@nexustech.in",
        customer_phone="+919844004400",
        amount=85000.0,
        failure_type="authentication_failed",
        expected_action="ESCALATE_TO_HUMAN",
    ),
]
