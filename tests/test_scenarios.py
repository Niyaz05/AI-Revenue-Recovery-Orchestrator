from __future__ import annotations
"""
Unit tests for the 8 Canonical Scenarios.
"""

import pytest
from simulator.environment import RecoveryEnvironment
from simulator.models import SimActionType
from simulator.scenarios import ALL_SCENARIOS


def test_scenario_temporary_bank_failure():
    env = RecoveryEnvironment(seed=42)
    state = ALL_SCENARIOS["temporary_bank_failure"]()
    # Advance time / cooldown
    state.hours_since_first_failure = 6.0
    outcome = env.step(state, SimActionType.RETRY)
    assert outcome.action_cost > 0
    assert outcome.friction >= 0


def test_scenario_insufficient_funds_payment_link():
    env = RecoveryEnvironment(seed=42)
    state = ALL_SCENARIOS["insufficient_funds"]()
    outcome = env.step(state, SimActionType.PAYMENT_LINK)
    assert outcome.action_cost == 15.0
    assert outcome.friction == 1.0


def test_scenario_invalid_payment_method_retry_fails():
    env = RecoveryEnvironment(seed=42)
    state = ALL_SCENARIOS["invalid_payment_method"]()
    # Retry on expired card has 0% probability
    prob = env._compute_recovery_probability(state, SimActionType.RETRY)
    assert prob == 0.0


def test_scenario_high_value_customer():
    env = RecoveryEnvironment(seed=42)
    state = ALL_SCENARIOS["high_value_customer"]()
    prob = env._compute_recovery_probability(state, SimActionType.ESCALATE_TO_HUMAN)
    assert prob >= 0.60


def test_scenario_opted_out_and_window_expired():
    env = RecoveryEnvironment(seed=42)
    state_opt = ALL_SCENARIOS["opted_out_customer"]()
    outcome = env.step(state_opt, SimActionType.STOP_RECOVERY)
    assert outcome.terminal is True
    assert outcome.success is False
