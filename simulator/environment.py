from __future__ import annotations
"""
Simulation Environment for Revenue Recovery Orchestration.

Transition Function:
  (state, action) → (next_state, success, recovered_amount, recovery_time, action_cost, friction, next_attempt_count)

Features realistic, correlated probability distributions:
- Temporary failures recover best after cooldown (WAIT -> RETRY)
- Insufficient funds recover via Payment Links after 24-72h
- Repeated failures (>2) degrade retry success sharply, favoring Payment Links
- Invalid payment methods require Alternate Payment Method request
- High-value subscriptions justify white-glove human escalation
- Excessive customer interventions cause fatigue and decrease reward
- Reliable-but-dormant customers frequently self-resolve when given time (WAIT)

This module supports TWO modes:
  1. Single-step transitions (existing `reset` / `step`) — used by the flat
     bandit / baseline comparisons.
  2. Full episode rollouts (`run_episode`) — used by the sequential RL policy,
     emitting (state, action, reward, next_state, done) tuples where the real
     reward is concentrated at episode termination.
"""

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from simulator.models import (
    ACTION_INDEX,
    SimActionType,
    SimCustomer,
    SimFailureType,
    SimOutcome,
    SimState,
    SimSubscription,
)
from simulator.reward import (
    ACTION_COSTS,
    ACTION_FRICTION,
    RewardWeights,
    compute_episode_reward,
    compute_reward,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────── Dynamics Config ────────────────────────────
#
# All "hidden physics" of the simulator live here so that
# `simulator/domain_randomization.py` can perturb them for robustness training
# and evaluation. Defaults reproduce the original hardcoded behaviour exactly.


@dataclass
class SimDynamics:
    # Fatigue: per-contact response degradation
    fatigue_coef: float = 0.08
    # Self-cure: reliable customers resolve themselves when given time (WAIT).
    # Effective self-cure prob on a WAIT = self_cure_base * lifetime_success_rate.
    self_cure_base: float = 0.45
    # Failure-type distribution (cumulative thresholds over U(0,1))
    fail_dist: List[float] = field(
        default_factory=lambda: [0.40, 0.70, 0.85, 0.92, 0.97]
    )
    # Per-(failure, action) success probabilities
    high_value_escalate_success: float = 0.75
    temp_retry_after_cd_base: float = 0.78
    temp_retry_after_cd_decay: float = 0.15
    temp_retry_immediate: float = 0.25
    temp_link: float = 0.50
    temp_escalate: float = 0.60
    insuff_retry_after48_base: float = 0.32
    insuff_retry_after48_decay: float = 0.10
    insuff_retry_immediate: float = 0.12
    insuff_link: float = 0.68
    insuff_reminder: float = 0.35
    insuff_escalate: float = 0.70
    invalid_alt_method: float = 0.72
    invalid_link: float = 0.58
    invalid_escalate: float = 0.65
    fraud_retry: float = 0.02
    fraud_escalate: float = 0.55
    fraud_link: float = 0.40
    base_p: float = 0.20


class RecoveryEnvironment:
    def __init__(
        self,
        seed: Optional[int] = 42,
        reward_weights: Optional[RewardWeights] = None,
        dynamics: Optional[SimDynamics] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.weights = reward_weights or RewardWeights()
        self.dynamics = dynamics or SimDynamics()
        self.seed = seed

    # ───────────────────────────── episode reset ────────────────────────────

    def reset(self, seed: Optional[int] = None) -> SimState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed

        cust_id = int(self.rng.integers(1000, 99999))
        archetype_p = self.rng.random()
        # Archetype mix: reliable 62%, occasional 20%, high_risk 10%, reliable_dormant 8%
        if archetype_p < 0.62:
            archetype = "reliable"
            fail_rate = 0.04
            ltv = float(self.rng.uniform(20000, 100000))
            tenure = int(self.rng.integers(180, 1200))
            total_pay = int(self.rng.integers(10, 40))
            succ_pay = total_pay - 1
            days_since_success = int(self.rng.integers(5, 45))
            first_failure = False
        elif archetype_p < 0.82:
            archetype = "occasional"
            fail_rate = 0.15
            ltv = float(self.rng.uniform(10000, 50000))
            tenure = int(self.rng.integers(90, 800))
            total_pay = int(self.rng.integers(5, 25))
            succ_pay = max(1, total_pay - int(self.rng.integers(2, 5)))
            days_since_success = int(self.rng.integers(10, 90))
            first_failure = False
        elif archetype_p < 0.92:
            archetype = "high_risk"
            fail_rate = 0.35
            ltv = float(self.rng.uniform(5000, 25000))
            tenure = int(self.rng.integers(30, 400))
            total_pay = int(self.rng.integers(4, 20))
            succ_pay = max(0, total_pay - int(self.rng.integers(4, 10)))
            days_since_success = int(self.rng.integers(30, 150))
            first_failure = False
        else:
            # Reliable-but-dormant: long tenure, near-perfect lifetime record,
            # last success a while ago, and this is their FIRST ever failure.
            archetype = "reliable_dormant"
            fail_rate = 0.02
            ltv = float(self.rng.uniform(40000, 150000))
            tenure = int(self.rng.integers(1000, 2200))   # ~3-6 years
            total_pay = int(self.rng.integers(30, 80))
            succ_pay = total_pay                           # clean record
            days_since_success = int(self.rng.integers(14, 60))  # dormant 2+ weeks
            first_failure = True

        customer = SimCustomer(
            id=cust_id,
            name=f"Customer {cust_id}",
            email=f"user_{cust_id}@example.com",
            phone=f"+9198{self.rng.integers(10000000, 99999999)}",
            opted_out=False,
            archetype=archetype,
            ltv=ltv,
            failure_rate=fail_rate,
            total_payments=total_pay,
            failed_payments=total_pay - succ_pay,
            account_tenure_days=tenure,
            successful_payments=succ_pay,
            days_since_last_success=days_since_success,
            historical_ltv=ltv,
            is_first_ever_failure=first_failure,
        )

        amount = float(self.rng.choice([999.0, 2499.0, 4999.0, 9999.0, 24999.0, 59999.0]))
        subscription = SimSubscription(
            id=f"sub_sim_{cust_id}",
            customer_id=cust_id,
            amount_per_period=amount,
            paid_count=int(self.rng.integers(1, 10)),
            remaining_count=int(self.rng.integers(2, 11)),
            total_count=12,
            status="pending",
        )

        # Failure distribution (domain-randomizable)
        fd = self.dynamics.fail_dist
        fail_p = self.rng.random()
        if fail_p < fd[0]:
            failure_type = SimFailureType.TEMPORARY_BANK_FAILURE
        elif fail_p < fd[1]:
            failure_type = SimFailureType.INSUFFICIENT_FUNDS
        elif fail_p < fd[2]:
            failure_type = SimFailureType.INVALID_PAYMENT_METHOD
        elif fail_p < fd[3]:
            failure_type = SimFailureType.NETWORK_ERROR
        elif fail_p < fd[4]:
            failure_type = SimFailureType.CARD_EXPIRED
        else:
            failure_type = SimFailureType.AUTHENTICATION_FAILED

        return SimState(
            customer=customer,
            subscription=subscription,
            failure_type=failure_type,
            amount=amount,
            attempt_count=1,
            hours_since_first_failure=0.0,
            interventions_count=0,
            consecutive_failures=1,
            is_weekend=bool(self.rng.random() < 0.28),
        )

    # ───────────────────────────── single-step (existing) ───────────────────

    def step(self, state: SimState, action: SimActionType) -> SimOutcome:
        """
        Applies action to state, transitioning to next_state or terminal recovery outcome.
        """
        if isinstance(action, str):
            action = SimActionType(action)
        action_name = action.value
        cost = ACTION_COSTS.get(action_name, 0.0)
        friction = ACTION_FRICTION.get(action_name, 0.0)

        # Handle STOP_RECOVERY immediately
        if action == SimActionType.STOP_RECOVERY:
            r = compute_reward(
                recovered_revenue=0.0,
                action_type=action_name,
                is_unnecessary_retry=False,
                is_policy_violation=False,
                excessive_intervention_count=state.interventions_count,
                weights=self.weights,
            )
            return SimOutcome(
                success=False,
                recovered_amount=0.0,
                recovery_time_hours=state.hours_since_first_failure,
                action_cost=cost,
                friction=friction,
                next_attempt_count=state.attempt_count,
                next_state=None,
                reward=r,
                terminal=True,
            )

        # Handle WAIT — may self-cure for reliable customers
        if action == SimActionType.WAIT:
            next_state = copy.deepcopy(state)
            wait_hours = 6.0 if state.failure_type == SimFailureType.TEMPORARY_BANK_FAILURE else 24.0
            next_state.hours_since_first_failure += wait_hours

            # Self-cure: reliable customers resolve themselves given time.
            self_cure_p = self.dynamics.self_cure_base * state.customer.lifetime_success_rate
            self_cured = bool(self.rng.random() < self_cure_p)
            # Never self-cure an invalid/expired card — that needs a method change.
            if state.failure_type in (
                SimFailureType.INVALID_PAYMENT_METHOD,
                SimFailureType.CARD_EXPIRED,
            ):
                self_cured = False

            if self_cured:
                return SimOutcome(
                    success=True,
                    recovered_amount=state.amount,
                    recovery_time_hours=next_state.hours_since_first_failure,
                    action_cost=cost,
                    friction=friction,
                    next_attempt_count=state.attempt_count,
                    next_state=None,
                    reward=0.0,
                    terminal=True,
                )

            r = compute_reward(
                recovered_revenue=0.0,
                action_type=action_name,
                is_unnecessary_retry=False,
                is_policy_violation=False,
                excessive_intervention_count=state.interventions_count,
                weights=self.weights,
            )
            terminal = next_state.hours_since_first_failure > 168.0
            return SimOutcome(
                success=False,
                recovered_amount=0.0,
                recovery_time_hours=next_state.hours_since_first_failure,
                action_cost=cost,
                friction=friction,
                next_attempt_count=state.attempt_count,
                next_state=None if terminal else next_state,
                reward=r,
                terminal=terminal,
            )

        # Compute success probability based on realistic correlated rules
        p_success = self._compute_recovery_probability(state, action)
        recovered = bool(self.rng.random() < p_success)

        if recovered:
            recovered_revenue = state.amount
            time_taken = state.hours_since_first_failure + self._action_latency_hours(action)
            r = compute_reward(
                recovered_revenue=recovered_revenue,
                action_type=action_name,
                is_unnecessary_retry=False,
                is_policy_violation=False,
                excessive_intervention_count=state.interventions_count + 1,
                weights=self.weights,
            )
            return SimOutcome(
                success=True,
                recovered_amount=recovered_revenue,
                recovery_time_hours=time_taken,
                action_cost=cost,
                friction=friction,
                next_attempt_count=state.attempt_count + (1 if action == SimActionType.RETRY else 0),
                next_state=None,
                reward=r,
                terminal=True,
            )
        else:
            next_state = copy.deepcopy(state)
            next_state.interventions_count += 1
            if action == SimActionType.RETRY:
                next_state.attempt_count += 1
            next_state.consecutive_failures += 1
            next_state.hours_since_first_failure += self._action_latency_hours(action)

            terminal = (
                next_state.attempt_count > 4
                or next_state.hours_since_first_failure >= 168.0
                or next_state.interventions_count >= 5
            )

            is_unnecessary = (action == SimActionType.RETRY and p_success < 0.15)
            r = compute_reward(
                recovered_revenue=0.0,
                action_type=action_name,
                is_unnecessary_retry=is_unnecessary,
                is_policy_violation=False,
                excessive_intervention_count=next_state.interventions_count,
                weights=self.weights,
            )

            return SimOutcome(
                success=False,
                recovered_amount=0.0,
                recovery_time_hours=next_state.hours_since_first_failure,
                action_cost=cost,
                friction=friction,
                next_attempt_count=next_state.attempt_count,
                next_state=None if terminal else next_state,
                reward=r,
                terminal=terminal,
            )

    # ───────────────────────────── episode rollout (RL) ─────────────────────

    def allowed_actions(self, state: SimState) -> Set[SimActionType]:
        """
        Apply the UNMODIFIED Tier-1 hard safety gate at the current state and
        return the surviving action subset. Mirrors safety_policy.py rules using
        the simulator's state; kept structurally identical (opt-out, retry cap,
        cooldown, recovery window, value thresholds, intervention caps).
        """
        from backend.policies.safety_policy import SafetyContext, SafetyPolicyEngine
        from backend.models.enums import ActionType
        from backend.models.merchant_policy import MerchantPolicy

        policy = MerchantPolicy()
        engine = SafetyPolicyEngine(merchant_policy=policy)
        now = datetime.utcnow()
        ctx = SafetyContext(
            customer_id=state.customer.id,
            amount=state.amount,
            attempt_count=state.attempt_count,
            interventions_today=state.interventions_count,
            global_retries_today=0,
            first_failure_time=now - timedelta(hours=state.hours_since_first_failure),
            last_attempt_time=(
                now - timedelta(hours=state.hours_since_first_failure)
                if state.attempt_count > 1
                else None
            ),
            is_customer_opted_out=state.customer.opted_out,
            current_time=now,
        )
        decision = engine.evaluate(ctx)
        return {SimActionType(a.value) for a in decision.allowed_actions}

    def run_episode(
        self,
        policy_fn: Callable[[SimState, Set[SimActionType]], SimActionType],
        seed: Optional[int] = None,
        max_steps: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Roll out one full recovery episode under `policy_fn`.

        At EVERY timestep the Tier-1 safety gate is re-applied; the policy only
        ever chooses from the surviving allowed subset. Rewards are ~0 at
        intermediate steps; the real signal is the terminal `episode_reward`.
        """
        state = self.reset(seed=seed)
        transitions: List[Dict[str, Any]] = []
        action_costs: List[float] = []
        contacts = 0
        resolved = False
        recovered_amount = 0.0

        for t in range(max_steps):
            allowed = self.allowed_actions(state)
            if not allowed:
                allowed = {SimActionType.STOP_RECOVERY}
            action = policy_fn(state, allowed)
            if action not in allowed:  # never let a policy bypass the gate
                action = SimActionType.STOP_RECOVERY

            # Behavior propensity (for off-policy evaluation). The logging policy
            # exposes `.propensity(state, action, allowed)`; deterministic eval
            # policies get propensity 1.0.
            propensity_fn = getattr(policy_fn, "propensity", None)
            behavior_prob = (
                float(propensity_fn(state, action, allowed)) if callable(propensity_fn) else 1.0
            )

            action_costs.append(ACTION_COSTS.get(action.value, 0.0))
            if action in (
                SimActionType.PAYMENT_LINK,
                SimActionType.SEND_REMINDER,
                SimActionType.REQUEST_ALTERNATE_METHOD,
                SimActionType.ESCALATE_TO_HUMAN,
            ):
                contacts += 1

            outcome = self.step(state, action)
            resolved = outcome.success
            recovered_amount = outcome.recovered_amount

            transitions.append(
                self._transition_record(
                    state, action, allowed, reward=0.0, done=False, behavior_prob=behavior_prob
                )
            )

            if outcome.terminal:
                done = True
                terminal_reward = compute_episode_reward(
                    recovered_amount=recovered_amount,
                    action_costs=action_costs,
                    contacts_count=contacts,
                    resolved=resolved,
                    weights=self.weights,
                )
                transitions[-1]["reward"] = terminal_reward
                transitions[-1]["done"] = True
                transitions[-1]["resolved"] = resolved
                transitions[-1]["recovered_amount"] = recovered_amount
                break

            state = outcome.next_state
        else:
            # Ran out of steps without termination → treat as window expiry.
            terminal_reward = compute_episode_reward(
                recovered_amount=recovered_amount,
                action_costs=action_costs,
                contacts_count=contacts,
                resolved=resolved,
                weights=self.weights,
            )
            transitions[-1]["reward"] = terminal_reward
            transitions[-1]["done"] = True
            transitions[-1]["resolved"] = resolved
            transitions[-1]["recovered_amount"] = recovered_amount

        return transitions

    @staticmethod
    def _transition_record(
        state: SimState,
        action: SimActionType,
        allowed: Set[SimActionType],
        reward: float,
        done: bool,
        behavior_prob: float = 1.0,
    ) -> Dict[str, Any]:
        return {
            "behavior_prob": behavior_prob,
            "failure_reason": state.failure_type.value,
            "amount": state.amount,
            "attempt_count": state.attempt_count,
            "hours_since_first_failure": state.hours_since_first_failure,
            "interventions_count": state.interventions_count,
            "consecutive_failures": state.consecutive_failures,
            "is_weekend": int(state.is_weekend),
            "customer_ltv": state.customer.ltv,
            "customer_failure_rate": state.customer.failure_rate,
            "subscription_paid_count": state.subscription.paid_count,
            "customer_archetype": state.customer.archetype,
            "opted_out": int(state.customer.opted_out),
            # long-horizon history
            "account_tenure_days": state.customer.account_tenure_days,
            "lifetime_success_rate": state.customer.lifetime_success_rate,
            "days_since_last_success": state.customer.days_since_last_success,
            "historical_ltv": state.customer.historical_ltv,
            "is_first_ever_failure": int(state.customer.is_first_ever_failure),
            # action + outcome
            "action": action.value,
            "allowed_actions": ",".join(sorted(a.value for a in allowed)),
            "reward": reward,
            "done": done,
            "resolved": False,
            "recovered_amount": 0.0,
        }

    # ─────────────────────────── hidden physics ─────────────────────────────

    def _compute_recovery_probability(self, state: SimState, action: SimActionType) -> float:
        """Realistic correlated simulation dynamics (domain-randomizable)."""
        d = self.dynamics
        ft = state.failure_type
        attempts = state.attempt_count
        hours = state.hours_since_first_failure
        fatigue = max(0.0, state.interventions_count * d.fatigue_coef)

        if state.amount > 50000.0 and action == SimActionType.ESCALATE_TO_HUMAN:
            return d.high_value_escalate_success

        if ft in (SimFailureType.TEMPORARY_BANK_FAILURE, SimFailureType.NETWORK_ERROR):
            if action == SimActionType.RETRY:
                if hours >= 4.0:
                    return max(0.05, d.temp_retry_after_cd_base - (attempts * d.temp_retry_after_cd_decay) - fatigue)
                return max(0.05, d.temp_retry_immediate - fatigue)
            elif action == SimActionType.PAYMENT_LINK:
                return max(0.05, d.temp_link - fatigue)
            elif action == SimActionType.ESCALATE_TO_HUMAN:
                return d.temp_escalate
            return 0.10

        elif ft == SimFailureType.INSUFFICIENT_FUNDS:
            if action == SimActionType.RETRY:
                if hours >= 48.0:
                    return max(0.05, d.insuff_retry_after48_base - (attempts * d.insuff_retry_after48_decay) - fatigue)
                return max(0.02, d.insuff_retry_immediate - fatigue)
            elif action == SimActionType.PAYMENT_LINK:
                return max(0.10, d.insuff_link - fatigue)
            elif action == SimActionType.SEND_REMINDER:
                return max(0.05, d.insuff_reminder - fatigue)
            elif action == SimActionType.ESCALATE_TO_HUMAN:
                return d.insuff_escalate
            return 0.05

        elif ft in (SimFailureType.INVALID_PAYMENT_METHOD, SimFailureType.CARD_EXPIRED):
            if action == SimActionType.RETRY:
                return 0.0  # an expired/invalid card will NEVER succeed on retry
            elif action == SimActionType.REQUEST_ALTERNATE_METHOD:
                return max(0.10, d.invalid_alt_method - fatigue)
            elif action == SimActionType.PAYMENT_LINK:
                return max(0.10, d.invalid_link - fatigue)
            elif action == SimActionType.ESCALATE_TO_HUMAN:
                return d.invalid_escalate
            return 0.05

        elif ft in (SimFailureType.FRAUD_SUSPECTED, SimFailureType.AUTHENTICATION_FAILED):
            if action == SimActionType.RETRY:
                return d.fraud_retry
            elif action == SimActionType.ESCALATE_TO_HUMAN:
                return d.fraud_escalate
            elif action == SimActionType.PAYMENT_LINK:
                return d.fraud_link
            return 0.05

        return max(0.01, d.base_p - fatigue)

    def _action_latency_hours(self, action: SimActionType) -> float:
        if action == SimActionType.WAIT:
            return 12.0
        elif action == SimActionType.RETRY:
            return 1.0
        elif action == SimActionType.PAYMENT_LINK:
            return 24.0
        elif action == SimActionType.SEND_REMINDER:
            return 12.0
        elif action == SimActionType.REQUEST_ALTERNATE_METHOD:
            return 36.0
        elif action == SimActionType.ESCALATE_TO_HUMAN:
            return 48.0
        return 0.0
