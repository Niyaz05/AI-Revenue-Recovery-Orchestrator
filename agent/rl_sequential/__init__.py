from __future__ import annotations
"""
Sequential RL agents for the AI Revenue Recovery Orchestrator (Tier 2 upgrade).

These reason over the entire multi-day recovery window as ONE episode, unlike
the single-shot LinUCB bandit. Both respect the Tier-1 safety mask at inference
(argmax is taken only over the allowed-actions subset, never over a forbidden
action — the mask is applied before argmax, not after).
"""

from agent.rl_sequential.cql import CQLAgent
from agent.rl_sequential.decision_transformer import DecisionTransformerAgent

__all__ = ["CQLAgent", "DecisionTransformerAgent"]
