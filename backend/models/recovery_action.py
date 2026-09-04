from __future__ import annotations
from typing import Optional

"""Recovery action model — every intervention decided by the system."""

import json
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base
from backend.models.enums import ActionStatus, ActionType, Provenance


class RecoveryAction(Base):
    __tablename__ = "recovery_actions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    payment_attempt_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("payment_attempts.id"), index=True, nullable=True
    )
    subscription_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subscriptions.id"), index=True, nullable=True
    )
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)

    action_type: Mapped[str] = mapped_column(Enum(ActionType))
    action_details: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    status: Mapped[str] = mapped_column(Enum(ActionStatus), default=ActionStatus.PENDING)

    # Safety layer tracking
    blocked_by_policy: Mapped[bool] = mapped_column(Boolean, default=False)
    block_reason: Mapped[str] = mapped_column(String(500), default="")

    # AI decision tracking (structured, not free-form)
    ai_recommendation: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    policy_decision: Mapped[str] = mapped_column(Text, default="{}")  # JSON

    # Execution tracking
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    result: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    recovered_amount: Mapped[float] = mapped_column(Float, default=0.0)
    razorpay_payment_link_id: Mapped[str] = mapped_column(String(50), default="")

    # Idempotency
    idempotency_key: Mapped[str] = mapped_column(String(100), unique=True, index=True)

    # Versioning
    model_version: Mapped[str] = mapped_column(String(50), default="v0")
    policy_version: Mapped[str] = mapped_column(String(50), default="v0")

    # Provenance — EVERY number must be tagged
    provenance: Mapped[str] = mapped_column(
        Enum(Provenance), default=Provenance.SIMULATED
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    payment_attempt = relationship("PaymentAttempt", back_populates="recovery_actions")
    subscription = relationship("Subscription", back_populates="recovery_actions")
    customer = relationship("Customer", back_populates="recovery_actions")
    outcome = relationship("RecoveryOutcome", back_populates="recovery_action", uselist=False)

    def get_ai_recommendation(self) -> dict:
        return json.loads(self.ai_recommendation) if self.ai_recommendation else {}

    def get_policy_decision(self) -> dict:
        return json.loads(self.policy_decision) if self.policy_decision else {}

    def __repr__(self):
        return (
            f"<RecoveryAction id={self.id} type={self.action_type} "
            f"status={self.status} blocked={self.blocked_by_policy}>"
        )
