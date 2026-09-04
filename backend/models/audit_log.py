from __future__ import annotations
from typing import Optional

"""Immutable audit log — every decision is recorded with full context."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base
from backend.models.enums import Provenance


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Context
    customer_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    payment_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    subscription_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # State snapshot (JSON)
    state_before: Mapped[str] = mapped_column(Text, default="{}")
    state_after: Mapped[str] = mapped_column(Text, default="{}")

    # Decision details (structured JSON, not free-form)
    ai_recommendation: Mapped[str] = mapped_column(Text, default="{}")
    policy_decision: Mapped[str] = mapped_column(Text, default="{}")
    executed_action: Mapped[str] = mapped_column(String(50), default="")
    result: Mapped[str] = mapped_column(Text, default="{}")

    recovered_amount: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")

    # Versioning
    model_version: Mapped[str] = mapped_column(String(50), default="v0")
    policy_version: Mapped[str] = mapped_column(String(50), default="v0")

    provenance: Mapped[str] = mapped_column(
        Enum(Provenance), default=Provenance.SIMULATED
    )

    def __repr__(self):
        return f"<AuditLog id={self.id} event={self.event_type} action={self.executed_action}>"
