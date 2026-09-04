from __future__ import annotations
from typing import Optional

"""Recovery outcome model — the final result of a recovery attempt."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base
from backend.models.enums import OutcomeType, Provenance


class RecoveryOutcome(Base):
    __tablename__ = "recovery_outcomes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    recovery_action_id: Mapped[int] = mapped_column(
        ForeignKey("recovery_actions.id"), unique=True, index=True
    )
    subscription_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subscriptions.id"), index=True, nullable=True
    )

    outcome_type: Mapped[str] = mapped_column(Enum(OutcomeType))
    recovered_amount: Mapped[float] = mapped_column(Float, default=0.0)
    recovery_time_seconds: Mapped[int] = mapped_column(Integer, default=0)
    total_attempts: Mapped[int] = mapped_column(Integer, default=0)
    total_interventions: Mapped[int] = mapped_column(Integer, default=0)
    reward: Mapped[float] = mapped_column(Float, default=0.0)

    provenance: Mapped[str] = mapped_column(
        Enum(Provenance), default=Provenance.SIMULATED
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    recovery_action = relationship("RecoveryAction", back_populates="outcome")

    def __repr__(self):
        return (
            f"<RecoveryOutcome id={self.id} type={self.outcome_type} "
            f"recovered={self.recovered_amount} reward={self.reward:.2f}>"
        )
