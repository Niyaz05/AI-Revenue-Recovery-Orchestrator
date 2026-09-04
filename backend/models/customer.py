from __future__ import annotations
from typing import Optional

"""Customer model — our internal representation, not a Razorpay mirror."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    razorpay_customer_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, index=True, nullable=True
    )
    email: Mapped[str] = mapped_column(String(255), index=True)
    name: Mapped[str] = mapped_column(String(255), default="")
    phone: Mapped[str] = mapped_column(String(20), default="")

    # Recovery-specific fields
    opted_out_recovery: Mapped[bool] = mapped_column(Boolean, default=False)
    risk_tier: Mapped[str] = mapped_column(String(20), default="standard")  # low, standard, high
    lifetime_value: Mapped[float] = mapped_column(Float, default=0.0)
    total_payments: Mapped[int] = mapped_column(default=0)
    failed_payments: Mapped[int] = mapped_column(default=0)
    recovered_payments: Mapped[int] = mapped_column(default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    subscriptions = relationship("Subscription", back_populates="customer")
    payment_attempts = relationship("PaymentAttempt", back_populates="customer")
    recovery_actions = relationship("RecoveryAction", back_populates="customer")

    @property
    def failure_rate(self) -> float:
        if self.total_payments == 0:
            return 0.0
        return self.failed_payments / self.total_payments

    def __repr__(self):
        return f"<Customer id={self.id} email={self.email} opted_out={self.opted_out_recovery}>"
