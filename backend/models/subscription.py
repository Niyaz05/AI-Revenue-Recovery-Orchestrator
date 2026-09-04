from __future__ import annotations
from typing import Optional

"""Subscription model — tracks recurring billing lifecycle."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base
from backend.models.enums import SubscriptionStatus


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    razorpay_subscription_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, index=True, nullable=True
    )
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    razorpay_plan_id: Mapped[str] = mapped_column(String(50), default="")
    plan_name: Mapped[str] = mapped_column(String(255), default="")
    amount_per_period: Mapped[float] = mapped_column(Float, default=0.0)  # in base currency unit (rupees)
    currency: Mapped[str] = mapped_column(String(3), default="INR")
    billing_period: Mapped[str] = mapped_column(String(20), default="monthly")  # daily/weekly/monthly/yearly
    billing_interval: Mapped[int] = mapped_column(Integer, default=1)

    status: Mapped[str] = mapped_column(
        Enum(SubscriptionStatus), default=SubscriptionStatus.CREATED
    )

    paid_count: Mapped[int] = mapped_column(Integer, default=0)
    remaining_count: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)

    current_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    charge_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    customer = relationship("Customer", back_populates="subscriptions")
    payment_attempts = relationship("PaymentAttempt", back_populates="subscription")
    recovery_actions = relationship("RecoveryAction", back_populates="subscription")

    @property
    def is_at_risk(self) -> bool:
        return self.status in (SubscriptionStatus.PENDING, SubscriptionStatus.HALTED)

    @property
    def at_risk_amount(self) -> float:
        if self.is_at_risk:
            return self.amount_per_period * self.remaining_count
        return 0.0

    def __repr__(self):
        return f"<Subscription id={self.id} rz_id={self.razorpay_subscription_id} status={self.status}>"
