from __future__ import annotations
from typing import Optional

"""Payment attempt model — every individual charge attempt."""

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base
from backend.models.enums import FailureReason, PaymentStatus


class PaymentAttempt(Base):
    __tablename__ = "payment_attempts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    razorpay_payment_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, index=True, nullable=True
    )
    subscription_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("subscriptions.id"), index=True, nullable=True
    )
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)

    amount: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(3), default="INR")

    status: Mapped[str] = mapped_column(Enum(PaymentStatus), default=PaymentStatus.CREATED)
    failure_reason: Mapped[Optional[str]] = mapped_column(
        Enum(FailureReason), nullable=True
    )
    error_code: Mapped[str] = mapped_column(String(100), default="")
    error_description: Mapped[str] = mapped_column(String(500), default="")
    error_source: Mapped[str] = mapped_column(String(50), default="")

    payment_method: Mapped[str] = mapped_column(String(50), default="")
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    subscription = relationship("Subscription", back_populates="payment_attempts")
    customer = relationship("Customer", back_populates="payment_attempts")
    recovery_actions = relationship("RecoveryAction", back_populates="payment_attempt")

    def __repr__(self):
        return (
            f"<PaymentAttempt id={self.id} rz_id={self.razorpay_payment_id} "
            f"status={self.status} attempt={self.attempt_number}>"
        )
