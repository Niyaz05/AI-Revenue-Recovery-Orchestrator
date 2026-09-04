from __future__ import annotations
from typing import Optional
"""Webhook event model — raw event persistence for audit/replay/debugging."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    razorpay_event_id: Mapped[str] = mapped_column(
        String(100), unique=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(100), index=True)

    raw_payload: Mapped[str] = mapped_column(Text)  # Full JSON preserved for audit

    signature_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    processing_error: Mapped[str] = mapped_column(String(1000), default="")
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True, index=True)

    received_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self):
        return (
            f"<WebhookEvent id={self.id} type={self.event_type} "
            f"processed={self.processed}>"
        )
