from __future__ import annotations
"""Merchant policy model — hard safety constraints that gate every action."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


class MerchantPolicy(Base):
    __tablename__ = "merchant_policies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), default="default")

    # Hard safety limits — the AI can NEVER bypass these
    max_retry_attempts: Mapped[int] = mapped_column(Integer, default=3)
    retry_cooldown_hours: Mapped[float] = mapped_column(Float, default=4.0)
    recovery_window_hours: Mapped[float] = mapped_column(Float, default=168.0)  # 7 days
    max_auto_recovery_amount: Mapped[float] = mapped_column(Float, default=50000.0)  # ₹50,000
    global_daily_retry_budget: Mapped[int] = mapped_column(Integer, default=100)
    human_approval_threshold_amount: Mapped[float] = mapped_column(Float, default=100000.0)  # ₹1,00,000
    daily_intervention_cap_per_customer: Mapped[int] = mapped_column(Integer, default=3)

    # Opt-out list (JSON array of customer IDs)
    opted_out_customer_ids: Mapped[str] = mapped_column(Text, default="[]")

    # Versioning
    version: Mapped[str] = mapped_column(String(20), default="v1")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __init__(
        self,
        name: str = "default",
        max_retry_attempts: int = 3,
        retry_cooldown_hours: float = 4.0,
        recovery_window_hours: float = 168.0,
        max_auto_recovery_amount: float = 50000.0,
        global_daily_retry_budget: int = 100,
        human_approval_threshold_amount: float = 100000.0,
        daily_intervention_cap_per_customer: int = 3,
        opted_out_customer_ids: str = "[]",
        version: str = "v1",
        is_active: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.max_retry_attempts = max_retry_attempts
        self.retry_cooldown_hours = retry_cooldown_hours
        self.recovery_window_hours = recovery_window_hours
        self.max_auto_recovery_amount = max_auto_recovery_amount
        self.global_daily_retry_budget = global_daily_retry_budget
        self.human_approval_threshold_amount = human_approval_threshold_amount
        self.daily_intervention_cap_per_customer = daily_intervention_cap_per_customer
        self.opted_out_customer_ids = opted_out_customer_ids
        self.version = version
        self.is_active = is_active

    def get_opted_out_ids(self) -> list[int]:
        import json
        return json.loads(self.opted_out_customer_ids) if self.opted_out_customer_ids else []

    def __repr__(self):
        return f"<MerchantPolicy name={self.name} version={self.version} active={self.is_active}>"
