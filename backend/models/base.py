from __future__ import annotations
from typing import Optional
"""
SQLAlchemy database setup.

Provides engine, session factory, and base class for all models.
Supports SQLite (dev) and PostgreSQL (production) via DATABASE_URL.
"""

import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./data/recovery.db")


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def _get_engine(url: Optional[str] = None):
    db_url = url or DATABASE_URL
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(db_url, connect_args=connect_args, echo=False)
    # Enable WAL mode for SQLite for better concurrent read/write
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


engine = _get_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """FastAPI dependency — yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(url: Optional[str] = None):
    """Create all tables. Call once at startup."""
    # Ensure all ORM models are registered in metadata
    import backend.models  # noqa: F401
    if url:
        eng = _get_engine(url)
        Base.metadata.create_all(bind=eng)
    else:
        Base.metadata.create_all(bind=engine)
