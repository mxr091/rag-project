"""Relational records. Snapshots, model results and reviews stay separate."""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "backend_jobs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_key: Mapped[str] = mapped_column(String(512), unique=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)


class JobSnapshot(Base):
    __tablename__ = "backend_job_snapshots"
    __table_args__ = (
        UniqueConstraint("job_id", "revision", name="uq_backend_snapshot_revision"),
        Index("ix_backend_snapshot_city", "city"),
    )
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("backend_jobs.id"))
    revision: Mapped[int] = mapped_column(Integer)
    city: Mapped[str] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(String(300))
    company: Mapped[str] = mapped_column(String(300))
    content: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Evaluation(Base):
    __tablename__ = "backend_evaluations"
    __table_args__ = (Index("ix_backend_evaluation_created", "created_at", "id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("backend_job_snapshots.id"))
    request: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24))
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    failure_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationReview(Base):
    __tablename__ = "backend_evaluation_reviews"
    __table_args__ = (UniqueConstraint("evaluation_id", "version", name="uq_backend_review_version"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("backend_evaluations.id"))
    version: Mapped[int] = mapped_column(Integer)
    decision: Mapped[str] = mapped_column(String(24))
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
