"""Create immutable snapshots, durable evaluations, and append-only reviews."""
from alembic import op
import sqlalchemy as sa

revision = "backend_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("backend_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("source_key", sa.String(512), nullable=False, unique=True),
        sa.Column("revision", sa.Integer, nullable=False))
    op.create_table("backend_job_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("job_id", sa.String(64), sa.ForeignKey("backend_jobs.id"), nullable=False),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("city", sa.String(100), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("company", sa.String(300), nullable=False),
        sa.Column("content", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("job_id", "revision", name="uq_backend_snapshot_revision"))
    op.create_index("ix_backend_snapshot_city", "backend_job_snapshots", ["city"])
    op.create_table("backend_evaluations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), nullable=False, unique=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("snapshot_id", sa.String(64), sa.ForeignKey("backend_job_snapshots.id"), nullable=False),
        sa.Column("request", sa.JSON, nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("result", sa.JSON, nullable=True),
        sa.Column("failure_type", sa.String(64), nullable=True),
        sa.Column("review_version", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_backend_evaluation_created", "backend_evaluations", ["created_at", "id"])
    op.create_table("backend_evaluation_reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("evaluation_id", sa.String(36), sa.ForeignKey("backend_evaluations.id"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("decision", sa.String(24), nullable=False),
        sa.Column("note", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("evaluation_id", "version", name="uq_backend_review_version"))


def downgrade():
    raise RuntimeError("destructive downgrade is disabled; back up and review a forward migration")
