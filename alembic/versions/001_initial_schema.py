"""Initial schema with jobs table.

Revision ID: 001
Revises:
Create Date: 2026-01-29

This migration creates the core jobs table with:
- UUID primary key
- Job type and input data
- Status state machine
- Retry tracking
- Token and cost metrics
- Timestamps
- Dead letter queue support
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# Revision identifiers
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the jobs table and indexes."""

    # Create the jobs table
    op.create_table(
        "jobs",
        # Primary key
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),

        # Job definition
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("input_data", postgresql.JSONB(), nullable=False),

        # State machine
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),

        # Results
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("error", postgresql.JSONB(), nullable=True),

        # Metrics
        sa.Column("input_tokens", sa.Integer(), server_default="0"),
        sa.Column("output_tokens", sa.Integer(), server_default="0"),
        sa.Column("total_tokens", sa.Integer(), server_default="0"),
        sa.Column("cost_usd", sa.Float(), server_default="0.0"),
        sa.Column("duration_ms", sa.Integer(), nullable=True),

        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),

        # Worker tracking
        sa.Column("worker_id", sa.String(50), nullable=True),

        # Dead letter queue
        sa.Column("moved_to_dlq_at", sa.DateTime(), nullable=True),
        sa.Column("dlq_reason", sa.Text(), nullable=True),

        # Constraints
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'dead')",
            name="valid_status",
        ),
    )

    # Create indexes for efficient queries

    # Basic indexes
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.create_index("idx_jobs_created_at", "jobs", ["created_at"])
    op.create_index("idx_jobs_job_type", "jobs", ["job_type"])

    # Composite index for filtering
    op.create_index("idx_jobs_type_status", "jobs", ["job_type", "status"])

    # Partial index for pending jobs (most common query)
    op.create_index(
        "idx_jobs_pending_created",
        "jobs",
        ["status", "created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    """Drop the jobs table and indexes."""

    # Drop indexes first
    op.drop_index("idx_jobs_pending_created", table_name="jobs")
    op.drop_index("idx_jobs_type_status", table_name="jobs")
    op.drop_index("idx_jobs_job_type", table_name="jobs")
    op.drop_index("idx_jobs_created_at", table_name="jobs")
    op.drop_index("idx_jobs_status", table_name="jobs")

    # Drop table
    op.drop_table("jobs")
