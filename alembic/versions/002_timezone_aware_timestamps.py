"""Change timestamp columns to timezone-aware

Revision ID: 002
Revises: 001
Create Date: 2026-02-01

This migration updates all timestamp columns from
TIMESTAMP WITHOUT TIME ZONE to TIMESTAMP WITH TIME ZONE
for proper timezone-aware datetime handling.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# Revision identifiers
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Convert timestamp columns to timezone-aware."""
    # Alter created_at column
    op.alter_column(
        "jobs",
        "created_at",
        existing_type=sa.DateTime(),
        type_=sa.TIMESTAMP(timezone=True),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # Alter started_at column
    op.alter_column(
        "jobs",
        "started_at",
        existing_type=sa.DateTime(),
        type_=sa.TIMESTAMP(timezone=True),
        existing_nullable=True,
        postgresql_using="started_at AT TIME ZONE 'UTC'",
    )

    # Alter completed_at column
    op.alter_column(
        "jobs",
        "completed_at",
        existing_type=sa.DateTime(),
        type_=sa.TIMESTAMP(timezone=True),
        existing_nullable=True,
        postgresql_using="completed_at AT TIME ZONE 'UTC'",
    )

    # Alter moved_to_dlq_at column
    op.alter_column(
        "jobs",
        "moved_to_dlq_at",
        existing_type=sa.DateTime(),
        type_=sa.TIMESTAMP(timezone=True),
        existing_nullable=True,
        postgresql_using="moved_to_dlq_at AT TIME ZONE 'UTC'",
    )


def downgrade() -> None:
    """Revert timestamp columns to timezone-naive."""
    # Alter created_at column back
    op.alter_column(
        "jobs",
        "created_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        type_=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # Alter started_at column back
    op.alter_column(
        "jobs",
        "started_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        type_=sa.DateTime(),
        existing_nullable=True,
        postgresql_using="started_at AT TIME ZONE 'UTC'",
    )

    # Alter completed_at column back
    op.alter_column(
        "jobs",
        "completed_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        type_=sa.DateTime(),
        existing_nullable=True,
        postgresql_using="completed_at AT TIME ZONE 'UTC'",
    )

    # Alter moved_to_dlq_at column back
    op.alter_column(
        "jobs",
        "moved_to_dlq_at",
        existing_type=sa.TIMESTAMP(timezone=True),
        type_=sa.DateTime(),
        existing_nullable=True,
        postgresql_using="moved_to_dlq_at AT TIME ZONE 'UTC'",
    )
