"""Database models and Pydantic schemas for the job queue."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase


# =============================================================================
# Enums
# =============================================================================
class JobStatus(str, Enum):
    """Possible states for a job in the queue."""
    PENDING = "pending"      # Waiting to be processed
    RUNNING = "running"      # Currently being processed by a worker
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"        # Failed after all retry attempts
    CANCELLED = "cancelled"  # Cancelled by user
    DEAD = "dead"            # Moved to dead letter queue


class JobType(str, Enum):
    """Supported job types."""
    LLM_COMPLETION = "llm.completion"  # Single prompt completion
    LLM_BATCH = "llm.batch"            # Batch of prompts processed concurrently


# =============================================================================
# SQLAlchemy Models
# =============================================================================
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class JobRecord(Base):
    """
    Database model for jobs.

    This represents the persistent state of a job in the queue.
    Jobs go through the following state machine:

        PENDING -> RUNNING -> COMPLETED
                          \\-> FAILED -> DEAD (after max retries)
                          \\-> PENDING (retry)
                          \\-> CANCELLED
        PENDING -> CANCELLED
    """
    __tablename__ = "jobs"

    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Job definition
    job_type = Column(String(50), nullable=False, index=True)
    input_data = Column(JSONB, nullable=False)

    # State machine
    status = Column(
        String(20),
        nullable=False,
        default=JobStatus.PENDING.value,
        index=True,
    )
    attempt = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)

    # Results
    result = Column(JSONB, nullable=True)
    error = Column(JSONB, nullable=True)

    # Metrics
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Integer, nullable=True)

    # Timestamps - Use TIMESTAMP WITH TIME ZONE for timezone-aware datetimes
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        # lambda makes this a callable so that this gets evaluated for every new job created
        default=lambda: datetime.now(UTC),
        index=True,
    )
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Worker tracking
    worker_id = Column(String(50), nullable=True)

    # Dead letter queue metadata
    moved_to_dlq_at = Column(TIMESTAMP(timezone=True), nullable=True)
    dlq_reason = Column(Text, nullable=True)

    # Table-level constraints and indexes
    __table_args__ = (
        # Partial index for efficient pending job queries
        Index(
            "idx_jobs_pending_created",
            "status",
            "created_at",
            postgresql_where=(Column("status") == JobStatus.PENDING.value),
        ),
        # Composite index for filtering by type and status
        Index("idx_jobs_type_status", "job_type", "status"),
        # Ensure status is always valid
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'dead')",
            name="valid_status",
        ),
    )

    def __repr__(self) -> str:
        return f"<Job {self.id} type={self.job_type} status={self.status}>"


# =============================================================================
# Pydantic Schemas - Input Validation
# =============================================================================
class CompletionInput(BaseModel):
    """Input schema for LLM completion jobs."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The prompt to send to the LLM",
    )
    model: str = Field(
        default="gpt-4o-mini",
        max_length=100,
        description="Model to use for completion",
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=4000,
        description="Maximum tokens in response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )


class BatchItem(BaseModel):
    """Single item in a batch job."""
    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for this item",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="The prompt for this item",
    )


class BatchInput(BaseModel):
    """Input schema for LLM batch jobs."""
    items: list[BatchItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of items to process",
    )
    model: str = Field(
        default="gpt-4o-mini",
        max_length=100,
        description="Model to use for all items",
    )
    max_tokens: int = Field(
        default=500,
        ge=1,
        le=4000,
        description="Maximum tokens per response",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )


# =============================================================================
# Pydantic Schemas - API Request/Response
# =============================================================================
class JobCreate(BaseModel):
    """Request schema for creating a new job."""
    job_type: JobType = Field(
        ...,
        description="Type of job to create",
    )
    input_data: dict[str, Any] = Field(
        ...,
        description="Input data for the job (schema depends on job_type)",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retry attempts",
    )

    @field_validator("input_data")
    @classmethod
    def validate_input_data(cls, v: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        """Validate that input_data matches the expected schema for job_type."""
        job_type = info.data.get("job_type")

        if job_type == JobType.LLM_COMPLETION:
            # Validate against CompletionInput schema
            CompletionInput(**v)
        elif job_type == JobType.LLM_BATCH:
            # Validate against BatchInput schema
            BatchInput(**v)

        return v


class JobResponse(BaseModel):
    """Response schema for job data."""
    id: UUID
    job_type: JobType
    input_data: dict[str, Any]
    status: JobStatus
    attempt: int
    max_attempts: int
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    duration_ms: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    worker_id: str | None = None

     # Pydantic V2 configuration using model_config
    model_config = {
        "from_attributes": True, # Allow creating from SQLAlchemy models
    }


class JobSubmitResponse(BaseModel):
    """Response schema after submitting a job."""
    id: UUID
    job_type: JobType
    status: JobStatus
    message: str = "Job submitted successfully"


class StatsResponse(BaseModel):
    """System statistics response"""
    total_jobs: int
    jobs_by_status: dict[str, int]
    jobs_by_type: dict[str, int]
    total_tokens: int
    total_cost_usd: float
    avg_duration_ms: float | None
    success_rate: float
    queue_depth: int
    dlq_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str
    redis: str
    version: str = "0.1.0"


# =============================================================================
# Quick Test
# =============================================================================
async def _test_models() -> None:
    """Quick test of model functionality."""

    # Test CompletionInput validation
    completion = CompletionInput(prompt="Hello, world!")
    print(f"CompletionInput: {completion}")

    # Test BatchInput validation
    batch = BatchInput(
        items=[
            BatchItem(id="1", prompt="Question 1"),
            BatchItem(id="2", prompt="Question 2"),
        ]
    )
    print(f"BatchInput: {batch}")

    # Test JobCreate validation
    job = JobCreate(
        job_type=JobType.LLM_COMPLETION,
        input_data={"prompt": "Test prompt"},
    )
    print(f"JobCreate: {job}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_models())
