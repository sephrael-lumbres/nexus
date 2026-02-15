"""Tests for FastAPI REST API endpoints.

These tests verify API functionality including
job submission, status checks, and error handling.
"""

from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from nexus.api import app
from nexus.database import Database, JobRepository, get_database, reset_database
from nexus.models import JobRecord, JobStatus, JobType
from nexus.queue import JobQueue, get_queue, reset_queue


# =============================================================================
# Fixtures
# =============================================================================
@pytest_asyncio.fixture
async def db():
    """Get a clean database for testing."""
    reset_database()
    database = get_database()
    yield database
    await database.close()
    reset_database()


@pytest_asyncio.fixture
async def queue():
    """Get a clean queue for testing."""
    reset_queue()
    q = get_queue()
    await q.connect()
    await q.clear_all()
    yield q
    await q.clear_all()
    await q.disconnect()
    reset_queue()


@pytest_asyncio.fixture
async def client(db: Database, queue: JobQueue):
    """Get async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def completion_payload():
    """Sample completion job payload."""
    return {
        "job_type": "llm.completion",
        "input_data": {
            "prompt": "What is Python?",
            "model": "gpt-4o-mini",
            "max_tokens": 100,
        },
    }


@pytest.fixture
def batch_payload():
    """Sample batch job payload."""
    return {
        "job_type": "llm.batch",
        "input_data": {
            "items": [
                {"id": "q1", "prompt": "What is Redis?"},
                {"id": "q2", "prompt": "What is PostgreSQL?"},
            ],
            "model": "gpt-4o-mini",
            "max_tokens": 50,
        },
    }


# =============================================================================
# Health Endpoint Tests
# =============================================================================
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """Test health check returns healthy status."""
        response = await client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["redis"] == "connected"

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client: AsyncClient):
        """Test root endpoint returns API info."""
        response = await client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "Nexus Job Queue API"
        assert "version" in data
        assert "docs" in data


# =============================================================================
# Job Submission Tests
# =============================================================================
class TestJobSubmission:
    """Tests for job submission endpoint."""

    @pytest.mark.asyncio
    async def test_submit_completion_job(
        self,
        client: AsyncClient,
        completion_payload: dict,
    ):
        """Test submitting a completion job."""
        response = await client.post("/jobs", json=completion_payload)

        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert data["job_type"] == "llm.completion"
        assert data["status"] == "pending"
        assert data["message"] == "Job submitted successfully"

    @pytest.mark.asyncio
    async def test_submit_batch_job(
        self,
        client: AsyncClient,
        batch_payload: dict,
    ):
        """Test submitting a batch job."""
        response = await client.post("/jobs", json=batch_payload)

        assert response.status_code == 201

        data = response.json()
        assert data["job_type"] == "llm.batch"
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_submit_job_enqueues(
        self,
        client: AsyncClient,
        queue: JobQueue,
        completion_payload: dict,
    ):
        """Test that submitted job is enqueued."""
        initial_count = await queue.pending_count()

        await client.post("/jobs", json=completion_payload)

        assert await queue.pending_count() == initial_count + 1

    @pytest.mark.asyncio
    async def test_submit_job_invalid_type(self, client: AsyncClient):
        """Test submitting job with invalid type."""
        payload = {
            "job_type": "invalid.type",
            "input_data": {"prompt": "test"},
        }

        response = await client.post("/jobs", json=payload)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_submit_job_missing_input(self, client: AsyncClient):
        """Test submitting job with missing input data."""
        payload = {
            "job_type": "llm.completion",
            # Missing input_data
        }

        response = await client.post("/jobs", json=payload)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_job_invalid_input(self, client: AsyncClient):
        """Test submitting job with invalid input data."""
        payload = {
            "job_type": "llm.completion",
            "input_data": {
                "prompt": "",  # Empty prompt not allowed
                "model": "gpt-4o-mini",
            },
        }

        response = await client.post("/jobs", json=payload)

        assert response.status_code == 422


# =============================================================================
# Job Retrieval Tests
# =============================================================================
class TestJobRetrieval:
    """Tests for job retrieval endpoints."""

    @pytest.mark.asyncio
    async def test_get_job_success(
        self,
        client: AsyncClient,
        completion_payload: dict,
    ):
        """Test getting an existing job."""
        # Create job
        submit_response = await client.post("/jobs", json=completion_payload)
        job_id = submit_response.json()["id"]

        # Get job
        response = await client.get(f"/jobs/{job_id}")

        assert response.status_code == 200

        data = response.json()
        assert data["id"] == job_id
        assert data["job_type"] == "llm.completion"
        assert data["status"] == "pending"
        assert "input_data" in data
        assert "created_at" in data

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, client: AsyncClient):
        """Test getting a non-existent job."""
        fake_id = uuid4()

        response = await client.get(f"/jobs/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_job_invalid_uuid(self, client: AsyncClient):
        """Test getting job with invalid UUID."""
        response = await client.get("/jobs/invalid-uuid")

        assert response.status_code == 422


# =============================================================================
# Job Listing Tests
# =============================================================================
class TestJobListing:
    """Tests for job listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, client: AsyncClient):
        """Test listing jobs when none exist."""
        response = await client.get("/jobs")

        assert response.status_code == 200

        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_jobs_returns_all(
        self,
        client: AsyncClient,
        completion_payload: dict,
    ):
        """Test listing returns all jobs."""
        # Create multiple jobs
        for _ in range(3):
            await client.post("/jobs", json=completion_payload)

        response = await client.get("/jobs")

        assert response.status_code == 200

        data = response.json()
        assert len(data["jobs"]) == 3
        assert data["total"] == 3

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_status(
        self,
        client: AsyncClient,
        db: Database,
        completion_payload: dict,
    ):
        """Test filtering jobs by status."""
        # Create a pending job via API
        await client.post("/jobs", json=completion_payload)

        # Create a completed job directly in database
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "test", "model": "gpt-4o-mini"},
                status=JobStatus.COMPLETED.value,
            )
            await repo.create(job)

        # Filter by pending
        response = await client.get("/jobs?status=pending")

        assert response.status_code == 200

        data = response.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_type(
        self,
        client: AsyncClient,
        completion_payload: dict,
        batch_payload: dict,
    ):
        """Test filtering jobs by type."""
        # Create both types
        await client.post("/jobs", json=completion_payload)
        await client.post("/jobs", json=batch_payload)

        # Filter by completion
        response = await client.get("/jobs?type=llm.completion")

        assert response.status_code == 200

        data = response.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_type"] == "llm.completion"

    @pytest.mark.asyncio
    async def test_list_jobs_pagination(
        self,
        client: AsyncClient,
        completion_payload: dict,
    ):
        """Test job listing pagination."""
        # Create 5 jobs
        for _ in range(5):
            await client.post("/jobs", json=completion_payload)

        # Get first 2
        response = await client.get("/jobs?limit=2&offset=0")

        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0

        # Get next 2
        response = await client.get("/jobs?limit=2&offset=2")

        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["offset"] == 2


# =============================================================================
# Job Cancellation Tests
# =============================================================================
class TestJobCancellation:
    """Tests for job cancellation endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_pending_job(
        self,
        client: AsyncClient,
        completion_payload: dict,
    ):
        """Test cancelling a pending job."""
        # Create job
        submit_response = await client.post("/jobs", json=completion_payload)
        job_id = submit_response.json()["id"]

        # Cancel job
        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 204

        # Verify cancelled
        get_response = await client.get(f"/jobs/{job_id}")
        assert get_response.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, client: AsyncClient):
        """Test cancelling a non-existent job."""
        fake_id = uuid4()

        response = await client.delete(f"/jobs/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_completed_job_fails(
        self,
        client: AsyncClient,
        db: Database,
    ):
        """Test that completed jobs cannot be cancelled."""
        # Create completed job
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "test", "model": "gpt-4o-mini"},
                status=JobStatus.COMPLETED.value,
            )
            job = await repo.create(job)
            job_id = job.id

        # Try to cancel
        response = await client.delete(f"/jobs/{job_id}")

        assert response.status_code == 409
        assert "cannot cancel" in response.json()["detail"].lower()


# =============================================================================
# Statistics Tests
# =============================================================================
class TestStatistics:
    """Tests for statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, client: AsyncClient):
        """Test stats with no jobs."""
        response = await client.get("/stats")

        assert response.status_code == 200

        data = response.json()
        assert data["total_jobs"] == 0
        assert data["queue_depth"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_with_jobs(
        self,
        client: AsyncClient,
        db: Database,
        completion_payload: dict,
    ):
        """Test stats with jobs."""
        # Create jobs via API
        for _ in range(3):
            await client.post("/jobs", json=completion_payload)

        # Create completed job with metrics
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "test", "model": "gpt-4o-mini"},
                status=JobStatus.COMPLETED.value,
                total_tokens=100,
                cost_usd=0.001,
            )
            await repo.create(job)

        response = await client.get("/stats")

        data = response.json()
        assert data["total_jobs"] == 4
        assert data["queue_depth"] == 3  # 3 pending
        assert data["total_tokens"] == 100
        assert data["total_cost_usd"] == 0.001


# =============================================================================
# DLQ Tests
# =============================================================================
class TestDLQ:
    """Tests for dead letter queue endpoints."""

    @pytest.mark.asyncio
    async def test_list_dlq_empty(self, client: AsyncClient):
        """Test listing empty DLQ."""
        response = await client.get("/dlq")

        assert response.status_code == 200

        data = response.json()
        assert data["entries"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_dlq_with_entries(
        self,
        client: AsyncClient,
        queue: JobQueue,
    ):
        """Test listing DLQ with entries."""
        # Add job to DLQ
        job_id = uuid4()
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()
        await queue.move_to_dlq(
            job_id,
            error={"type": "TestError"},
            reason="Test failure",
        )

        response = await client.get("/dlq")

        assert response.status_code == 200

        data = response.json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["job_id"] == str(job_id)
        assert data["entries"][0]["reason"] == "Test failure"

    @pytest.mark.asyncio
    async def test_replay_from_dlq(
        self,
        client: AsyncClient,
        db: Database,
        queue: JobQueue,
    ):
        """Test replaying a job from DLQ."""
        # Create job in database
        async with db.session() as session:
            repo = JobRepository(session)
            job = JobRecord(
                job_type=JobType.LLM_COMPLETION.value,
                input_data={"prompt": "test", "model": "gpt-4o-mini"},
                status=JobStatus.DEAD.value,
            )
            job = await repo.create(job)
            job_id = job.id

        # Add to queue DLQ
        await queue.enqueue(job_id)
        await queue.dequeue_nonblocking()
        await queue.move_to_dlq(job_id, reason="Test")

        # Replay
        response = await client.post(f"/dlq/{job_id}/replay")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "pending"
        assert "replayed" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_replay_nonexistent_dlq_entry(self, client: AsyncClient):
        """Test replaying job not in DLQ."""
        fake_id = uuid4()

        response = await client.post(f"/dlq/{fake_id}/replay")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_clear_dlq(
        self,
        client: AsyncClient,
        queue: JobQueue,
    ):
        """Test clearing the DLQ."""
        # Add jobs to DLQ
        for _ in range(3):
            job_id = uuid4()
            await queue.enqueue(job_id)
            await queue.dequeue_nonblocking()
            await queue.move_to_dlq(job_id)

        assert await queue.dlq_count() == 3

        # Clear DLQ
        response = await client.delete("/dlq")

        assert response.status_code == 204
        assert await queue.dlq_count() == 0


# =============================================================================
# Queue Endpoint Tests
# =============================================================================
class TestQueueEndpoints:
    """Tests for queue management endpoints."""

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, client: AsyncClient):
        """Test getting queue statistics."""
        response = await client.get("/queue/stats")

        assert response.status_code == 200

        data = response.json()
        assert "pending" in data
        assert "processing" in data
        assert "dlq" in data

    @pytest.mark.asyncio
    async def test_peek_pending(
        self,
        client: AsyncClient,
        completion_payload: dict,
    ):
        """Test peeking at pending jobs."""
        # Create jobs
        for _ in range(3):
            await client.post("/jobs", json=completion_payload)

        response = await client.get("/queue/pending?count=2")

        assert response.status_code == 200

        data = response.json()
        assert data["pending_count"] == 3
        assert len(data["peeked"]) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================
class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json(self, client: AsyncClient):
        """Test handling of invalid JSON."""
        response = await client.post(
            "/jobs",
            content="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_method_not_allowed(self, client: AsyncClient):
        """Test handling of unsupported methods."""
        response = await client.put("/jobs")

        assert response.status_code == 405
