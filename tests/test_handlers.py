"""Tests for job handlers.

These tests verify handler functionality including
job execution, error handling, and result formatting.
"""

from uuid import uuid4

import pytest

from nexus.handlers import (
    HANDLERS,
    BatchHandler,
    CompletionHandler,
    HandlerResult,
    get_handler,
    list_handlers,
)
from nexus.models import JobRecord, JobType
from nexus.providers import MockLLMProvider


class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = HandlerResult(
            success=True,
            result={"content": "Test"},
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            duration_ms=100,
        )

        assert result.success is True
        assert result.result["content"] == "Test"
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = HandlerResult(
            success=False,
            error={"type": "TestError", "message": "Test failed"},
        )

        assert result.success is False
        assert result.result is None
        assert result.error["type"] == "TestError"

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = HandlerResult(
            success=True,
            result={"content": "Test"},
            total_tokens=30,
            cost_usd=0.001,
            duration_ms=100,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["result"]["content"] == "Test"
        assert data["total_tokens"] == 30


class TestCompletionHandler:
    """Tests for CompletionHandler."""

    @pytest.fixture
    def provider(self):
        """Get mock provider for testing."""
        return MockLLMProvider(min_latency_ms=1, max_latency_ms=10)

    @pytest.fixture
    def handler(self, provider):
        """Get completion handler with mock provider."""
        return CompletionHandler(provider=provider)

    @pytest.fixture
    def completion_job(self):
        """Create a sample completion job."""
        return JobRecord(
            id=uuid4(),
            job_type=JobType.LLM_COMPLETION.value,
            input_data={
                "prompt": "What is Python?",
                "model": "gpt-4o-mini",
                "max_tokens": 500,
                "temperature": 0.7,
            },
        )

    def test_job_type(self, handler):
        """Test handler job type property."""
        assert handler.job_type == JobType.LLM_COMPLETION

    @pytest.mark.asyncio
    async def test_execute_success(self, handler, completion_job):
        """Test successful job execution."""
        result = await handler.execute(completion_job)

        assert result.success is True
        assert result.result is not None
        assert "content" in result.result
        assert "model" in result.result
        assert result.total_tokens > 0
        assert result.cost_usd > 0
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_returns_content(self, handler, completion_job):
        """Test that execution returns generated content."""
        result = await handler.execute(completion_job)

        assert len(result.result["content"]) > 0

    @pytest.mark.asyncio
    async def test_execute_tracks_tokens(self, handler, completion_job):
        """Test that execution tracks token counts."""
        result = await handler.execute(completion_job)

        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.total_tokens == result.input_tokens + result.output_tokens

    @pytest.mark.asyncio
    async def test_execute_with_provider_error(self, completion_job):
        """Test handling of provider errors."""
        failing_provider = MockLLMProvider(failure_rate=1.0)
        handler = CompletionHandler(provider=failing_provider)

        result = await handler.execute(completion_job)

        assert result.success is False
        assert result.error is not None
        assert "type" in result.error
        assert "message" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_invalid_input(self, handler):
        """Test handling of invalid input data."""
        invalid_job = JobRecord(
            id=uuid4(),
            job_type=JobType.LLM_COMPLETION.value,
            input_data={
                # Missing required 'prompt' field
                "model": "gpt-4o-mini",
            },
        )

        result = await handler.execute(invalid_job)

        assert result.success is False
        assert result.error is not None


class TestBatchHandler:
    """Tests for BatchHandler."""

    @pytest.fixture
    def provider(self):
        """Get mock provider for testing."""
        return MockLLMProvider(min_latency_ms=1, max_latency_ms=10)

    @pytest.fixture
    def handler(self, provider):
        """Get batch handler with mock provider."""
        return BatchHandler(provider=provider)

    @pytest.fixture
    def batch_job(self):
        """Create a sample batch job."""
        return JobRecord(
            id=uuid4(),
            job_type=JobType.LLM_BATCH.value,
            input_data={
                "items": [
                    {"id": "q1", "prompt": "What is Python?"},
                    {"id": "q2", "prompt": "What is FastAPI?"},
                    {"id": "q3", "prompt": "What is Redis?"},
                ],
                "model": "gpt-4o-mini",
                "max_tokens": 200,
                "temperature": 0.7,
            },
        )

    def test_job_type(self, handler):
        """Test handler job type property."""
        assert handler.job_type == JobType.LLM_BATCH

    @pytest.mark.asyncio
    async def test_execute_success(self, handler, batch_job):
        """Test successful batch execution."""
        result = await handler.execute(batch_job)

        assert result.success is True
        assert result.result is not None
        assert result.result["total_items"] == 3
        assert result.result["successful_count"] == 3
        assert result.result["failed_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_returns_all_items(self, handler, batch_job):
        """Test that all items are processed."""
        result = await handler.execute(batch_job)

        successful = result.result["successful"]
        assert len(successful) == 3

        ids = [item["id"] for item in successful]
        assert "q1" in ids
        assert "q2" in ids
        assert "q3" in ids

    @pytest.mark.asyncio
    async def test_execute_aggregates_tokens(self, handler, batch_job):
        """Test that tokens are aggregated across items."""
        result = await handler.execute(batch_job)

        # Should have tokens from all 3 items
        assert result.total_tokens > 0

        # Verify aggregation
        total_from_items = sum(
            item["input_tokens"] + item["output_tokens"]
            for item in result.result["successful"]
        )
        assert result.total_tokens == total_from_items

    @pytest.mark.asyncio
    async def test_execute_aggregates_cost(self, handler, batch_job):
        """Test that cost is aggregated across items."""
        result = await handler.execute(batch_job)

        assert result.cost_usd > 0

        # Verify aggregation
        total_cost_from_items = sum(
            item["cost_usd"] for item in result.result["successful"]
        )
        assert result.cost_usd == pytest.approx(total_cost_from_items, rel=0.01)

    @pytest.mark.asyncio
    async def test_execute_concurrent_processing(self, handler):
        """Test that batch processing is concurrent (faster than sequential)."""
        import time

        # Create job with many items
        large_batch_job = JobRecord(
            id=uuid4(),
            job_type=JobType.LLM_BATCH.value,
            input_data={
                "items": [
                    {"id": f"q{i}", "prompt": f"Question {i}"}
                    for i in range(10)
                ],
                "model": "gpt-4o-mini",
                "max_tokens": 100,
                "temperature": 0.7,
            },
        )

        # Use provider with noticeable latency
        slow_provider = MockLLMProvider(min_latency_ms=50, max_latency_ms=100)
        slow_handler = BatchHandler(provider=slow_provider)

        start = time.time()
        result = await slow_handler.execute(large_batch_job)
        duration = time.time() - start

        assert result.success is True

        # Sequential would take 10 * ~75ms = ~750ms
        # Concurrent should be much faster (limited by MAX_CONCURRENCY)
        # With MAX_CONCURRENCY=5, should take ~2 batches * ~75ms = ~150ms
        assert duration < 0.5  # Should be under 500ms

    @pytest.mark.asyncio
    async def test_execute_with_partial_failure(self, batch_job):
        """Test handling when some items fail."""
        # Create a provider that sometimes fails
        flaky_provider = MockLLMProvider(failure_rate=0.5)
        handler = BatchHandler(provider=flaky_provider)

        result = await handler.execute(batch_job)

        # With 50% failure rate, some should fail
        # Result reflects partial failure
        assert result.result["total_items"] == 3
        assert result.result["successful_count"] + result.result["failed_count"] == 3

    @pytest.mark.asyncio
    async def test_execute_with_invalid_input(self, handler):
        """Test handling of invalid input data."""
        invalid_job = JobRecord(
            id=uuid4(),
            job_type=JobType.LLM_BATCH.value,
            input_data={
                # Empty items list
                "items": [],
                "model": "gpt-4o-mini",
            },
        )

        result = await handler.execute(invalid_job)

        assert result.success is False
        assert result.error is not None


class TestHandlerRegistry:
    """Tests for handler registry functions."""

    def test_get_handler_completion(self):
        """Test getting completion handler."""
        handler = get_handler(JobType.LLM_COMPLETION)

        assert isinstance(handler, CompletionHandler)
        assert handler.job_type == JobType.LLM_COMPLETION

    def test_get_handler_batch(self):
        """Test getting batch handler."""
        handler = get_handler(JobType.LLM_BATCH)

        assert isinstance(handler, BatchHandler)
        assert handler.job_type == JobType.LLM_BATCH

    def test_get_handler_with_provider(self):
        """Test getting handler with custom provider."""
        provider = MockLLMProvider()
        handler = get_handler(JobType.LLM_COMPLETION, provider=provider)

        assert handler.provider is provider

    def test_get_handler_invalid_type(self):
        """Test getting handler for invalid type raises error."""
        with pytest.raises(ValueError) as exc_info:
            get_handler("invalid.type")

        assert "No handler registered" in str(exc_info.value)

    def test_list_handlers(self):
        """Test listing registered handlers."""
        handlers = list_handlers()

        assert JobType.LLM_COMPLETION in handlers
        assert JobType.LLM_BATCH in handlers
        assert len(handlers) == 2

    def test_handlers_dict_matches_job_types(self):
        """Test that all job types have handlers."""
        for job_type in JobType:
            assert job_type in HANDLERS
