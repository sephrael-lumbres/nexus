"""
Tests for Pydantic models and validation.

These tests ensure our API schemas properly validate
input data and catch errors before they reach the database.
"""

import pytest
from pydantic import ValidationError

from nexus.models import (
    BatchInput,
    BatchItem,
    CompletionInput,
    JobCreate,
    JobStatus,
    JobType,
)


class TestCompletionInput:
    """Tests for CompletionInput schema."""

    def test_valid_minimal_input(self):
        """Test with only required fields."""
        data = CompletionInput(prompt="Hello, world!")

        assert data.prompt == "Hello, world!"
        assert data.model == "gpt-4o-mini"  # default
        assert data.max_tokens == 500  # default
        assert data.temperature == 0.7  # default

    def test_valid_full_input(self):
        """Test with all fields specified."""
        data = CompletionInput(
            prompt="Explain quantum computing",
            model="gpt-4-turbo",
            max_tokens=1000,
            temperature=0.9,
        )

        assert data.prompt == "Explain quantum computing"
        assert data.model == "gpt-4-turbo"
        assert data.max_tokens == 1000
        assert data.temperature == 0.9

    def test_empty_prompt_rejected(self):
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CompletionInput(prompt="")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("prompt",) for e in errors)

    def test_prompt_too_long_rejected(self):
        """Test that overly long prompt is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CompletionInput(prompt="x" * 100001)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("prompt",) for e in errors)

    def test_max_tokens_bounds(self):
        """Test max_tokens validation bounds."""
        # Valid: at minimum
        data = CompletionInput(prompt="test", max_tokens=1)
        assert data.max_tokens == 1

        # Valid: at maximum
        data = CompletionInput(prompt="test", max_tokens=4000)
        assert data.max_tokens == 4000

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            CompletionInput(prompt="test", max_tokens=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            CompletionInput(prompt="test", max_tokens=4001)

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid: at minimum
        data = CompletionInput(prompt="test", temperature=0.0)
        assert data.temperature == 0.0

        # Valid: at maximum
        data = CompletionInput(prompt="test", temperature=2.0)
        assert data.temperature == 2.0

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            CompletionInput(prompt="test", temperature=-0.1)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            CompletionInput(prompt="test", temperature=2.1)


class TestBatchInput:
    """Tests for BatchInput schema."""

    def test_valid_batch_input(self):
        """Test valid batch input with multiple items."""
        data = BatchInput(
            items=[
                BatchItem(id="1", prompt="Question 1"),
                BatchItem(id="2", prompt="Question 2"),
            ]
        )

        assert len(data.items) == 2
        assert data.items[0].id == "1"
        assert data.items[1].prompt == "Question 2"

    def test_empty_items_rejected(self):
        """Test that empty items list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            BatchInput(items=[])

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("items",) for e in errors)

    def test_too_many_items_rejected(self):
        """Test that more than 100 items is rejected."""
        items = [BatchItem(id=str(i), prompt=f"Q{i}") for i in range(101)]

        with pytest.raises(ValidationError) as exc_info:
            BatchInput(items=items)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("items",) for e in errors)

    def test_duplicate_ids_allowed(self):
        """Test that duplicate IDs are allowed (no unique constraint)."""
        # This should not raise - uniqueness is application logic
        data = BatchInput(
            items=[
                BatchItem(id="same", prompt="Q1"),
                BatchItem(id="same", prompt="Q2"),
            ]
        )
        assert len(data.items) == 2


class TestBatchItem:
    """Tests for BatchItem schema."""

    def test_valid_item(self):
        """Test valid batch item."""
        item = BatchItem(id="test-123", prompt="What is Python?")

        assert item.id == "test-123"
        assert item.prompt == "What is Python?"

    def test_empty_id_rejected(self):
        """Test that empty ID is rejected."""
        with pytest.raises(ValidationError):
            BatchItem(id="", prompt="test")

    def test_id_too_long_rejected(self):
        """Test that overly long ID is rejected."""
        with pytest.raises(ValidationError):
            BatchItem(id="x" * 101, prompt="test")


class TestJobCreate:
    """Tests for JobCreate schema."""

    def test_valid_completion_job(self):
        """Test creating a completion job."""
        job = JobCreate(
            job_type=JobType.LLM_COMPLETION,
            input_data={"prompt": "Hello"},
        )

        assert job.job_type == JobType.LLM_COMPLETION
        assert job.input_data["prompt"] == "Hello"
        assert job.max_attempts == 3  # default

    def test_valid_batch_job(self):
        """Test creating a batch job."""
        job = JobCreate(
            job_type=JobType.LLM_BATCH,
            input_data={
                "items": [
                    {"id": "1", "prompt": "Q1"},
                    {"id": "2", "prompt": "Q2"},
                ]
            },
        )

        assert job.job_type == JobType.LLM_BATCH
        assert len(job.input_data["items"]) == 2

    def test_invalid_completion_input_rejected(self):
        """Test that invalid completion input is rejected."""
        with pytest.raises(ValidationError):
            JobCreate(
                job_type=JobType.LLM_COMPLETION,
                input_data={"prompt": ""},  # Empty prompt
            )

    def test_invalid_batch_input_rejected(self):
        """Test that invalid batch input is rejected."""
        with pytest.raises(ValidationError):
            JobCreate(
                job_type=JobType.LLM_BATCH,
                input_data={"items": []},  # Empty items
            )

    def test_max_attempts_bounds(self):
        """Test max_attempts validation."""
        # Valid
        job = JobCreate(
            job_type=JobType.LLM_COMPLETION,
            input_data={"prompt": "test"},
            max_attempts=1,
        )
        assert job.max_attempts == 1

        # Invalid: too low
        with pytest.raises(ValidationError):
            JobCreate(
                job_type=JobType.LLM_COMPLETION,
                input_data={"prompt": "test"},
                max_attempts=0,
            )

        # Invalid: too high
        with pytest.raises(ValidationError):
            JobCreate(
                job_type=JobType.LLM_COMPLETION,
                input_data={"prompt": "test"},
                max_attempts=11,
            )


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        expected = {"pending", "running", "completed", "failed", "cancelled", "dead"}
        actual = {s.value for s in JobStatus}

        assert actual == expected

    def test_status_string_values(self):
        """Test that status values are lowercase strings."""
        for status in JobStatus:
            assert status.value == status.value.lower()
            assert isinstance(status.value, str)


class TestJobType:
    """Tests for JobType enum."""

    def test_all_types_defined(self):
        """Test that all expected types are defined."""
        expected = {"llm.completion", "llm.batch"}
        actual = {t.value for t in JobType}

        assert actual == expected
