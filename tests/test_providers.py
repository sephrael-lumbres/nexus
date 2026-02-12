"""Tests for LLM providers.

These tests verify provider functionality including
response generation, token counting, and cost calculation.
"""

import pytest

from nexus.config import LLMProvider
from nexus.providers import (
    LLMAuthenticationError,
    LLMProviderError,
    LLMResponse,
    MockLLMProvider,
    OpenAIProvider,
    get_provider,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Test content",
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            duration_ms=100,
        )

        assert response.content == "Test content"
        assert response.model == "gpt-4o-mini"
        assert response.total_tokens == 30

    def test_to_dict(self):
        """Test converting response to dictionary."""
        response = LLMResponse(
            content="Test",
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            duration_ms=100,
        )

        data = response.to_dict()

        assert data["content"] == "Test"
        assert data["model"] == "gpt-4o-mini"
        assert data["total_tokens"] == 30


class TestMockLLMProvider:
    """Tests for MockLLMProvider."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        """Test that complete returns a valid response."""
        provider = MockLLMProvider()

        response = await provider.complete(
            prompt="Hello, world!",
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_complete_returns_mock_model(self):
        """Test that mock provider returns mock model name."""
        provider = MockLLMProvider()

        response = await provider.complete(
            prompt="Test",
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7,
        )

        assert response.model.startswith("mock-")

    @pytest.mark.asyncio
    async def test_complete_counts_tokens(self):
        """Test that tokens are counted correctly."""
        provider = MockLLMProvider()

        response = await provider.complete(
            prompt="This is a test prompt with several words.",
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7,
        )

        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.total_tokens == response.input_tokens + response.output_tokens

    @pytest.mark.asyncio
    async def test_complete_calculates_cost(self):
        """Test that cost is calculated."""
        provider = MockLLMProvider()

        response = await provider.complete(
            prompt="Test prompt",
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7,
        )

        assert response.cost_usd > 0

    @pytest.mark.asyncio
    async def test_complete_respects_max_tokens(self):
        """Test that output respects max_tokens limit."""
        provider = MockLLMProvider()

        response = await provider.complete(
            prompt="Write a very long essay about everything.",
            model="gpt-4o-mini",
            max_tokens=50,
            temperature=0.7,
        )

        # Should be roughly within the limit
        assert response.output_tokens <= 60  # Some tolerance

    @pytest.mark.asyncio
    async def test_complete_tracks_duration(self):
        """Test that duration is tracked."""
        provider = MockLLMProvider(min_latency_ms=10, max_latency_ms=50)

        response = await provider.complete(
            prompt="Test",
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7,
        )

        assert response.duration_ms >= 10

    @pytest.mark.asyncio
    async def test_complete_with_failure_rate(self):
        """Test that failure rate causes errors."""
        provider = MockLLMProvider(failure_rate=1.0)  # Always fail

        with pytest.raises(LLMProviderError):
            await provider.complete(
                prompt="Test",
                model="gpt-4o-mini",
                max_tokens=100,
                temperature=0.7,
            )

    @pytest.mark.asyncio
    async def test_topic_specific_response(self):
        """Test that topic-specific responses are used."""
        provider = MockLLMProvider()

        response = await provider.complete(
            prompt="What is Redis?",
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.7,
        )

        # Should contain Redis-specific content
        assert "redis" in response.content.lower()


class TestMockLLMProviderTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens_simple(self):
        """Test counting tokens in simple text."""
        provider = MockLLMProvider()

        tokens = provider.count_tokens("Hello, world!")

        assert tokens > 0
        assert tokens < 10  # Should be a few tokens

    def test_count_tokens_longer_text(self):
        """Test counting tokens in longer text."""
        provider = MockLLMProvider()

        short_text = "Hello"
        long_text = "Hello " * 100

        short_tokens = provider.count_tokens(short_text)
        long_tokens = provider.count_tokens(long_text)

        assert long_tokens > short_tokens

    def test_count_tokens_code(self):
        """Test counting tokens in code."""
        provider = MockLLMProvider()

        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        tokens = provider.count_tokens(code)

        assert tokens > 0


class TestMockLLMProviderCostCalculation:
    """Tests for cost calculation functionality."""

    def test_calculate_cost_gpt4o_mini(self):
        """Test cost calculation for gpt-4o-mini."""
        provider = MockLLMProvider()

        # gpt-4o-mini: $0.00015/1K input, $0.0006/1K output
        cost = provider.calculate_cost("gpt-4o-mini", 1000, 1000)

        expected = (1000 * 0.00015 / 1000) + (1000 * 0.0006 / 1000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for gpt-4o."""
        provider = MockLLMProvider()

        # gpt-4o: $0.005/1K input, $0.015/1K output
        cost = provider.calculate_cost("gpt-4o", 1000, 1000)

        expected = (1000 * 0.005 / 1000) + (1000 * 0.015 / 1000)
        assert cost == pytest.approx(expected, rel=0.01)

    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model uses default."""
        provider = MockLLMProvider()

        cost = provider.calculate_cost("unknown-model", 1000, 1000)

        # Should use default pricing
        assert cost > 0

    def test_get_pricing_known_model(self):
        """Test getting pricing for known model."""
        provider = MockLLMProvider()

        input_price, output_price = provider.get_pricing("gpt-4o-mini")

        assert input_price == 0.00015
        assert output_price == 0.0006

    def test_get_pricing_unknown_model(self):
        """Test getting pricing for unknown model returns default."""
        provider = MockLLMProvider()

        input_price, output_price = provider.get_pricing("unknown-model")

        assert input_price == provider.DEFAULT_PRICING[0]
        assert output_price == provider.DEFAULT_PRICING[1]


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises custom error."""
        from unittest.mock import MagicMock, patch

        # Create a fake settings object with no OpenAI API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = None

        # Patch get_settings so the provider sees our fake settings,
        # regardless of env vars or .env file
        with patch("nexus.providers.get_settings", return_value=mock_settings):
            with pytest.raises(LLMAuthenticationError):
                OpenAIProvider(api_key=None)

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = OpenAIProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.client is not None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing the provider."""
        provider = OpenAIProvider(api_key="test-key")

        await provider.close()
        # Client should now be closed (can't easily verify, but shouldn't raise)


class TestGetProvider:
    """Tests for get_provider factory function."""

    def test_get_mock_provider(self):
        """Test getting mock provider."""
        provider = get_provider(LLMProvider.MOCK)

        assert isinstance(provider, MockLLMProvider)

    def test_get_openai_provider_with_key(self):
        """Test getting OpenAI provider with explicit key."""
        provider = get_provider(LLMProvider.OPENAI, api_key="test-key")

        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_with_kwargs(self):
        """Test passing kwargs to provider."""
        provider = get_provider(
            LLMProvider.MOCK,
            min_latency_ms=10,
            max_latency_ms=20,
            failure_rate=0.5,
        )

        assert isinstance(provider, MockLLMProvider)
        assert provider.min_latency_ms == 10
        assert provider.max_latency_ms == 20
        assert provider.failure_rate == 0.5
