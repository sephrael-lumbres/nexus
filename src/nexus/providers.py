"""LLM provider abstraction layer.

This module provides:
- Base class defining the LLM interface
- Mock provider for testing and demos (no API costs)
- OpenAI provider for production use
- Token counting and cost calculation

Usage:
    # Get provider based on settings
    provider = get_provider()

    # Generate completion
    response = await provider.complete(
        prompt="Explain microservices",
        model="gpt-4o-mini",
        max_tokens=500,
        temperature=0.7,
    )

    print(response.content)
    print(f"Cost: ${response.cost_usd}")
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import structlog
import tiktoken

from nexus.config import LLMProvider, get_settings

logger = structlog.get_logger()


# =============================================================================
# Response Dataclass
# =============================================================================
@dataclass
class LLMResponse:
    """Standardized response from any LLM provider.

    Attributes:
        content: Generated text content
        model: Model that generated the response
        input_tokens: Number of tokens in the prompt
        output_tokens: Number of tokens in the response
        total_tokens: Sum of input and output tokens
        cost_usd: Estimated cost in USD
        duration_ms: Time taken for the request in milliseconds
    """
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# Base Provider Class
# =============================================================================
class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All providers must implement the `complete` method and can
    optionally override the pricing and token counting methods.
    """

    # Pricing per 1K tokens (input_price, output_price) in USD
    # Updated as of 2024 - verify current pricing at platform websites
    PRICING: dict[str, tuple[float, float]] = {
        # OpenAI models
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4": (0.03, 0.06),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        # Mock models (same pricing as real for accurate simulation)
        "mock-gpt-4o": (0.005, 0.015),
        "mock-gpt-4o-mini": (0.00015, 0.0006),
        "mock-gpt-4-turbo": (0.01, 0.03),
    }

    # Default pricing for unknown models
    DEFAULT_PRICING: tuple[float, float] = (0.001, 0.002)

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt
            model: Model identifier (e.g., "gpt-4o-mini")
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0)

        Returns:
            LLMResponse with generated content and metrics

        Raises:
            LLMProviderError: If the request fails
        """
        pass

    def get_pricing(self, model: str) -> tuple[float, float]:
        """Get pricing for a model.

        Args:
            model: Model identifier

        Returns:
            Tuple of (input_price_per_1k, output_price_per_1k)
        """
        return self.PRICING.get(model, self.DEFAULT_PRICING)

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost in USD for a completion.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD (rounded to 6 decimal places)
        """
        input_price, output_price = self.get_pricing(model)
        cost = (input_tokens * input_price / 1000) + (output_tokens * output_price / 1000)
        return round(cost, 6)

    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization

        Returns:
            Number of tokens
        """
        try:
            # Try to get encoding for specific model
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base (used by GPT-4 and GPT-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))


# =============================================================================
# Custom Exceptions
# =============================================================================
class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when rate limited by the provider."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when authentication fails."""
    pass


class LLMInvalidRequestError(LLMProviderError):
    """Raised when the request is invalid."""
    pass


# =============================================================================
# Mock Provider
# =============================================================================
class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing and demos.

    Generates realistic-looking responses without making
    actual API calls. Useful for:
    - Local development without API costs
    - Testing job queue functionality
    - Demo presentations
    - Load testing

    Features:
    - Realistic response timing (configurable)
    - Accurate token counting
    - Cost calculation matching real pricing
    - Configurable failure rate for testing retries
    """

    # Pre-defined responses for realistic demos
    RESPONSES: list[str] = [
        "Based on my analysis, the key considerations are: First, we need to "
        "understand the fundamental principles at play. Second, the practical "
        "implications should be carefully evaluated. Third, implementation "
        "requires a systematic approach that accounts for edge cases and "
        "potential failure modes.",

        "This is an interesting question that touches on several important "
        "concepts. Let me break it down into manageable components and address "
        "each one systematically. The core issue relates to how systems interact "
        "and maintain consistency under various conditions.",

        "To provide a comprehensive answer, I'll examine this from multiple "
        "perspectives: technical feasibility, practical implementation, and "
        "long-term sustainability. Each aspect contributes to the overall "
        "solution architecture.",

        "The solution involves several interconnected steps. Starting with the "
        "foundation, we establish core principles. Building upon that, we "
        "implement specific mechanisms. Finally, we validate through testing "
        "and iteration to ensure reliability.",

        "When approaching this problem, it's essential to consider both the "
        "immediate requirements and future scalability. The architecture should "
        "be flexible enough to accommodate changes while maintaining performance "
        "and reliability standards.",
    ]

    # Topic-specific responses for more realistic demos
    TOPIC_RESPONSES: dict[str, str] = {
        "microservices": (
            "Microservices architecture is a design approach where an application "
            "is built as a collection of loosely coupled, independently deployable "
            "services. Each service focuses on a specific business capability and "
            "communicates via well-defined APIs. Key benefits include independent "
            "scaling, technology flexibility, and fault isolation."
        ),
        "redis": (
            "Redis is an open-source, in-memory data structure store used as a "
            "database, cache, message broker, and queue. It supports various data "
            "structures like strings, hashes, lists, sets, and sorted sets. Redis "
            "is known for its exceptional performance, typically achieving "
            "sub-millisecond response times."
        ),
        "postgresql": (
            "PostgreSQL is a powerful, open-source object-relational database "
            "system with over 35 years of active development. It's known for "
            "reliability, feature robustness, and performance. Key features "
            "include ACID compliance, MVCC, complex queries, and extensibility."
        ),
        "fastapi": (
            "FastAPI is a modern, fast web framework for building APIs with "
            "Python based on standard Python type hints. It provides automatic "
            "API documentation, data validation, serialization, and async "
            "support. FastAPI is one of the fastest Python frameworks available."
        ),
        "docker": (
            "Docker is a platform for developing, shipping, and running "
            "applications in containers. Containers package an application with "
            "all its dependencies, ensuring consistency across environments. "
            "Docker simplifies deployment and enables microservices architectures."
        ),
        "kubernetes": (
            "Kubernetes is an open-source container orchestration platform that "
            "automates deploying, scaling, and managing containerized applications. "
            "It provides features like service discovery, load balancing, storage "
            "orchestration, and self-healing capabilities."
        ),
    }

    def __init__(
        self,
        min_latency_ms: int = 50,
        max_latency_ms: int = 200,
        failure_rate: float = 0.0,
    ):
        """Initialize mock provider.

        Args:
            min_latency_ms: Minimum simulated latency
            max_latency_ms: Maximum simulated latency
            failure_rate: Probability of simulated failure (0.0-1.0)
        """
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.failure_rate = failure_rate

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate a mock completion.

        Simulates realistic API behavior including latency
        and optional failures for testing retry logic.
        """
        start_time = time.time()

        # Simulate API latency
        latency_ms = random.randint(self.min_latency_ms, self.max_latency_ms)
        await asyncio.sleep(latency_ms / 1000)

        # Simulate random failures if configured
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            raise LLMProviderError("Simulated API error for testing")

        # Generate response
        response_text = self._generate_response(prompt, temperature)

        # Count tokens
        input_tokens = self.count_tokens(prompt, model)
        output_tokens = self.count_tokens(response_text, model)

        # Trim response if it exceeds max_tokens (approximate)
        if output_tokens > max_tokens:
            # Rough approximation: 1 token ≈ 4 characters
            char_limit = max_tokens * 4
            response_text = response_text[:char_limit].rsplit(" ", 1)[0] + "..."
            output_tokens = self.count_tokens(response_text, model)

        # Use mock model name
        mock_model = f"mock-{model}" if not model.startswith("mock-") else model

        duration_ms = int((time.time() - start_time) * 1000)

        logger.debug(
            "Mock completion generated",
            model=mock_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
        )

        return LLMResponse(
            content=response_text,
            model=mock_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=self.calculate_cost(mock_model, input_tokens, output_tokens),
            duration_ms=duration_ms,
        )

    def _generate_response(self, prompt: str, temperature: float) -> str:
        """Generate a contextually appropriate response."""
        prompt_lower = prompt.lower()

        # Check for topic-specific responses
        for topic, response in self.TOPIC_RESPONSES.items():
            if topic in prompt_lower:
                # Add some variation based on temperature
                if temperature > 0.5 and random.random() > 0.5:
                    response += (
                        f"\n\nAdditionally, when considering {topic}, "
                        "it's important to evaluate your specific use case "
                        "and requirements to make the best architectural decisions."
                    )
                return response

        # Fall back to generic response
        base_response = random.choice(self.RESPONSES)

        # Add context from prompt if temperature allows creativity
        if temperature > 0.5 and len(prompt) > 20:
            prompt_preview = prompt[:50].replace("\n", " ")
            base_response += (
                f"\n\nIn the context of your question about "
                f"'{prompt_preview}...', these principles apply directly "
                "to help solve the problem at hand."
            )

        return base_response


# =============================================================================
# OpenAI Provider
# =============================================================================
class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for production use.

    Provides access to GPT-4 and GPT-3.5 models via the
    OpenAI API. Handles authentication, retries, and error
    mapping to custom exceptions.

    Features:
    - Async HTTP client with connection pooling
    - Automatic retry with exponential backoff
    - Detailed error handling and logging
    - Token usage from API response
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to settings)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient errors
        """
        self.api_key = api_key or get_settings().openai_api_key

        if not self.api_key:
            raise LLMAuthenticationError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self.timeout = timeout
        self.max_retries = max_retries

        # Create async HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Generate completion using OpenAI API.

        Uses the chat completions endpoint with automatic
        retry for transient errors.
        """
        start_time = time.time()

        # Build request payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Retry loop for transient errors
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    "/chat/completions",
                    json=payload,
                )

                # Handle HTTP errors
                if response.status_code == 401:
                    raise LLMAuthenticationError("Invalid API key")
                elif response.status_code == 429:
                    raise LLMRateLimitError("Rate limit exceeded")
                elif response.status_code == 400:
                    error_data = response.json()
                    raise LLMInvalidRequestError(
                        error_data.get("error", {}).get("message", "Invalid request")
                    )
                elif response.status_code >= 500:
                    # Server error - retry
                    raise LLMProviderError(f"Server error: {response.status_code}")

                response.raise_for_status()

                # Parse response
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data["usage"]

                duration_ms = int((time.time() - start_time) * 1000)

                logger.debug(
                    "OpenAI completion generated",
                    model=model,
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    duration_ms=duration_ms,
                )

                return LLMResponse(
                    content=content,
                    model=model,
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    cost_usd=self.calculate_cost(
                        model,
                        usage["prompt_tokens"],
                        usage["completion_tokens"],
                    ),
                    duration_ms=duration_ms,
                )

            except (LLMAuthenticationError, LLMInvalidRequestError):
                # Don't retry auth or validation errors
                raise
            except LLMRateLimitError as e:
                last_error = e
                # Exponential backoff for rate limits
                wait_time = (2 ** attempt) + random.random()
                logger.warning(
                    "Rate limited, retrying",
                    attempt=attempt + 1,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + random.random()
                    logger.warning(
                        "Request failed, retrying",
                        attempt=attempt + 1,
                        error=str(e),
                        wait_seconds=wait_time,
                    )
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        raise LLMProviderError(f"Request failed after {self.max_retries} attempts: {last_error}")

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


# =============================================================================
# Factory Function
# =============================================================================
def get_provider(
    provider_type: LLMProvider | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """Get an LLM provider instance based on configuration.

    Args:
        provider_type: Override provider type from settings
        **kwargs: Additional arguments passed to provider constructor

    Returns:
        BaseLLMProvider instance

    Example:
        # Use configured provider
        provider = get_provider()

        # Force mock provider
        provider = get_provider(LLMProvider.MOCK)

        # Mock with custom failure rate for testing
        provider = get_provider(LLMProvider.MOCK, failure_rate=0.1)
    """
    settings = get_settings()
    provider_type = provider_type or settings.llm_provider

    if provider_type == LLMProvider.OPENAI:
        return OpenAIProvider(**kwargs)
    else:
        return MockLLMProvider(**kwargs)


# =============================================================================
# Quick Test
# =============================================================================
async def _test_providers() -> None:
    """Quick test of provider functionality."""

    print("=" * 60)
    print("Testing Mock Provider")
    print("=" * 60)

    mock_provider = MockLLMProvider()

    # Test single completion
    response = await mock_provider.complete(
        prompt="Explain microservices architecture in simple terms.",
        model="gpt-4o-mini",
        max_tokens=500,
        temperature=0.7,
    )

    print(f"Model: {response.model}")
    print(f"Content: {response.content[:100]}...")
    print(f"Input tokens: {response.input_tokens}")
    print(f"Output tokens: {response.output_tokens}")
    print(f"Total tokens: {response.total_tokens}")
    print(f"Cost: ${response.cost_usd:.6f}")
    print(f"Duration: {response.duration_ms}ms")

    print("\n" + "=" * 60)
    print("Testing Token Counting")
    print("=" * 60)

    test_texts = [
        "Hello, world!",
        "This is a longer sentence with more tokens.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    for text in test_texts:
        tokens = mock_provider.count_tokens(text)
        print(f"'{text[:30]}...' -> {tokens} tokens")

    print("\n" + "=" * 60)
    print("Testing Cost Calculation")
    print("=" * 60)

    models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    for model in models:
        cost = mock_provider.calculate_cost(model, 1000, 500)
        print(f"{model}: 1000 input + 500 output = ${cost:.6f}")

    print("\nProvider tests complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_providers())
