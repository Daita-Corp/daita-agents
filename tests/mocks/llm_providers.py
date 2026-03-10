"""
Mock LLM providers for testing without API calls.

Provides predictable responses for testing agent behavior.
"""

import asyncio
import json
import random
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timezone


class MockLLMProvider:
    """
    Mock LLM provider that returns predictable responses.

    Simulates different LLM behaviors for comprehensive testing.
    """

    def __init__(
        self,
        provider_name: str = "mock",
        model: str = "mock-model-1",
        response_delay: float = 0.1,
        failure_rate: float = 0.0,
    ):
        """
        Initialize mock LLM provider.

        Args:
            provider_name: Name of the provider to simulate
            model: Model name to simulate
            response_delay: Simulated response delay in seconds
            failure_rate: Probability of simulated failures (0.0 to 1.0)
        """
        self.provider_name = provider_name
        self.model = model
        self.response_delay = response_delay
        self.failure_rate = failure_rate

        # Tracking
        self.call_count = 0
        self.total_tokens = 0
        self.agent_id = None
        self.call_history = []

        # Response templates
        self.response_templates = {
            "analyze": self._analyze_response,
            "summarize": self._summarize_response,
            "process": self._process_response,
            "transform": self._transform_response,
            "generate": self._generate_response,
            "classify": self._classify_response,
            "extract": self._extract_response,
        }

    def set_agent_id(self, agent_id: str) -> None:
        """Set the agent ID for tracking."""
        self.agent_id = agent_id

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate a response to the given prompt."""
        # Simulate network delay
        await asyncio.sleep(self.response_delay)

        # Simulate failures
        if random.random() < self.failure_rate:
            raise Exception(f"Mock {self.provider_name} API failure")

        # Track the call
        self.call_count += 1
        call_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "system_prompt": (
                system_prompt[:50] + "..."
                if system_prompt and len(system_prompt) > 50
                else system_prompt
            ),
            "kwargs": kwargs,
        }
        self.call_history.append(call_info)

        # Determine response type based on prompt keywords
        response_type = self._detect_response_type(prompt)
        response_generator = self.response_templates.get(
            response_type, self._default_response
        )

        # Generate response
        response_text = response_generator(prompt, system_prompt, **kwargs)

        # Calculate mock token usage
        prompt_tokens = len((prompt + (system_prompt or "")).split())
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens
        self.total_tokens += total_tokens

        return {
            "response": response_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model": self.model,
            "provider": self.provider_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        response = await self.generate(prompt, system_prompt, **kwargs)
        response_text = response["response"]

        # Stream word by word
        words = response_text.split()
        for word in words:
            await asyncio.sleep(0.01)  # Small delay between words
            yield word + " "

    def get_token_usage(self) -> Dict[str, int]:
        """Get total token usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "average_tokens_per_call": self.total_tokens // max(1, self.call_count),
        }

    def reset_tracking(self) -> None:
        """Reset all tracking counters."""
        self.call_count = 0
        self.total_tokens = 0
        self.call_history.clear()

    def _detect_response_type(self, prompt: str) -> str:
        """Detect the type of response needed based on prompt."""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["analyze", "analysis"]):
            return "analyze"
        elif any(word in prompt_lower for word in ["summarize", "summary"]):
            return "summarize"
        elif any(word in prompt_lower for word in ["process", "processing"]):
            return "process"
        elif any(word in prompt_lower for word in ["transform", "convert"]):
            return "transform"
        elif any(word in prompt_lower for word in ["generate", "create"]):
            return "generate"
        elif any(word in prompt_lower for word in ["classify", "categorize"]):
            return "classify"
        elif any(word in prompt_lower for word in ["extract", "find"]):
            return "extract"
        else:
            return "default"

    def _analyze_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate analysis response."""
        return """Based on the provided data, I've identified several key insights:

1. **Trends**: The data shows a generally positive trend with some fluctuations
2. **Patterns**: There are recurring patterns in the dataset that suggest seasonality
3. **Outliers**: A few data points appear to be outliers that may need investigation
4. **Recommendations**:
   - Focus on the strongest performing segments
   - Address the areas showing decline
   - Monitor the outlier cases for potential issues

The analysis indicates overall healthy performance with opportunities for optimization."""

    def _summarize_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate summary response."""
        return """**Summary:**

The provided information covers multiple key areas with the following highlights:
- Main findings show positive indicators across most metrics
- Several action items have been identified for improvement
- Timeline for implementation spans the next quarter
- Expected outcomes include 15-20% improvement in key performance indicators

**Next Steps:** Review recommendations and begin implementation of priority items."""

    def _process_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate processing response."""
        return """Processing completed successfully. Here are the results:

**Input Processing:**
- Data validation:  Passed
- Format conversion:  Completed
- Quality checks:  All clear

**Output:**
- Processed records: 147
- Success rate: 98.6%
- Processing time: 2.3 seconds
- Status: Ready for next stage

The processed data is now available for further analysis or downstream systems."""

    def _transform_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate transformation response."""
        return """Data transformation completed. Summary of changes:

**Transformations Applied:**
- Format standardization: JSON → Structured format
- Data cleaning: Removed 3 invalid entries
- Field mapping: Applied business rules
- Validation: All records now conform to schema

**Result:**
```json
{
  "status": "success",
  "total_records": 144,
  "transformed_records": 144,
  "errors": 0,
  "warnings": 2
}
```

Transformation pipeline executed successfully."""

    def _generate_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate creative/generative response."""
        return """I've generated the requested content based on your specifications:

**Generated Content:**
Here is a comprehensive response tailored to your requirements. The content includes relevant details, maintains consistency with your guidelines, and follows best practices for the specified format.

Key elements included:
- Clear structure and organization
- Relevant examples and use cases
- Actionable recommendations
- Supporting data and evidence

The generated content is ready for review and can be easily modified if adjustments are needed."""

    def _classify_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate classification response."""
        return """Classification Results:

**Primary Category:** Business Intelligence
**Confidence Score:** 0.92

**Secondary Categories:**
- Data Analysis (0.78)
- Reporting (0.65)
- Decision Support (0.59)

**Key Indicators:**
- Contains numerical data
- Business context present
- Actionable insights required
- Time-sensitive information

**Recommendation:** Route to Business Intelligence workflow for detailed analysis."""

    def _extract_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate extraction response."""
        return """Extraction completed. Found the following information:

**Extracted Data:**
- **Names:** John Doe, Jane Smith, Bob Johnson
- **Dates:** 2024-03-15, 2024-02-28, 2024-03-01
- **Numbers:** 15000, 22000, 18000, 25000
- **Categories:** Premium, Standard, Enterprise
- **Status Values:** Active, Pending, Completed

**Confidence Levels:**
- Names: 95% accuracy
- Dates: 98% accuracy
- Numbers: 99% accuracy
- Categories: 87% accuracy

All extracted data has been validated and structured for further processing."""

    def _default_response(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Generate default response for unclassified prompts."""
        return f"""I've processed your request successfully. Based on the input provided, I've analyzed the content and generated appropriate output.

**Request Summary:**
- Input processed:
- Analysis completed:
- Response generated:

**Key Points:**
- The provided information has been thoroughly reviewed
- All relevant aspects have been considered
- The response addresses the core requirements
- Additional context has been included where beneficial

If you need any clarification or have additional requirements, please let me know."""


class MockLLMProviderFactory:
    """Factory for creating different types of mock LLM providers."""

    @staticmethod
    def create_openai_mock(failure_rate: float = 0.0) -> MockLLMProvider:
        """Create a mock OpenAI provider."""
        return MockLLMProvider(
            provider_name="openai",
            model="gpt-4",
            response_delay=0.5,
            failure_rate=failure_rate,
        )

    @staticmethod
    def create_anthropic_mock(failure_rate: float = 0.0) -> MockLLMProvider:
        """Create a mock Anthropic provider."""
        return MockLLMProvider(
            provider_name="anthropic",
            model="claude-3-sonnet",
            response_delay=0.3,
            failure_rate=failure_rate,
        )

    @staticmethod
    def create_fast_mock() -> MockLLMProvider:
        """Create a fast mock provider for quick tests."""
        return MockLLMProvider(
            provider_name="fast_mock",
            model="fast-model",
            response_delay=0.01,
            failure_rate=0.0,
        )

    @staticmethod
    def create_slow_mock() -> MockLLMProvider:
        """Create a slow mock provider for timeout testing."""
        return MockLLMProvider(
            provider_name="slow_mock",
            model="slow-model",
            response_delay=2.0,
            failure_rate=0.0,
        )

    @staticmethod
    def create_unreliable_mock() -> MockLLMProvider:
        """Create an unreliable mock provider for error testing."""
        return MockLLMProvider(
            provider_name="unreliable_mock",
            model="unreliable-model",
            response_delay=0.2,
            failure_rate=0.3,  # 30% failure rate
        )


# Mock external services
class MockExternalService:
    """Mock external service for testing integrations."""

    def __init__(self, service_name: str, base_url: str = "http://mock-service"):
        self.service_name = service_name
        self.base_url = base_url
        self.call_count = 0
        self.responses = {}
        self.default_response = {"status": "success", "data": "mock_response"}

    def set_response(self, endpoint: str, response: Dict[str, Any]) -> None:
        """Set a mock response for a specific endpoint."""
        self.responses[endpoint] = response

    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Mock GET request."""
        self.call_count += 1
        return self.responses.get(endpoint, self.default_response)

    async def post(self, endpoint: str, data: Any = None, **kwargs) -> Dict[str, Any]:
        """Mock POST request."""
        self.call_count += 1
        response = self.responses.get(endpoint, self.default_response.copy())
        if data:
            response["received_data"] = data
        return response

    def reset(self) -> None:
        """Reset call count and responses."""
        self.call_count = 0
        self.responses.clear()
