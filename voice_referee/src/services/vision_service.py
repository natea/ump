"""Vision service for screen content analysis.

This module provides a wrapper for vision APIs to analyze screen captures
and return natural language descriptions of the content.
"""

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class VisionAnalysisResult:
    """Result from vision analysis."""
    description: str
    confidence: float
    latency_ms: float
    provider: str
    model: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    requires_commentary: bool = False
    detected_elements: list[str] = field(default_factory=list)


class VisionService:
    """Service for analyzing screen content using vision models.

    Supports multiple providers:
    - anthropic: Claude 3.5 Sonnet (recommended)
    - openai: GPT-4 Vision
    - google: Gemini Pro Vision
    """

    # Cost per 1M tokens (approximate)
    COST_PER_1M_TOKENS = {
        "anthropic": {"input": 3.0, "output": 15.0},
        "openai": {"input": 10.0, "output": 30.0},
        "google": {"input": 0.5, "output": 1.5},
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet-20241022",
        api_key: str = "",
        max_tokens: int = 150,
    ):
        """Initialize vision service.

        Args:
            provider: Vision API provider (anthropic, openai, google)
            model: Model identifier
            api_key: API key for the provider
            max_tokens: Maximum tokens for response
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the API client based on provider."""
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Anthropic vision client initialized with model {self.model}")
            except ImportError:
                logger.error("anthropic package not installed")
                raise
        elif self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"OpenAI vision client initialized with model {self.model}")
            except ImportError:
                logger.error("openai package not installed")
                raise
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
                logger.info(f"Google vision client initialized with model {self.model}")
            except ImportError:
                logger.error("google-generativeai package not installed")
                raise
        else:
            raise ValueError(f"Unsupported vision provider: {self.provider}")

    async def analyze_image(
        self,
        image_data: bytes,
        prompt: str = "Describe what you see on this screen. Focus on key elements, changes, or notable content.",
        format: str = "RGB",
        size: tuple = None,
    ) -> VisionAnalysisResult:
        """Analyze an image and return a description.

        Args:
            image_data: Raw image bytes
            prompt: Analysis prompt
            format: Image format (RGB, RGBA, etc.)
            size: Image dimensions (width, height)

        Returns:
            VisionAnalysisResult with description and metadata
        """
        start_time = time.time()

        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            if self.provider == "anthropic":
                result = await self._analyze_anthropic(image_b64, prompt)
            elif self.provider == "openai":
                result = await self._analyze_openai(image_b64, prompt)
            elif self.provider == "google":
                result = await self._analyze_google(image_data, prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms

            logger.info(f"Vision analysis completed in {latency_ms:.0f}ms: {result.description[:100]}...")

            return result

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}", exc_info=True)
            latency_ms = (time.time() - start_time) * 1000
            return VisionAnalysisResult(
                description=f"Analysis failed: {str(e)}",
                confidence=0.0,
                latency_ms=latency_ms,
                provider=self.provider,
                model=self.model,
            )

    async def _analyze_anthropic(self, image_b64: str, prompt: str) -> VisionAnalysisResult:
        """Analyze image using Anthropic Claude Vision."""
        import asyncio

        # Run synchronous client in executor
        loop = asyncio.get_event_loop()

        def _call_api():
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            return response

        response = await loop.run_in_executor(None, _call_api)

        description = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        cost = self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens)

        return VisionAnalysisResult(
            description=description,
            confidence=0.9,  # Claude doesn't provide confidence scores
            latency_ms=0,  # Will be set by caller
            provider=self.provider,
            model=self.model,
            tokens_used=tokens_used,
            cost_usd=cost,
            requires_commentary=self._should_comment(description),
        )

    async def _analyze_openai(self, image_b64: str, prompt: str) -> VisionAnalysisResult:
        """Analyze image using OpenAI GPT-4 Vision."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _call_api():
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            )
            return response

        response = await loop.run_in_executor(None, _call_api)

        description = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        cost = self._calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )

        return VisionAnalysisResult(
            description=description,
            confidence=0.85,
            latency_ms=0,
            provider=self.provider,
            model=self.model,
            tokens_used=tokens_used,
            cost_usd=cost,
            requires_commentary=self._should_comment(description),
        )

    async def _analyze_google(self, image_data: bytes, prompt: str) -> VisionAnalysisResult:
        """Analyze image using Google Gemini Vision."""
        import asyncio
        from PIL import Image
        import io

        loop = asyncio.get_event_loop()

        def _call_api():
            # Convert bytes to PIL Image for Gemini
            image = Image.open(io.BytesIO(image_data))
            response = self._client.generate_content([prompt, image])
            return response

        response = await loop.run_in_executor(None, _call_api)

        description = response.text

        return VisionAnalysisResult(
            description=description,
            confidence=0.85,
            latency_ms=0,
            provider=self.provider,
            model=self.model,
            tokens_used=0,  # Gemini doesn't always report tokens
            cost_usd=0.001,  # Estimate
            requires_commentary=self._should_comment(description),
        )

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        rates = self.COST_PER_1M_TOKENS.get(self.provider, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def _should_comment(self, description: str) -> bool:
        """Determine if the description warrants spoken commentary."""
        # Keywords that suggest notable content
        notable_keywords = [
            "error", "warning", "alert", "important", "changed",
            "new", "different", "significant", "problem", "issue",
            "code", "chart", "graph", "data", "document"
        ]

        description_lower = description.lower()
        for keyword in notable_keywords:
            if keyword in description_lower:
                return True

        return False


def create_vision_service(
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    api_key: str = "",
) -> VisionService:
    """Factory function to create a vision service.

    Args:
        provider: Vision provider (anthropic, openai, google)
        model: Model identifier
        api_key: API key

    Returns:
        Configured VisionService instance
    """
    return VisionService(
        provider=provider,
        model=model,
        api_key=api_key,
    )
