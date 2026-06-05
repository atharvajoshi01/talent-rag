"""
Generator Module.

This module provides the LLM-based generation component
for the RAG pipeline with support for multiple providers (OpenAI, Ollama).
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterator
from dataclasses import dataclass, field

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompt_builder import PromptBuilder, PromptContext, QueryType


@dataclass
class GenerationResult:
    """
    Result from the generator.

    Attributes:
        response: Generated response text
        model: Model used for generation
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        latency_ms: Generation latency in milliseconds
        citations: List of citations found in response
        metadata: Additional metadata
    """
    response: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> tuple[str, dict[str, Any]]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dictionaries
            temperature: Generation temperature
            max_tokens: Maximum tokens in response

        Returns:
            Tuple of (response_text, usage_info)
        """
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Iterator[str]:
        """
        Stream a chat completion request.

        Args:
            messages: List of message dictionaries
            temperature: Generation temperature
            max_tokens: Maximum tokens in response

        Yields:
            Response chunks
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: OpenAI model name
            api_key: API key (uses env var if not provided)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self._model = model
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

        logger.info(f"OpenAI provider initialized with model={model}")

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> tuple[str, dict[str, Any]]:
        """Send chat completion to OpenAI."""
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        latency_ms = (time.time() - start_time) * 1000

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "latency_ms": latency_ms
        }

        return response.choices[0].message.content, usage

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Iterator[str]:
        """Stream chat completion from OpenAI."""
        stream = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local inference."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Ollama model name (e.g., 'llama3.2', 'mistral', 'codellama')
            host: Ollama server host URL
        """
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("requests is required. Install with: pip install requests")

        self._model = model
        self._host = host.rstrip('/')
        self._api_url = f"{self._host}/api/chat"

        # Verify Ollama is running
        self._verify_connection()

        logger.info(f"Ollama provider initialized with model={model} at {host}")

    def _verify_connection(self):
        """Verify Ollama server is accessible."""
        try:
            response = self._requests.get(f"{self._host}/api/tags", timeout=5)
            if response.status_code != 200:
                logger.warning(f"Ollama server responded with status {response.status_code}")
        except self._requests.exceptions.ConnectionError:
            logger.warning(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve"
            )

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> tuple[str, dict[str, Any]]:
        """Send chat completion to Ollama."""
        start_time = time.time()

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        response = self._requests.post(
            self._api_url,
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        data = response.json()
        latency_ms = (time.time() - start_time) * 1000

        # Ollama provides token counts
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            "latency_ms": latency_ms
        }

        return data["message"]["content"], usage

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> Iterator[str]:
        """Stream chat completion from Ollama."""
        import json

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        response = self._requests.post(
            self._api_url,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    yield data["message"]["content"]


def get_llm_provider(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None
) -> BaseLLMProvider:
    """
    Factory function to get an LLM provider.

    Args:
        provider: Provider name ('openai' or 'ollama')
        model: Model name (uses defaults if not specified)
        api_key: API key for OpenAI
        host: Host URL for Ollama

    Returns:
        BaseLLMProvider instance
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider(
            model=model or "gpt-4o",
            api_key=api_key
        )
    elif provider == "ollama":
        return OllamaProvider(
            model=model or "llama3.2",
            host=host or "http://localhost:11434"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'ollama'.")


class Generator:
    """
    LLM-based response generator for RAG.

    This class handles the generation of responses using
    an LLM with the constructed prompts.

    Attributes:
        provider: LLM provider instance
        temperature: Temperature for generation
        max_tokens: Maximum tokens in response
        prompt_builder: PromptBuilder instance
    """

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        model: str = "llama3.2",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        llm_provider: str = "ollama",
        ollama_host: str = "http://localhost:11434",
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize the generator.

        Args:
            provider: Pre-configured LLM provider
            model: Model name to use
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            api_key: API key for OpenAI (if using OpenAI)
            llm_provider: Provider name ('openai' or 'ollama')
            ollama_host: Ollama server host URL
            prompt_builder: PromptBuilder instance
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        if provider:
            self.provider = provider
        else:
            if llm_provider == "openai":
                self.provider = OpenAIProvider(model=model, api_key=api_key)
            else:
                self.provider = OllamaProvider(model=model, host=ollama_host)

        self.prompt_builder = prompt_builder or PromptBuilder()

        logger.info(
            f"Generator initialized with provider={llm_provider}, "
            f"model={self.provider.model_name}, temperature={temperature}"
        )

    @property
    def model(self) -> str:
        """Return the model name."""
        return self.provider.model_name

    def _extract_citations(self, text: str) -> list[str]:
        """
        Extract citation markers from response text.

        Args:
            text: Response text

        Returns:
            List of citation IDs
        """
        import re

        # Find all citation patterns like [CAND_001] or [ROLE_003]
        pattern = r'\[([A-Z]+_\d+)\]'
        matches = re.findall(pattern, text)

        return list(set(matches))

    def generate(
        self,
        context: PromptContext,
        variation: int = 0
    ) -> GenerationResult:
        """
        Generate a response for the given context.

        Args:
            context: PromptContext with query and evidence
            variation: Variation number for ensemble generation

        Returns:
            GenerationResult with response and metadata
        """
        # Build messages
        messages = self.prompt_builder.build_messages(context, variation)

        # Call LLM
        response_text, usage = self.provider.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Extract citations
        citations = self._extract_citations(response_text)

        return GenerationResult(
            response=response_text,
            model=self.model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            latency_ms=usage["latency_ms"],
            citations=citations,
            metadata={
                "query_type": context.query_type.value,
                "variation": variation,
                "evidence_count": len(context.evidence)
            }
        )

    def generate_with_custom_prompt(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> GenerationResult:
        """
        Generate with custom prompts.

        Args:
            system_prompt: Custom system prompt
            user_prompt: Custom user prompt

        Returns:
            GenerationResult with response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text, usage = self.provider.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        citations = self._extract_citations(response_text)

        return GenerationResult(
            response=response_text,
            model=self.model,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            latency_ms=usage["latency_ms"],
            citations=citations,
            metadata={"custom_prompt": True}
        )

    def generate_candidate_summary(
        self,
        candidate_evidence: list[Any],
        role_context: Optional[dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Generate a summary for a candidate.

        Args:
            candidate_evidence: Evidence chunks for the candidate
            role_context: Optional role to evaluate against

        Returns:
            GenerationResult with candidate summary
        """
        if not candidate_evidence:
            return GenerationResult(
                response="No evidence available for this candidate.",
                model=self.model
            )

        query = "Provide a comprehensive summary of this candidate's qualifications, experience, and suitability."

        if role_context:
            query += f" Evaluate their fit for the {role_context.get('title', 'role')} position."

        context = PromptContext(
            query=query,
            query_type=QueryType.CANDIDATE_SEARCH,
            evidence=candidate_evidence,
            role=role_context
        )

        return self.generate(context)

    def generate_comparison(
        self,
        candidates_evidence: dict[str, list[Any]],
        role: dict[str, Any]
    ) -> GenerationResult:
        """
        Generate a comparison of multiple candidates.

        Args:
            candidates_evidence: Dict mapping candidate IDs to their evidence
            role: Role to compare against

        Returns:
            GenerationResult with comparison
        """
        # Flatten evidence
        all_evidence = []
        for evidence_list in candidates_evidence.values():
            all_evidence.extend(evidence_list)

        candidate_ids = list(candidates_evidence.keys())
        query = (
            f"Compare candidates {', '.join(candidate_ids)} for the "
            f"{role.get('title', 'role')} position. "
            "Provide a detailed analysis and final recommendation."
        )

        context = PromptContext(
            query=query,
            query_type=QueryType.CANDIDATE_COMPARE,
            evidence=all_evidence,
            role=role
        )

        return self.generate(context)

    def generate_batch(
        self,
        contexts: list[PromptContext],
        variations: list[int] = None
    ) -> list[GenerationResult]:
        """
        Generate responses for multiple contexts.

        Note: This is sequential; for true parallelism,
        use async implementation.

        Args:
            contexts: List of PromptContext objects
            variations: Optional list of variation numbers

        Returns:
            List of GenerationResult objects
        """
        if variations is None:
            variations = [0] * len(contexts)

        results = []
        for context, variation in zip(contexts, variations):
            try:
                result = self.generate(context, variation)
                results.append(result)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                results.append(GenerationResult(
                    response=f"Generation failed: {str(e)}",
                    model=self.model,
                    metadata={"error": str(e)}
                ))

        return results

    def stream_generate(
        self,
        context: PromptContext,
        variation: int = 0
    ):
        """
        Stream generate a response.

        Args:
            context: PromptContext with query and evidence
            variation: Variation number

        Yields:
            Response chunks as they are generated
        """
        messages = self.prompt_builder.build_messages(context, variation)

        start_time = time.time()

        for chunk in self.provider.stream_chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        ):
            yield chunk

        # Log completion
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Streaming complete in {latency_ms:.0f}ms")
