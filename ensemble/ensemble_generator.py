"""
Ensemble Generator Module.

This module provides ensemble generation capabilities,
generating multiple response variations for improved quality.
"""

from typing import Any, Optional
from dataclasses import dataclass, field
import time

from loguru import logger

from ..rag.prompt_builder import PromptBuilder, PromptContext
from ..rag.generator import Generator, GenerationResult
from .judge import ResponseJudge


@dataclass
class EnsembleResult:
    """
    Result from ensemble generation.

    Attributes:
        selected_response: The selected best response
        all_responses: List of all generated responses
        selected_index: Index of selected response
        judge_scores: Scores from the judge
        latency_ms: Total ensemble latency
        metadata: Additional metadata
    """
    selected_response: GenerationResult
    all_responses: list[GenerationResult]
    selected_index: int
    judge_scores: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EnsembleGenerator:
    """
    Ensemble generator with multiple response variations.

    This class generates multiple responses with slight prompt
    variations and uses a judge to select the best one.

    Attributes:
        generator: Base generator instance
        judge: ResponseJudge for selection
        num_variations: Number of response variations to generate
        prompt_builder: PromptBuilder instance
    """

    # System prompt variations for diversity
    PROMPT_VARIATIONS = [
        "",  # Base prompt
        "\n\nBe particularly thorough in citing evidence for every claim.",
        "\n\nFocus on providing actionable, practical insights.",
        "\n\nEmphasize both strengths and potential concerns in your analysis.",
        "\n\nStructure your response with clear sections and bullet points."
    ]

    def __init__(
        self,
        generator: Optional[Generator] = None,
        judge: Optional[ResponseJudge] = None,
        num_variations: int = 3,
        model: str = "llama3.2",
        judge_model: str = "llama3.2",
        llm_provider: str = "ollama",
        ollama_host: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        """
        Initialize the ensemble generator.

        Args:
            generator: Pre-initialized generator
            judge: Pre-initialized judge
            num_variations: Number of response variations
            model: Model for generation
            judge_model: Model for judging
            llm_provider: Provider name ('openai' or 'ollama')
            ollama_host: Ollama server host URL
            api_key: API key for OpenAI
        """
        self.llm_provider = llm_provider
        self.ollama_host = ollama_host
        self.api_key = api_key

        self.generator = generator or Generator(
            model=model,
            llm_provider=llm_provider,
            ollama_host=ollama_host,
            api_key=api_key
        )
        self.judge = judge or ResponseJudge(
            model=judge_model,
            llm_provider=llm_provider,
            ollama_host=ollama_host,
            api_key=api_key
        )
        self.num_variations = min(num_variations, len(self.PROMPT_VARIATIONS))
        self.prompt_builder = PromptBuilder()

        logger.info(
            f"EnsembleGenerator initialized with {self.num_variations} variations, "
            f"provider={llm_provider}"
        )

    def generate(
        self,
        context: PromptContext,
        use_judge: bool = True
    ) -> EnsembleResult:
        """
        Generate ensemble responses and select the best.

        Args:
            context: PromptContext with query and evidence
            use_judge: Whether to use judge for selection

        Returns:
            EnsembleResult with selected response
        """
        start_time = time.time()

        # Generate multiple responses with variations
        responses = []
        for i in range(self.num_variations):
            try:
                result = self.generator.generate(context, variation=i)
                responses.append(result)
            except Exception as e:
                logger.warning(f"Generation {i} failed: {e}")
                # Continue with remaining variations

        if not responses:
            raise RuntimeError("All generation attempts failed")

        # Select best response
        if use_judge and len(responses) > 1:
            selected_idx, judge_scores = self.judge.select_best(
                responses=[r.response for r in responses],
                query=context.query,
                evidence=context.evidence
            )
        else:
            selected_idx = 0
            judge_scores = []

        latency_ms = (time.time() - start_time) * 1000

        return EnsembleResult(
            selected_response=responses[selected_idx],
            all_responses=responses,
            selected_index=selected_idx,
            judge_scores=judge_scores,
            latency_ms=latency_ms,
            metadata={
                "num_variations": len(responses),
                "used_judge": use_judge
            }
        )

    def generate_with_fallback(
        self,
        context: PromptContext,
        fallback_on_low_score: float = 0.5
    ) -> EnsembleResult:
        """
        Generate with fallback for low-scoring responses.

        If the best response scores below threshold, generate
        additional variations.

        Args:
            context: PromptContext
            fallback_on_low_score: Score threshold for fallback

        Returns:
            EnsembleResult
        """
        # Initial generation
        result = self.generate(context)

        # Check if we need fallback
        if result.judge_scores:
            best_score = max(s.get("overall", 0) for s in result.judge_scores)

            if best_score < fallback_on_low_score:
                logger.info(
                    f"Best score {best_score} below threshold, "
                    "generating additional variations"
                )

                # Generate more with different temperature
                additional_generator = Generator(
                    model=self.generator.model,
                    temperature=0.3,  # Slightly higher temperature
                    llm_provider=self.llm_provider,
                    ollama_host=self.ollama_host,
                    api_key=self.api_key
                )

                try:
                    additional_result = additional_generator.generate(context)
                    result.all_responses.append(additional_result)

                    # Re-judge with additional response
                    selected_idx, judge_scores = self.judge.select_best(
                        responses=[r.response for r in result.all_responses],
                        query=context.query,
                        evidence=context.evidence
                    )

                    result.selected_index = selected_idx
                    result.selected_response = result.all_responses[selected_idx]
                    result.judge_scores = judge_scores

                except Exception as e:
                    logger.warning(f"Fallback generation failed: {e}")

        return result

    def generate_diverse(
        self,
        context: PromptContext,
        temperatures: list[float] = [0.1, 0.3, 0.5]
    ) -> EnsembleResult:
        """
        Generate responses with diverse temperature settings.

        Args:
            context: PromptContext
            temperatures: List of temperatures to use

        Returns:
            EnsembleResult
        """
        start_time = time.time()

        responses = []
        for temp in temperatures:
            try:
                temp_generator = Generator(
                    model=self.generator.model,
                    temperature=temp,
                    llm_provider=self.llm_provider,
                    ollama_host=self.ollama_host,
                    api_key=self.api_key
                )
                result = temp_generator.generate(context)
                result.metadata["temperature"] = temp
                responses.append(result)
            except Exception as e:
                logger.warning(f"Generation at temp={temp} failed: {e}")

        if not responses:
            raise RuntimeError("All generation attempts failed")

        # Select best
        selected_idx, judge_scores = self.judge.select_best(
            responses=[r.response for r in responses],
            query=context.query,
            evidence=context.evidence
        )

        latency_ms = (time.time() - start_time) * 1000

        return EnsembleResult(
            selected_response=responses[selected_idx],
            all_responses=responses,
            selected_index=selected_idx,
            judge_scores=judge_scores,
            latency_ms=latency_ms,
            metadata={
                "generation_type": "diverse_temperature",
                "temperatures": temperatures
            }
        )


class VotingEnsemble:
    """
    Ensemble using majority voting on key decisions.

    This class generates multiple responses and uses voting
    to reach consensus on recommendations.
    """

    def __init__(
        self,
        generator: Generator,
        num_voters: int = 3
    ):
        """
        Initialize the voting ensemble.

        Args:
            generator: Generator instance
            num_voters: Number of voting responses
        """
        self.generator = generator
        self.num_voters = num_voters

        logger.info(f"VotingEnsemble initialized with {num_voters} voters")

    def vote_on_candidates(
        self,
        context: PromptContext,
        candidate_ids: list[str]
    ) -> dict[str, Any]:
        """
        Vote on best candidate from multiple responses.

        Args:
            context: PromptContext with comparison query
            candidate_ids: List of candidate IDs to vote on

        Returns:
            Dictionary with voting results
        """

        votes = {cid: 0 for cid in candidate_ids}
        responses = []

        for i in range(self.num_voters):
            try:
                result = self.generator.generate(context, variation=i)
                responses.append(result.response)

                # Extract recommendation from response
                response_lower = result.response.lower()

                for cid in candidate_ids:
                    # Check if candidate is recommended
                    if any(phrase in response_lower for phrase in [
                        f"recommend {cid.lower()}",
                        f"{cid.lower()} is the best",
                        f"top choice: {cid.lower()}",
                        f"first choice.*{cid.lower()}"
                    ]):
                        votes[cid] += 2  # Strong vote
                    elif cid.lower() in response_lower:
                        votes[cid] += 1  # Weak vote

            except Exception as e:
                logger.warning(f"Voter {i} failed: {e}")

        # Determine winner
        winner = max(votes, key=votes.get) if votes else None

        return {
            "winner": winner,
            "votes": votes,
            "num_voters": len(responses),
            "responses": responses,
            "confidence": votes[winner] / (self.num_voters * 2) if winner else 0
        }
