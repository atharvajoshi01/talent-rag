"""
Response Judge Module.

This module provides LLM-based judging for selecting
the best response from multiple candidates.
"""

from typing import Any, Optional
from dataclasses import dataclass

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from ..rag.generator import BaseLLMProvider, OllamaProvider, OpenAIProvider


@dataclass
class JudgeScore:
    """
    Score from the judge for a single response.

    Attributes:
        response_index: Index of the response
        completeness: Score for completeness
        faithfulness: Score for faithfulness to evidence
        evidence_usage: Score for proper evidence citation
        overall: Overall quality score
        explanation: Judge's explanation
    """
    response_index: int
    completeness: float
    faithfulness: float
    evidence_usage: float
    overall: float
    explanation: str


class ResponseJudge:
    """
    LLM-based judge for response selection.

    This class evaluates multiple responses and selects
    the best one based on completeness, faithfulness,
    and evidence usage.

    Attributes:
        provider: LLM provider instance
        model: Model for judging
    """

    JUDGE_PROMPT = """You are an expert judge evaluating AI responses for a talent intelligence assistant.

You must evaluate the following responses to the same query and select the BEST one.

## Evaluation Criteria (score each 1-10):

1. **Completeness**: Does the response fully address all aspects of the query?
2. **Faithfulness**: Is the response accurate and faithful to the provided evidence?
3. **Evidence Usage**: Does the response properly cite and reference the evidence?

## Query:
{query}

## Available Evidence:
{evidence}

## Responses to Evaluate:

{responses}

## Your Task:
Evaluate each response and select the best one.

Respond in EXACTLY this format:

RESPONSE_1_SCORES:
- Completeness: [1-10]
- Faithfulness: [1-10]
- Evidence Usage: [1-10]
- Overall: [1-10]

RESPONSE_2_SCORES:
- Completeness: [1-10]
- Faithfulness: [1-10]
- Evidence Usage: [1-10]
- Overall: [1-10]

RESPONSE_3_SCORES:
- Completeness: [1-10]
- Faithfulness: [1-10]
- Evidence Usage: [1-10]
- Overall: [1-10]

BEST_RESPONSE: [1, 2, or 3]
EXPLANATION: [Brief explanation of why this response is best]"""

    def __init__(
        self,
        provider: Optional[BaseLLMProvider] = None,
        model: str = "llama3.2",
        llm_provider: str = "ollama",
        ollama_host: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        """
        Initialize the response judge.

        Args:
            provider: Pre-configured LLM provider
            model: Model to use for judging
            llm_provider: Provider name ('openai' or 'ollama')
            ollama_host: Ollama server host URL
            api_key: API key for OpenAI
        """
        if provider:
            self.provider = provider
        else:
            if llm_provider == "openai":
                self.provider = OpenAIProvider(model=model, api_key=api_key)
            else:
                self.provider = OllamaProvider(model=model, host=ollama_host)

        self.model = model

        logger.info(f"ResponseJudge initialized with provider={llm_provider}, model={model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM for judging.

        Args:
            prompt: Judge prompt

        Returns:
            LLM response text
        """
        messages = [
            {
                "role": "system",
                "content": "You are an impartial judge evaluating AI responses. Be precise and follow the format exactly."
            },
            {"role": "user", "content": prompt}
        ]

        response_text, _ = self.provider.chat(
            messages,
            temperature=0,
            max_tokens=1000
        )

        return response_text

    def _parse_score(self, text: str, key: str) -> float:
        """Parse a score from judge output."""
        import re

        pattern = rf'{key}:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return float(match.group(1)) / 10.0  # Normalize to 0-1

        return 0.5

    def _parse_best_response(self, text: str) -> int:
        """Parse the best response selection."""
        import re

        pattern = r'BEST_RESPONSE:\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return int(match.group(1)) - 1  # Convert to 0-indexed

        return 0

    def _format_evidence(self, evidence: list[Any]) -> str:
        """Format evidence for the judge prompt."""
        evidence_parts = []
        for i, e in enumerate(evidence[:5]):  # Limit for context length
            text = e.text if hasattr(e, 'text') else str(e)
            evidence_parts.append(f"[{i+1}] {text[:500]}")

        return "\n\n".join(evidence_parts)

    def _format_responses(self, responses: list[str]) -> str:
        """Format responses for the judge prompt."""
        response_parts = []
        for i, response in enumerate(responses):
            response_parts.append(f"### Response {i+1}:\n{response}")

        return "\n\n".join(response_parts)

    def select_best(
        self,
        responses: list[str],
        query: str,
        evidence: list[Any]
    ) -> tuple[int, list[dict[str, Any]]]:
        """
        Select the best response from multiple candidates.

        Args:
            responses: List of response texts
            query: Original query
            evidence: List of evidence items

        Returns:
            Tuple of (best_index, scores_list)
        """
        if len(responses) == 1:
            return 0, [{"overall": 1.0}]

        # Build judge prompt
        prompt = self.JUDGE_PROMPT.format(
            query=query,
            evidence=self._format_evidence(evidence),
            responses=self._format_responses(responses)
        )

        # Get judge evaluation
        judge_output = self._call_llm(prompt)

        # Parse scores for each response
        scores = []
        for i in range(len(responses)):
            section_pattern = rf'RESPONSE_{i+1}_SCORES:(.*?)(?=RESPONSE_|BEST_RESPONSE|\Z)'
            section_match = __import__('re').search(
                section_pattern,
                judge_output,
                __import__('re').DOTALL | __import__('re').IGNORECASE
            )

            if section_match:
                section = section_match.group(1)
                score = {
                    "response_index": i,
                    "completeness": self._parse_score(section, "Completeness"),
                    "faithfulness": self._parse_score(section, "Faithfulness"),
                    "evidence_usage": self._parse_score(section, "Evidence Usage"),
                    "overall": self._parse_score(section, "Overall")
                }
            else:
                score = {
                    "response_index": i,
                    "completeness": 0.5,
                    "faithfulness": 0.5,
                    "evidence_usage": 0.5,
                    "overall": 0.5
                }

            scores.append(score)

        # Get best response selection
        best_idx = self._parse_best_response(judge_output)

        # Validate best_idx
        if best_idx >= len(responses):
            # Fallback to highest overall score
            best_idx = max(range(len(scores)), key=lambda i: scores[i]["overall"])

        logger.debug(
            f"Judge selected response {best_idx + 1} with score "
            f"{scores[best_idx]['overall']:.2f}"
        )

        return best_idx, scores

    def evaluate_single(
        self,
        response: str,
        query: str,
        evidence: list[Any]
    ) -> dict[str, Any]:
        """
        Evaluate a single response.

        Args:
            response: Response text
            query: Original query
            evidence: List of evidence items

        Returns:
            Dictionary with scores
        """
        _, scores = self.select_best([response], query, evidence)
        return scores[0] if scores else {"overall": 0.5}

    def rank_responses(
        self,
        responses: list[str],
        query: str,
        evidence: list[Any]
    ) -> list[tuple[int, dict[str, Any]]]:
        """
        Rank all responses by quality.

        Args:
            responses: List of response texts
            query: Original query
            evidence: List of evidence items

        Returns:
            List of (index, scores) tuples sorted by overall score
        """
        _, scores = self.select_best(responses, query, evidence)

        # Sort by overall score
        ranked = [(i, s) for i, s in enumerate(scores)]
        ranked.sort(key=lambda x: x[1]["overall"], reverse=True)

        return ranked


class ConsensusJudge:
    """
    Judge using multiple evaluations for consensus.

    This class runs multiple judge evaluations and uses
    consensus to select the best response.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        num_evaluations: int = 3,
        llm_provider: str = "ollama",
        ollama_host: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        """
        Initialize the consensus judge.

        Args:
            model: Model for judging
            num_evaluations: Number of evaluation rounds
            llm_provider: Provider name ('openai' or 'ollama')
            ollama_host: Ollama server host URL
            api_key: API key for OpenAI
        """
        self.judge = ResponseJudge(
            model=model,
            llm_provider=llm_provider,
            ollama_host=ollama_host,
            api_key=api_key
        )
        self.num_evaluations = num_evaluations

        logger.info(
            f"ConsensusJudge initialized with {num_evaluations} evaluations"
        )

    def select_best_consensus(
        self,
        responses: list[str],
        query: str,
        evidence: list[Any]
    ) -> tuple[int, dict[str, Any]]:
        """
        Select best response using consensus from multiple evaluations.

        Args:
            responses: List of response texts
            query: Original query
            evidence: List of evidence items

        Returns:
            Tuple of (best_index, aggregated_scores)
        """
        votes = [0] * len(responses)
        all_scores = [[] for _ in responses]

        for _ in range(self.num_evaluations):
            try:
                best_idx, scores = self.judge.select_best(
                    responses, query, evidence
                )
                votes[best_idx] += 1

                for i, score in enumerate(scores):
                    all_scores[i].append(score)

            except Exception as e:
                logger.warning(f"Evaluation round failed: {e}")

        # Find consensus winner
        winner_idx = votes.index(max(votes))

        # Aggregate scores
        aggregated_scores = {}
        if all_scores[winner_idx]:
            keys = all_scores[winner_idx][0].keys()
            for key in keys:
                if key != "response_index":
                    values = [s[key] for s in all_scores[winner_idx] if key in s]
                    aggregated_scores[key] = sum(values) / len(values) if values else 0.5

        aggregated_scores["votes"] = votes[winner_idx]
        aggregated_scores["total_votes"] = sum(votes)

        return winner_idx, aggregated_scores
