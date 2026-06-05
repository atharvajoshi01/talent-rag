"""
Response Evaluator Module.

This module provides evaluation capabilities for RAG responses,
including groundedness checking and quality assessment.
"""

from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class EvaluationCriteria(Enum):
    """Criteria for evaluating responses."""
    GROUNDEDNESS = "groundedness"
    COMPLETENESS = "completeness"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"


@dataclass
class EvaluationScore:
    """
    Score for a single evaluation criterion.

    Attributes:
        criterion: The evaluation criterion
        score: Score from 0 to 1
        explanation: Explanation for the score
    """
    criterion: EvaluationCriteria
    score: float
    explanation: str


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a response.

    Attributes:
        response: The evaluated response
        overall_score: Overall quality score
        scores: Individual criterion scores
        is_grounded: Whether response is grounded in evidence
        unsupported_claims: List of claims not supported by evidence
        metadata: Additional metadata
    """
    response: str
    overall_score: float
    scores: list[EvaluationScore] = field(default_factory=list)
    is_grounded: bool = True
    unsupported_claims: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResponseEvaluator:
    """
    Evaluator for RAG response quality.

    This class uses an LLM as a "judge" to evaluate response
    quality, groundedness, and faithfulness to evidence.

    Attributes:
        client: OpenAI client instance
        model: Model for evaluation
        strict_mode: Whether to use strict evaluation criteria
    """

    GROUNDEDNESS_PROMPT = """You are a strict fact-checker evaluating whether a response is grounded in the provided evidence.

Evidence:
{evidence}

Response to evaluate:
{response}

Evaluate the response for groundedness. For EACH claim in the response:
1. Is it directly supported by the evidence?
2. Is it a reasonable inference from the evidence?
3. Is it completely unsupported (hallucinated)?

Respond in the following format:
GROUNDEDNESS_SCORE: [0-10]
UNSUPPORTED_CLAIMS: [list any claims not supported by evidence, or "None"]
EXPLANATION: [brief explanation]"""

    QUALITY_PROMPT = """You are evaluating the quality of an AI assistant's response.

User Query: {query}

Evidence Provided:
{evidence}

Response to evaluate:
{response}

Evaluate the response on these criteria (score each 0-10):

1. COMPLETENESS: Does the response fully address the query?
2. FAITHFULNESS: Does the response accurately represent the evidence?
3. RELEVANCE: Is the response focused on what was asked?
4. COHERENCE: Is the response well-organized and clear?

Respond in this exact format:
COMPLETENESS: [0-10]
FAITHFULNESS: [0-10]
RELEVANCE: [0-10]
COHERENCE: [0-10]
OVERALL: [0-10]
EXPLANATION: [brief overall assessment]"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        strict_mode: bool = False
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to use for evaluation
            api_key: OpenAI API key
            strict_mode: Whether to use strict evaluation
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self.model = model
        self.strict_mode = strict_mode

        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

        logger.info(f"ResponseEvaluator initialized with model={model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM for evaluation.

        Args:
            prompt: Evaluation prompt

        Returns:
            LLM response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise evaluation assistant. Follow the format exactly."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )

        return response.choices[0].message.content

    def _parse_score(self, text: str, key: str) -> float:
        """
        Parse a score from evaluation output.

        Args:
            text: Evaluation output text
            key: Score key to look for

        Returns:
            Parsed score normalized to 0-1
        """
        import re

        pattern = rf'{key}:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            score = float(match.group(1))
            return min(max(score / 10.0, 0.0), 1.0)

        return 0.5  # Default score

    def _parse_list(self, text: str, key: str) -> list[str]:
        """
        Parse a list from evaluation output.

        Args:
            text: Evaluation output text
            key: Key to look for

        Returns:
            Parsed list of items
        """
        import re

        pattern = rf'{key}:\s*(.+?)(?=\n[A-Z]|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            content = match.group(1).strip()
            if content.lower() == "none":
                return []

            # Parse as list items
            items = re.findall(r'[-•*]\s*(.+?)(?=\n|$)', content)
            if items:
                return [item.strip() for item in items]

            # Try comma-separated
            if ',' in content:
                return [item.strip() for item in content.split(',')]

            return [content] if content else []

        return []

    def evaluate_groundedness(
        self,
        response: str,
        evidence: list[Any]
    ) -> EvaluationResult:
        """
        Evaluate whether a response is grounded in evidence.

        Args:
            response: Response text to evaluate
            evidence: List of evidence items

        Returns:
            EvaluationResult with groundedness assessment
        """
        # Format evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {e.text if hasattr(e, 'text') else str(e)}"
            for i, e in enumerate(evidence[:10])  # Limit evidence
        ])

        prompt = self.GROUNDEDNESS_PROMPT.format(
            evidence=evidence_text,
            response=response
        )

        result_text = self._call_llm(prompt)

        # Parse results
        groundedness_score = self._parse_score(result_text, "GROUNDEDNESS_SCORE")
        unsupported_claims = self._parse_list(result_text, "UNSUPPORTED_CLAIMS")

        # Determine if grounded (threshold depends on strict_mode)
        threshold = 0.8 if self.strict_mode else 0.6
        is_grounded = groundedness_score >= threshold and len(unsupported_claims) == 0

        return EvaluationResult(
            response=response,
            overall_score=groundedness_score,
            scores=[
                EvaluationScore(
                    criterion=EvaluationCriteria.GROUNDEDNESS,
                    score=groundedness_score,
                    explanation=result_text
                )
            ],
            is_grounded=is_grounded,
            unsupported_claims=unsupported_claims,
            metadata={"evaluation_type": "groundedness"}
        )

    def evaluate_quality(
        self,
        response: str,
        query: str,
        evidence: list[Any]
    ) -> EvaluationResult:
        """
        Evaluate overall response quality.

        Args:
            response: Response text to evaluate
            query: Original query
            evidence: List of evidence items

        Returns:
            EvaluationResult with quality assessment
        """
        # Format evidence
        evidence_text = "\n\n".join([
            f"[{i+1}] {e.text if hasattr(e, 'text') else str(e)}"
            for i, e in enumerate(evidence[:10])
        ])

        prompt = self.QUALITY_PROMPT.format(
            query=query,
            evidence=evidence_text,
            response=response
        )

        result_text = self._call_llm(prompt)

        # Parse scores
        scores = [
            EvaluationScore(
                criterion=EvaluationCriteria.COMPLETENESS,
                score=self._parse_score(result_text, "COMPLETENESS"),
                explanation=""
            ),
            EvaluationScore(
                criterion=EvaluationCriteria.FAITHFULNESS,
                score=self._parse_score(result_text, "FAITHFULNESS"),
                explanation=""
            ),
            EvaluationScore(
                criterion=EvaluationCriteria.RELEVANCE,
                score=self._parse_score(result_text, "RELEVANCE"),
                explanation=""
            ),
            EvaluationScore(
                criterion=EvaluationCriteria.COHERENCE,
                score=self._parse_score(result_text, "COHERENCE"),
                explanation=""
            )
        ]

        overall_score = self._parse_score(result_text, "OVERALL")

        return EvaluationResult(
            response=response,
            overall_score=overall_score,
            scores=scores,
            is_grounded=scores[1].score >= 0.6,  # Faithfulness as proxy
            metadata={
                "evaluation_type": "quality",
                "raw_output": result_text
            }
        )

    def evaluate(
        self,
        response: str,
        query: str,
        evidence: list[Any],
        check_groundedness: bool = True,
        check_quality: bool = True
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation.

        Args:
            response: Response text to evaluate
            query: Original query
            evidence: List of evidence items
            check_groundedness: Whether to check groundedness
            check_quality: Whether to check quality

        Returns:
            Combined EvaluationResult
        """
        all_scores = []
        unsupported_claims = []
        is_grounded = True

        if check_groundedness:
            groundedness_result = self.evaluate_groundedness(response, evidence)
            all_scores.extend(groundedness_result.scores)
            unsupported_claims = groundedness_result.unsupported_claims
            is_grounded = groundedness_result.is_grounded

        if check_quality:
            quality_result = self.evaluate_quality(response, query, evidence)
            all_scores.extend(quality_result.scores)

        # Calculate overall score
        if all_scores:
            overall_score = sum(s.score for s in all_scores) / len(all_scores)
        else:
            overall_score = 0.5

        return EvaluationResult(
            response=response,
            overall_score=overall_score,
            scores=all_scores,
            is_grounded=is_grounded,
            unsupported_claims=unsupported_claims,
            metadata={
                "evaluation_type": "comprehensive",
                "num_criteria": len(all_scores)
            }
        )

    def compare_responses(
        self,
        responses: list[str],
        query: str,
        evidence: list[Any]
    ) -> int:
        """
        Compare multiple responses and select the best.

        Args:
            responses: List of response texts
            query: Original query
            evidence: List of evidence items

        Returns:
            Index of the best response
        """
        if len(responses) == 1:
            return 0

        evaluations = []
        for response in responses:
            try:
                eval_result = self.evaluate(
                    response=response,
                    query=query,
                    evidence=evidence,
                    check_groundedness=True,
                    check_quality=True
                )
                evaluations.append(eval_result)
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                evaluations.append(EvaluationResult(
                    response=response,
                    overall_score=0.0
                ))

        # Find best response
        best_idx = 0
        best_score = evaluations[0].overall_score

        for i, eval_result in enumerate(evaluations[1:], 1):
            # Prefer grounded responses
            if eval_result.is_grounded and not evaluations[best_idx].is_grounded:
                best_idx = i
                best_score = eval_result.overall_score
            elif eval_result.overall_score > best_score:
                if eval_result.is_grounded or not evaluations[best_idx].is_grounded:
                    best_idx = i
                    best_score = eval_result.overall_score

        logger.info(
            f"Selected response {best_idx} with score {best_score:.2f}"
        )

        return best_idx
