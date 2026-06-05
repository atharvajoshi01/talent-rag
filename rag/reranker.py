"""
Reranker Module.

This module provides reranking functionality to improve
retrieval quality by re-scoring retrieved documents.
"""

from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from ..vectorstore.faiss_store import SearchResult


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.

    Rerankers take retrieved results and re-score them
    for improved relevance ranking.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5
    ) -> list[SearchResult]:
        """
        Rerank search results.

        Args:
            query: Original query string
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of SearchResult objects
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using a cross-encoder model.

    Cross-encoders process query-document pairs together,
    providing more accurate relevance scores than bi-encoders.

    Attributes:
        model: Cross-encoder model instance
        model_name: Name of the cross-encoder model
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5
    ) -> list[SearchResult]:
        """
        Rerank results using cross-encoder scoring.

        Args:
            query: Original query string
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        # Create query-document pairs
        pairs = [(query, result.text) for result in results]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Combine with results
        scored_results = list(zip(scores, results))
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (score, result) in enumerate(scored_results[:top_k]):
            reranked.append(SearchResult(
                id=result.id,
                score=float(score),
                text=result.text,
                metadata=result.metadata,
                rank=rank
            ))

        logger.debug(f"Reranked {len(results)} results to top {len(reranked)}")
        return reranked


class LLMReranker(BaseReranker):
    """
    Reranker using an LLM for relevance scoring.

    This reranker uses an LLM to score document relevance,
    providing high-quality but slower reranking.

    Attributes:
        client: OpenAI client instance
        model: LLM model name
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM reranker.

        Args:
            model: LLM model to use for scoring
            api_key: OpenAI API key
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        self.model = model
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()

        logger.info(f"LLMReranker initialized with model: {model}")

    def _score_document(self, query: str, document: str) -> float:
        """
        Score a single document's relevance to the query.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score between 0 and 1
        """
        prompt = f"""Rate the relevance of the following document to the query.
Return ONLY a number between 0 and 10, where:
- 0 means completely irrelevant
- 10 means perfectly relevant

Query: {query}

Document: {document[:1000]}

Relevance score (0-10):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=5
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text) / 10.0  # Normalize to 0-1
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            return 0.5  # Default score on error

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5
    ) -> list[SearchResult]:
        """
        Rerank results using LLM scoring.

        Args:
            query: Original query string
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        # Score each document
        scored_results = []
        for result in results:
            score = self._score_document(query, result.text)
            scored_results.append((score, result))

        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (score, result) in enumerate(scored_results[:top_k]):
            reranked.append(SearchResult(
                id=result.id,
                score=score,
                text=result.text,
                metadata=result.metadata,
                rank=rank
            ))

        return reranked


class HybridScoreReranker(BaseReranker):
    """
    Lightweight reranker using combined scoring heuristics.

    This reranker doesn't require additional models and uses
    a combination of the original score with keyword and
    metadata-based adjustments.

    Attributes:
        keyword_boost: Weight for keyword matching
        recency_boost: Weight for document recency
        length_penalty: Penalty for very short documents
    """

    def __init__(
        self,
        keyword_boost: float = 0.2,
        skill_match_boost: float = 0.3,
        length_penalty: float = 0.1
    ):
        """
        Initialize the hybrid score reranker.

        Args:
            keyword_boost: Boost for keyword matches
            skill_match_boost: Boost for skill matches
            length_penalty: Penalty for short documents
        """
        self.keyword_boost = keyword_boost
        self.skill_match_boost = skill_match_boost
        self.length_penalty = length_penalty

        logger.info("HybridScoreReranker initialized")

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract important keywords from text."""
        import re

        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Filter common words
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'would', 'could', 'there', 'their', 'will',
            'when', 'who', 'with', 'this', 'from', 'that', 'what'
        }

        return {w for w in words if w not in stopwords}

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
        target_skills: Optional[list[str]] = None
    ) -> list[SearchResult]:
        """
        Rerank results using hybrid scoring.

        Args:
            query: Original query string
            results: List of search results to rerank
            top_k: Number of top results to return
            target_skills: Optional skills to boost matches for

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        query_keywords = self._extract_keywords(query)
        target_skills_lower = {s.lower() for s in (target_skills or [])}

        scored_results = []

        for result in results:
            # Start with original score (normalized)
            base_score = result.score

            # Keyword matching boost
            doc_keywords = self._extract_keywords(result.text)
            if query_keywords and doc_keywords:
                keyword_overlap = len(query_keywords & doc_keywords) / len(query_keywords)
                base_score += self.keyword_boost * keyword_overlap

            # Skill matching boost
            if target_skills_lower:
                doc_skills = {s.lower() for s in result.metadata.get("skills", [])}
                if doc_skills:
                    skill_overlap = len(target_skills_lower & doc_skills) / len(target_skills_lower)
                    base_score += self.skill_match_boost * skill_overlap

            # Length penalty for very short documents
            doc_length = len(result.text.split())
            if doc_length < 50:
                base_score -= self.length_penalty * (1 - doc_length / 50)

            scored_results.append((base_score, result))

        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Build reranked results
        reranked = []
        for rank, (score, result) in enumerate(scored_results[:top_k]):
            reranked.append(SearchResult(
                id=result.id,
                score=score,
                text=result.text,
                metadata=result.metadata,
                rank=rank
            ))

        logger.debug(f"Reranked {len(results)} results to top {len(reranked)}")
        return reranked


class Reranker:
    """
    Main reranker class with multiple strategy support.

    This class provides a unified interface for reranking with
    support for different reranking strategies.

    Attributes:
        strategy: Current reranking strategy
        reranker: Underlying reranker implementation
    """

    STRATEGIES = {
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker,
        "hybrid_score": HybridScoreReranker
    }

    def __init__(
        self,
        strategy: str = "hybrid_score",
        **kwargs
    ):
        """
        Initialize the reranker.

        Args:
            strategy: Reranking strategy to use
            **kwargs: Arguments passed to the underlying reranker
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(self.STRATEGIES.keys())}"
            )

        self.strategy = strategy
        self.reranker = self.STRATEGIES[strategy](**kwargs)

        logger.info(f"Reranker initialized with strategy: {strategy}")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
        **kwargs
    ) -> list[SearchResult]:
        """
        Rerank search results.

        Args:
            query: Original query string
            results: List of search results to rerank
            top_k: Number of top results to return
            **kwargs: Additional arguments for the reranker

        Returns:
            Reranked list of SearchResult objects
        """
        return self.reranker.rerank(query, results, top_k, **kwargs)
