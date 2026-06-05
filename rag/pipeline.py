"""
RAG Pipeline Module.

This module provides the complete RAG pipeline integrating
retrieval, reranking, generation, and evaluation.
"""

import time
from typing import Any, Optional
from dataclasses import dataclass, field

from loguru import logger

from ..embeddings import EmbeddingModel
from ..vectorstore import FAISSVectorStore
from .retriever import RAGRetriever, RetrievalResult
from .reranker import Reranker
from .prompt_builder import PromptBuilder, PromptContext, QueryType
from .generator import Generator, GenerationResult
from .evaluator import ResponseEvaluator, EvaluationResult


@dataclass
class RAGResponse:
    """
    Complete RAG pipeline response.

    Attributes:
        answer: Generated answer text
        query: Original query
        evidence_used: List of evidence chunks used
        citations: List of citation IDs in the response
        retrieval_result: Full retrieval result
        generation_result: Full generation result
        evaluation_result: Optional evaluation result
        latency_ms: Total pipeline latency
        metadata: Additional metadata
    """
    answer: str
    query: str
    evidence_used: list[dict[str, Any]]
    citations: list[str]
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    evaluation_result: Optional[EvaluationResult] = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "answer": self.answer,
            "query": self.query,
            "evidence_used": self.evidence_used,
            "citations": self.citations,
            "latency_ms": self.latency_ms,
            "metadata": {
                **self.metadata,
                "retrieval_count": self.retrieval_result.total_retrieved,
                "model": self.generation_result.model,
                "prompt_tokens": self.generation_result.prompt_tokens,
                "completion_tokens": self.generation_result.completion_tokens
            }
        }


class RAGPipeline:
    """
    Complete RAG pipeline for the Talent Intelligence Assistant.

    This class orchestrates the full RAG workflow from query
    to response, including retrieval, reranking, generation,
    and optional evaluation.

    Attributes:
        retriever: RAG retriever instance
        reranker: Reranker instance
        generator: Generator instance
        evaluator: Optional evaluator instance
        prompt_builder: PromptBuilder instance
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        generator_model: str = "llama3.2",
        reranker_strategy: str = "hybrid_score",
        use_evaluator: bool = False,
        retrieval_k: int = 10,
        rerank_k: int = 5,
        llm_provider: str = "ollama",
        ollama_host: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: FAISS vector store with indexed documents
            embedding_model: Embedding model for queries
            generator_model: LLM model for generation
            reranker_strategy: Reranking strategy to use
            use_evaluator: Whether to evaluate responses
            retrieval_k: Number of documents to retrieve
            rerank_k: Number of documents after reranking
            llm_provider: LLM provider ('openai' or 'ollama')
            ollama_host: Ollama server host URL
            api_key: API key for LLM services
        """
        # Initialize components
        self.retriever = RAGRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            default_k=retrieval_k
        )

        self.reranker = Reranker(strategy=reranker_strategy)

        self.prompt_builder = PromptBuilder()

        self.generator = Generator(
            model=generator_model,
            llm_provider=llm_provider,
            ollama_host=ollama_host,
            api_key=api_key,
            prompt_builder=self.prompt_builder
        )

        self.evaluator = None
        if use_evaluator:
            self.evaluator = ResponseEvaluator(api_key=api_key)

        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k

        logger.info(
            f"RAGPipeline initialized with provider={llm_provider}, "
            f"model={generator_model}, retrieval_k={retrieval_k}, rerank_k={rerank_k}"
        )

    def query(
        self,
        query: str,
        role_context: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
        evaluate: bool = False
    ) -> RAGResponse:
        """
        Process a query through the full RAG pipeline.

        Args:
            query: User's query string
            role_context: Optional role context for matching
            filters: Optional filters for retrieval
            evaluate: Whether to evaluate the response

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()

        # Detect query type
        query_type = self.prompt_builder.detect_query_type(query)

        # Retrieve relevant documents
        if filters:
            retrieval_result = self.retriever.retrieve_for_candidate_search(
                query=query,
                k=self.retrieval_k,
                required_skills=filters.get("skills"),
                location=filters.get("location"),
                seniority=filters.get("seniority"),
                min_experience=filters.get("min_experience"),
                max_experience=filters.get("max_experience")
            )
        else:
            retrieval_result = self.retriever.retrieve(
                query=query,
                k=self.retrieval_k
            )

        # Rerank results
        reranked_results = self.reranker.rerank(
            query=query,
            results=retrieval_result.results,
            top_k=self.rerank_k
        )

        # Build prompt context
        context = PromptContext(
            query=query,
            query_type=query_type,
            evidence=reranked_results,
            role=role_context
        )

        # Generate response
        generation_result = self.generator.generate(context)

        # Build evidence used list
        evidence_used = [
            {
                "id": r.id,
                "text": r.text[:500],
                "score": r.score,
                "metadata": r.metadata
            }
            for r in reranked_results
        ]

        # Optionally evaluate
        evaluation_result = None
        if evaluate and self.evaluator:
            evaluation_result = self.evaluator.evaluate(
                response=generation_result.response,
                query=query,
                evidence=reranked_results
            )

        latency_ms = (time.time() - start_time) * 1000

        return RAGResponse(
            answer=generation_result.response,
            query=query,
            evidence_used=evidence_used,
            citations=generation_result.citations,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            evaluation_result=evaluation_result,
            latency_ms=latency_ms,
            metadata={
                "query_type": query_type.value,
                "has_role_context": role_context is not None,
                "has_filters": filters is not None
            }
        )

    def search_candidates(
        self,
        query: str,
        role: Optional[dict[str, Any]] = None,
        skills: Optional[list[str]] = None,
        location: Optional[str] = None,
        seniority: Optional[list[str]] = None,
        min_experience: Optional[int] = None,
        max_experience: Optional[int] = None,
        k: int = 10
    ) -> RAGResponse:
        """
        Search for candidates matching criteria.

        Args:
            query: Search query
            role: Optional role to match against
            skills: Required skills
            location: Location filter
            seniority: Seniority levels
            min_experience: Minimum experience
            max_experience: Maximum experience
            k: Number of candidates

        Returns:
            RAGResponse with candidate matches
        """
        # Build filters from role if provided
        if role:
            skills = skills or role.get("required_skills", [])
            if role.get("nice_to_have_skills"):
                skills = skills + role["nice_to_have_skills"]
            location = location or role.get("location")
            seniority = seniority or ([role.get("seniority")] if role.get("seniority") else None)
            min_experience = min_experience or role.get("years_experience_required")

        filters = {
            "skills": skills,
            "location": location,
            "seniority": seniority,
            "min_experience": min_experience,
            "max_experience": max_experience
        }

        # Remove None values
        filters = {k: v for k, v in filters.items() if v is not None}

        return self.query(
            query=query,
            role_context=role,
            filters=filters if filters else None
        )

    def compare_candidates(
        self,
        candidate_ids: list[str],
        role: dict[str, Any]
    ) -> RAGResponse:
        """
        Compare multiple candidates for a role.

        Args:
            candidate_ids: List of candidate IDs to compare
            role: Role to compare against

        Returns:
            RAGResponse with comparison analysis
        """
        start_time = time.time()

        # Retrieve evidence for each candidate
        all_evidence = []
        for candidate_id in candidate_ids:
            evidence = self.retriever.retrieve_candidate_details(candidate_id)
            all_evidence.extend(evidence)

        # Rerank all evidence
        query = f"Compare candidates {', '.join(candidate_ids)} for {role.get('title', 'the role')}"
        reranked = self.reranker.rerank(
            query=query,
            results=all_evidence,
            top_k=self.rerank_k * len(candidate_ids)  # More evidence for comparison
        )

        # Build context
        context = self.prompt_builder.build_comparison_prompt(
            candidates=[{"candidate_id": cid} for cid in candidate_ids],
            role=role,
            evidence=reranked
        )

        # Generate comparison
        generation_result = self.generator.generate(context)

        # Build evidence used
        evidence_used = [
            {
                "id": r.id,
                "text": r.text[:500],
                "score": r.score,
                "metadata": r.metadata
            }
            for r in reranked
        ]

        latency_ms = (time.time() - start_time) * 1000

        # Create retrieval result for consistency
        retrieval_result = RetrievalResult(
            query=query,
            results=reranked,
            total_retrieved=len(all_evidence),
            retrieval_type="comparison"
        )

        return RAGResponse(
            answer=generation_result.response,
            query=query,
            evidence_used=evidence_used,
            citations=generation_result.citations,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            latency_ms=latency_ms,
            metadata={
                "query_type": "comparison",
                "candidate_ids": candidate_ids,
                "role_id": role.get("id")
            }
        )

    def ask(
        self,
        question: str,
        context_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Answer a general question about candidates/roles.

        Args:
            question: User's question
            context_filter: Optional filter ('candidates', 'roles', or None)

        Returns:
            RAGResponse with answer
        """
        # Add context filter to retrieval if needed
        filters = None
        if context_filter == "candidates":
            filters = {"source": ["resume", "interview"]}
        elif context_filter == "roles":
            filters = {"source": "role"}

        return self.query(question, filters=filters)

    def stream_query(
        self,
        query: str,
        role_context: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None
    ):
        """
        Stream a query response.

        Args:
            query: User's query
            role_context: Optional role context
            filters: Optional filters

        Yields:
            Response chunks as they are generated
        """
        # Detect query type
        query_type = self.prompt_builder.detect_query_type(query)

        # Retrieve and rerank
        if filters:
            retrieval_result = self.retriever.retrieve_for_candidate_search(
                query=query,
                k=self.retrieval_k,
                **filters
            )
        else:
            retrieval_result = self.retriever.retrieve(
                query=query,
                k=self.retrieval_k
            )

        reranked_results = self.reranker.rerank(
            query=query,
            results=retrieval_result.results,
            top_k=self.rerank_k
        )

        # Build context
        context = PromptContext(
            query=query,
            query_type=query_type,
            evidence=reranked_results,
            role=role_context
        )

        # Stream generation
        yield from self.generator.stream_generate(context)
