"""
RAG Retriever Module.

This module provides the retriever component for the RAG pipeline,
responsible for fetching relevant context from the vector store.
"""

from typing import Any, Optional
from dataclasses import dataclass, field

from loguru import logger

from ..embeddings import EmbeddingModel
from ..vectorstore import FAISSVectorStore
from ..vectorstore.faiss_store import SearchResult
from ..retrieval import SemanticRetriever, HybridRetriever


@dataclass
class RetrievalResult:
    """
    Result from the RAG retriever.

    Attributes:
        query: Original query string
        results: List of search results
        total_retrieved: Total number of results retrieved
        retrieval_type: Type of retrieval used ('semantic' or 'hybrid')
        metadata: Additional metadata about the retrieval
    """
    query: str
    results: list[SearchResult]
    total_retrieved: int
    retrieval_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGRetriever:
    """
    RAG-specific retriever with multi-strategy support.

    This class provides retrieval capabilities optimized for RAG,
    supporting both semantic and hybrid retrieval strategies.

    Attributes:
        semantic_retriever: Semantic search retriever
        hybrid_retriever: Hybrid search retriever
        default_k: Default number of results to retrieve
        use_hybrid: Whether to use hybrid search by default
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        default_k: int = 10,
        use_hybrid: bool = True
    ):
        """
        Initialize the RAG retriever.

        Args:
            vector_store: FAISS vector store with indexed documents
            embedding_model: Embedding model for query encoding
            default_k: Default number of results to retrieve
            use_hybrid: Whether to use hybrid search by default
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.default_k = default_k
        self.use_hybrid = use_hybrid

        # Initialize retrievers
        self.semantic_retriever = SemanticRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model,
            default_k=default_k
        )

        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_model=embedding_model
        )

        logger.info(
            f"RAGRetriever initialized with default_k={default_k}, "
            f"use_hybrid={use_hybrid}"
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query string
            k: Number of results to retrieve
            use_hybrid: Whether to use hybrid search (overrides default)
            **kwargs: Additional arguments passed to the retriever

        Returns:
            RetrievalResult with retrieved documents
        """
        k = k or self.default_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        if use_hybrid:
            results = self.hybrid_retriever.retrieve(query, k=k, **kwargs)
            retrieval_type = "hybrid"
        else:
            results = self.semantic_retriever.retrieve(query, k=k)
            retrieval_type = "semantic"

        return RetrievalResult(
            query=query,
            results=results,
            total_retrieved=len(results),
            retrieval_type=retrieval_type,
            metadata={"k": k, "use_hybrid": use_hybrid}
        )

    def retrieve_for_candidate_search(
        self,
        query: str,
        k: int = 20,
        required_skills: Optional[list[str]] = None,
        location: Optional[str] = None,
        seniority: Optional[list[str]] = None,
        min_experience: Optional[int] = None,
        max_experience: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve candidates matching search criteria.

        Args:
            query: Search query describing ideal candidate
            k: Number of results to retrieve
            required_skills: Required skills to filter by
            location: Location filter
            seniority: Seniority level filter
            min_experience: Minimum years of experience
            max_experience: Maximum years of experience

        Returns:
            RetrievalResult with matching candidates
        """
        results = self.hybrid_retriever.retrieve(
            query=query,
            k=k,
            target_skills=required_skills,
            target_location=location,
            target_seniority=seniority,
            min_experience=min_experience,
            max_experience=max_experience,
            source_filter="resume"  # Focus on resumes for candidate search
        )

        return RetrievalResult(
            query=query,
            results=results,
            total_retrieved=len(results),
            retrieval_type="hybrid_candidate_search",
            metadata={
                "required_skills": required_skills,
                "location": location,
                "seniority": seniority,
                "experience_range": (min_experience, max_experience)
            }
        )

    def retrieve_for_role_matching(
        self,
        role: dict[str, Any],
        k: int = 20,
        include_interviews: bool = True
    ) -> RetrievalResult:
        """
        Retrieve candidates matching a specific role.

        Args:
            role: Role dictionary with requirements
            k: Number of candidates to retrieve
            include_interviews: Whether to include interview data

        Returns:
            RetrievalResult with matching candidates
        """
        # Build query from role
        query_parts = [role.get("title", "")]

        if role.get("description"):
            query_parts.append(role["description"][:500])

        if role.get("required_skills"):
            query_parts.append(f"Skills: {', '.join(role['required_skills'])}")

        query = " ".join(query_parts)

        results = self.hybrid_retriever.retrieve(
            query=query,
            k=k,
            target_skills=role.get("required_skills", []) + role.get("nice_to_have_skills", []),
            target_location=role.get("location"),
            target_seniority=[role.get("seniority")] if role.get("seniority") else None,
            min_experience=role.get("years_experience_required"),
            source_filter=None if include_interviews else "resume"
        )

        return RetrievalResult(
            query=query,
            results=results,
            total_retrieved=len(results),
            retrieval_type="role_matching",
            metadata={"role_id": role.get("id"), "role_title": role.get("title")}
        )

    def retrieve_candidate_details(
        self,
        candidate_id: str,
        include_interview: bool = True
    ) -> list[SearchResult]:
        """
        Retrieve all chunks for a specific candidate.

        Args:
            candidate_id: Candidate ID to retrieve
            include_interview: Whether to include interview chunks

        Returns:
            List of SearchResult objects for the candidate
        """
        results = []

        for doc_id in self.vector_store.get_all_ids():
            if not doc_id.startswith(candidate_id):
                continue

            doc = self.vector_store.get_document(doc_id)
            if not doc:
                continue

            # Filter by source if needed
            if not include_interview and doc["metadata"].get("source") == "interview":
                continue

            results.append(SearchResult(
                id=doc_id,
                score=1.0,  # Full match
                text=doc["text"],
                metadata=doc["metadata"],
                rank=len(results)
            ))

        return results

    def multi_query_retrieve(
        self,
        queries: list[str],
        k_per_query: int = 5,
        deduplicate: bool = True
    ) -> RetrievalResult:
        """
        Retrieve using multiple query variations.

        This method retrieves results for multiple query formulations
        and combines them for improved recall.

        Args:
            queries: List of query variations
            k_per_query: Results per query
            deduplicate: Whether to deduplicate results

        Returns:
            Combined RetrievalResult
        """
        all_results = []
        seen_ids = set()

        for query in queries:
            results = self.semantic_retriever.retrieve(query, k=k_per_query)

            for result in results:
                if deduplicate and result.id in seen_ids:
                    continue

                seen_ids.add(result.id)
                all_results.append(result)

        # Re-rank by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for i, result in enumerate(all_results):
            result.rank = i

        return RetrievalResult(
            query=queries[0],  # Primary query
            results=all_results,
            total_retrieved=len(all_results),
            retrieval_type="multi_query",
            metadata={"num_queries": len(queries), "queries": queries}
        )
