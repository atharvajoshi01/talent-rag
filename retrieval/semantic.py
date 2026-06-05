"""
Semantic Retrieval Module.

This module provides semantic search functionality using
vector similarity for the Talent RAG system.
"""

from typing import Any, Optional
import numpy as np

from loguru import logger

from ..embeddings import EmbeddingModel
from ..vectorstore import FAISSVectorStore
from ..vectorstore.faiss_store import SearchResult


class SemanticRetriever:
    """
    Semantic retriever using vector similarity search.

    This class provides semantic search capabilities by embedding
    queries and finding similar documents in the vector store.

    Attributes:
        vector_store: FAISS vector store instance
        embedding_model: Model for generating query embeddings
        default_k: Default number of results to return
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        default_k: int = 10
    ):
        """
        Initialize the semantic retriever.

        Args:
            vector_store: FAISS vector store with indexed documents
            embedding_model: Embedding model for query encoding
            default_k: Default number of results to return
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.default_k = default_k

        logger.info(
            f"SemanticRetriever initialized with "
            f"{len(vector_store)} documents"
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: float = 0.0
    ) -> list[SearchResult]:
        """
        Retrieve documents semantically similar to the query.

        Args:
            query: Search query text
            k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of SearchResult objects sorted by relevance
        """
        k = k or self.default_k

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)

        # Filter by minimum score
        if min_score > 0:
            results = [r for r in results if r.score >= min_score]

        logger.debug(
            f"Semantic search for '{query[:50]}...' returned {len(results)} results"
        )

        return results

    def retrieve_for_candidate_matching(
        self,
        role_description: str,
        k: int = 20,
        source_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Retrieve candidate chunks matching a role description.

        Args:
            role_description: Job role description text
            k: Number of results to return
            source_filter: Filter by source type ('resume' or 'interview')

        Returns:
            List of SearchResult objects
        """
        query_embedding = self.embedding_model.embed_query(role_description)

        if source_filter:
            results = self.vector_store.search_with_metadata_filter(
                query_embedding,
                k=k,
                filters={"source": source_filter}
            )
        else:
            results = self.vector_store.search(query_embedding, k=k)

        return results

    def retrieve_by_skills(
        self,
        query: str,
        required_skills: list[str],
        k: int = 10,
        skill_match_threshold: float = 0.5
    ) -> list[SearchResult]:
        """
        Retrieve documents matching query and skill requirements.

        Args:
            query: Search query text
            required_skills: List of required skills
            k: Number of results to return
            skill_match_threshold: Minimum fraction of skills to match

        Returns:
            List of SearchResult objects
        """
        query_embedding = self.embedding_model.embed_query(query)

        results = self.vector_store.search_by_skills(
            query_embedding,
            required_skills=required_skills,
            k=k,
            match_threshold=skill_match_threshold
        )

        return results

    def retrieve_with_filters(
        self,
        query: str,
        k: int = 10,
        location: Optional[str] = None,
        seniority: Optional[list[str]] = None,
        min_experience: Optional[int] = None,
        max_experience: Optional[int] = None
    ) -> list[SearchResult]:
        """
        Retrieve documents with metadata filters.

        Args:
            query: Search query text
            k: Number of results
            location: Filter by location
            seniority: Filter by seniority levels
            min_experience: Minimum years of experience
            max_experience: Maximum years of experience

        Returns:
            List of SearchResult objects
        """
        query_embedding = self.embedding_model.embed_query(query)

        # Build filters
        filters = {}

        if location:
            filters["location"] = location

        if seniority:
            filters["seniority"] = seniority

        if min_experience is not None or max_experience is not None:
            exp_filter = {}
            if min_experience is not None:
                exp_filter["min"] = min_experience
            if max_experience is not None:
                exp_filter["max"] = max_experience
            filters["years_experience"] = exp_filter

        results = self.vector_store.search_with_metadata_filter(
            query_embedding,
            k=k,
            filters=filters if filters else None
        )

        return results

    def find_similar_candidates(
        self,
        candidate_id: str,
        k: int = 5
    ) -> list[SearchResult]:
        """
        Find candidates similar to a given candidate.

        Args:
            candidate_id: ID of the reference candidate
            k: Number of similar candidates to find

        Returns:
            List of SearchResult objects (excluding the reference candidate)
        """
        # Get the candidate's resume chunks
        candidate_chunks = [
            doc_id for doc_id in self.vector_store.get_all_ids()
            if doc_id.startswith(candidate_id) and "resume" in doc_id
        ]

        if not candidate_chunks:
            logger.warning(f"No resume found for candidate {candidate_id}")
            return []

        # Get the first resume chunk for similarity search
        ref_doc = self.vector_store.get_document(candidate_chunks[0])
        if not ref_doc:
            return []

        # Search using the resume text
        query_embedding = self.embedding_model.embed_query(ref_doc["text"])

        # Get more results to filter out the reference candidate
        results = self.vector_store.search(query_embedding, k=k * 3)

        # Filter out chunks from the reference candidate
        filtered_results = []
        seen_candidates = set()

        for result in results:
            # Extract candidate ID from chunk ID
            chunk_candidate_id = result.metadata.get("candidate_id", "")

            if chunk_candidate_id == candidate_id:
                continue

            if chunk_candidate_id in seen_candidates:
                continue

            seen_candidates.add(chunk_candidate_id)
            filtered_results.append(result)

            if len(filtered_results) >= k:
                break

        return filtered_results

    def batch_retrieve(
        self,
        queries: list[str],
        k: int = 10
    ) -> list[list[SearchResult]]:
        """
        Retrieve results for multiple queries efficiently.

        Args:
            queries: List of query strings
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        all_results = []

        # Batch embed queries
        query_embeddings = self.embedding_model.embed_batch(queries)

        for query_embedding in query_embeddings:
            results = self.vector_store.search(query_embedding, k=k)
            all_results.append(results)

        return all_results
