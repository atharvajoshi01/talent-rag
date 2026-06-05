"""
Hybrid Retrieval Module.

This module provides hybrid search combining semantic similarity
with keyword matching and metadata filtering.
"""

import re
from collections import defaultdict
from typing import Any, Optional
from dataclasses import dataclass

from loguru import logger

from ..embeddings import EmbeddingModel
from ..vectorstore import FAISSVectorStore
from ..vectorstore.faiss_store import SearchResult


@dataclass
class HybridSearchConfig:
    """
    Configuration for hybrid search.

    Attributes:
        semantic_weight: Weight for semantic similarity score
        keyword_weight: Weight for keyword matching score
        metadata_weight: Weight for metadata matching score
        skill_match_boost: Bonus for skill matches
        location_match_boost: Bonus for location matches
        seniority_match_boost: Bonus for seniority matches
    """
    semantic_weight: float = 0.6
    keyword_weight: float = 0.2
    metadata_weight: float = 0.2
    skill_match_boost: float = 0.1
    location_match_boost: float = 0.05
    seniority_match_boost: float = 0.05


class HybridRetriever:
    """
    Hybrid retriever combining semantic and lexical search.

    This class provides advanced retrieval by combining vector
    similarity with keyword matching and metadata-based boosting.

    Attributes:
        vector_store: FAISS vector store instance
        embedding_model: Model for generating query embeddings
        config: Hybrid search configuration
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        config: Optional[HybridSearchConfig] = None
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_store: FAISS vector store with indexed documents
            embedding_model: Embedding model for query encoding
            config: Hybrid search configuration
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or HybridSearchConfig()

        logger.info(
            f"HybridRetriever initialized with weights: "
            f"semantic={self.config.semantic_weight}, "
            f"keyword={self.config.keyword_weight}, "
            f"metadata={self.config.metadata_weight}"
        )

    def _extract_keywords(self, text: str) -> set[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text

        Returns:
            Set of lowercase keywords
        """
        # Simple keyword extraction - remove common words
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why',
            'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should'
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

        # Filter stopwords and return
        return {w for w in words if w not in stopwords}

    def _calculate_keyword_score(
        self,
        query_keywords: set[str],
        document_text: str
    ) -> float:
        """
        Calculate keyword matching score.

        Args:
            query_keywords: Set of query keywords
            document_text: Document text to match against

        Returns:
            Score between 0 and 1
        """
        if not query_keywords:
            return 0.0

        doc_keywords = self._extract_keywords(document_text)

        if not doc_keywords:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(query_keywords & doc_keywords)
        union = len(query_keywords | doc_keywords)

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_metadata_score(
        self,
        metadata: dict[str, Any],
        target_skills: Optional[list[str]] = None,
        target_location: Optional[str] = None,
        target_seniority: Optional[list[str]] = None,
        min_experience: Optional[int] = None,
        max_experience: Optional[int] = None
    ) -> float:
        """
        Calculate metadata matching score.

        Args:
            metadata: Document metadata
            target_skills: Required skills
            target_location: Required location
            target_seniority: Required seniority levels
            min_experience: Minimum years of experience
            max_experience: Maximum years of experience

        Returns:
            Score between 0 and 1
        """
        score = 0.0
        max_possible = 0.0

        # Skill matching
        if target_skills:
            max_possible += 1.0
            doc_skills = {s.lower() for s in metadata.get("skills", [])}
            target_skills_lower = {s.lower() for s in target_skills}

            if doc_skills and target_skills_lower:
                skill_overlap = len(doc_skills & target_skills_lower)
                skill_score = skill_overlap / len(target_skills_lower)
                score += skill_score

        # Location matching
        if target_location:
            max_possible += 0.5
            doc_location = metadata.get("location", "").lower()
            if target_location.lower() in doc_location or doc_location in target_location.lower():
                score += 0.5
            elif "remote" in doc_location:
                score += 0.3  # Partial credit for remote

        # Seniority matching
        if target_seniority:
            max_possible += 0.5
            doc_seniority = metadata.get("seniority", "").lower()
            if any(s.lower() == doc_seniority for s in target_seniority):
                score += 0.5

        # Experience range matching
        if min_experience is not None or max_experience is not None:
            max_possible += 0.5
            doc_experience = metadata.get("years_experience", 0)

            in_range = True
            if min_experience is not None and doc_experience < min_experience:
                in_range = False
            if max_experience is not None and doc_experience > max_experience:
                in_range = False

            if in_range:
                score += 0.5

        # Normalize score
        if max_possible > 0:
            return score / max_possible
        return 0.0

    def retrieve(
        self,
        query: str,
        k: int = 10,
        target_skills: Optional[list[str]] = None,
        target_location: Optional[str] = None,
        target_seniority: Optional[list[str]] = None,
        min_experience: Optional[int] = None,
        max_experience: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Perform hybrid retrieval combining semantic and metadata search.

        Args:
            query: Search query text
            k: Number of results to return
            target_skills: Required skills for filtering/boosting
            target_location: Required location
            target_seniority: Required seniority levels
            min_experience: Minimum years of experience
            max_experience: Maximum years of experience
            source_filter: Filter by document source

        Returns:
            List of SearchResult objects with combined scores
        """
        # Get more candidates for reranking
        candidate_k = k * 3

        # Semantic search
        query_embedding = self.embedding_model.embed_query(query)

        if source_filter:
            candidates = self.vector_store.search_with_metadata_filter(
                query_embedding,
                k=candidate_k,
                filters={"source": source_filter}
            )
        else:
            candidates = self.vector_store.search(query_embedding, k=candidate_k)

        if not candidates:
            return []

        # Extract query keywords for lexical matching
        query_keywords = self._extract_keywords(query)

        # Add skill terms to keywords if provided
        if target_skills:
            query_keywords.update(s.lower() for s in target_skills)

        # Rerank with hybrid scoring
        scored_results = []

        # Normalize semantic scores to 0-1 range
        max_semantic = max(c.score for c in candidates) if candidates else 1.0
        min_semantic = min(c.score for c in candidates) if candidates else 0.0
        semantic_range = max_semantic - min_semantic if max_semantic != min_semantic else 1.0

        for candidate in candidates:
            # Normalize semantic score
            semantic_score = (candidate.score - min_semantic) / semantic_range

            # Calculate keyword score
            keyword_score = self._calculate_keyword_score(
                query_keywords,
                candidate.text
            )

            # Calculate metadata score
            metadata_score = self._calculate_metadata_score(
                candidate.metadata,
                target_skills=target_skills,
                target_location=target_location,
                target_seniority=target_seniority,
                min_experience=min_experience,
                max_experience=max_experience
            )

            # Combine scores
            combined_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score +
                self.config.metadata_weight * metadata_score
            )

            scored_results.append((combined_score, candidate))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Build final results with updated scores
        results = []
        for rank, (score, candidate) in enumerate(scored_results[:k]):
            result = SearchResult(
                id=candidate.id,
                score=score,
                text=candidate.text,
                metadata=candidate.metadata,
                rank=rank
            )
            results.append(result)

        logger.debug(
            f"Hybrid search for '{query[:50]}...' returned {len(results)} results"
        )

        return results

    def search_candidates_for_role(
        self,
        role: dict[str, Any],
        k: int = 10,
        include_interviews: bool = True
    ) -> list[dict[str, Any]]:
        """
        Search for candidates matching a specific role.

        This method aggregates results across candidate chunks and
        returns deduplicated candidate-level results.

        Args:
            role: Role dictionary with required_skills, location, seniority, etc.
            k: Number of candidates to return
            include_interviews: Whether to include interview transcripts

        Returns:
            List of candidate match dictionaries with scores and evidence
        """
        # Extract role requirements
        query = f"{role.get('title', '')} {role.get('description', '')}"
        required_skills = role.get("required_skills", [])
        nice_to_have = role.get("nice_to_have_skills", [])
        all_skills = required_skills + nice_to_have
        location = role.get("location")
        seniority = [role.get("seniority")] if role.get("seniority") else None
        min_exp = role.get("years_experience_required")

        # Search for matching chunks
        source_filter = None if include_interviews else "resume"

        results = self.retrieve(
            query=query,
            k=k * 5,  # Get more to deduplicate
            target_skills=all_skills,
            target_location=location,
            target_seniority=seniority,
            min_experience=min_exp,
            source_filter=source_filter
        )

        # Aggregate by candidate
        candidate_scores: dict[str, dict] = defaultdict(
            lambda: {"scores": [], "evidence": [], "metadata": {}}
        )

        for result in results:
            candidate_id = result.metadata.get("candidate_id", "unknown")

            candidate_scores[candidate_id]["scores"].append(result.score)
            candidate_scores[candidate_id]["evidence"].append({
                "chunk_id": result.id,
                "text": result.text[:500],  # Truncate for readability
                "score": result.score,
                "source": result.metadata.get("source", "unknown")
            })

            # Store candidate metadata
            if not candidate_scores[candidate_id]["metadata"]:
                candidate_scores[candidate_id]["metadata"] = {
                    "name": result.metadata.get("candidate_name", ""),
                    "location": result.metadata.get("location", ""),
                    "seniority": result.metadata.get("seniority", ""),
                    "years_experience": result.metadata.get("years_experience", 0),
                    "skills": result.metadata.get("skills", [])
                }

        # Calculate aggregate scores
        candidate_results = []

        for candidate_id, data in candidate_scores.items():
            if not data["scores"]:
                continue

            # Use max score with count bonus
            max_score = max(data["scores"])
            count_bonus = min(0.1, len(data["scores"]) * 0.02)
            aggregate_score = max_score + count_bonus

            # Calculate skill match
            candidate_skills = set(s.lower() for s in data["metadata"].get("skills", []))
            required_match = len(candidate_skills & set(s.lower() for s in required_skills))
            nice_match = len(candidate_skills & set(s.lower() for s in nice_to_have))

            candidate_results.append({
                "candidate_id": candidate_id,
                "match_score": aggregate_score,
                "metadata": data["metadata"],
                "skill_match": {
                    "required_matched": required_match,
                    "required_total": len(required_skills),
                    "nice_to_have_matched": nice_match,
                    "nice_to_have_total": len(nice_to_have)
                },
                "evidence": data["evidence"][:3]  # Top 3 evidence chunks
            })

        # Sort by match score
        candidate_results.sort(key=lambda x: x["match_score"], reverse=True)

        return candidate_results[:k]

    def compare_candidates(
        self,
        candidate_ids: list[str],
        role: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compare multiple candidates against a role.

        Args:
            candidate_ids: List of candidate IDs to compare
            role: Role to compare against

        Returns:
            Comparison dictionary with scores and analysis
        """
        comparison = {
            "role": {
                "id": role.get("id"),
                "title": role.get("title"),
                "required_skills": role.get("required_skills", []),
                "nice_to_have_skills": role.get("nice_to_have_skills", [])
            },
            "candidates": []
        }

        required_skills = role.get("required_skills", [])
        nice_to_have = role.get("nice_to_have_skills", [])

        for candidate_id in candidate_ids:
            # Get all chunks for this candidate
            candidate_chunks = [
                self.vector_store.get_document(doc_id)
                for doc_id in self.vector_store.get_all_ids()
                if doc_id.startswith(candidate_id)
            ]

            if not candidate_chunks:
                continue

            # Get metadata from first chunk
            metadata = candidate_chunks[0]["metadata"] if candidate_chunks else {}
            candidate_skills = set(s.lower() for s in metadata.get("skills", []))

            # Calculate skill matches
            required_matches = [
                s for s in required_skills
                if s.lower() in candidate_skills
            ]
            nice_matches = [
                s for s in nice_to_have
                if s.lower() in candidate_skills
            ]
            missing_required = [
                s for s in required_skills
                if s.lower() not in candidate_skills
            ]

            # Build candidate comparison entry
            candidate_entry = {
                "candidate_id": candidate_id,
                "name": metadata.get("candidate_name", ""),
                "location": metadata.get("location", ""),
                "seniority": metadata.get("seniority", ""),
                "years_experience": metadata.get("years_experience", 0),
                "all_skills": metadata.get("skills", []),
                "skill_analysis": {
                    "required_matched": required_matches,
                    "required_missing": missing_required,
                    "nice_to_have_matched": nice_matches,
                    "match_percentage": (
                        len(required_matches) / len(required_skills) * 100
                        if required_skills else 100
                    )
                }
            }

            comparison["candidates"].append(candidate_entry)

        # Sort by match percentage
        comparison["candidates"].sort(
            key=lambda x: x["skill_analysis"]["match_percentage"],
            reverse=True
        )

        return comparison
