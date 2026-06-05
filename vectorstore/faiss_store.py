"""
FAISS Vector Store Module.

This module provides a FAISS-based vector store implementation
with support for both flat and HNSW indices.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")


@dataclass
class SearchResult:
    """
    Result from a vector search.

    Attributes:
        id: Document/chunk ID
        score: Similarity score
        text: Document text content
        metadata: Document metadata
        rank: Rank in search results (0-indexed)
    """
    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
            "rank": self.rank
        }


class FAISSVectorStore:
    """
    FAISS-based vector store with metadata support.

    This class provides a complete vector storage solution using FAISS
    for efficient similarity search, with additional metadata storage
    for filtering and retrieval.

    Attributes:
        dimension: Dimensionality of stored vectors
        index_type: Type of FAISS index ('flat' or 'hnsw')
        index: FAISS index object
        documents: Mapping of internal IDs to document data
        id_to_idx: Mapping of document IDs to internal indices
        idx_to_id: Mapping of internal indices to document IDs
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 128
    ):
        """
        Initialize the FAISS vector store.

        Args:
            dimension: Dimensionality of vectors
            index_type: Type of index ('flat' for IndexFlatIP, 'hnsw' for HNSW)
            hnsw_m: HNSW M parameter (connections per layer)
            hnsw_ef_construction: HNSW ef_construction parameter
            hnsw_ef_search: HNSW ef_search parameter for queries
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        # Initialize index
        self.index = self._create_index()

        # Document storage
        self.documents: dict[str, dict[str, Any]] = {}  # id -> {text, metadata}
        self.id_to_idx: dict[str, int] = {}  # document id -> faiss index
        self.idx_to_id: dict[int, str] = {}  # faiss index -> document id

        logger.info(
            f"Initialized FAISSVectorStore with dimension={dimension}, "
            f"index_type={index_type}"
        )

    def _create_index(self) -> "faiss.Index":
        """
        Create the FAISS index based on configuration.

        Returns:
            FAISS index object
        """
        if self.index_type == "flat":
            # IndexFlatIP for inner product (cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(self.dimension)
            logger.debug("Created IndexFlatIP")
        elif self.index_type == "hnsw":
            # HNSW index for approximate search
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search
            logger.debug(
                f"Created IndexHNSWFlat with M={self.hnsw_m}, "
                f"efConstruction={self.hnsw_ef_construction}"
            )
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: List of document IDs
            embeddings: Numpy array of embeddings, shape (n, dimension)
            texts: List of document texts
            metadatas: Optional list of metadata dictionaries
        """
        if len(ids) != len(embeddings) or len(ids) != len(texts):
            raise ValueError("ids, embeddings, and texts must have the same length")

        if metadatas is None:
            metadatas = [{} for _ in ids]

        # Ensure embeddings are float32 and C-contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Get starting index
        start_idx = self.index.ntotal

        # Add to FAISS index
        self.index.add(embeddings)

        # Store document data and mappings
        for i, (doc_id, text, metadata) in enumerate(zip(ids, texts, metadatas)):
            idx = start_idx + i
            self.documents[doc_id] = {
                "text": text,
                "metadata": metadata
            }
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id

        logger.info(f"Added {len(ids)} documents to vector store (total: {self.index.ntotal})")

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents from a list of document dictionaries.

        Args:
            documents: List of dicts with 'id', 'text', and 'metadata' keys
            embeddings: Corresponding embeddings array
        """
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.add(ids, embeddings, texts, metadatas)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector, shape (dimension,) or (1, dimension)
            k: Number of results to return
            filter_fn: Optional function to filter results by metadata

        Returns:
            List of SearchResult objects
        """
        if self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        # Search more results if filtering, to ensure enough after filter
        search_k = k * 3 if filter_fn else k
        search_k = min(search_k, self.index.ntotal)

        # Perform search
        scores, indices = self.index.search(query_embedding, search_k)

        # Convert to SearchResult objects
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc_id = self.idx_to_id.get(idx)
            if doc_id is None:
                continue

            doc = self.documents.get(doc_id)
            if doc is None:
                continue

            # Apply filter if provided
            if filter_fn and not filter_fn(doc["metadata"]):
                continue

            result = SearchResult(
                id=doc_id,
                score=float(score),
                text=doc["text"],
                metadata=doc["metadata"],
                rank=len(results)
            )
            results.append(result)

            if len(results) >= k:
                break

        return results

    def search_with_metadata_filter(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[dict[str, Any]] = None
    ) -> list[SearchResult]:
        """
        Search with metadata filtering.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Dictionary of metadata filters
                - Exact match: {"field": "value"}
                - List membership: {"field": ["val1", "val2"]}
                - Range: {"field": {"min": 0, "max": 10}}

        Returns:
            List of SearchResult objects
        """
        if not filters:
            return self.search(query_embedding, k)

        def filter_fn(metadata: dict[str, Any]) -> bool:
            for field, condition in filters.items():
                value = metadata.get(field)

                if isinstance(condition, dict):
                    # Range filter
                    if "min" in condition and value < condition["min"]:
                        return False
                    if "max" in condition and value > condition["max"]:
                        return False
                elif isinstance(condition, list):
                    # List membership
                    if value not in condition:
                        return False
                else:
                    # Exact match
                    if value != condition:
                        return False

            return True

        return self.search(query_embedding, k, filter_fn=filter_fn)

    def search_by_skills(
        self,
        query_embedding: np.ndarray,
        required_skills: list[str],
        k: int = 10,
        match_threshold: float = 0.5
    ) -> list[SearchResult]:
        """
        Search with skill matching.

        Args:
            query_embedding: Query vector
            required_skills: List of required skills
            k: Number of results
            match_threshold: Minimum fraction of skills that must match

        Returns:
            List of SearchResult objects
        """
        required_skills_lower = {s.lower() for s in required_skills}
        min_matches = int(len(required_skills) * match_threshold)

        def skill_filter(metadata: dict[str, Any]) -> bool:
            doc_skills = metadata.get("skills", [])
            doc_skills_lower = {s.lower() for s in doc_skills}
            matches = len(required_skills_lower & doc_skills_lower)
            return matches >= min_matches

        return self.search(query_embedding, k, filter_fn=skill_filter)

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document dict with 'text' and 'metadata', or None
        """
        return self.documents.get(doc_id)

    def get_all_ids(self) -> list[str]:
        """Get all document IDs in the store."""
        return list(self.documents.keys())

    def __len__(self) -> int:
        """Return number of documents in store."""
        return len(self.documents)

    def save(self, path: Path | str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata and mappings
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "documents": self.documents,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": {str(k): v for k, v in self.idx_to_id.items()}
        }

        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved vector store to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "FAISSVectorStore":
        """
        Load a vector store from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded FAISSVectorStore instance
        """
        path = Path(path)

        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # Create instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            hnsw_m=metadata.get("hnsw_m", 32),
            hnsw_ef_construction=metadata.get("hnsw_ef_construction", 200),
            hnsw_ef_search=metadata.get("hnsw_ef_search", 128)
        )

        # Load FAISS index
        index_path = path / "index.faiss"
        store.index = faiss.read_index(str(index_path))

        # Restore mappings
        store.documents = metadata["documents"]
        store.id_to_idx = metadata["id_to_idx"]
        store.idx_to_id = {int(k): v for k, v in metadata["idx_to_id"].items()}

        logger.info(
            f"Loaded vector store from {path} "
            f"({store.index.ntotal} vectors)"
        )

        return store

    def clear(self) -> None:
        """Clear all documents from the store."""
        self.index = self._create_index()
        self.documents.clear()
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        logger.info("Cleared vector store")

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.index.is_trained
        }
