"""
Tests for vector store module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from talent_rag.vectorstore import FAISSVectorStore


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Create a vector store fixture."""
        return FAISSVectorStore(dimension=128, index_type="flat")

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_docs = 10
        dim = 128

        embeddings = np.random.randn(n_docs, dim).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        ids = [f"doc_{i}" for i in range(n_docs)]
        texts = [f"This is document {i}" for i in range(n_docs)]
        metadatas = [{"index": i, "category": "test"} for i in range(n_docs)]

        return ids, embeddings, texts, metadatas

    def test_init(self, vector_store):
        """Test vector store initialization."""
        assert vector_store.dimension == 128
        assert vector_store.index_type == "flat"
        assert len(vector_store) == 0

    def test_add_documents(self, vector_store, sample_data):
        """Test adding documents."""
        ids, embeddings, texts, metadatas = sample_data

        vector_store.add(ids, embeddings, texts, metadatas)

        assert len(vector_store) == 10
        assert vector_store.index.ntotal == 10

    def test_search_basic(self, vector_store, sample_data):
        """Test basic search."""
        ids, embeddings, texts, metadatas = sample_data
        vector_store.add(ids, embeddings, texts, metadatas)

        # Search with the first embedding
        query = embeddings[0]
        results = vector_store.search(query, k=5)

        assert len(results) == 5
        # First result should be the same document (highest similarity)
        assert results[0].id == "doc_0"
        assert results[0].score >= results[1].score  # Sorted by score

    def test_search_with_filter(self, vector_store, sample_data):
        """Test search with metadata filter."""
        ids, embeddings, texts, metadatas = sample_data

        # Modify metadata for some docs
        for i in range(5):
            metadatas[i]["category"] = "A"
        for i in range(5, 10):
            metadatas[i]["category"] = "B"

        vector_store.add(ids, embeddings, texts, metadatas)

        # Search with filter
        query = embeddings[0]
        results = vector_store.search_with_metadata_filter(
            query, k=5, filters={"category": "B"}
        )

        # All results should be category B
        for result in results:
            assert result.metadata["category"] == "B"

    def test_get_document(self, vector_store, sample_data):
        """Test retrieving a document by ID."""
        ids, embeddings, texts, metadatas = sample_data
        vector_store.add(ids, embeddings, texts, metadatas)

        doc = vector_store.get_document("doc_0")

        assert doc is not None
        assert doc["text"] == "This is document 0"
        assert doc["metadata"]["index"] == 0

    def test_get_nonexistent_document(self, vector_store):
        """Test retrieving a nonexistent document."""
        doc = vector_store.get_document("nonexistent")
        assert doc is None

    def test_save_and_load(self, vector_store, sample_data):
        """Test saving and loading the store."""
        ids, embeddings, texts, metadatas = sample_data
        vector_store.add(ids, embeddings, texts, metadatas)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index"
            vector_store.save(path)

            # Load into new store
            loaded_store = FAISSVectorStore.load(path)

            assert len(loaded_store) == 10
            assert loaded_store.dimension == 128

            # Verify search works
            query = embeddings[0]
            results = loaded_store.search(query, k=5)
            assert len(results) == 5

    def test_clear(self, vector_store, sample_data):
        """Test clearing the store."""
        ids, embeddings, texts, metadatas = sample_data
        vector_store.add(ids, embeddings, texts, metadatas)

        assert len(vector_store) == 10

        vector_store.clear()

        assert len(vector_store) == 0
        assert vector_store.index.ntotal == 0

    def test_get_stats(self, vector_store, sample_data):
        """Test getting store statistics."""
        ids, embeddings, texts, metadatas = sample_data
        vector_store.add(ids, embeddings, texts, metadatas)

        stats = vector_store.get_stats()

        assert stats["total_vectors"] == 10
        assert stats["total_documents"] == 10
        assert stats["dimension"] == 128
        assert stats["index_type"] == "flat"

    def test_search_by_skills(self, vector_store, sample_data):
        """Test skill-based search."""
        ids, embeddings, texts, metadatas = sample_data

        # Add skills to metadata
        metadatas[0]["skills"] = ["Python", "ML", "NLP"]
        metadatas[1]["skills"] = ["Python", "ML"]
        metadatas[2]["skills"] = ["JavaScript", "React"]

        vector_store.add(ids, embeddings, texts, metadatas)

        query = embeddings[0]
        results = vector_store.search_by_skills(
            query,
            required_skills=["Python", "ML"],
            k=5,
            match_threshold=0.5
        )

        # Should find docs with Python and ML
        skill_matches = [
            r for r in results
            if "Python" in r.metadata.get("skills", [])
        ]
        assert len(skill_matches) > 0
