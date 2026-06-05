"""
Index Builder Module.

This module provides utilities for building and managing
vector indices for the Talent RAG system.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..embeddings import EmbeddingModel, get_embedding_model
from ..vectorstore import FAISSVectorStore
from ..preprocessing import DocumentProcessor


class IndexBuilder:
    """
    Builder for creating and managing vector indices.

    This class orchestrates the process of loading documents,
    generating embeddings, and building searchable indices.

    Attributes:
        embedding_model: Model for generating embeddings
        vector_store: FAISS vector store instance
        document_processor: Processor for chunking documents
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        use_openai_embeddings: bool = False,
        use_ollama_embeddings: bool = False,
        ollama_host: str = "http://localhost:11434",
        index_type: str = "flat",
        chunk_size_resume: int = 400,
        chunk_size_role: int = 250,
        chunk_size_interview: int = 200,
        chunk_overlap: int = 50
    ):
        """
        Initialize the index builder.

        Args:
            embedding_model: Pre-initialized embedding model
            use_openai_embeddings: Whether to use OpenAI embeddings
            use_ollama_embeddings: Whether to use Ollama embeddings (recommended)
            ollama_host: Ollama server host URL
            index_type: FAISS index type ('flat' or 'hnsw')
            chunk_size_resume: Chunk size for resumes
            chunk_size_role: Chunk size for roles
            chunk_size_interview: Chunk size for interviews
            chunk_overlap: Overlap between chunks
        """
        # Initialize embedding model
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = get_embedding_model(
                use_openai=use_openai_embeddings,
                use_ollama=use_ollama_embeddings,
                ollama_host=ollama_host
            )

        # Store configuration
        self.index_type = index_type

        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size_resume=chunk_size_resume,
            chunk_size_role=chunk_size_role,
            chunk_size_interview=chunk_size_interview,
            chunk_overlap=chunk_overlap
        )

        # Vector store will be created during build
        self.vector_store: Optional[FAISSVectorStore] = None

        logger.info(
            f"IndexBuilder initialized with embedding model: "
            f"{self.embedding_model.model_name}"
        )

    def build_index(
        self,
        candidates_path: Optional[Path | str] = None,
        roles_path: Optional[Path | str] = None,
        candidates: Optional[list[dict[str, Any]]] = None,
        roles: Optional[list[dict[str, Any]]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> FAISSVectorStore:
        """
        Build vector index from candidates and/or roles.

        Args:
            candidates_path: Path to candidates JSON file
            roles_path: Path to roles JSON file
            candidates: Pre-loaded candidate data
            roles: Pre-loaded role data
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar

        Returns:
            Built FAISSVectorStore instance
        """
        logger.info("Starting index build...")

        # Process documents into chunks
        candidate_chunks, role_chunks = self.document_processor.process_all(
            candidates_path=candidates_path,
            roles_path=roles_path,
            candidates=candidates,
            roles=roles
        )

        all_chunks = candidate_chunks + role_chunks

        if not all_chunks:
            raise ValueError("No chunks to index. Check input data.")

        logger.info(f"Processing {len(all_chunks)} chunks for embedding...")

        # Convert chunks to documents
        documents = self.document_processor.chunks_to_documents(all_chunks)

        # Generate embeddings in batches
        texts = [doc["text"] for doc in documents]
        embeddings = self._generate_embeddings(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Create vector store
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_model.dimension,
            index_type=self.index_type
        )

        # Add documents to store
        self.vector_store.add_documents(documents, embeddings)

        logger.info(
            f"Index built successfully with {len(documents)} documents"
        )

        return self.vector_store

    def _generate_embeddings(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")

        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed_batch(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def build_candidate_index(
        self,
        candidates_path: Optional[Path | str] = None,
        candidates: Optional[list[dict[str, Any]]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> FAISSVectorStore:
        """
        Build index for candidates only.

        Args:
            candidates_path: Path to candidates JSON file
            candidates: Pre-loaded candidate data
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar

        Returns:
            Built FAISSVectorStore instance
        """
        return self.build_index(
            candidates_path=candidates_path,
            candidates=candidates,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def build_role_index(
        self,
        roles_path: Optional[Path | str] = None,
        roles: Optional[list[dict[str, Any]]] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> FAISSVectorStore:
        """
        Build index for roles only.

        Args:
            roles_path: Path to roles JSON file
            roles: Pre-loaded role data
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress bar

        Returns:
            Built FAISSVectorStore instance
        """
        return self.build_index(
            roles_path=roles_path,
            roles=roles,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def add_candidates(
        self,
        candidates: list[dict[str, Any]],
        batch_size: int = 32
    ) -> int:
        """
        Add new candidates to existing index.

        Args:
            candidates: List of candidate dictionaries
            batch_size: Batch size for embedding generation

        Returns:
            Number of chunks added
        """
        if self.vector_store is None:
            raise ValueError("No index exists. Call build_index first.")

        # Process new candidates
        chunks = self.document_processor.process_all_candidates(candidates)
        documents = self.document_processor.chunks_to_documents(chunks)

        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = self._generate_embeddings(texts, batch_size=batch_size)

        # Add to store
        self.vector_store.add_documents(documents, embeddings)

        logger.info(f"Added {len(documents)} chunks from {len(candidates)} candidates")
        return len(documents)

    def add_roles(
        self,
        roles: list[dict[str, Any]],
        batch_size: int = 32
    ) -> int:
        """
        Add new roles to existing index.

        Args:
            roles: List of role dictionaries
            batch_size: Batch size for embedding generation

        Returns:
            Number of chunks added
        """
        if self.vector_store is None:
            raise ValueError("No index exists. Call build_index first.")

        # Process new roles
        chunks = self.document_processor.process_all_roles(roles)
        documents = self.document_processor.chunks_to_documents(chunks)

        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = self._generate_embeddings(texts, batch_size=batch_size)

        # Add to store
        self.vector_store.add_documents(documents, embeddings)

        logger.info(f"Added {len(documents)} chunks from {len(roles)} roles")
        return len(documents)

    def save_index(self, path: Path | str) -> None:
        """
        Save the index to disk.

        Args:
            path: Directory path to save the index
        """
        if self.vector_store is None:
            raise ValueError("No index to save. Call build_index first.")

        self.vector_store.save(path)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: Path | str) -> FAISSVectorStore:
        """
        Load an index from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded FAISSVectorStore instance
        """
        self.vector_store = FAISSVectorStore.load(path)
        logger.info(f"Index loaded from {path}")
        return self.vector_store

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Dictionary with index statistics
        """
        if self.vector_store is None:
            return {"status": "No index built"}

        stats = self.vector_store.get_stats()
        stats["embedding_model"] = self.embedding_model.model_name
        stats["embedding_dimension"] = self.embedding_model.dimension
        return stats


def build_index_from_data(
    candidates_path: Path | str,
    roles_path: Path | str,
    output_path: Path | str,
    use_openai_embeddings: bool = False,
    index_type: str = "flat"
) -> FAISSVectorStore:
    """
    Convenience function to build and save an index.

    Args:
        candidates_path: Path to candidates JSON
        roles_path: Path to roles JSON
        output_path: Path to save the index
        use_openai_embeddings: Whether to use OpenAI embeddings
        index_type: FAISS index type

    Returns:
        Built FAISSVectorStore instance
    """
    builder = IndexBuilder(
        use_openai_embeddings=use_openai_embeddings,
        index_type=index_type
    )

    vector_store = builder.build_index(
        candidates_path=candidates_path,
        roles_path=roles_path
    )

    builder.save_index(output_path)

    return vector_store
