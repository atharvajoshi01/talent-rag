"""
Configuration module for the Talent RAG system.

This module provides centralized configuration management using pydantic-settings
for environment variable handling and type validation.
"""

from pathlib import Path
from typing import Literal
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes:
        openai_api_key: API key for OpenAI services
        embedding_model: Name of the embedding model to use
        embedding_dim: Dimensionality of embeddings
        llm_model: Name of the LLM model for generation
        llm_temperature: Temperature for LLM generation (lower = more deterministic)
        top_k_retrieval: Number of documents to retrieve
        rerank_top_k: Number of documents after reranking
        faiss_index_type: Type of FAISS index to use
        chunk_size_resume: Target chunk size for resume text
        chunk_size_role: Target chunk size for role descriptions
        chunk_size_interview: Target chunk size for interview transcripts
        chunk_overlap: Overlap between chunks
        api_host: Host for FastAPI server
        api_port: Port for FastAPI server
        log_level: Logging level
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Embedding Configuration
    embedding_model: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence transformer model for embeddings"
    )
    embedding_dim: int = Field(default=384, description="Embedding dimensionality")
    use_openai_embeddings: bool = Field(
        default=False,
        description="Use OpenAI embeddings instead of local"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model name"
    )

    # LLM Configuration
    llm_provider: Literal["openai", "ollama"] = Field(
        default="ollama",
        description="LLM provider to use"
    )
    llm_model: str = Field(default="llama3.2", description="LLM model for generation")
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    llm_max_tokens: int = Field(default=2048, description="Max tokens for LLM response")

    # Ollama Configuration
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )

    # Retrieval Configuration
    top_k_retrieval: int = Field(default=10, description="Top-k for initial retrieval")
    rerank_top_k: int = Field(default=5, description="Top-k after reranking")

    # FAISS Configuration
    faiss_index_type: Literal["flat", "hnsw"] = Field(
        default="flat",
        description="FAISS index type"
    )
    faiss_hnsw_m: int = Field(default=32, description="HNSW M parameter")
    faiss_hnsw_ef_construction: int = Field(
        default=200,
        description="HNSW ef_construction parameter"
    )
    faiss_hnsw_ef_search: int = Field(default=128, description="HNSW ef_search parameter")

    # Chunking Configuration
    chunk_size_resume: int = Field(default=400, description="Chunk size for resumes")
    chunk_size_role: int = Field(default=250, description="Chunk size for roles")
    chunk_size_interview: int = Field(
        default=200,
        description="Chunk size for interviews"
    )
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Paths
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"

    @property
    def index_dir(self) -> Path:
        """Get vector index directory path."""
        return self.project_root / "data" / "indices"

    @property
    def candidates_path(self) -> Path:
        """Get candidates JSON file path."""
        return self.data_dir / "candidates.json"

    @property
    def roles_path(self) -> Path:
        """Get roles JSON file path."""
        return self.data_dir / "roles.json"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings singleton
    """
    return Settings()


# Convenience access
settings = get_settings()
