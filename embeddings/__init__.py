"""
Embeddings Module.

This module provides embedding model wrappers for generating
vector representations of text.
"""

from .embedding_model import (
    EmbeddingModel,
    OllamaEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
    get_embedding_model,
    create_embedding_model
)

__all__ = [
    "EmbeddingModel",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "get_embedding_model",
    "create_embedding_model"
]
