"""
Vector Store Module.

This module provides FAISS-based vector storage and retrieval
functionality for the Talent RAG system.
"""

from .faiss_store import FAISSVectorStore

__all__ = ["FAISSVectorStore"]
