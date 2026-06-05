"""
API Module.

This module provides the FastAPI backend for the
Talent Intelligence Assistant.
"""

from .main import app, create_app
from .schemas import (
    SearchCandidatesRequest,
    CompareCandidatesRequest,
    AskRequest,
    RAGResponseSchema
)

__all__ = [
    "app",
    "create_app",
    "SearchCandidatesRequest",
    "CompareCandidatesRequest",
    "AskRequest",
    "RAGResponseSchema"
]
