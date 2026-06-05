"""
Data Generation Module.

This module provides utilities for generating synthetic candidate and role data
for testing and development of the Talent RAG system.
"""

from .candidate_generator import CandidateGenerator
from .role_generator import RoleGenerator
from .generate_all import generate_all_data

__all__ = ["CandidateGenerator", "RoleGenerator", "generate_all_data"]
