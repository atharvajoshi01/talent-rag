"""
API Schemas Module.

This module provides Pydantic models for request/response
validation in the FastAPI backend.
"""

from typing import Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SeniorityLevel(str, Enum):
    """Seniority level options."""
    JUNIOR = "Junior"
    MID_LEVEL = "Mid-Level"
    SENIOR = "Senior"
    STAFF = "Staff"
    PRINCIPAL = "Principal"
    DIRECTOR = "Director"


# ============== Request Schemas ==============

class SearchFilters(BaseModel):
    """Filters for candidate search."""
    skills: Optional[list[str]] = Field(
        default=None,
        description="Required skills to filter by"
    )
    location: Optional[str] = Field(
        default=None,
        description="Location filter"
    )
    seniority: Optional[list[SeniorityLevel]] = Field(
        default=None,
        description="Seniority levels to include"
    )
    min_experience: Optional[int] = Field(
        default=None,
        ge=0,
        description="Minimum years of experience"
    )
    max_experience: Optional[int] = Field(
        default=None,
        ge=0,
        description="Maximum years of experience"
    )


class SearchCandidatesRequest(BaseModel):
    """Request schema for candidate search."""
    query: str = Field(
        ...,
        min_length=3,
        description="Search query describing ideal candidate"
    )
    filters: Optional[SearchFilters] = Field(
        default=None,
        description="Optional filters to apply"
    )
    role_id: Optional[str] = Field(
        default=None,
        description="Optional role ID to match against"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of candidates to return"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Senior ML engineer with NLP experience",
                    "filters": {
                        "skills": ["Python", "PyTorch", "NLP"],
                        "seniority": ["Senior", "Staff"],
                        "min_experience": 5
                    },
                    "k": 10
                }
            ]
        }
    }


class CompareCandidatesRequest(BaseModel):
    """Request schema for candidate comparison."""
    candidate_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of candidate IDs to compare"
    )
    role_id: str = Field(
        ...,
        description="Role ID to compare candidates against"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "candidate_ids": ["CAND_001", "CAND_002", "CAND_003"],
                    "role_id": "ROLE_001"
                }
            ]
        }
    }


class AskRequest(BaseModel):
    """Request schema for general RAG queries."""
    question: str = Field(
        ...,
        min_length=5,
        description="Question to ask"
    )
    context_filter: Optional[str] = Field(
        default=None,
        description="Filter context to 'candidates' or 'roles'"
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to include evidence in response"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Which candidates have experience with large-scale distributed systems?",
                    "context_filter": "candidates",
                    "include_evidence": True
                }
            ]
        }
    }


# ============== Response Schemas ==============

class EvidenceItem(BaseModel):
    """Schema for a single evidence item."""
    id: str = Field(..., description="Chunk ID")
    text: str = Field(..., description="Evidence text")
    score: float = Field(..., description="Relevance score")
    source: str = Field(..., description="Source type (resume/interview/role)")
    candidate_id: Optional[str] = Field(
        default=None,
        description="Associated candidate ID"
    )
    candidate_name: Optional[str] = Field(
        default=None,
        description="Associated candidate name"
    )


class CandidateMatch(BaseModel):
    """Schema for a candidate match result."""
    candidate_id: str = Field(..., description="Candidate ID")
    name: str = Field(..., description="Candidate name")
    match_score: float = Field(..., description="Match score (0-1)")
    location: str = Field(..., description="Candidate location")
    seniority: str = Field(..., description="Seniority level")
    years_experience: int = Field(..., description="Years of experience")
    skills: list[str] = Field(..., description="Candidate skills")
    skill_match: dict[str, Any] = Field(
        ...,
        description="Skill matching details"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Supporting evidence"
    )


class MetaInfo(BaseModel):
    """Schema for response metadata."""
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    model: str = Field(..., description="LLM model used")
    retrieval_count: int = Field(..., description="Number of documents retrieved")
    prompt_tokens: int = Field(default=0, description="Prompt tokens used")
    completion_tokens: int = Field(default=0, description="Completion tokens used")


class RAGResponseSchema(BaseModel):
    """Schema for RAG response."""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    evidence_used: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence used in response"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Citation IDs found in response"
    )
    meta: MetaInfo = Field(..., description="Response metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Based on the evidence, the top candidates for this role are...",
                    "query": "Find senior ML engineers",
                    "evidence_used": [
                        {
                            "id": "CAND_001_resume_chunk_0",
                            "text": "Senior ML Engineer with 8 years...",
                            "score": 0.92,
                            "source": "resume",
                            "candidate_id": "CAND_001",
                            "candidate_name": "John Smith"
                        }
                    ],
                    "citations": ["CAND_001"],
                    "meta": {
                        "latency_ms": 1250.5,
                        "model": "gpt-4o",
                        "retrieval_count": 10,
                        "prompt_tokens": 2500,
                        "completion_tokens": 500
                    }
                }
            ]
        }
    }


class CandidateSearchResponse(BaseModel):
    """Response schema for candidate search."""
    matches: list[CandidateMatch] = Field(
        ...,
        description="Matching candidates"
    )
    total_found: int = Field(..., description="Total matches found")
    query: str = Field(..., description="Original query")
    filters_applied: Optional[SearchFilters] = Field(
        default=None,
        description="Filters that were applied"
    )
    meta: MetaInfo = Field(..., description="Response metadata")


class ComparisonResponse(BaseModel):
    """Response schema for candidate comparison."""
    analysis: str = Field(..., description="Comparison analysis")
    candidates: list[CandidateMatch] = Field(
        ...,
        description="Compared candidates with scores"
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Recommended candidate"
    )
    role_title: str = Field(..., description="Role being compared against")
    meta: MetaInfo = Field(..., description="Response metadata")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    index_status: dict[str, Any] = Field(
        ...,
        description="Vector index status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )
    code: str = Field(..., description="Error code")
