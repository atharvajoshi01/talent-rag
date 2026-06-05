"""
FastAPI Main Application Module.

This module provides the FastAPI backend for the
Talent Intelligence Assistant.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .schemas import (
    SearchCandidatesRequest,
    CompareCandidatesRequest,
    AskRequest,
    RAGResponseSchema,
    CandidateSearchResponse,
    ComparisonResponse,
    HealthResponse,
    ErrorResponse,
    EvidenceItem,
    CandidateMatch,
    MetaInfo
)

# Global state for the RAG pipeline
_rag_pipeline = None
_roles_data = None
_candidates_data = None


def get_pipeline():
    """Get the RAG pipeline instance."""
    if _rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please wait for startup."
        )
    return _rag_pipeline


def get_roles():
    """Get the roles data."""
    if _roles_data is None:
        raise HTTPException(
            status_code=503,
            detail="Roles data not loaded."
        )
    return _roles_data


def get_candidates():
    """Get the candidates data."""
    if _candidates_data is None:
        raise HTTPException(
            status_code=503,
            detail="Candidates data not loaded."
        )
    return _candidates_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Initializes the RAG pipeline on startup.
    """
    global _rag_pipeline, _roles_data, _candidates_data

    logger.info("Starting Talent RAG API...")

    try:
        # Import here to avoid circular imports
        from ..config import settings
        from ..embeddings import get_embedding_model
        from ..vectorstore import FAISSVectorStore
        from ..rag import RAGPipeline

        # Check if index exists
        index_path = settings.index_dir

        if index_path.exists():
            logger.info(f"Loading existing index from {index_path}")
            embedding_model = get_embedding_model(
                use_openai=settings.use_openai_embeddings
            )
            vector_store = FAISSVectorStore.load(index_path)
        else:
            logger.info("No existing index found. Building new index...")

            # Check if data exists
            if not settings.candidates_path.exists():
                logger.warning("No candidates data found. Generating synthetic data...")
                from ..data_generation import generate_all_data
                generate_all_data(
                    output_dir=settings.data_dir,
                    num_candidates=100,
                    num_roles=20
                )

            # Build index
            from ..retrieval import IndexBuilder

            builder = IndexBuilder(
                use_openai_embeddings=settings.use_openai_embeddings
            )

            vector_store = builder.build_index(
                candidates_path=settings.candidates_path,
                roles_path=settings.roles_path
            )

            # Save index
            index_path.mkdir(parents=True, exist_ok=True)
            builder.save_index(index_path)

            embedding_model = builder.embedding_model

        # Initialize RAG pipeline
        _rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_model=embedding_model,
            generator_model=settings.llm_model,
            llm_provider=settings.llm_provider,
            ollama_host=settings.ollama_host,
            api_key=settings.openai_api_key if settings.llm_provider == "openai" else None,
            retrieval_k=settings.top_k_retrieval,
            rerank_k=settings.rerank_top_k
        )

        # Load raw data for lookups
        if settings.roles_path.exists():
            with open(settings.roles_path) as f:
                _roles_data = {r["id"]: r for r in json.load(f)}

        if settings.candidates_path.exists():
            with open(settings.candidates_path) as f:
                _candidates_data = {c["id"]: c for c in json.load(f)}

        logger.info("Talent RAG API initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Continue anyway for health checks

    yield

    # Cleanup
    logger.info("Shutting down Talent RAG API...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application
    """
    application = FastAPI(
        title="Talent Intelligence Assistant API",
        description="""
        A production-grade RAG-based API for talent intelligence.

        ## Features
        - **Candidate Search**: Find candidates matching specific criteria
        - **Candidate Comparison**: Compare multiple candidates against a role
        - **General Q&A**: Ask questions about candidates and roles

        ## Authentication
        API key authentication (configure via headers)
        """,
        version="1.0.0",
        lifespan=lifespan,
        responses={
            500: {"model": ErrorResponse, "description": "Internal server error"}
        }
    )

    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return application


# Create the app
app = create_app()


# ============== Endpoints ==============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check the health status of the API.

    Returns service status and index information.
    """
    index_status = {"status": "not_initialized", "documents": 0}

    if _rag_pipeline:
        try:
            stats = _rag_pipeline.retriever.vector_store.get_stats()
            index_status = {
                "status": "ready",
                "documents": stats.get("total_documents", 0),
                "vectors": stats.get("total_vectors", 0),
                "index_type": stats.get("index_type", "unknown")
            }
        except Exception as e:
            index_status = {"status": "error", "error": str(e)}

    return HealthResponse(
        status="healthy" if _rag_pipeline else "initializing",
        version="1.0.0",
        index_status=index_status
    )


@app.post(
    "/search_candidates",
    response_model=RAGResponseSchema,
    tags=["Search"]
)
async def search_candidates(
    request: SearchCandidatesRequest,
    pipeline=Depends(get_pipeline)
):
    """
    Search for candidates matching criteria.

    Uses semantic search with optional filters to find
    the best matching candidates.
    """
    start_time = time.time()

    try:
        # Build filters
        filters = None
        if request.filters:
            filters = {
                "skills": request.filters.skills,
                "location": request.filters.location,
                "seniority": [s.value for s in request.filters.seniority] if request.filters.seniority else None,
                "min_experience": request.filters.min_experience,
                "max_experience": request.filters.max_experience
            }
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}

        # Get role context if provided
        role_context = None
        if request.role_id and _roles_data:
            role_context = _roles_data.get(request.role_id)

        # Execute search
        response = pipeline.search_candidates(
            query=request.query,
            role=role_context,
            k=request.k,
            **(filters or {})
        )

        # Build evidence items
        evidence_items = [
            EvidenceItem(
                id=e["id"],
                text=e["text"],
                score=e["score"],
                source=e["metadata"].get("source", "unknown"),
                candidate_id=e["metadata"].get("candidate_id"),
                candidate_name=e["metadata"].get("candidate_name")
            )
            for e in response.evidence_used
        ]

        return RAGResponseSchema(
            answer=response.answer,
            query=request.query,
            evidence_used=evidence_items,
            citations=response.citations,
            meta=MetaInfo(
                latency_ms=response.latency_ms,
                model=response.generation_result.model,
                retrieval_count=response.retrieval_result.total_retrieved,
                prompt_tokens=response.generation_result.prompt_tokens,
                completion_tokens=response.generation_result.completion_tokens
            )
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/compare_candidates",
    response_model=RAGResponseSchema,
    tags=["Search"]
)
async def compare_candidates(
    request: CompareCandidatesRequest,
    pipeline=Depends(get_pipeline),
    roles=Depends(get_roles)
):
    """
    Compare multiple candidates for a specific role.

    Provides a detailed side-by-side analysis of candidates
    against role requirements.
    """
    try:
        # Get role
        role = roles.get(request.role_id)
        if not role:
            raise HTTPException(
                status_code=404,
                detail=f"Role {request.role_id} not found"
            )

        # Execute comparison
        response = pipeline.compare_candidates(
            candidate_ids=request.candidate_ids,
            role=role
        )

        # Build evidence items
        evidence_items = [
            EvidenceItem(
                id=e["id"],
                text=e["text"],
                score=e["score"],
                source=e["metadata"].get("source", "unknown"),
                candidate_id=e["metadata"].get("candidate_id"),
                candidate_name=e["metadata"].get("candidate_name")
            )
            for e in response.evidence_used
        ]

        return RAGResponseSchema(
            answer=response.answer,
            query=response.query,
            evidence_used=evidence_items,
            citations=response.citations,
            meta=MetaInfo(
                latency_ms=response.latency_ms,
                model=response.generation_result.model,
                retrieval_count=response.retrieval_result.total_retrieved,
                prompt_tokens=response.generation_result.prompt_tokens,
                completion_tokens=response.generation_result.completion_tokens
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/ask",
    response_model=RAGResponseSchema,
    tags=["Chat"]
)
async def ask_question(
    request: AskRequest,
    pipeline=Depends(get_pipeline)
):
    """
    Ask a general question about candidates or roles.

    Uses RAG to retrieve relevant context and generate
    an informed response.
    """
    try:
        response = pipeline.ask(
            question=request.question,
            context_filter=request.context_filter
        )

        # Build evidence items
        evidence_items = []
        if request.include_evidence:
            evidence_items = [
                EvidenceItem(
                    id=e["id"],
                    text=e["text"],
                    score=e["score"],
                    source=e["metadata"].get("source", "unknown"),
                    candidate_id=e["metadata"].get("candidate_id"),
                    candidate_name=e["metadata"].get("candidate_name")
                )
                for e in response.evidence_used
            ]

        return RAGResponseSchema(
            answer=response.answer,
            query=request.question,
            evidence_used=evidence_items,
            citations=response.citations,
            meta=MetaInfo(
                latency_ms=response.latency_ms,
                model=response.generation_result.model,
                retrieval_count=response.retrieval_result.total_retrieved,
                prompt_tokens=response.generation_result.prompt_tokens,
                completion_tokens=response.generation_result.completion_tokens
            )
        )

    except Exception as e:
        logger.error(f"Ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/roles", tags=["Data"])
async def list_roles(roles=Depends(get_roles)):
    """
    List all available roles.

    Returns a summary of all roles in the system.
    """
    return {
        "roles": [
            {
                "id": r["id"],
                "title": r["title"],
                "company": r["company"],
                "location": r["location"],
                "seniority": r["seniority"],
                "required_skills": r.get("required_skills", [])
            }
            for r in roles.values()
        ],
        "total": len(roles)
    }


@app.get("/roles/{role_id}", tags=["Data"])
async def get_role(role_id: str, roles=Depends(get_roles)):
    """
    Get details for a specific role.
    """
    role = roles.get(role_id)
    if not role:
        raise HTTPException(status_code=404, detail=f"Role {role_id} not found")
    return role


@app.get("/candidates", tags=["Data"])
async def list_candidates(
    skip: int = 0,
    limit: int = 20,
    candidates=Depends(get_candidates)
):
    """
    List all candidates with pagination.

    Returns a summary of candidates in the system.
    """
    all_candidates = list(candidates.values())
    paginated = all_candidates[skip:skip + limit]

    return {
        "candidates": [
            {
                "id": c["id"],
                "name": c["name"],
                "location": c["location"],
                "seniority": c["seniority"],
                "years_of_experience": c["years_of_experience"],
                "skills": c.get("skills", [])[:10]  # Top 10 skills
            }
            for c in paginated
        ],
        "total": len(all_candidates),
        "skip": skip,
        "limit": limit
    }


@app.get("/candidates/{candidate_id}", tags=["Data"])
async def get_candidate(
    candidate_id: str,
    candidates=Depends(get_candidates)
):
    """
    Get details for a specific candidate.
    """
    candidate = candidates.get(candidate_id)
    if not candidate:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {candidate_id} not found"
        )
    return candidate


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    from ..config import settings

    uvicorn.run(
        "talent_rag.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
