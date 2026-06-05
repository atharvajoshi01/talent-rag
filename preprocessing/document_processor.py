"""
Document Processor Module.

This module provides a unified interface for processing candidates
and roles into chunks ready for embedding.
"""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .text_cleaner import TextCleaner
from .chunker import DocumentChunker, Chunk


class DocumentProcessor:
    """
    Unified document processor for the Talent RAG system.

    This class handles the complete preprocessing pipeline from raw
    JSON data to chunks ready for embedding and indexing.

    Attributes:
        cleaner: TextCleaner instance for text normalization
        chunker: DocumentChunker instance for text chunking
    """

    def __init__(
        self,
        chunk_size_resume: int = 400,
        chunk_size_role: int = 250,
        chunk_size_interview: int = 200,
        chunk_overlap: int = 50,
        clean_urls: bool = False,
        clean_emails: bool = False
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size_resume: Target tokens for resume chunks
            chunk_size_role: Target tokens for role chunks
            chunk_size_interview: Target tokens for interview chunks
            chunk_overlap: Token overlap between chunks
            clean_urls: Whether to remove URLs during cleaning
            clean_emails: Whether to remove emails during cleaning
        """
        self.cleaner = TextCleaner(
            remove_urls=clean_urls,
            remove_emails=clean_emails,
            normalize_whitespace=True
        )

        self.chunker = DocumentChunker(
            chunk_size_resume=chunk_size_resume,
            chunk_size_role=chunk_size_role,
            chunk_size_interview=chunk_size_interview,
            overlap=chunk_overlap
        )

        logger.info("DocumentProcessor initialized")

    def load_candidates(self, path: Path | str) -> list[dict[str, Any]]:
        """
        Load candidates from a JSON file.

        Args:
            path: Path to candidates JSON file

        Returns:
            List of candidate dictionaries
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        logger.info(f"Loaded {len(candidates)} candidates from {path}")
        return candidates

    def load_roles(self, path: Path | str) -> list[dict[str, Any]]:
        """
        Load roles from a JSON file.

        Args:
            path: Path to roles JSON file

        Returns:
            List of role dictionaries
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            roles = json.load(f)
        logger.info(f"Loaded {len(roles)} roles from {path}")
        return roles

    def process_candidate(self, candidate: dict[str, Any]) -> list[Chunk]:
        """
        Process a single candidate into chunks.

        Processes both resume text and interview transcript, creating
        separate chunks with appropriate metadata.

        Args:
            candidate: Candidate dictionary with resume_text and interview_transcript

        Returns:
            List of Chunk objects from both resume and interview
        """
        candidate_id = candidate["id"]
        chunks = []

        # Base metadata for all chunks from this candidate
        base_metadata = {
            "candidate_id": candidate_id,
            "candidate_name": candidate.get("name", ""),
            "location": candidate.get("location", ""),
            "seniority": candidate.get("seniority", ""),
            "years_experience": candidate.get("years_of_experience", 0),
            "skills": candidate.get("skills", [])
        }

        # Process resume
        resume_text = candidate.get("resume_text", "")
        if resume_text:
            cleaned_resume = self.cleaner.clean(resume_text)
            resume_metadata = {**base_metadata, "source": "resume"}
            resume_chunks = self.chunker.chunk_resume(
                cleaned_resume,
                document_id=f"{candidate_id}_resume",
                metadata=resume_metadata
            )
            chunks.extend(resume_chunks)

        # Process interview transcript
        interview_text = candidate.get("interview_transcript", "")
        if interview_text:
            cleaned_interview = self.cleaner.clean(interview_text)
            interview_metadata = {**base_metadata, "source": "interview"}
            interview_chunks = self.chunker.chunk_interview(
                cleaned_interview,
                document_id=f"{candidate_id}_interview",
                metadata=interview_metadata
            )
            chunks.extend(interview_chunks)

        logger.debug(
            f"Processed candidate {candidate_id}: "
            f"{len(resume_chunks) if resume_text else 0} resume chunks, "
            f"{len(interview_chunks) if interview_text else 0} interview chunks"
        )

        return chunks

    def process_role(self, role: dict[str, Any]) -> list[Chunk]:
        """
        Process a single role into chunks.

        Args:
            role: Role dictionary with description and other fields

        Returns:
            List of Chunk objects
        """
        role_id = role["id"]

        # Build full text from role fields
        text_parts = []

        # Title and company
        if role.get("title"):
            text_parts.append(f"# {role['title']}")
        if role.get("company"):
            text_parts.append(f"**Company:** {role['company']}")
        if role.get("location"):
            text_parts.append(f"**Location:** {role['location']}")
        if role.get("seniority"):
            text_parts.append(f"**Level:** {role['seniority']}")
        if role.get("salary_range"):
            text_parts.append(f"**Salary:** {role['salary_range']}")

        # Description
        if role.get("description"):
            text_parts.append(f"\n{role['description']}")

        # Responsibilities
        if role.get("responsibilities"):
            text_parts.append("\n## Responsibilities")
            for resp in role["responsibilities"]:
                text_parts.append(f"- {resp}")

        # Qualifications
        if role.get("qualifications"):
            text_parts.append("\n## Qualifications")
            for qual in role["qualifications"]:
                text_parts.append(f"- {qual}")

        # Skills
        if role.get("required_skills"):
            text_parts.append(
                f"\n**Required Skills:** {', '.join(role['required_skills'])}"
            )
        if role.get("nice_to_have_skills"):
            text_parts.append(
                f"**Nice to Have:** {', '.join(role['nice_to_have_skills'])}"
            )

        # Benefits
        if role.get("benefits"):
            text_parts.append("\n## Benefits")
            for benefit in role["benefits"]:
                text_parts.append(f"- {benefit}")

        full_text = "\n".join(text_parts)
        cleaned_text = self.cleaner.clean(full_text)

        # Metadata
        metadata = {
            "role_id": role_id,
            "title": role.get("title", ""),
            "company": role.get("company", ""),
            "location": role.get("location", ""),
            "seniority": role.get("seniority", ""),
            "required_skills": role.get("required_skills", []),
            "nice_to_have_skills": role.get("nice_to_have_skills", []),
            "years_experience_required": role.get("years_experience_required", 0),
            "source": "role"
        }

        chunks = self.chunker.chunk_role(
            cleaned_text,
            document_id=role_id,
            metadata=metadata
        )

        logger.debug(f"Processed role {role_id}: {len(chunks)} chunks")
        return chunks

    def process_all_candidates(
        self,
        candidates: list[dict[str, Any]]
    ) -> list[Chunk]:
        """
        Process all candidates into chunks.

        Args:
            candidates: List of candidate dictionaries

        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for candidate in candidates:
            chunks = self.process_candidate(candidate)
            all_chunks.extend(chunks)

        logger.info(
            f"Processed {len(candidates)} candidates into {len(all_chunks)} chunks"
        )
        return all_chunks

    def process_all_roles(self, roles: list[dict[str, Any]]) -> list[Chunk]:
        """
        Process all roles into chunks.

        Args:
            roles: List of role dictionaries

        Returns:
            List of all Chunk objects
        """
        all_chunks = []
        for role in roles:
            chunks = self.process_role(role)
            all_chunks.extend(chunks)

        logger.info(f"Processed {len(roles)} roles into {len(all_chunks)} chunks")
        return all_chunks

    def process_all(
        self,
        candidates_path: Optional[Path | str] = None,
        roles_path: Optional[Path | str] = None,
        candidates: Optional[list[dict[str, Any]]] = None,
        roles: Optional[list[dict[str, Any]]] = None
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Process all documents from files or provided data.

        Args:
            candidates_path: Path to candidates JSON file
            roles_path: Path to roles JSON file
            candidates: Pre-loaded candidate data
            roles: Pre-loaded role data

        Returns:
            Tuple of (candidate_chunks, role_chunks)
        """
        # Load data if paths provided
        if candidates_path and not candidates:
            candidates = self.load_candidates(candidates_path)
        if roles_path and not roles:
            roles = self.load_roles(roles_path)

        candidate_chunks = []
        role_chunks = []

        if candidates:
            candidate_chunks = self.process_all_candidates(candidates)

        if roles:
            role_chunks = self.process_all_roles(roles)

        logger.info(
            f"Total processing complete: "
            f"{len(candidate_chunks)} candidate chunks, "
            f"{len(role_chunks)} role chunks"
        )

        return candidate_chunks, role_chunks

    def chunks_to_documents(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        """
        Convert chunks to document format for vector store.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of document dictionaries with id, text, and metadata
        """
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": {
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "document_type": chunk.document_type.value,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count
                }
            }
            for chunk in chunks
        ]
