"""
Document Chunking Module.

This module provides intelligent chunking strategies for different
document types in the Talent RAG system.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from loguru import logger

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to approximate token counting")


class DocumentType(Enum):
    """Types of documents in the talent RAG system."""
    RESUME = "resume"
    ROLE = "role"
    INTERVIEW = "interview"


@dataclass
class Chunk:
    """
    A chunk of text with metadata.

    Attributes:
        id: Unique chunk identifier
        text: The chunk text content
        document_id: ID of the source document
        document_type: Type of the source document
        chunk_index: Position of chunk in the document
        metadata: Additional metadata about the chunk
        token_count: Number of tokens in the chunk
    """
    id: str
    text: str
    document_id: str
    document_type: DocumentType
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "token_count": self.token_count
        }


class DocumentChunker:
    """
    Intelligent document chunker with type-specific strategies.

    This class implements different chunking strategies optimized for
    resumes, job roles, and interview transcripts.

    Attributes:
        chunk_sizes: Target chunk sizes by document type
        overlap: Token overlap between chunks
        encoding: Tiktoken encoding for token counting
    """

    # Default chunk sizes by document type (in tokens)
    DEFAULT_CHUNK_SIZES = {
        DocumentType.RESUME: 400,
        DocumentType.ROLE: 250,
        DocumentType.INTERVIEW: 200
    }

    def __init__(
        self,
        chunk_size_resume: int = 400,
        chunk_size_role: int = 250,
        chunk_size_interview: int = 200,
        overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size_resume: Target tokens for resume chunks
            chunk_size_role: Target tokens for role chunks
            chunk_size_interview: Target tokens for interview chunks
            overlap: Token overlap between consecutive chunks
            encoding_name: Tiktoken encoding name for token counting
        """
        self.chunk_sizes = {
            DocumentType.RESUME: chunk_size_resume,
            DocumentType.ROLE: chunk_size_role,
            DocumentType.INTERVIEW: chunk_size_interview
        }
        self.overlap = overlap

        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding: {e}")
                self.encoding = None
        else:
            self.encoding = None

        logger.info(
            f"DocumentChunker initialized with sizes: resume={chunk_size_resume}, "
            f"role={chunk_size_role}, interview={chunk_size_interview}, overlap={overlap}"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken or approximation.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens
        """
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Approximate: ~4 characters per token for English
            return len(text) // 4

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on sentence boundaries while preserving the delimiter
        sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split on double newlines or markdown section breaks
        paragraphs = re.split(r'\n\n+|\n(?=##)', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_by_tokens(
        self,
        text: str,
        target_size: int,
        preserve_sentences: bool = True
    ) -> list[str]:
        """
        Chunk text by token count.

        Args:
            text: Text to chunk
            target_size: Target tokens per chunk
            preserve_sentences: Whether to avoid splitting mid-sentence

        Returns:
            List of text chunks
        """
        if preserve_sentences:
            units = self._split_into_sentences(text)
        else:
            # Split by words
            units = text.split()

        chunks = []
        current_chunk = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            # If single unit exceeds target, split it
            if unit_tokens > target_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large unit by words
                words = unit.split()
                word_chunk = []
                word_tokens = 0
                for word in words:
                    wt = self.count_tokens(word)
                    if word_tokens + wt > target_size:
                        if word_chunk:
                            chunks.append(' '.join(word_chunk))
                        word_chunk = [word]
                        word_tokens = wt
                    else:
                        word_chunk.append(word)
                        word_tokens += wt
                if word_chunk:
                    current_chunk = word_chunk
                    current_tokens = word_tokens
                continue

            # Check if adding this unit would exceed target
            if current_tokens + unit_tokens > target_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    # Include some overlap from previous chunk
                    overlap_text = ' '.join(current_chunk)
                    overlap_tokens = self.count_tokens(overlap_text)
                    if overlap_tokens > self.overlap:
                        # Take last few sentences/units
                        overlap_units = []
                        overlap_count = 0
                        for u in reversed(current_chunk):
                            ut = self.count_tokens(u)
                            if overlap_count + ut <= self.overlap:
                                overlap_units.insert(0, u)
                                overlap_count += ut
                            else:
                                break
                        current_chunk = overlap_units + [unit]
                        current_tokens = overlap_count + unit_tokens
                    else:
                        current_chunk = [unit]
                        current_tokens = unit_tokens
                else:
                    current_chunk = [unit]
                    current_tokens = unit_tokens
            else:
                current_chunk.append(unit)
                current_tokens += unit_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_resume(
        self,
        text: str,
        document_id: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> list[Chunk]:
        """
        Chunk a resume document with context preservation.

        Resumes are chunked to preserve section context (experience,
        education, skills) while maintaining reasonable chunk sizes.

        Args:
            text: Resume text content
            document_id: Unique document identifier
            metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        target_size = self.chunk_sizes[DocumentType.RESUME]
        metadata = metadata or {}

        # Split into major sections first
        sections = self._split_into_paragraphs(text)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_tokens = self.count_tokens(section)

            if section_tokens <= target_size:
                # Section fits in one chunk
                chunk = Chunk(
                    id=f"{document_id}_chunk_{chunk_index}",
                    text=section,
                    document_id=document_id,
                    document_type=DocumentType.RESUME,
                    chunk_index=chunk_index,
                    metadata={**metadata, "section": self._detect_section(section)},
                    token_count=section_tokens
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Section needs splitting
                sub_chunks = self._chunk_by_tokens(section, target_size)
                section_name = self._detect_section(section)

                for sub_text in sub_chunks:
                    chunk = Chunk(
                        id=f"{document_id}_chunk_{chunk_index}",
                        text=sub_text,
                        document_id=document_id,
                        document_type=DocumentType.RESUME,
                        chunk_index=chunk_index,
                        metadata={**metadata, "section": section_name},
                        token_count=self.count_tokens(sub_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        logger.debug(f"Created {len(chunks)} chunks from resume {document_id}")
        return chunks

    def chunk_role(
        self,
        text: str,
        document_id: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> list[Chunk]:
        """
        Chunk a job role description.

        Role descriptions are chunked to preserve logical sections like
        responsibilities, requirements, and benefits.

        Args:
            text: Role description text
            document_id: Unique document identifier
            metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        target_size = self.chunk_sizes[DocumentType.ROLE]
        metadata = metadata or {}

        # Split into sections
        sections = self._split_into_paragraphs(text)

        chunks = []
        chunk_index = 0

        for section in sections:
            section_tokens = self.count_tokens(section)

            if section_tokens <= target_size:
                chunk = Chunk(
                    id=f"{document_id}_chunk_{chunk_index}",
                    text=section,
                    document_id=document_id,
                    document_type=DocumentType.ROLE,
                    chunk_index=chunk_index,
                    metadata={**metadata, "section": self._detect_role_section(section)},
                    token_count=section_tokens
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                sub_chunks = self._chunk_by_tokens(section, target_size)
                section_name = self._detect_role_section(section)

                for sub_text in sub_chunks:
                    chunk = Chunk(
                        id=f"{document_id}_chunk_{chunk_index}",
                        text=sub_text,
                        document_id=document_id,
                        document_type=DocumentType.ROLE,
                        chunk_index=chunk_index,
                        metadata={**metadata, "section": section_name},
                        token_count=self.count_tokens(sub_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        logger.debug(f"Created {len(chunks)} chunks from role {document_id}")
        return chunks

    def chunk_interview(
        self,
        text: str,
        document_id: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> list[Chunk]:
        """
        Chunk an interview transcript keeping Q&A pairs together.

        Interview transcripts are chunked to preserve question-answer
        pairs and maintain conversation context.

        Args:
            text: Interview transcript text
            document_id: Unique document identifier
            metadata: Additional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        target_size = self.chunk_sizes[DocumentType.INTERVIEW]
        metadata = metadata or {}

        # Split by Q&A pairs
        qa_pattern = re.compile(r'(\*\*Q:.*?\*\*.*?)(?=\*\*Q:|---|\Z)', re.DOTALL)
        qa_matches = qa_pattern.findall(text)

        # Also capture intro/metadata sections
        intro_match = re.match(r'^(.*?)(?=\*\*Q:)', text, re.DOTALL)
        sections = []
        if intro_match:
            intro = intro_match.group(1).strip()
            if intro:
                sections.append(("intro", intro))

        for qa in qa_matches:
            qa = qa.strip()
            if qa:
                sections.append(("qa", qa))

        # Capture interviewer notes at the end
        notes_match = re.search(r'## Interviewer Notes.*', text, re.DOTALL)
        if notes_match:
            sections.append(("notes", notes_match.group(0).strip()))

        chunks = []
        chunk_index = 0

        for section_type, section_text in sections:
            section_tokens = self.count_tokens(section_text)

            if section_tokens <= target_size:
                chunk = Chunk(
                    id=f"{document_id}_chunk_{chunk_index}",
                    text=section_text,
                    document_id=document_id,
                    document_type=DocumentType.INTERVIEW,
                    chunk_index=chunk_index,
                    metadata={**metadata, "section_type": section_type},
                    token_count=section_tokens
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large sections while trying to preserve context
                sub_chunks = self._chunk_by_tokens(section_text, target_size)

                for sub_text in sub_chunks:
                    chunk = Chunk(
                        id=f"{document_id}_chunk_{chunk_index}",
                        text=sub_text,
                        document_id=document_id,
                        document_type=DocumentType.INTERVIEW,
                        chunk_index=chunk_index,
                        metadata={**metadata, "section_type": section_type},
                        token_count=self.count_tokens(sub_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        logger.debug(f"Created {len(chunks)} chunks from interview {document_id}")
        return chunks

    def _detect_section(self, text: str) -> str:
        """Detect the section type of resume text."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ['experience', 'work history', 'employment']):
            return "experience"
        elif any(kw in text_lower for kw in ['education', 'degree', 'university']):
            return "education"
        elif any(kw in text_lower for kw in ['skill', 'technologies', 'technical']):
            return "skills"
        elif any(kw in text_lower for kw in ['summary', 'objective', 'about']):
            return "summary"
        elif any(kw in text_lower for kw in ['project', 'portfolio']):
            return "projects"
        elif any(kw in text_lower for kw in ['publication', 'research', 'paper']):
            return "publications"
        else:
            return "general"

    def _detect_role_section(self, text: str) -> str:
        """Detect the section type of role description text."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ['about the role', 'about this']):
            return "overview"
        elif any(kw in text_lower for kw in ['responsibilit', 'you will', "what you'll do"]):
            return "responsibilities"
        elif any(kw in text_lower for kw in ['requirement', 'qualification', 'you have']):
            return "requirements"
        elif any(kw in text_lower for kw in ['benefit', 'perk', 'offer']):
            return "benefits"
        elif any(kw in text_lower for kw in ['about us', 'about the company', 'who we are']):
            return "company"
        elif any(kw in text_lower for kw in ['tech stack', 'technologies']):
            return "tech_stack"
        else:
            return "general"

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_type: DocumentType,
        metadata: Optional[dict[str, Any]] = None
    ) -> list[Chunk]:
        """
        Chunk a document using the appropriate strategy.

        Args:
            text: Document text content
            document_id: Unique document identifier
            document_type: Type of document
            metadata: Additional metadata

        Returns:
            List of Chunk objects
        """
        if document_type == DocumentType.RESUME:
            return self.chunk_resume(text, document_id, metadata)
        elif document_type == DocumentType.ROLE:
            return self.chunk_role(text, document_id, metadata)
        elif document_type == DocumentType.INTERVIEW:
            return self.chunk_interview(text, document_id, metadata)
        else:
            raise ValueError(f"Unknown document type: {document_type}")
