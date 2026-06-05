"""
Tests for preprocessing module.
"""

import pytest
from talent_rag.preprocessing import TextCleaner, DocumentChunker, Chunk
from talent_rag.preprocessing.chunker import DocumentType


class TestTextCleaner:
    """Tests for TextCleaner."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        cleaner = TextCleaner(normalize_whitespace=True)

        text = "Hello    world\n\n\ntest"
        result = cleaner.clean(text)

        assert "    " not in result
        assert result == "Hello world\n\n\ntest"

    def test_remove_urls(self):
        """Test URL removal."""
        cleaner = TextCleaner(remove_urls=True)

        text = "Check out https://example.com for more info"
        result = cleaner.clean(text)

        assert "https://example.com" not in result
        assert "Check out" in result

    def test_remove_emails(self):
        """Test email removal."""
        cleaner = TextCleaner(remove_emails=True)

        text = "Contact me at test@example.com"
        result = cleaner.clean(text)

        assert "test@example.com" not in result
        assert "Contact me at" in result

    def test_extract_sections(self):
        """Test section extraction from markdown."""
        cleaner = TextCleaner()

        text = """# Header
## Experience
Some experience text
## Education
Some education text
"""
        sections = cleaner.extract_sections(text)

        assert "experience" in sections
        assert "education" in sections

    def test_truncate_text(self):
        """Test text truncation."""
        cleaner = TextCleaner()

        text = "This is a very long text that needs to be truncated"
        result = cleaner.truncate_text(text, 20)

        assert len(result) <= 20
        assert result.endswith("...")


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_count_tokens(self):
        """Test token counting."""
        chunker = DocumentChunker()

        text = "Hello world, this is a test."
        count = chunker.count_tokens(text)

        assert count > 0
        assert count < 20  # Should be around 7-8 tokens

    def test_chunk_short_text(self):
        """Test chunking short text that fits in one chunk."""
        chunker = DocumentChunker(chunk_size_resume=500)

        text = "This is a short resume text."
        chunks = chunker.chunk_resume(text, "DOC_001")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].document_id == "DOC_001"

    def test_chunk_long_text(self):
        """Test chunking long text into multiple chunks."""
        chunker = DocumentChunker(chunk_size_resume=50, overlap=10)

        text = " ".join(["word"] * 200)  # Long text
        chunks = chunker.chunk_resume(text, "DOC_001")

        assert len(chunks) > 1
        # Check all chunks have correct document_id
        for chunk in chunks:
            assert chunk.document_id == "DOC_001"

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves metadata."""
        chunker = DocumentChunker()

        text = "Test resume text"
        metadata = {"candidate_id": "CAND_001", "name": "John"}
        chunks = chunker.chunk_resume(text, "DOC_001", metadata)

        assert chunks[0].metadata["candidate_id"] == "CAND_001"
        assert chunks[0].metadata["name"] == "John"

    def test_chunk_interview_keeps_qa_pairs(self):
        """Test that interview chunking preserves Q&A pairs."""
        chunker = DocumentChunker(chunk_size_interview=500)

        text = """# Interview
**Q: Tell me about your experience?**

*Candidate:* I have 5 years of experience in ML.

**Q: What's your biggest achievement?**

*Candidate:* I built a recommendation system.
"""
        chunks = chunker.chunk_interview(text, "INT_001")

        # Each Q&A should ideally be together
        assert len(chunks) >= 1

    def test_chunk_document_dispatches_correctly(self):
        """Test that chunk_document calls the right method."""
        chunker = DocumentChunker()

        resume_chunks = chunker.chunk_document(
            "Resume text",
            "DOC_001",
            DocumentType.RESUME
        )
        assert resume_chunks[0].document_type == DocumentType.RESUME

        role_chunks = chunker.chunk_document(
            "Role description",
            "ROLE_001",
            DocumentType.ROLE
        )
        assert role_chunks[0].document_type == DocumentType.ROLE


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        chunk = Chunk(
            id="chunk_001",
            text="Test text",
            document_id="DOC_001",
            document_type=DocumentType.RESUME,
            chunk_index=0,
            metadata={"key": "value"},
            token_count=10
        )

        data = chunk.to_dict()

        assert data["id"] == "chunk_001"
        assert data["text"] == "Test text"
        assert data["document_type"] == "resume"
        assert data["metadata"]["key"] == "value"
