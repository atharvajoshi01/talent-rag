"""
Text Cleaning Module.

This module provides utilities for cleaning and normalizing text
before chunking and embedding.
"""

import re
from typing import Optional

from loguru import logger


class TextCleaner:
    """
    Text cleaning and normalization utility.

    This class provides methods for cleaning text content from various
    sources including resumes, job descriptions, and interview transcripts.

    Attributes:
        remove_urls: Whether to remove URLs from text
        remove_emails: Whether to remove email addresses from text
        normalize_whitespace: Whether to normalize whitespace
        remove_special_chars: Whether to remove special characters
    """

    def __init__(
        self,
        remove_urls: bool = False,
        remove_emails: bool = False,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False
    ):
        """
        Initialize the text cleaner.

        Args:
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
            remove_special_chars: Whether to remove special characters
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars

        # Compile regex patterns
        self._url_pattern = re.compile(
            r'https?://\S+|www\.\S+',
            re.IGNORECASE
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self._whitespace_pattern = re.compile(r'\s+')
        self._special_chars_pattern = re.compile(r'[^\w\s\-.,;:!?\'\"()\[\]{}]')

        logger.debug("TextCleaner initialized")

    def clean(self, text: str) -> str:
        """
        Apply all configured cleaning operations to text.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text string
        """
        if not text:
            return ""

        result = text

        # Remove URLs if configured
        if self.remove_urls:
            result = self._remove_urls(result)

        # Remove emails if configured
        if self.remove_emails:
            result = self._remove_emails(result)

        # Remove special characters if configured
        if self.remove_special_chars:
            result = self._remove_special_characters(result)

        # Normalize whitespace if configured
        if self.normalize_whitespace:
            result = self._normalize_whitespace(result)

        return result.strip()

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self._url_pattern.sub('', text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self._email_pattern.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces to single, trim lines)."""
        # Replace multiple spaces with single space
        text = self._whitespace_pattern.sub(' ', text)
        # Normalize line breaks
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        return '\n'.join(lines)

    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving basic punctuation."""
        return self._special_chars_pattern.sub('', text)

    def remove_markdown_headers(self, text: str) -> str:
        """
        Remove markdown header markers from text.

        Args:
            text: Text with markdown headers

        Returns:
            Text with header markers removed
        """
        # Remove # markers but preserve the text
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.startswith('#'):
                # Remove leading # and spaces
                cleaned_line = line.lstrip('#').strip()
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def extract_sections(self, text: str) -> dict[str, str]:
        """
        Extract sections from markdown-formatted text.

        Args:
            text: Markdown text with ## headers

        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        current_section = "intro"
        current_content = []

        for line in text.split('\n'):
            if line.startswith('## '):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                # Start new section
                current_section = line.lstrip('#').strip().lower()
                current_content = []
            elif line.startswith('### '):
                # Subsection - include in current section
                current_content.append(line.lstrip('#').strip())
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        return sections

    def normalize_skills(self, skills: list[str]) -> list[str]:
        """
        Normalize skill names for consistent matching.

        Args:
            skills: List of skill names

        Returns:
            List of normalized skill names
        """
        normalized = []
        for skill in skills:
            # Convert to lowercase and strip
            norm = skill.strip().lower()
            # Common normalizations
            replacements = {
                'javascript': 'javascript',
                'js': 'javascript',
                'typescript': 'typescript',
                'ts': 'typescript',
                'python3': 'python',
                'py': 'python',
                'postgres': 'postgresql',
                'k8s': 'kubernetes',
                'tf': 'terraform',
                'aws lambda': 'aws',
                'gcp': 'google cloud',
                'ml': 'machine learning',
                'dl': 'deep learning',
                'nlp': 'natural language processing',
                'cv': 'computer vision',
            }
            if norm in replacements:
                norm = replacements[norm]
            normalized.append(norm)
        return normalized

    def truncate_text(
        self,
        text: str,
        max_chars: int,
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to maximum characters while preserving word boundaries.

        Args:
            text: Text to truncate
            max_chars: Maximum number of characters
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text

        # Find last space before max_chars
        truncated = text[:max_chars - len(suffix)]
        last_space = truncated.rfind(' ')

        if last_space > max_chars // 2:
            truncated = truncated[:last_space]

        return truncated + suffix
