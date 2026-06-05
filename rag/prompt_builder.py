"""
Prompt Builder Module.

This module provides prompt construction utilities for the RAG pipeline,
building context-aware prompts with evidence and citations.
"""

from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ..vectorstore.faiss_store import SearchResult


class QueryType(Enum):
    """Types of queries the system can handle."""
    CANDIDATE_SEARCH = "candidate_search"
    CANDIDATE_COMPARE = "candidate_compare"
    ROLE_ANALYSIS = "role_analysis"
    GENERAL_QUESTION = "general_question"
    SKILL_ANALYSIS = "skill_analysis"


@dataclass
class PromptContext:
    """
    Context information for prompt building.

    Attributes:
        query: User's query
        query_type: Type of query
        evidence: List of evidence chunks
        role: Optional role context
        candidates: Optional candidate contexts
        additional_context: Any additional context
    """
    query: str
    query_type: QueryType
    evidence: list[SearchResult]
    role: Optional[dict[str, Any]] = None
    candidates: Optional[list[dict[str, Any]]] = None
    additional_context: Optional[str] = None


class PromptBuilder:
    """
    Builder for RAG prompts with evidence and citations.

    This class constructs prompts that include retrieved evidence
    with proper citation markers for grounded generation.

    Attributes:
        system_prompt_template: Template for system prompts
        max_evidence_tokens: Maximum tokens for evidence
        include_reasoning: Whether to include chain-of-thought
    """

    # Base system prompt for the Talent Intelligence Assistant
    BASE_SYSTEM_PROMPT = """You are an expert AI Talent Intelligence Assistant designed to help recruiters find and evaluate candidates. You have access to candidate resumes, interview transcripts, and job role descriptions.

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided evidence to answer questions. Do not make up information.
2. Always cite your sources using the provided citation markers (e.g., [CAND_001], [ROLE_003]).
3. If the evidence doesn't contain enough information to answer, say so clearly.
4. Be objective and professional in your assessments.
5. Highlight both strengths and potential concerns when evaluating candidates.
6. Consider both technical skills and soft skills from interview transcripts.

When comparing candidates:
- Provide a balanced analysis of each candidate
- Highlight unique strengths of each
- Note any gaps relative to role requirements
- Give a clear recommendation with justification"""

    # Query-specific prompt additions
    QUERY_PROMPTS = {
        QueryType.CANDIDATE_SEARCH: """
You are searching for candidates matching specific criteria.
Analyze the evidence and identify the best matching candidates.
For each candidate, explain:
1. How well they match the requirements
2. Key relevant experience and skills
3. Any potential concerns or gaps
4. Overall recommendation""",

        QueryType.CANDIDATE_COMPARE: """
You are comparing multiple candidates for a role.
Provide a detailed side-by-side analysis covering:
1. Technical skills alignment
2. Experience relevance
3. Interview performance insights
4. Cultural fit indicators
5. Final ranking with justification""",

        QueryType.ROLE_ANALYSIS: """
You are analyzing a job role and its requirements.
Provide insights on:
1. Key requirements and their importance
2. Ideal candidate profile
3. Potential challenges in filling this role
4. Suggestions for candidate sourcing""",

        QueryType.GENERAL_QUESTION: """
Answer the user's question based on the provided evidence.
Be thorough but concise.
Always cite specific evidence for your claims.""",

        QueryType.SKILL_ANALYSIS: """
You are analyzing skills and experience.
Provide detailed insights on:
1. Skill proficiency levels based on evidence
2. Practical application examples
3. Growth trajectory
4. Recommendations"""
    }

    def __init__(
        self,
        max_evidence_tokens: int = 3000,
        include_reasoning: bool = True,
        citation_format: str = "[{id}]"
    ):
        """
        Initialize the prompt builder.

        Args:
            max_evidence_tokens: Maximum tokens for evidence section
            include_reasoning: Whether to include reasoning instructions
            citation_format: Format string for citations
        """
        self.max_evidence_tokens = max_evidence_tokens
        self.include_reasoning = include_reasoning
        self.citation_format = citation_format

        logger.info(
            f"PromptBuilder initialized with max_evidence_tokens={max_evidence_tokens}"
        )

    def _format_evidence(
        self,
        evidence: list[SearchResult],
        max_chars: int = 12000
    ) -> str:
        """
        Format evidence chunks with citations.

        Args:
            evidence: List of search results
            max_chars: Maximum characters for evidence

        Returns:
            Formatted evidence string
        """
        if not evidence:
            return "No relevant evidence found."

        evidence_parts = []
        total_chars = 0

        for result in evidence:
            # Extract candidate/role ID for citation
            doc_id = result.metadata.get("candidate_id") or result.metadata.get("role_id") or result.id
            source_type = result.metadata.get("source", "document")

            # Format citation marker
            citation = self.citation_format.format(id=doc_id)

            # Build evidence entry
            entry_parts = [f"--- Evidence {citation} ---"]
            entry_parts.append(f"Source: {source_type}")

            if result.metadata.get("candidate_name"):
                entry_parts.append(f"Candidate: {result.metadata['candidate_name']}")

            if result.metadata.get("seniority"):
                entry_parts.append(f"Level: {result.metadata['seniority']}")

            if result.metadata.get("skills"):
                entry_parts.append(f"Skills: {', '.join(result.metadata['skills'][:10])}")

            entry_parts.append(f"\n{result.text}")
            entry_parts.append("---\n")

            entry = "\n".join(entry_parts)

            # Check length limit
            if total_chars + len(entry) > max_chars:
                break

            evidence_parts.append(entry)
            total_chars += len(entry)

        return "\n".join(evidence_parts)

    def _format_role_context(self, role: dict[str, Any]) -> str:
        """
        Format role information for context.

        Args:
            role: Role dictionary

        Returns:
            Formatted role context string
        """
        parts = [
            "=== Target Role ===",
            f"Title: {role.get('title', 'Unknown')}",
            f"Company: {role.get('company', 'Unknown')}",
            f"Location: {role.get('location', 'Unknown')}",
            f"Seniority: {role.get('seniority', 'Unknown')}",
        ]

        if role.get("required_skills"):
            parts.append(f"Required Skills: {', '.join(role['required_skills'])}")

        if role.get("nice_to_have_skills"):
            parts.append(f"Nice to Have: {', '.join(role['nice_to_have_skills'])}")

        if role.get("years_experience_required"):
            parts.append(f"Experience Required: {role['years_experience_required']}+ years")

        parts.append("==================\n")

        return "\n".join(parts)

    def _build_reasoning_instruction(self) -> str:
        """Build chain-of-thought reasoning instruction."""
        return """
Before providing your final answer, think through the following (but do not include this reasoning in your response):
1. What are the key points from the evidence?
2. How does the evidence relate to the query?
3. What are the strengths of the evidence for answering this question?
4. Are there any gaps or limitations in the evidence?
5. What is the most accurate and helpful response based on this evidence?

Now provide your response:"""

    def build_system_prompt(
        self,
        query_type: QueryType,
        variation: int = 0
    ) -> str:
        """
        Build the system prompt.

        Args:
            query_type: Type of query
            variation: Variation number for ensemble (0 = default)

        Returns:
            System prompt string
        """
        base = self.BASE_SYSTEM_PROMPT

        # Add query-specific instructions
        if query_type in self.QUERY_PROMPTS:
            base += "\n\n" + self.QUERY_PROMPTS[query_type]

        # Add variations for ensemble
        if variation == 1:
            base += "\n\nBe particularly thorough in citing evidence."
        elif variation == 2:
            base += "\n\nFocus on practical, actionable insights."

        return base

    def build_user_prompt(
        self,
        context: PromptContext
    ) -> str:
        """
        Build the user prompt with evidence.

        Args:
            context: PromptContext with query and evidence

        Returns:
            User prompt string
        """
        parts = []

        # Add role context if present
        if context.role:
            parts.append(self._format_role_context(context.role))

        # Add evidence section
        parts.append("=== Retrieved Evidence ===")
        parts.append(self._format_evidence(context.evidence))
        parts.append("=========================\n")

        # Add additional context if present
        if context.additional_context:
            parts.append(f"Additional Context: {context.additional_context}\n")

        # Add the query
        parts.append(f"User Query: {context.query}")

        # Add reasoning instruction if enabled
        if self.include_reasoning:
            parts.append(self._build_reasoning_instruction())

        return "\n".join(parts)

    def build_prompt(
        self,
        context: PromptContext,
        variation: int = 0
    ) -> tuple[str, str]:
        """
        Build complete prompt (system + user).

        Args:
            context: PromptContext with all information
            variation: Variation number for ensemble

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.build_system_prompt(context.query_type, variation)
        user_prompt = self.build_user_prompt(context)

        return system_prompt, user_prompt

    def build_messages(
        self,
        context: PromptContext,
        variation: int = 0
    ) -> list[dict[str, str]]:
        """
        Build OpenAI-compatible messages list.

        Args:
            context: PromptContext with all information
            variation: Variation number for ensemble

        Returns:
            List of message dictionaries
        """
        system_prompt, user_prompt = self.build_prompt(context, variation)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def build_comparison_prompt(
        self,
        candidates: list[dict[str, Any]],
        role: dict[str, Any],
        evidence: list[SearchResult]
    ) -> PromptContext:
        """
        Build a prompt context for candidate comparison.

        Args:
            candidates: List of candidate metadata dicts
            role: Role to compare against
            evidence: Retrieved evidence chunks

        Returns:
            PromptContext for comparison
        """
        # Build comparison query
        candidate_names = [c.get("name", c.get("candidate_id", "Unknown")) for c in candidates]
        query = (
            f"Compare the following candidates for the {role.get('title', 'role')}: "
            f"{', '.join(candidate_names)}. "
            f"Provide a detailed analysis and recommendation."
        )

        return PromptContext(
            query=query,
            query_type=QueryType.CANDIDATE_COMPARE,
            evidence=evidence,
            role=role,
            candidates=candidates
        )

    def detect_query_type(self, query: str) -> QueryType:
        """
        Detect the type of query from the text.

        Args:
            query: User's query string

        Returns:
            Detected QueryType
        """
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["compare", "versus", "vs", "between", "difference"]):
            return QueryType.CANDIDATE_COMPARE

        if any(kw in query_lower for kw in ["find", "search", "looking for", "candidates for", "who"]):
            return QueryType.CANDIDATE_SEARCH

        if any(kw in query_lower for kw in ["role", "position", "job", "requirements"]):
            return QueryType.ROLE_ANALYSIS

        if any(kw in query_lower for kw in ["skill", "experience", "proficiency", "expertise"]):
            return QueryType.SKILL_ANALYSIS

        return QueryType.GENERAL_QUESTION
