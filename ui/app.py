"""
Streamlit UI Application.

This module provides the Streamlit frontend for the
Talent Intelligence Assistant.
"""

from typing import Optional

import streamlit as st
import requests

# API Configuration
API_BASE_URL = "http://localhost:8000"


def init_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = None
    if "comparison_candidates" not in st.session_state:
        st.session_state.comparison_candidates = []


def fetch_roles() -> list[dict]:
    """Fetch available roles from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/roles", timeout=10)
        if response.status_code == 200:
            return response.json().get("roles", [])
    except Exception as e:
        st.error(f"Failed to fetch roles: {e}")
    return []


def fetch_candidates(skip: int = 0, limit: int = 50) -> list[dict]:
    """Fetch candidates from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/candidates",
            params={"skip": skip, "limit": limit},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("candidates", [])
    except Exception as e:
        st.error(f"Failed to fetch candidates: {e}")
    return []


def search_candidates(query: str, filters: dict, k: int = 10) -> dict:
    """Search for candidates via API."""
    try:
        payload = {
            "query": query,
            "filters": filters if filters else None,
            "k": k
        }
        if st.session_state.selected_role:
            payload["role_id"] = st.session_state.selected_role["id"]

        response = requests.post(
            f"{API_BASE_URL}/search_candidates",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
    return {}


def compare_candidates(candidate_ids: list[str], role_id: str) -> dict:
    """Compare candidates via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/compare_candidates",
            json={
                "candidate_ids": candidate_ids,
                "role_id": role_id
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Comparison failed: {e}")
    return {}


def ask_question(question: str, context_filter: Optional[str] = None) -> dict:
    """Ask a general question via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={
                "question": question,
                "context_filter": context_filter,
                "include_evidence": True
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Question failed: {e}")
    return {}


def render_candidate_card(candidate: dict, match_score: Optional[float] = None):
    """Render a candidate card."""
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {candidate.get('name', candidate.get('candidate_id', 'Unknown'))}")
            st.markdown(f"**{candidate.get('seniority', '')}** | {candidate.get('location', '')}")
            st.markdown(f"Experience: {candidate.get('years_of_experience', candidate.get('years_experience', 0))} years")

            # Skills
            skills = candidate.get("skills", [])[:8]
            if skills:
                st.markdown("**Skills:** " + ", ".join(skills))

        with col2:
            if match_score is not None:
                score_pct = int(match_score * 100)
                st.metric("Match Score", f"{score_pct}%")

        # Add to comparison button
        candidate_id = candidate.get("id") or candidate.get("candidate_id")
        if candidate_id:
            if st.button("Add to Compare", key=f"add_{candidate_id}"):
                if candidate_id not in st.session_state.comparison_candidates:
                    st.session_state.comparison_candidates.append(candidate_id)
                    st.success(f"Added {candidate.get('name', candidate_id)} to comparison")

        st.divider()


def render_evidence(evidence: list[dict]):
    """Render evidence section."""
    if not evidence:
        return

    with st.expander("View Evidence", expanded=False):
        for i, e in enumerate(evidence):
            st.markdown(f"**Source {i+1}:** [{e.get('source', 'unknown')}]")
            if e.get("candidate_name"):
                st.markdown(f"*{e.get('candidate_name')}*")
            st.markdown(f"> {e.get('text', '')[:500]}...")
            st.markdown(f"*Relevance Score: {e.get('score', 0):.2f}*")
            st.divider()


def render_chat_message(role: str, content: str, evidence: Optional[list] = None):
    """Render a chat message."""
    with st.chat_message(role):
        st.markdown(content)
        if evidence:
            render_evidence(evidence)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Talent Intelligence Assistant",
        page_icon="",
        layout="wide"
    )

    init_session_state()

    # Header
    st.title("Talent Intelligence Assistant")
    st.markdown("*AI-powered recruitment copilot*")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Role Selection
        st.subheader("Target Role")
        roles = fetch_roles()

        if roles:
            role_options = ["None"] + [f"{r['title']} ({r['id']})" for r in roles]
            selected = st.selectbox("Select a role to match against:", role_options)

            if selected != "None":
                role_id = selected.split("(")[-1].rstrip(")")
                st.session_state.selected_role = next(
                    (r for r in roles if r["id"] == role_id), None
                )
            else:
                st.session_state.selected_role = None

            if st.session_state.selected_role:
                role = st.session_state.selected_role
                st.info(f"**{role['title']}**\n\n"
                       f"Company: {role.get('company', 'N/A')}\n\n"
                       f"Skills: {', '.join(role.get('required_skills', [])[:5])}")

        st.divider()

        # Filters
        st.subheader("Search Filters")

        skills_input = st.text_input(
            "Required Skills (comma-separated)",
            placeholder="Python, Machine Learning, AWS"
        )

        location = st.selectbox(
            "Location",
            ["Any", "San Francisco, CA", "New York, NY", "Seattle, WA",
             "Austin, TX", "Remote", "Boston, MA"]
        )

        seniority = st.multiselect(
            "Seniority Level",
            ["Junior", "Mid-Level", "Senior", "Staff", "Principal", "Director"]
        )

        min_exp = st.slider("Minimum Experience (years)", 0, 20, 0)

        st.divider()

        # Comparison Section
        st.subheader("Candidate Comparison")
        if st.session_state.comparison_candidates:
            st.write(f"Selected: {len(st.session_state.comparison_candidates)} candidates")
            for cid in st.session_state.comparison_candidates:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(cid)
                with col2:
                    if st.button("X", key=f"remove_{cid}"):
                        st.session_state.comparison_candidates.remove(cid)
                        st.rerun()

            if len(st.session_state.comparison_candidates) >= 2:
                if st.button("Compare Selected"):
                    if st.session_state.selected_role:
                        st.session_state.show_comparison = True
                    else:
                        st.warning("Please select a role first")

            if st.button("Clear Selection"):
                st.session_state.comparison_candidates = []
                st.rerun()
        else:
            st.write("No candidates selected")

    # Main content area - Tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Search", "Browse"])

    # Chat Tab
    with tab1:
        st.header("Recruiter Copilot")

        # Display chat history
        for message in st.session_state.messages:
            render_chat_message(
                message["role"],
                message["content"],
                message.get("evidence")
            )

        # Chat input
        if prompt := st.chat_input("Ask about candidates, roles, or search for talent..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            render_chat_message("user", prompt)

            # Get response
            with st.spinner("Thinking..."):
                response = ask_question(prompt)

            if response:
                answer = response.get("answer", "I couldn't find an answer.")
                evidence = response.get("evidence_used", [])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence
                })
                render_chat_message("assistant", answer, evidence)

                # Show metadata
                meta = response.get("meta", {})
                st.caption(
                    f"Latency: {meta.get('latency_ms', 0):.0f}ms | "
                    f"Retrieved: {meta.get('retrieval_count', 0)} docs"
                )

    # Search Tab
    with tab2:
        st.header("Candidate Search")

        # Search input
        search_query = st.text_area(
            "Describe your ideal candidate:",
            placeholder="e.g., Senior machine learning engineer with experience in NLP and production ML systems..."
        )

        num_results = st.slider("Number of results", 5, 20, 10)

        if st.button("Search", type="primary"):
            if search_query:
                # Build filters
                filters = {}
                if skills_input:
                    filters["skills"] = [s.strip() for s in skills_input.split(",")]
                if location != "Any":
                    filters["location"] = location
                if seniority:
                    filters["seniority"] = seniority
                if min_exp > 0:
                    filters["min_experience"] = min_exp

                with st.spinner("Searching..."):
                    results = search_candidates(search_query, filters, num_results)

                if results:
                    st.success("Found matching candidates")

                    # Display answer
                    st.markdown("### AI Analysis")
                    st.markdown(results.get("answer", ""))

                    # Display evidence as candidate cards
                    st.markdown("### Retrieved Candidates")
                    evidence = results.get("evidence_used", [])

                    # Group evidence by candidate
                    candidates_seen = set()
                    for e in evidence:
                        cid = e.get("candidate_id")
                        if cid and cid not in candidates_seen:
                            candidates_seen.add(cid)
                            render_candidate_card({
                                "candidate_id": cid,
                                "id": cid,
                                "name": e.get("candidate_name", cid),
                                "seniority": e.get("metadata", {}).get("seniority", ""),
                                "location": e.get("metadata", {}).get("location", ""),
                                "years_experience": e.get("metadata", {}).get("years_experience", 0),
                                "skills": e.get("metadata", {}).get("skills", [])
                            }, e.get("score"))

                    # Show full evidence
                    render_evidence(evidence)

                    # Metadata
                    meta = results.get("meta", {})
                    st.caption(
                        f"Search completed in {meta.get('latency_ms', 0):.0f}ms"
                    )
            else:
                st.warning("Please enter a search query")

    # Browse Tab
    with tab3:
        st.header("Browse Candidates")

        candidates = fetch_candidates(limit=50)

        if candidates:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                filter_seniority = st.selectbox(
                    "Filter by Seniority",
                    ["All"] + list(set(c.get("seniority", "") for c in candidates))
                )
            with col2:
                filter_location = st.selectbox(
                    "Filter by Location",
                    ["All"] + list(set(c.get("location", "") for c in candidates))
                )

            # Apply filters
            filtered = candidates
            if filter_seniority != "All":
                filtered = [c for c in filtered if c.get("seniority") == filter_seniority]
            if filter_location != "All":
                filtered = [c for c in filtered if c.get("location") == filter_location]

            st.write(f"Showing {len(filtered)} candidates")

            # Display candidates
            for candidate in filtered:
                render_candidate_card(candidate)
        else:
            st.info("No candidates available. Make sure the API is running.")

    # Comparison Modal
    if hasattr(st.session_state, 'show_comparison') and st.session_state.show_comparison:
        st.header("Candidate Comparison")

        with st.spinner("Generating comparison..."):
            comparison = compare_candidates(
                st.session_state.comparison_candidates,
                st.session_state.selected_role["id"]
            )

        if comparison:
            st.markdown(comparison.get("answer", ""))
            render_evidence(comparison.get("evidence_used", []))

        st.session_state.show_comparison = False


if __name__ == "__main__":
    main()
