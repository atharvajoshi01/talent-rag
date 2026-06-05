"""
Tests for data generation module.
"""

from talent_rag.data_generation import CandidateGenerator, RoleGenerator


class TestCandidateGenerator:
    """Tests for CandidateGenerator."""

    def test_init_with_seed(self):
        """Test initialization with seed for reproducibility."""
        gen1 = CandidateGenerator(seed=42)
        gen2 = CandidateGenerator(seed=42)

        c1 = gen1.generate_candidate(1)
        c2 = gen2.generate_candidate(1)

        assert c1.name == c2.name
        assert c1.skills == c2.skills

    def test_generate_single_candidate(self):
        """Test generating a single candidate."""
        gen = CandidateGenerator(seed=42)
        candidate = gen.generate_candidate(1)

        assert candidate.id == "CAND_001"
        assert candidate.name
        assert candidate.email
        assert candidate.location
        assert isinstance(candidate.years_of_experience, int)
        assert len(candidate.skills) > 0
        assert candidate.seniority
        assert len(candidate.resume_text) > 300
        assert len(candidate.interview_transcript) > 0

    def test_generate_multiple_candidates(self):
        """Test generating multiple candidates."""
        gen = CandidateGenerator(seed=42)
        candidates = gen.generate_candidates(count=10)

        assert len(candidates) == 10
        ids = [c.id for c in candidates]
        assert len(set(ids)) == 10  # All unique IDs

    def test_candidate_skills_vary_by_seniority(self):
        """Test that skills count varies by seniority level."""
        gen = CandidateGenerator(seed=42)

        # Generate many candidates and check skill distribution
        candidates = gen.generate_candidates(count=50)

        junior_skills = [len(c.skills) for c in candidates if c.seniority == "Junior"]
        senior_skills = [len(c.skills) for c in candidates if c.seniority == "Senior"]

        if junior_skills and senior_skills:
            # Seniors should have more skills on average
            assert sum(senior_skills) / len(senior_skills) >= sum(junior_skills) / len(junior_skills)

    def test_candidate_to_dict(self):
        """Test candidate serialization."""
        gen = CandidateGenerator(seed=42)
        candidate = gen.generate_candidate(1)

        data = candidate.to_dict()

        assert data["id"] == "CAND_001"
        assert "name" in data
        assert "skills" in data
        assert isinstance(data["education"], list)
        assert isinstance(data["experience"], list)


class TestRoleGenerator:
    """Tests for RoleGenerator."""

    def test_init_with_seed(self):
        """Test initialization with seed."""
        gen1 = RoleGenerator(seed=42)
        gen2 = RoleGenerator(seed=42)

        r1 = gen1.generate_role(1)
        r2 = gen2.generate_role(1)

        assert r1.title == r2.title

    def test_generate_single_role(self):
        """Test generating a single role."""
        gen = RoleGenerator(seed=42)
        role = gen.generate_role(1)

        assert role.id == "ROLE_001"
        assert role.title
        assert role.company
        assert role.location
        assert len(role.required_skills) > 0
        assert len(role.description) > 100
        assert len(role.responsibilities) > 0
        assert len(role.qualifications) > 0

    def test_generate_multiple_roles(self):
        """Test generating multiple roles."""
        gen = RoleGenerator(seed=42)
        roles = gen.generate_roles(count=5)

        assert len(roles) == 5
        ids = [r.id for r in roles]
        assert len(set(ids)) == 5

    def test_role_has_salary_range(self):
        """Test that roles have salary ranges."""
        gen = RoleGenerator(seed=42)
        role = gen.generate_role(1)

        assert role.salary_range
        assert "$" in role.salary_range

    def test_role_to_dict(self):
        """Test role serialization."""
        gen = RoleGenerator(seed=42)
        role = gen.generate_role(1)

        data = role.to_dict()

        assert data["id"] == "ROLE_001"
        assert "title" in data
        assert "required_skills" in data
        assert isinstance(data["responsibilities"], list)
