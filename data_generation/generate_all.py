"""
Main data generation script.

This module orchestrates the generation of all synthetic data required
for the Talent RAG system.
"""

from pathlib import Path

from loguru import logger

from .candidate_generator import CandidateGenerator
from .role_generator import RoleGenerator


def generate_all_data(
    output_dir: Path | str = None,
    num_candidates: int = 100,
    num_roles: int = 20,
    seed: int = 42
) -> tuple[Path, Path]:
    """
    Generate all synthetic data for the Talent RAG system.

    This function generates candidate profiles with resumes and interview
    transcripts, as well as job role postings, and saves them to JSON files.

    Args:
        output_dir: Directory to save generated data. Defaults to data/
        num_candidates: Number of candidates to generate
        num_roles: Number of roles to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (candidates_path, roles_path)
    """
    # Default output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Talent RAG Data Generation")
    logger.info("=" * 60)

    # Generate candidates
    logger.info(f"Generating {num_candidates} candidates with seed={seed}")
    candidate_gen = CandidateGenerator(seed=seed)
    candidates = candidate_gen.generate_candidates(count=num_candidates)
    candidates_path = output_dir / "candidates.json"
    candidate_gen.save_candidates(candidates, candidates_path)

    # Generate roles
    logger.info(f"Generating {num_roles} roles with seed={seed}")
    role_gen = RoleGenerator(seed=seed)
    roles = role_gen.generate_roles(count=num_roles)
    roles_path = output_dir / "roles.json"
    role_gen.save_roles(roles, roles_path)

    logger.info("=" * 60)
    logger.info("Data Generation Complete!")
    logger.info(f"  Candidates: {candidates_path}")
    logger.info(f"  Roles: {roles_path}")
    logger.info("=" * 60)

    return candidates_path, roles_path


if __name__ == "__main__":
    # Run data generation when executed directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    generate_all_data()
