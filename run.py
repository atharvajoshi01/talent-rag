#!/usr/bin/env python3
"""
Main entry point for the Talent RAG system.

This script provides CLI commands for common operations.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>"
    )


def cmd_generate_data(args):
    """Generate synthetic data."""
    from data_generation import generate_all_data

    logger.info("Generating synthetic data...")
    generate_all_data(
        num_candidates=args.candidates,
        num_roles=args.roles,
        seed=args.seed
    )
    logger.info("Data generation complete!")


def cmd_build_index(args):
    """Build the vector index."""
    from config import settings
    from retrieval import IndexBuilder

    logger.info("Building vector index...")

    builder = IndexBuilder(
        use_openai_embeddings=args.openai_embeddings
    )

    vector_store = builder.build_index(
        candidates_path=settings.candidates_path,
        roles_path=settings.roles_path
    )

    output_path = Path(args.output) if args.output else settings.index_dir
    builder.save_index(output_path)

    logger.info(f"Index saved to {output_path}")


def cmd_run_api(args):
    """Run the FastAPI server."""
    import uvicorn

    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def cmd_run_ui(args):
    """Run the Streamlit UI."""
    import subprocess

    logger.info("Starting Streamlit UI...")
    subprocess.run([
        "streamlit", "run", "ui/app.py",
        "--server.port", str(args.port),
        "--server.address", args.host
    ])


def cmd_run_benchmark(args):
    """Run the evaluation benchmark."""
    from config import settings
    from embeddings import get_embedding_model
    from vectorstore import FAISSVectorStore
    from rag import RAGPipeline
    from evaluation import RAGBenchmark

    logger.info("Loading pipeline for benchmark...")

    # Load index
    embedding_model = get_embedding_model()
    vector_store = FAISSVectorStore.load(settings.index_dir)

    # Create pipeline
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_model=embedding_model
    )

    # Run benchmark
    benchmark = RAGBenchmark(pipeline)
    results = benchmark.run(
        include_generation=not args.retrieval_only,
        verbose=True
    )

    if args.output:
        benchmark.save_results(results, args.output)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Talent RAG - Retrieval-Augmented Talent Intelligence"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate data command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    gen_parser.add_argument("--candidates", type=int, default=100, help="Number of candidates")
    gen_parser.add_argument("--roles", type=int, default=20, help="Number of roles")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Build index command
    build_parser = subparsers.add_parser("build-index", help="Build vector index")
    build_parser.add_argument("--output", type=str, help="Output directory")
    build_parser.add_argument("--openai-embeddings", action="store_true", help="Use OpenAI embeddings")

    # Run API command
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Run UI command
    ui_parser = subparsers.add_parser("ui", help="Run Streamlit UI")
    ui_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port to bind")

    # Run benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run evaluation benchmark")
    bench_parser.add_argument("--output", type=str, help="Output file for results")
    bench_parser.add_argument("--retrieval-only", action="store_true", help="Only test retrieval")

    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.command == "generate":
        cmd_generate_data(args)
    elif args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "api":
        cmd_run_api(args)
    elif args.command == "ui":
        cmd_run_ui(args)
    elif args.command == "benchmark":
        cmd_run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
