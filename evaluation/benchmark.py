"""
RAG Benchmark Module.

This module provides benchmarking utilities for evaluating
the complete RAG pipeline performance.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from .metrics import MetricsCalculator


@dataclass
class BenchmarkQuery:
    """
    A single benchmark query with ground truth.

    Attributes:
        query: The query string
        relevant_candidate_ids: List of relevant candidate IDs
        relevant_role_ids: List of relevant role IDs
        expected_skills: Expected skills in response
        query_type: Type of query
        difficulty: Difficulty level
    """
    query: str
    relevant_candidate_ids: list[str] = field(default_factory=list)
    relevant_role_ids: list[str] = field(default_factory=list)
    expected_skills: list[str] = field(default_factory=list)
    query_type: str = "general"
    difficulty: str = "medium"


@dataclass
class BenchmarkResult:
    """
    Result from running a benchmark.

    Attributes:
        timestamp: When the benchmark was run
        total_queries: Number of queries evaluated
        metrics: Aggregated metrics
        per_query_results: Results for each query
        latency_stats: Latency statistics
        config: Configuration used
    """
    timestamp: str
    total_queries: int
    metrics: dict[str, float]
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    latency_stats: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


# Sample benchmark queries for talent search
SAMPLE_BENCHMARK_QUERIES = [
    BenchmarkQuery(
        query="Find senior machine learning engineers with NLP experience",
        expected_skills=["Python", "NLP", "Machine Learning", "Deep Learning"],
        query_type="candidate_search",
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Who are the best candidates for a Staff ML Engineer role requiring PyTorch and distributed systems experience?",
        expected_skills=["PyTorch", "Distributed Systems", "Python"],
        query_type="candidate_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Find candidates with Kubernetes and cloud infrastructure experience for a platform engineering role",
        expected_skills=["Kubernetes", "AWS", "Docker", "Terraform"],
        query_type="candidate_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Which candidates have demonstrated leadership experience in their interviews?",
        query_type="behavioral_analysis",
        difficulty="hard"
    ),
    BenchmarkQuery(
        query="Find full-stack engineers with React and Python experience located in San Francisco",
        expected_skills=["React", "Python", "JavaScript"],
        query_type="candidate_search",
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Who has experience building recommendation systems at scale?",
        expected_skills=["Machine Learning", "Recommendation Systems"],
        query_type="candidate_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Find candidates who have worked on real-time data processing systems",
        expected_skills=["Kafka", "Spark", "Real-time"],
        query_type="candidate_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Which candidates have PhD degrees in machine learning or related fields?",
        query_type="education_filter",
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Find candidates with startup experience who can work in fast-paced environments",
        query_type="culture_fit",
        difficulty="hard"
    ),
    BenchmarkQuery(
        query="Who has experience with LLMs and RAG systems?",
        expected_skills=["LLMs", "RAG", "NLP"],
        query_type="candidate_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Find senior engineers with experience mentoring junior developers",
        query_type="behavioral_analysis",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Which candidates have the strongest system design skills based on their interviews?",
        query_type="technical_analysis",
        difficulty="hard"
    ),
    BenchmarkQuery(
        query="Find remote-friendly candidates with 5+ years of experience",
        query_type="filter_search",
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Who has experience with A/B testing and experimentation platforms?",
        expected_skills=["A/B Testing", "Statistics", "Experimentation"],
        query_type="candidate_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Find candidates who have transitioned from individual contributor to leadership roles",
        query_type="career_analysis",
        difficulty="hard"
    ),
    BenchmarkQuery(
        query="Which candidates have published papers or given conference talks?",
        query_type="achievement_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Find backend engineers with experience in microservices architecture",
        expected_skills=["Microservices", "Docker", "Kubernetes", "API Design"],
        query_type="candidate_search",
        difficulty="easy"
    ),
    BenchmarkQuery(
        query="Who would be the best fit for a principal engineer role requiring cross-team influence?",
        query_type="role_matching",
        difficulty="hard"
    ),
    BenchmarkQuery(
        query="Find candidates with experience in both frontend and ML/data work",
        expected_skills=["React", "Python", "Machine Learning"],
        query_type="hybrid_search",
        difficulty="medium"
    ),
    BenchmarkQuery(
        query="Which candidates have demonstrated ability to handle technical conflicts constructively?",
        query_type="behavioral_analysis",
        difficulty="hard"
    )
]


class RAGBenchmark:
    """
    Benchmark suite for evaluating RAG pipeline performance.

    This class runs a set of queries and measures retrieval
    quality, generation quality, and latency.

    Attributes:
        pipeline: RAG pipeline to evaluate
        metrics_calculator: MetricsCalculator instance
        queries: List of benchmark queries
    """

    def __init__(
        self,
        pipeline,
        queries: Optional[list[BenchmarkQuery]] = None,
        k_values: list[int] = None
    ):
        """
        Initialize the benchmark.

        Args:
            pipeline: RAG pipeline to evaluate
            queries: Custom benchmark queries (uses samples if None)
            k_values: K values for metrics calculation
        """
        self.pipeline = pipeline
        self.queries = queries or SAMPLE_BENCHMARK_QUERIES
        self.metrics_calculator = MetricsCalculator(k_values or [1, 3, 5, 10])

        logger.info(f"RAGBenchmark initialized with {len(self.queries)} queries")

    def run(
        self,
        include_generation: bool = True,
        verbose: bool = True
    ) -> BenchmarkResult:
        """
        Run the complete benchmark.

        Args:
            include_generation: Whether to include LLM generation
            verbose: Whether to log progress

        Returns:
            BenchmarkResult with all metrics
        """
        start_time = time.time()
        latencies = []
        per_query_results = []

        for i, query in enumerate(self.queries):
            if verbose:
                logger.info(f"Running query {i+1}/{len(self.queries)}: {query.query[:50]}...")

            query_start = time.time()

            try:
                # Run the query
                if include_generation:
                    response = self.pipeline.query(query.query)
                else:
                    # Just retrieval
                    retrieval_result = self.pipeline.retriever.retrieve(query.query)
                    response = type('obj', (object,), {
                        'retrieval_result': retrieval_result,
                        'evidence_used': [
                            {'id': r.id, 'metadata': r.metadata}
                            for r in retrieval_result.results
                        ]
                    })()

                query_latency = (time.time() - query_start) * 1000
                latencies.append(query_latency)

                # Extract retrieved IDs
                retrieved_ids = [
                    e.get("metadata", {}).get("candidate_id") or e.get("id")
                    for e in response.evidence_used
                ]
                retrieved_ids = [rid for rid in retrieved_ids if rid]

                # Calculate metrics if we have ground truth
                relevant_ids = set(query.relevant_candidate_ids)

                if relevant_ids:
                    metrics = self.metrics_calculator.calculate_query_metrics(
                        query=query.query,
                        retrieved_ids=retrieved_ids,
                        relevant_ids=relevant_ids
                    )
                else:
                    metrics = None

                # Check skill coverage
                skill_coverage = 0.0
                if query.expected_skills and hasattr(response, 'answer'):
                    found_skills = sum(
                        1 for skill in query.expected_skills
                        if skill.lower() in response.answer.lower()
                    )
                    skill_coverage = found_skills / len(query.expected_skills)

                per_query_results.append({
                    "query": query.query,
                    "query_type": query.query_type,
                    "difficulty": query.difficulty,
                    "latency_ms": query_latency,
                    "num_retrieved": len(retrieved_ids),
                    "skill_coverage": skill_coverage,
                    "metrics": {
                        "mrr": metrics.mrr if metrics else None,
                        "recall@5": metrics.recall_at_k.get(5) if metrics else None,
                        "ndcg@5": metrics.ndcg_at_k.get(5) if metrics else None
                    } if metrics else {}
                })

            except Exception as e:
                logger.error(f"Query failed: {e}")
                per_query_results.append({
                    "query": query.query,
                    "error": str(e)
                })

        # Aggregate metrics
        aggregated_metrics = self.metrics_calculator.aggregate_metrics()

        # Calculate latency stats
        latency_stats = {}
        if latencies:
            import numpy as np
            latency_stats = {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies))
            }

        total_time = time.time() - start_time

        result = BenchmarkResult(
            timestamp=datetime.utcnow().isoformat(),
            total_queries=len(self.queries),
            metrics=aggregated_metrics,
            per_query_results=per_query_results,
            latency_stats=latency_stats,
            config={
                "include_generation": include_generation,
                "total_time_seconds": total_time
            }
        )

        if verbose:
            self._print_summary(result)

        return result

    def _print_summary(self, result: BenchmarkResult):
        """Print a summary of benchmark results."""
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total queries: {result.total_queries}")
        logger.info(f"Total time: {result.config.get('total_time_seconds', 0):.2f}s")
        logger.info("")
        logger.info("RETRIEVAL METRICS:")
        for key, value in result.metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
        logger.info("")
        logger.info("LATENCY STATS:")
        for key, value in result.latency_stats.items():
            logger.info(f"  {key}: {value:.2f}")
        logger.info("=" * 60)

    def save_results(
        self,
        result: BenchmarkResult,
        path: Path | str
    ):
        """
        Save benchmark results to a JSON file.

        Args:
            result: BenchmarkResult to save
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": result.timestamp,
            "total_queries": result.total_queries,
            "metrics": result.metrics,
            "latency_stats": result.latency_stats,
            "config": result.config,
            "per_query_results": result.per_query_results
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {path}")

    @classmethod
    def load_results(cls, path: Path | str) -> BenchmarkResult:
        """
        Load benchmark results from a JSON file.

        Args:
            path: Path to results file

        Returns:
            BenchmarkResult instance
        """
        with open(path) as f:
            data = json.load(f)

        return BenchmarkResult(
            timestamp=data["timestamp"],
            total_queries=data["total_queries"],
            metrics=data["metrics"],
            per_query_results=data.get("per_query_results", []),
            latency_stats=data.get("latency_stats", {}),
            config=data.get("config", {})
        )
