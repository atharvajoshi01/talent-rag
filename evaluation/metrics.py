"""
Retrieval Metrics Module.

This module provides metrics calculation for evaluating
retrieval quality in the RAG pipeline.
"""

import math
from typing import Any, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class RetrievalMetrics:
    """
    Collection of retrieval metrics for a single query.

    Attributes:
        query: The query string
        recall_at_k: Recall@k scores for different k values
        precision_at_k: Precision@k scores for different k values
        ndcg_at_k: NDCG@k scores for different k values
        mrr: Mean Reciprocal Rank
        hit_rate: Whether any relevant document was retrieved
        avg_precision: Average precision across all k
    """
    query: str
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0
    avg_precision: float = 0.0


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """
    Calculate Recall@k.

    Recall@k measures the fraction of relevant documents
    that were retrieved in the top k results.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall@k score between 0 and 1
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids

    return len(relevant_retrieved) / len(relevant_ids)


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """
    Calculate Precision@k.

    Precision@k measures the fraction of retrieved documents
    in the top k that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@k score between 0 and 1
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_retrieved = retrieved_at_k & relevant_ids

    return len(relevant_retrieved) / k


def calculate_ndcg(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    NDCG measures ranking quality by considering the position
    of relevant documents and their graded relevance.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevance_scores: Dict mapping doc IDs to relevance scores
        k: Number of top results to consider

    Returns:
        NDCG@k score between 0 and 1
    """
    if not retrieved_ids or not relevance_scores:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = relevance_scores.get(doc_id, 0.0)
        # Using log2(i+2) because positions are 0-indexed
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # Calculate IDCG (ideal DCG)
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += (2 ** rel - 1) / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR measures the average of reciprocal ranks of the first
    relevant document across queries.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank)
        relevant_ids: Set of relevant document IDs

    Returns:
        MRR score between 0 and 1
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)

    return 0.0


def calculate_average_precision(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    """
    Calculate Average Precision (AP).

    AP is the average of precision values at each position
    where a relevant document is found.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs

    Returns:
        Average Precision score between 0 and 1
    """
    if not relevant_ids:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant_ids)


def calculate_hit_rate(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    """
    Calculate Hit Rate@k.

    Hit Rate measures whether at least one relevant document
    was retrieved in the top k.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    retrieved_at_k = set(retrieved_ids[:k])
    return 1.0 if retrieved_at_k & relevant_ids else 0.0


class MetricsCalculator:
    """
    Calculator for comprehensive retrieval metrics.

    This class calculates and aggregates metrics across
    multiple queries for evaluation.

    Attributes:
        k_values: List of k values to calculate metrics for
        results: List of per-query metrics
    """

    def __init__(self, k_values: list[int] = None):
        """
        Initialize the metrics calculator.

        Args:
            k_values: List of k values for @k metrics
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.results: list[RetrievalMetrics] = []

        logger.info(f"MetricsCalculator initialized with k_values={self.k_values}")

    def calculate_query_metrics(
        self,
        query: str,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        relevance_scores: Optional[dict[str, float]] = None
    ) -> RetrievalMetrics:
        """
        Calculate all metrics for a single query.

        Args:
            query: Query string
            retrieved_ids: List of retrieved document IDs
            relevant_ids: Set of relevant document IDs
            relevance_scores: Optional graded relevance scores

        Returns:
            RetrievalMetrics for the query
        """
        # Default relevance scores (binary) if not provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}

        metrics = RetrievalMetrics(query=query)

        # Calculate metrics for each k
        for k in self.k_values:
            metrics.recall_at_k[k] = calculate_recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            metrics.precision_at_k[k] = calculate_precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            metrics.ndcg_at_k[k] = calculate_ndcg(
                retrieved_ids, relevance_scores, k
            )

        # Calculate other metrics
        metrics.mrr = calculate_mrr(retrieved_ids, relevant_ids)
        metrics.hit_rate = calculate_hit_rate(
            retrieved_ids, relevant_ids, max(self.k_values)
        )
        metrics.avg_precision = calculate_average_precision(
            retrieved_ids, relevant_ids
        )

        self.results.append(metrics)
        return metrics

    def aggregate_metrics(self) -> dict[str, Any]:
        """
        Aggregate metrics across all queries.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results:
            return {}

        n = len(self.results)

        aggregated = {
            "num_queries": n,
            "mean_mrr": sum(r.mrr for r in self.results) / n,
            "mean_hit_rate": sum(r.hit_rate for r in self.results) / n,
            "mean_avg_precision": sum(r.avg_precision for r in self.results) / n
        }

        # Aggregate @k metrics
        for k in self.k_values:
            aggregated[f"mean_recall@{k}"] = sum(
                r.recall_at_k.get(k, 0) for r in self.results
            ) / n
            aggregated[f"mean_precision@{k}"] = sum(
                r.precision_at_k.get(k, 0) for r in self.results
            ) / n
            aggregated[f"mean_ndcg@{k}"] = sum(
                r.ndcg_at_k.get(k, 0) for r in self.results
            ) / n

        return aggregated

    def reset(self):
        """Reset collected results."""
        self.results = []

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.

        Returns:
            DataFrame with per-query metrics
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        rows = []
        for r in self.results:
            row = {
                "query": r.query,
                "mrr": r.mrr,
                "hit_rate": r.hit_rate,
                "avg_precision": r.avg_precision
            }
            for k in self.k_values:
                row[f"recall@{k}"] = r.recall_at_k.get(k, 0)
                row[f"precision@{k}"] = r.precision_at_k.get(k, 0)
                row[f"ndcg@{k}"] = r.ndcg_at_k.get(k, 0)
            rows.append(row)

        return pd.DataFrame(rows)
