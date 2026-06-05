"""
Tests for evaluation metrics module.
"""

import pytest
from talent_rag.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_ndcg,
    calculate_mrr,
    calculate_average_precision,
    calculate_hit_rate,
    MetricsCalculator
)


class TestRetrievalMetrics:
    """Tests for individual metric functions."""

    def test_recall_at_k_perfect(self):
        """Test recall with perfect retrieval."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c"}

        recall = calculate_recall_at_k(retrieved, relevant, k=5)
        assert recall == 1.0

    def test_recall_at_k_partial(self):
        """Test recall with partial retrieval."""
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = {"a", "b", "c", "d"}

        recall = calculate_recall_at_k(retrieved, relevant, k=5)
        assert recall == 0.5  # 2 out of 4 relevant found

    def test_recall_at_k_none(self):
        """Test recall with no relevant found."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}

        recall = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 0.0

    def test_recall_at_k_empty_relevant(self):
        """Test recall with empty relevant set."""
        retrieved = ["a", "b", "c"]
        relevant = set()

        recall = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 0.0

    def test_precision_at_k_perfect(self):
        """Test precision with perfect retrieval."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c", "d", "e"}

        precision = calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision == 1.0

    def test_precision_at_k_partial(self):
        """Test precision with mixed results."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}

        precision = calculate_precision_at_k(retrieved, relevant, k=5)
        assert precision == 0.6  # 3 out of 5

    def test_precision_at_k_none(self):
        """Test precision with no relevant found."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}

        precision = calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision == 0.0

    def test_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}

        mrr = calculate_mrr(retrieved, relevant)
        assert mrr == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        retrieved = ["x", "a", "b"]
        relevant = {"a"}

        mrr = calculate_mrr(retrieved, relevant)
        assert mrr == 0.5

    def test_mrr_not_found(self):
        """Test MRR when relevant doc not found."""
        retrieved = ["x", "y", "z"]
        relevant = {"a"}

        mrr = calculate_mrr(retrieved, relevant)
        assert mrr == 0.0

    def test_ndcg_perfect(self):
        """Test NDCG with perfect ranking."""
        retrieved = ["a", "b", "c"]
        relevance_scores = {"a": 3, "b": 2, "c": 1}

        ndcg = calculate_ndcg(retrieved, relevance_scores, k=3)
        assert ndcg == 1.0  # Perfect ranking

    def test_ndcg_reversed(self):
        """Test NDCG with reversed ranking."""
        retrieved = ["c", "b", "a"]
        relevance_scores = {"a": 3, "b": 2, "c": 1}

        ndcg = calculate_ndcg(retrieved, relevance_scores, k=3)
        assert ndcg < 1.0  # Not perfect

    def test_ndcg_empty(self):
        """Test NDCG with empty inputs."""
        assert calculate_ndcg([], {}, k=3) == 0.0
        assert calculate_ndcg(["a"], {}, k=3) == 0.0

    def test_average_precision(self):
        """Test average precision calculation."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}

        ap = calculate_average_precision(retrieved, relevant)

        # Manual calculation:
        # P@1 = 1/1 = 1.0 (a is relevant)
        # P@3 = 2/3 (b is relevant)
        # P@5 = 3/5 (c is relevant)
        # AP = (1.0 + 2/3 + 3/5) / 3
        expected = (1.0 + 2/3 + 3/5) / 3
        assert abs(ap - expected) < 0.001

    def test_hit_rate_hit(self):
        """Test hit rate when there's a hit."""
        retrieved = ["x", "a", "y"]
        relevant = {"a", "b"}

        hit = calculate_hit_rate(retrieved, relevant, k=3)
        assert hit == 1.0

    def test_hit_rate_miss(self):
        """Test hit rate when there's no hit."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}

        hit = calculate_hit_rate(retrieved, relevant, k=3)
        assert hit == 0.0


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_calculate_query_metrics(self):
        """Test calculating metrics for a single query."""
        calc = MetricsCalculator(k_values=[1, 3, 5])

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}

        metrics = calc.calculate_query_metrics(
            query="test query",
            retrieved_ids=retrieved,
            relevant_ids=relevant
        )

        assert metrics.query == "test query"
        assert 1 in metrics.recall_at_k
        assert 3 in metrics.recall_at_k
        assert 5 in metrics.recall_at_k
        assert metrics.mrr > 0
        assert metrics.hit_rate == 1.0

    def test_aggregate_metrics(self):
        """Test aggregating metrics across queries."""
        calc = MetricsCalculator(k_values=[1, 3, 5])

        # Add some query results
        calc.calculate_query_metrics(
            "query1",
            ["a", "b", "c"],
            {"a"}
        )
        calc.calculate_query_metrics(
            "query2",
            ["x", "a", "b"],
            {"a"}
        )

        aggregated = calc.aggregate_metrics()

        assert aggregated["num_queries"] == 2
        assert "mean_mrr" in aggregated
        assert "mean_recall@5" in aggregated
        assert "mean_ndcg@5" in aggregated

    def test_reset(self):
        """Test resetting the calculator."""
        calc = MetricsCalculator()

        calc.calculate_query_metrics("q1", ["a"], {"a"})
        assert len(calc.results) == 1

        calc.reset()
        assert len(calc.results) == 0

    def test_to_dataframe(self):
        """Test converting results to DataFrame."""
        calc = MetricsCalculator(k_values=[1, 3])

        calc.calculate_query_metrics("q1", ["a", "b"], {"a"})
        calc.calculate_query_metrics("q2", ["x", "a"], {"a"})

        df = calc.to_dataframe()

        assert len(df) == 2
        assert "query" in df.columns
        assert "mrr" in df.columns
        assert "recall@1" in df.columns
