"""Tests for evaluation metrics module."""

import pytest
import math
from src.utils.evaluation import RankingEvaluator, EvaluationResult, HiddenGemDetector
from unittest.mock import MagicMock


class TestRankingEvaluator:

    def setup_method(self):
        self.evaluator = RankingEvaluator(k_values=[1, 3, 5, 10])

    # ─── Precision@K ───────────────────────────────────────────────────────

    def test_precision_at_k_perfect_ranking(self):
        """Perfect ranking — all top-k are relevant."""
        labels = [1, 1, 1, 0, 0, 0]
        result = self.evaluator.evaluate(labels)
        assert result.precision_at_k[1] == pytest.approx(1.0)
        assert result.precision_at_k[3] == pytest.approx(1.0)

    def test_precision_at_k_worst_ranking(self):
        """Worst ranking — all relevant items at the bottom."""
        labels = [0, 0, 0, 1, 1, 1]
        result = self.evaluator.evaluate(labels)
        assert result.precision_at_k[3] == pytest.approx(0.0)

    def test_precision_at_k_mixed(self):
        """Mixed ranking — 1 relevant in top 3."""
        labels = [1, 0, 0, 1, 0, 1]
        result = self.evaluator.evaluate(labels)
        assert result.precision_at_k[3] == pytest.approx(1 / 3, abs=0.001)

    def test_precision_at_k_greater_than_n_skipped(self):
        """K > n should be skipped, not cause errors."""
        labels = [1, 0, 1]
        result = self.evaluator.evaluate(labels)
        assert 10 not in result.precision_at_k  # k=10 > n=3

    # ─── Recall@K ──────────────────────────────────────────────────────────

    def test_recall_at_k_all_relevant_in_top(self):
        labels = [1, 1, 1, 0, 0, 0]  # all 3 relevant in top 3
        result = self.evaluator.evaluate(labels)
        assert result.recall_at_k[3] == pytest.approx(1.0)

    def test_recall_at_k_none_in_top(self):
        labels = [0, 0, 0, 1, 1]
        result = self.evaluator.evaluate(labels)
        assert result.recall_at_k[3] == pytest.approx(0.0)

    def test_recall_at_k_partial(self):
        labels = [1, 0, 1, 0, 1]  # 2 of 3 relevant in top 5
        result = self.evaluator.evaluate(labels)
        assert result.recall_at_k[5] == pytest.approx(1.0)  # all 3 found in top 5

    # ─── NDCG@K ────────────────────────────────────────────────────────────

    def test_ndcg_perfect_ranking_is_one(self):
        labels = [1, 1, 1, 0, 0]
        result = self.evaluator.evaluate(labels)
        assert result.ndcg_at_k[3] == pytest.approx(1.0)

    def test_ndcg_worst_ranking_is_zero(self):
        labels = [0, 0, 0, 1, 1]
        result = self.evaluator.evaluate(labels)
        assert result.ndcg_at_k[3] == pytest.approx(0.0)

    def test_ndcg_values_in_zero_one(self):
        labels = [1, 0, 1, 0, 1]
        result = self.evaluator.evaluate(labels)
        for k, val in result.ndcg_at_k.items():
            assert 0.0 <= val <= 1.0, f"NDCG@{k} = {val} out of [0, 1]"

    def test_ndcg_decreasing_with_displaced_relevant(self):
        perfect = [1, 1, 0, 0, 0]
        imperfect = [1, 0, 1, 0, 0]  # relevant item moved from rank 2 to rank 3
        r_perfect = self.evaluator.evaluate(perfect)
        r_imperfect = self.evaluator.evaluate(imperfect)
        assert r_perfect.ndcg_at_k[3] >= r_imperfect.ndcg_at_k[3]

    # ─── Average Precision ─────────────────────────────────────────────────

    def test_average_precision_perfect(self):
        labels = [1, 1, 1, 0, 0]
        result = self.evaluator.evaluate(labels)
        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert result.average_precision == pytest.approx(1.0)

    def test_average_precision_no_relevant(self):
        labels = [0, 0, 0, 0]
        result = self.evaluator.evaluate(labels)
        assert result.average_precision == pytest.approx(0.0)

    def test_average_precision_single_relevant_at_top(self):
        labels = [1, 0, 0, 0]
        result = self.evaluator.evaluate(labels)
        assert result.average_precision == pytest.approx(1.0)

    def test_average_precision_single_relevant_at_bottom(self):
        labels = [0, 0, 0, 1]
        result = self.evaluator.evaluate(labels)
        assert result.average_precision == pytest.approx(1 / 4, abs=0.001)

    # ─── Scores-based sorting ──────────────────────────────────────────────

    def test_evaluate_with_scores_sorts_correctly(self):
        """When scores provided, should rank by score descending."""
        ground_truth = [0, 1, 0, 1, 0]  # items 1 and 3 are relevant
        scores = [0.2, 0.9, 0.1, 0.8, 0.3]  # item 1 → 0.9 (rank 1), item 3 → 0.8 (rank 2)

        result = self.evaluator.evaluate(ground_truth, scores=scores)
        # Both relevant items should be in top 2
        assert result.precision_at_k[3] >= 2 / 3
        assert result.ndcg_at_k[3] == pytest.approx(1.0)

    def test_evaluate_empty_ground_truth(self):
        result = self.evaluator.evaluate([])
        assert result.total_candidates == 0

    # ─── Multi-query ───────────────────────────────────────────────────────

    def test_map_perfect_queries(self):
        queries = [
            {"ground_truth": [1, 1, 0, 0]},
            {"ground_truth": [1, 0, 1, 0]},
        ]
        multi = self.evaluator.evaluate_multiple_queries(queries)
        assert multi["map"] > 0.0
        assert multi["num_queries"] == 2

    def test_map_empty_queries(self):
        multi = self.evaluator.evaluate_multiple_queries([])
        assert multi["map"] == 0.0


class TestHiddenGemDetector:

    def _make_candidate(self, name: str, semantic: float, skill: float, hybrid: float, rank: int):
        c = MagicMock()
        c.semantic_score = semantic
        c.skill_score = skill
        c.hybrid_score = hybrid
        c.rank = rank
        c.resume.contact_info.name = name
        return c

    def test_detects_hidden_gem(self):
        detector = HiddenGemDetector(keyword_threshold=0.3, semantic_threshold=0.6)
        candidates = [
            self._make_candidate("Alice", semantic=0.82, skill=0.15, hybrid=0.62, rank=3),
            self._make_candidate("Bob", semantic=0.90, skill=0.88, hybrid=0.89, rank=1),
        ]
        gems = detector.detect(candidates)
        assert len(gems) == 1
        assert gems[0]["name"] == "Alice"

    def test_no_hidden_gems_when_all_have_high_skill(self):
        detector = HiddenGemDetector()
        candidates = [
            self._make_candidate("Alice", semantic=0.9, skill=0.8, hybrid=0.87, rank=1),
            self._make_candidate("Bob", semantic=0.7, skill=0.6, hybrid=0.67, rank=2),
        ]
        gems = detector.detect(candidates)
        assert gems == []

    def test_sorted_by_score_gap_descending(self):
        detector = HiddenGemDetector(keyword_threshold=0.4, semantic_threshold=0.5)
        candidates = [
            self._make_candidate("Gem A", semantic=0.9, skill=0.1, hybrid=0.66, rank=2),  # gap 0.8
            self._make_candidate("Gem B", semantic=0.7, skill=0.2, hybrid=0.55, rank=4),  # gap 0.5
        ]
        gems = detector.detect(candidates)
        assert len(gems) == 2
        assert gems[0]["name"] == "Gem A"  # larger gap first
        assert gems[1]["name"] == "Gem B"

    def test_insight_string_present(self):
        detector = HiddenGemDetector()
        candidates = [
            self._make_candidate("Dr. Sarah", semantic=0.85, skill=0.10, hybrid=0.63, rank=5),
        ]
        gems = detector.detect(candidates)
        assert len(gems) == 1
        assert "hidden gem" in gems[0]["insight"].lower()

    def test_empty_candidates(self):
        detector = HiddenGemDetector()
        assert detector.detect([]) == []
