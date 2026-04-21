"""Evaluation metrics for ranking quality assessment."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metric results."""
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    average_precision: float = 0.0
    f1_scores: Dict[int, float] = field(default_factory=dict)
    total_candidates: int = 0
    total_relevant: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "precision": self.precision_at_k,
            "recall": self.recall_at_k,
            "ndcg": self.ndcg_at_k,
            "average_precision": round(self.average_precision, 4),
            "f1": self.f1_scores,
            "total_candidates": self.total_candidates,
            "total_relevant": self.total_relevant,
        }


class RankingEvaluator:
    """
    Evaluates ranking quality using standard IR metrics.

    Metrics implemented:
    - Precision@K: fraction of top-K that are relevant
    - Recall@K: fraction of all relevant items in top-K
    - NDCG@K: Normalised Discounted Cumulative Gain
    - Mean Average Precision (MAP)
    - F1@K: harmonic mean of Precision@K and Recall@K

    Usage::

        evaluator = RankingEvaluator()
        # ground_truth: list of 1/0 relevance labels in rank order
        # scores: list of predicted scores for each candidate
        result = evaluator.evaluate(ground_truth=[1,1,0,1,0,0], scores=[0.9,0.85,0.4,0.7,0.3,0.1])
        print(result.summary())
    """

    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Args:
            k_values: List of K values to compute metrics at.
                      Defaults to [1, 3, 5, 10].
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        ground_truth: List[int],
        scores: Optional[List[float]] = None,
        ranked_indices: Optional[List[int]] = None,
    ) -> EvaluationResult:
        """
        Compute full evaluation metrics.

        Args:
            ground_truth: Relevance labels in their *original* order (1=relevant, 0=not).
            scores: Predicted scores in the same order as ground_truth.
                    If provided, will be used to sort candidates.
            ranked_indices: Pre-sorted indices (highest to lowest score).
                            If provided, `scores` is ignored for ranking.

        Returns:
            EvaluationResult with all computed metrics.
        """
        if not ground_truth:
            return EvaluationResult()

        n = len(ground_truth)

        # Determine the ranked order
        if ranked_indices is not None:
            ranked_labels = [ground_truth[i] for i in ranked_indices]
        elif scores is not None:
            order = sorted(range(n), key=lambda i: scores[i], reverse=True)
            ranked_labels = [ground_truth[i] for i in order]
        else:
            # Assume ground_truth is already in ranked order
            ranked_labels = list(ground_truth)

        total_relevant = sum(ground_truth)
        result = EvaluationResult(
            total_candidates=n,
            total_relevant=total_relevant,
        )

        for k in self.k_values:
            if k > n:
                continue
            top_k = ranked_labels[:k]
            tp = sum(top_k)

            precision = tp / k
            recall = tp / max(total_relevant, 1)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            result.precision_at_k[k] = round(precision, 4)
            result.recall_at_k[k] = round(recall, 4)
            result.f1_scores[k] = round(f1, 4)
            result.ndcg_at_k[k] = round(self._ndcg(ranked_labels, k), 4)

        result.average_precision = round(self._average_precision(ranked_labels, total_relevant), 4)
        return result

    def evaluate_multiple_queries(
        self,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute Mean Average Precision and mean metrics over multiple queries.

        Args:
            queries: List of dicts, each with keys:
                     - "ground_truth": List[int]
                     - "scores": List[float] (optional)
                     - "ranked_indices": List[int] (optional)

        Returns:
            Dict with MAP and mean metric values.
        """
        all_results = [self.evaluate(**q) for q in queries]

        ap_values = [r.average_precision for r in all_results]
        map_score = float(np.mean(ap_values)) if ap_values else 0.0

        mean_ndcg: Dict[int, float] = {}
        mean_precision: Dict[int, float] = {}
        for k in self.k_values:
            ndcg_vals = [r.ndcg_at_k.get(k, 0.0) for r in all_results if k in r.ndcg_at_k]
            p_vals = [r.precision_at_k.get(k, 0.0) for r in all_results if k in r.precision_at_k]
            if ndcg_vals:
                mean_ndcg[k] = round(float(np.mean(ndcg_vals)), 4)
            if p_vals:
                mean_precision[k] = round(float(np.mean(p_vals)), 4)

        return {
            "map": round(map_score, 4),
            "mean_ndcg": mean_ndcg,
            "mean_precision": mean_precision,
            "num_queries": len(queries),
        }

    # ─── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _dcg(labels: List[int], k: int) -> float:
        """Compute Discounted Cumulative Gain at k."""
        dcg = 0.0
        for i, rel in enumerate(labels[:k], start=1):
            dcg += rel / math.log2(i + 1)
        return dcg

    def _ndcg(self, labels: List[int], k: int) -> float:
        """Compute Normalised Discounted Cumulative Gain at k."""
        ideal = sorted(labels, reverse=True)
        idcg = self._dcg(ideal, k)
        if idcg == 0:
            return 0.0
        return self._dcg(labels, k) / idcg

    @staticmethod
    def _average_precision(ranked_labels: List[int], total_relevant: int) -> float:
        """Compute Average Precision for a single query."""
        if total_relevant == 0:
            return 0.0
        ap = 0.0
        hits = 0
        for i, rel in enumerate(ranked_labels, start=1):
            if rel == 1:
                hits += 1
                ap += hits / i
        return ap / total_relevant


class HiddenGemDetector:
    """
    Detects 'hidden gem' candidates — those who score high semantically
    but would be ranked low by naive keyword matching.

    The core value-add of semantic embeddings: finding candidates
    whose vocabulary differs from the JD but whose experience matches.
    """

    def __init__(self, keyword_threshold: float = 0.3, semantic_threshold: float = 0.6):
        """
        Args:
            keyword_threshold: Skill-match score below which a candidate is 'keyword-poor'.
            semantic_threshold: Semantic score above which a candidate is a potential gem.
        """
        self.keyword_threshold = keyword_threshold
        self.semantic_threshold = semantic_threshold

    def detect(self, candidates: List[Any]) -> List[Dict[str, Any]]:
        """
        Find candidates with high semantic but low skill-match scores.

        Args:
            candidates: List of RankedCandidate objects.

        Returns:
            List of dicts describing hidden gems with their score gap.
        """
        gems = []
        for candidate in candidates:
            if (candidate.semantic_score >= self.semantic_threshold
                    and candidate.skill_score < self.keyword_threshold):
                gems.append({
                    "name": (candidate.resume.contact_info.name
                             if candidate.resume.contact_info else "Unknown"),
                    "rank": candidate.rank,
                    "semantic_score": round(candidate.semantic_score, 3),
                    "skill_score": round(candidate.skill_score, 3),
                    "hybrid_score": round(candidate.hybrid_score, 3),
                    "score_gap": round(candidate.semantic_score - candidate.skill_score, 3),
                    "insight": (
                        f"Semantic score ({candidate.semantic_score:.1%}) is "
                        f"{candidate.semantic_score - candidate.skill_score:.1%} higher than "
                        f"skill-match ({candidate.skill_score:.1%}). May be a hidden gem — "
                        "review resume manually."
                    ),
                })
        return sorted(gems, key=lambda g: g["score_gap"], reverse=True)
