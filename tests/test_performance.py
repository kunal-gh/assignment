"""Performance tests for critical path — ranking speed, skill matching throughput."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models.ranking import RankedCandidate
from src.models.resume import ContactInfo, ResumeData
from src.ranking.fairness_checker import FairnessChecker
from src.ranking.skill_matcher import SkillMatcher


def _make_resume(name: str, skills: list, score: float) -> RankedCandidate:
    contact = ContactInfo(name=name, email=f"{name}@t.com")
    resume = ResumeData(
        contact_info=contact,
        skills=skills,
        raw_text=f"Resume of {name}",
        embedding=np.random.rand(384).astype(np.float32),
    )
    return RankedCandidate(
        resume=resume,
        semantic_score=score,
        skill_score=score * 0.9,
        hybrid_score=score,
        rank=1,
    )


class TestSkillMatcherPerformance:
    """Throughput tests for skill matching — must be fast enough for real-time use."""

    def setup_method(self):
        self.matcher = SkillMatcher()
        # Warm up JIT / imports
        self.matcher.calculate_skill_match(["python"], ["python"])

    def test_1000_skill_matches_under_5s(self):
        """1000 skill match computations must complete in under 5 seconds."""
        required = ["python", "pytorch", "fastapi", "docker", "kubernetes", "aws"]
        preferred = ["spacy", "faiss", "streamlit", "redis"]
        resume_skills = ["python", "docker", "fastapi", "spacy", "postgresql", "git"]

        start = time.perf_counter()
        for _ in range(1000):
            self.matcher.calculate_skill_match(resume_skills, required, preferred)
        elapsed = time.perf_counter() - start

        print(f"\n1000 skill matches: {elapsed:.3f}s ({elapsed * 1000:.1f}ms avg/match)")
        assert elapsed < 5.0, f"Skill matching too slow: {elapsed:.3f}s"

    def test_100_analyze_skill_match_under_3s(self):
        """100 full skill analysis computations (with categorisation) under 3s."""
        required = ["python", "pytorch", "spacy", "faiss", "fastapi"]
        preferred = ["kubernetes", "aws", "streamlit"]
        resume_skills = ["python", "docker", "spacy", "postgresql", "git", "react"]

        start = time.perf_counter()
        for _ in range(100):
            self.matcher.analyze_skill_match(resume_skills, required, preferred)
        elapsed = time.perf_counter() - start

        print(f"\n100 analyze_skill_match calls: {elapsed:.3f}s ({elapsed * 10:.1f}ms avg)")
        assert elapsed < 3.0, f"Skill analysis too slow: {elapsed:.3f}s"

    def test_batch_50_resumes_skill_matching_under_2s(self):
        """Skill matching for 50 resumes must complete under 2 seconds."""
        required = ["python", "pytorch", "fastapi", "docker"]

        resumes = [
            [f"python", f"skill_{i}", f"tool_{i}", "git"]
            for i in range(50)
        ]

        start = time.perf_counter()
        scores = [self.matcher.calculate_skill_match(r, required) for r in resumes]
        elapsed = time.perf_counter() - start

        print(f"\n50 resume skill matches: {elapsed:.3f}s")
        assert elapsed < 2.0
        assert len(scores) == 50


class TestNormalisationPerformance:

    def setup_method(self):
        self.matcher = SkillMatcher()

    def test_normalise_large_skill_list_under_100ms(self):
        """Normalising 100 skills must be near-instant."""
        skills = [f"skill_{i}" for i in range(80)] + [
            "js", "nodejs", "postgres", "k8s", "tensorflow", "pytorch"
        ]

        start = time.perf_counter()
        for _ in range(200):
            self.matcher._normalize_skills(skills)
        elapsed = time.perf_counter() - start

        print(f"\n200 × normalise(106 skills): {elapsed:.3f}s")
        assert elapsed < 2.0


class TestFairnessCheckerPerformance:

    def setup_method(self):
        self.checker = FairnessChecker()
        np.random.seed(42)

    def test_fairness_report_20_candidates_under_1s(self):
        """Fairness report for 20 candidates under 1 second."""
        candidates = [
            _make_resume(f"Candidate {i}", ["python", "docker"], float(i) / 20)
            for i in range(20)
        ]

        start = time.perf_counter()
        for _ in range(10):
            self.checker.generate_fairness_report(candidates, top_k=10)
        elapsed = time.perf_counter() - start

        print(f"\n10 × fairness_report(20 candidates): {elapsed:.3f}s")
        assert elapsed < 5.0

    def test_fairness_metrics_50_candidates_under_2s(self):
        """Fairness metrics for 50 candidates under 2 seconds."""
        candidates = [
            _make_resume(f"Person {i}", ["python"], float(i) / 50)
            for i in range(50)
        ]

        start = time.perf_counter()
        self.checker.calculate_fairness_metrics(candidates)
        elapsed = time.perf_counter() - start

        print(f"\nFairness metrics (50 candidates): {elapsed:.3f}s")
        assert elapsed < 2.0


class TestEvaluationMetricsPerformance:

    def test_10000_ndcg_computations_under_3s(self):
        """10,000 NDCG computations must complete in under 3 seconds."""
        from src.utils.evaluation import RankingEvaluator
        evaluator = RankingEvaluator()
        labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

        start = time.perf_counter()
        for _ in range(10_000):
            evaluator.evaluate(labels)
        elapsed = time.perf_counter() - start

        print(f"\n10,000 NDCG evaluations: {elapsed:.3f}s")
        assert elapsed < 3.0
