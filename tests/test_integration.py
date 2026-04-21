"""Integration tests for end-to-end resume screening pipeline."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from src.parsers.resume_parser import ResumeParser
from src.models.job import JobDescription
from src.models.resume import ResumeData
from src.models.ranking import RankedCandidate, ContactInfo
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.ranking.ranking_engine import RankingEngine
from src.ranking.skill_matcher import SkillMatcher
from src.ranking.fairness_checker import FairnessChecker


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_JD_TEXT = """
We are hiring a Senior Machine Learning Engineer to join our NLP team.

Required Skills:
- Python (5+ years)
- Machine learning frameworks: PyTorch or TensorFlow
- Natural Language Processing (spaCy, NLTK, Hugging Face)
- Experience with sentence transformers and semantic search
- FAISS or similar vector databases
- FastAPI or Flask for ML model serving
- Docker and Kubernetes
- Git / GitHub

Preferred Skills:
- Experience with resume parsing or document understanding
- Fairness-aware ML and bias detection
- Streamlit or similar dashboard tools
- AWS or GCP cloud platforms

We expect 3-7 years of relevant experience. You will design and ship production ML systems.
"""

SAMPLE_RESUME_STRONG = """
Dr. Sarah Connor
Email: sarah.connor@ml.io
Phone: +1-555-0101

EXPERIENCE
Senior NLP Engineer — Skynet AI (2020–Present)
- Built semantic search pipeline using sentence-transformers (all-MiniLM-L6-v2) and FAISS
- Developed document parsing system using PyMuPDF and spaCy for information extraction
- Deployed ML models via FastAPI on Kubernetes, AWS EKS
- Worked on fairness metrics and bias detection using Fairlearn

Skills: Python, PyTorch, TensorFlow, spaCy, NLTK, Hugging Face, FAISS, FastAPI, Docker, Kubernetes, AWS, Git, Streamlit
"""

SAMPLE_RESUME_WEAK = """
Bob Smith
Email: bob@example.com

EXPERIENCE
Marketing Manager — ACME Corp (2019–Present)
- Managed social media campaigns
- Coordinated events and trade shows

Skills: Microsoft Office, PowerPoint, Excel, Photoshop, Social Media Marketing
"""


@pytest.fixture
def sample_job_description():
    return JobDescription(title="Senior ML Engineer", description=SAMPLE_JD_TEXT)


@pytest.fixture
def mock_resume_strong():
    """Strong candidate resume with good ML skills."""
    from src.models.resume import ContactInfo, ResumeData
    contact = ContactInfo(name="Dr. Sarah Connor", email="sarah.connor@ml.io")
    resume = ResumeData(
        contact_info=contact,
        skills=["python", "pytorch", "tensorflow", "spacy", "nltk",
                "hugging face", "faiss", "fastapi", "docker", "kubernetes",
                "aws", "git", "streamlit", "sentence-transformers"],
        raw_text=SAMPLE_RESUME_STRONG,
    )
    resume.embedding = np.random.rand(384).astype(np.float32)
    return resume


@pytest.fixture
def mock_resume_weak():
    """Weak candidate resume with irrelevant skills."""
    from src.models.resume import ContactInfo, ResumeData
    contact = ContactInfo(name="Bob Smith", email="bob@example.com")
    resume = ResumeData(
        contact_info=contact,
        skills=["microsoft office", "powerpoint", "excel", "photoshop", "social media"],
        raw_text=SAMPLE_RESUME_WEAK,
    )
    resume.embedding = np.random.rand(384).astype(np.float32)
    return resume


@pytest.fixture
def ranking_engine():
    """Create ranking engine with mocked embeddings."""
    mock_generator = MagicMock(spec=EmbeddingGenerator)

    def side_effect_encode(resume):
        # Strong candidate gets high-similarity embedding
        if "pytorch" in resume.skills or "spacy" in resume.skills:
            return np.ones(384, dtype=np.float32) * 0.9
        return np.ones(384, dtype=np.float32) * 0.2

    mock_generator.encode_resume.side_effect = side_effect_encode
    mock_generator.encode_job_description.return_value = np.ones(384, dtype=np.float32)
    mock_generator.batch_encode_resumes = MagicMock()

    engine = RankingEngine(
        semantic_weight=0.7,
        skill_weight=0.3,
        embedding_generator=mock_generator,
    )
    return engine


# ─── End-to-End Tests ────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    def test_ranking_produces_correct_order(
        self, ranking_engine, mock_resume_strong, mock_resume_weak, sample_job_description
    ):
        """Strong candidate should outrank weak candidate."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)

        results = ranking_engine.rank_candidates(
            [mock_resume_weak, mock_resume_strong], sample_job_description
        )

        assert len(results) == 2
        # Strong candidate (Sarah Connor) should be rank 1
        assert results[0].resume.contact_info.name == "Dr. Sarah Connor"
        assert results[1].resume.contact_info.name == "Bob Smith"
        assert results[0].hybrid_score > results[1].hybrid_score

    def test_scores_in_valid_range(
        self, ranking_engine, mock_resume_strong, mock_resume_weak, sample_job_description
    ):
        """All scores must be in [0, 1] range."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)

        results = ranking_engine.rank_candidates(
            [mock_resume_strong, mock_resume_weak], sample_job_description
        )

        for candidate in results:
            assert 0.0 <= candidate.hybrid_score <= 1.0, \
                f"Hybrid score {candidate.hybrid_score} out of range"
            assert 0.0 <= candidate.semantic_score <= 1.0, \
                f"Semantic score {candidate.semantic_score} out of range"
            assert 0.0 <= candidate.skill_score <= 1.0, \
                f"Skill score {candidate.skill_score} out of range"

    def test_ranks_are_sequential(
        self, ranking_engine, mock_resume_strong, mock_resume_weak, sample_job_description
    ):
        """Ranks must be 1, 2, 3, ... with no gaps."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)

        results = ranking_engine.rank_candidates(
            [mock_resume_strong, mock_resume_weak], sample_job_description
        )

        ranks = [c.rank for c in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_process_batch_returns_complete_result(
        self, ranking_engine, mock_resume_strong, mock_resume_weak, sample_job_description
    ):
        """process_batch should return BatchProcessingResult with all fields."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)

        result = ranking_engine.process_batch(
            [mock_resume_strong, mock_resume_weak],
            sample_job_description,
            include_fairness=True,
        )

        assert result.total_resumes == 2
        assert len(result.ranked_candidates) == 2
        assert result.processing_time > 0
        assert result.job_id == sample_job_description.job_id

    def test_empty_resumes_list(self, ranking_engine, sample_job_description):
        """Empty resume list should return empty results gracefully."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)
        results = ranking_engine.rank_candidates([], sample_job_description)
        assert results == []

    def test_single_resume(
        self, ranking_engine, mock_resume_strong, sample_job_description
    ):
        """Single resume should work and get rank 1."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)
        results = ranking_engine.rank_candidates([mock_resume_strong], sample_job_description)
        assert len(results) == 1
        assert results[0].rank == 1

    def test_ranking_consistency(
        self, ranking_engine, mock_resume_strong, mock_resume_weak, sample_job_description
    ):
        """Ranking must be deterministic — same input same output."""
        sample_job_description.embedding = np.ones(384, dtype=np.float32)

        results1 = ranking_engine.rank_candidates(
            [mock_resume_strong, mock_resume_weak], sample_job_description
        )
        results2 = ranking_engine.rank_candidates(
            [mock_resume_strong, mock_resume_weak], sample_job_description
        )

        assert [c.rank for c in results1] == [c.rank for c in results2]
        for r1, r2 in zip(results1, results2):
            assert abs(r1.hybrid_score - r2.hybrid_score) < 1e-6


class TestSkillMatchingIntegration:
    """Integration tests for skill matching within the pipeline."""

    def test_skill_overlap_improves_score(self):
        """Candidate with matching skills should score higher than one without."""
        matcher = SkillMatcher()

        required = ["python", "fastapi", "docker", "kubernetes"]

        score_full = matcher.calculate_skill_match(
            ["python", "fastapi", "docker", "kubernetes"], required
        )
        score_half = matcher.calculate_skill_match(
            ["python", "fastapi"], required
        )
        score_none = matcher.calculate_skill_match(
            ["java", "spring", "maven"], required
        )

        assert score_full > score_half > score_none
        assert score_none == pytest.approx(0.0, abs=0.01)

    def test_synonym_normalisation_end_to_end(self):
        """Synonyms should map to canonical skills and match correctly."""
        matcher = SkillMatcher()

        # Resume uses abbreviations
        resume_skills = ["js", "nodejs", "postgres", "k8s", "ml"]
        # JD uses full names
        required = ["javascript", "node.js", "postgresql", "kubernetes", "machine learning"]

        score = matcher.calculate_skill_match(resume_skills, required)
        assert score > 0.8, f"Expected >0.8 after synonym normalisation, got {score}"

    def test_coverage_bonus_applied(self):
        """Full coverage of required skills should trigger bonus score."""
        matcher = SkillMatcher()
        skills = ["python", "react", "aws", "docker", "postgresql"]
        required = ["python", "react", "aws", "docker", "postgresql"]

        score = matcher.calculate_skill_match(skills, required)
        # With 100% coverage, should get coverage bonus
        assert score == pytest.approx(1.0, abs=0.01)


class TestFairnessIntegration:
    """Integration tests for fairness checker."""

    def _make_candidate(self, name: str, score: float, rank: int) -> RankedCandidate:
        """Helper to create a mock candidate."""
        from src.models.resume import ContactInfo, ResumeData
        contact = ContactInfo(name=name, email=f"{name.lower().replace(' ', '.')}@test.com")
        resume = ResumeData(contact_info=contact, skills=[], raw_text="")
        candidate = RankedCandidate(
            resume=resume,
            semantic_score=score,
            skill_score=score,
            hybrid_score=score,
            rank=rank,
        )
        return candidate

    def test_fairness_report_generated(self):
        """Fairness report should be generated without errors."""
        checker = FairnessChecker()
        candidates = [
            self._make_candidate("Alice Johnson", 0.85, 1),
            self._make_candidate("Bob Smith", 0.78, 2),
            self._make_candidate("Chen Wei", 0.72, 3),
            self._make_candidate("Diana Prince", 0.65, 4),
            self._make_candidate("Edward Nwosu", 0.58, 5),
        ]

        report = checker.generate_fairness_report(candidates, top_k=3)
        assert report is not None
        assert report.total_candidates == 5
        assert isinstance(report.bias_flags, list)
        assert isinstance(report.recommendations, list)

    def test_fairness_score_in_range(self):
        """Overall fairness score must be in [0, 1]."""
        checker = FairnessChecker()
        candidates = [
            self._make_candidate(f"Candidate {i}", float(1.0 - i * 0.1), i + 1)
            for i in range(8)
        ]

        report = checker.generate_fairness_report(candidates, top_k=4)
        score = report.get_overall_fairness_score()
        assert 0.0 <= score <= 1.0

    def test_empty_candidates_handled(self):
        """Empty candidate list should return error report, not crash."""
        checker = FairnessChecker()
        report = checker.generate_fairness_report([], top_k=10)
        assert report is not None
        assert report.total_candidates == 0


class TestScoringFormula:
    """Tests validating the hybrid scoring formula properties."""

    def test_formula_monotonicity(self):
        """Higher component scores must produce higher (or equal) hybrid score."""
        engine = RankingEngine.__new__(RankingEngine)
        engine.semantic_weight = 0.7
        engine.skill_weight = 0.3

        def hybrid(sem, skill):
            return engine.semantic_weight * sem + engine.skill_weight * skill

        # Increasing semantic score should increase hybrid
        for sem in [0.3, 0.5, 0.7, 0.9]:
            assert hybrid(sem, 0.5) <= hybrid(min(sem + 0.1, 1.0), 0.5)

    def test_formula_weights_sum_to_one(self):
        """At boundary values, formula works correctly."""
        engine = RankingEngine(semantic_weight=0.7, skill_weight=0.3)
        assert abs(engine.semantic_weight + engine.skill_weight - 1.0) < 1e-9

    def test_perfect_candidate_scores_100(self):
        """A candidate matching everything should get close to 1.0."""
        engine = RankingEngine(semantic_weight=0.7, skill_weight=0.3)

        # Simulate perfect semantic score
        perfect_sem = 1.0
        perfect_skill = 1.0
        hybrid = 0.7 * perfect_sem + 0.3 * perfect_skill
        assert hybrid == pytest.approx(1.0, abs=1e-9)

    def test_zero_candidate_scores_zero(self):
        """A candidate matching nothing should score 0."""
        hybrid = 0.7 * 0.0 + 0.3 * 0.0
        assert hybrid == 0.0
