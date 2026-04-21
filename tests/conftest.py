"""Shared pytest fixtures and configuration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.job import JobDescription
from src.models.ranking import RankedCandidate
from src.models.resume import ContactInfo, ResumeData
from src.ranking.fairness_checker import FairnessChecker
from src.ranking.skill_matcher import SkillMatcher

# ─── Resume fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def skill_matcher():
    """Shared SkillMatcher instance (session-scoped for speed)."""
    return SkillMatcher()


@pytest.fixture(scope="session")
def fairness_checker():
    """Shared FairnessChecker instance."""
    return FairnessChecker()


@pytest.fixture
def sample_contact():
    return ContactInfo(name="Jane Doe", email="jane.doe@example.com", phone="+1-555-0199")


@pytest.fixture
def sample_resume(sample_contact):
    """Minimal valid ResumeData."""
    return ResumeData(
        contact_info=sample_contact,
        skills=["python", "docker", "postgresql", "fastapi", "git"],
        raw_text="Jane Doe. Python developer with 4 years experience. Skills: Python, Docker.",
        embedding=np.random.rand(384).astype(np.float32),
    )


@pytest.fixture
def strong_ml_resume():
    """Resume with strong ML/NLP skills matching typical JD."""
    contact = ContactInfo(name="Priya ML", email="priya.ml@example.com")
    return ResumeData(
        contact_info=contact,
        skills=[
            "python",
            "pytorch",
            "tensorflow",
            "spacy",
            "nltk",
            "hugging face",
            "faiss",
            "fastapi",
            "docker",
            "kubernetes",
            "aws",
            "git",
            "streamlit",
            "sentence-transformers",
            "scikit-learn",
        ],
        raw_text="Senior ML Engineer with 6 years NLP experience. Built semantic search and resume screening systems.",
        embedding=np.ones(384, dtype=np.float32) * 0.95,
    )


@pytest.fixture
def weak_resume():
    """Resume with no relevant ML skills."""
    contact = ContactInfo(name="Bob Non-Tech", email="bob@example.com")
    return ResumeData(
        contact_info=contact,
        skills=["microsoft office", "powerpoint", "excel", "event planning"],
        raw_text="Marketing coordinator with experience in social media and event management.",
        embedding=np.ones(384, dtype=np.float32) * 0.05,
    )


@pytest.fixture
def sample_job_description():
    return JobDescription(
        title="Senior ML Engineer",
        description=(
            "We are looking for a Senior ML Engineer with strong Python skills. "
            "Required: Python, PyTorch or TensorFlow, spaCy, FAISS, FastAPI, Docker, "
            "Kubernetes, Git. Preferred: AWS, Streamlit, sentence-transformers. "
            "5+ years experience in NLP and ML systems."
        ),
    )


# ─── Candidate fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_ranked_candidate(strong_ml_resume):
    return RankedCandidate(
        resume=strong_ml_resume,
        semantic_score=0.87,
        skill_score=0.80,
        hybrid_score=0.849,
        rank=1,
        explanation="Excellent candidate with strong ML/NLP background.",
    )


@pytest.fixture
def ranked_candidates_list(strong_ml_resume, weak_resume):
    """List of 5 ranked candidates from strong to weak."""
    candidates = []
    for i, (name, score) in enumerate(
        [
            ("Priya ML", 0.88),
            ("Alex Mid", 0.70),
            ("Chris OK", 0.55),
            ("Dana Low", 0.35),
            ("Eric Weak", 0.15),
        ],
        start=1,
    ):
        contact = ContactInfo(name=name, email=f"{name.lower().replace(' ', '.')}@test.com")
        resume = ResumeData(
            contact_info=contact,
            skills=["python"] if score > 0.5 else ["excel"],
            raw_text=f"Resume of {name}",
            embedding=np.ones(384) * score,
        )
        candidates.append(
            RankedCandidate(
                resume=resume,
                semantic_score=score,
                skill_score=score * 0.9,
                hybrid_score=score,
                rank=i,
            )
        )
    return candidates


# ─── Numpy fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def unit_vector():
    vec = np.ones(384, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def zero_vector():
    return np.zeros(384, dtype=np.float32)


@pytest.fixture
def random_embedding():
    rng = np.random.default_rng(42)
    return rng.random(384).astype(np.float32)
