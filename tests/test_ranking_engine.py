"""Tests for ranking engine functionality."""

import pytest
import numpy as np

from src.ranking.ranking_engine import RankingEngine
from src.models.resume import ResumeData, ContactInfo
from src.models.job import JobDescription
from src.models.ranking import RankedCandidate


class TestRankingEngine:
    """Test cases for RankingEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RankingEngine(semantic_weight=0.7, skill_weight=0.3)
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine is not None
        assert self.engine.semantic_weight == 0.7
        assert self.engine.skill_weight == 0.3
        assert hasattr(self.engine, 'embedding_generator')
        assert hasattr(self.engine, 'skill_matcher')
        assert hasattr(self.engine, 'fairness_checker')
    
    def test_invalid_weights(self):
        """Test initialization with invalid weights."""
        with pytest.raises(ValueError):
            RankingEngine(semantic_weight=0.8, skill_weight=0.3)  # Sum > 1
        
        with pytest.raises(ValueError):
            RankingEngine(semantic_weight=-0.1, skill_weight=1.1)  # Negative weight
        
        with pytest.raises(ValueError):
            RankingEngine(semantic_weight=1.5, skill_weight=0.5)  # Weight > 1
    
    def test_calculate_hybrid_score_no_embeddings(self):
        """Test hybrid score calculation without pre-existing embeddings."""
        resume = ResumeData(
            file_name="test.pdf",
            raw_text="Python developer with machine learning experience",
            contact_info=ContactInfo(name="John Doe"),
            skills=["Python", "Machine Learning"],
            experience=[],
            education=[]
        )
        
        job_desc = JobDescription(
            title="ML Engineer",
            description="Looking for Python and ML skills",
            required_skills=["Python", "Machine Learning"]
        )
        
        scores = self.engine.calculate_hybrid_score(resume, job_desc)
        
        assert isinstance(scores, dict)
        assert 'semantic_score' in scores
        assert 'skill_score' in scores
        assert 'hybrid_score' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_calculate_hybrid_score_with_embeddings(self):
        """Test hybrid score calculation with pre-existing embeddings."""
        resume = ResumeData(
            file_name="test.pdf",
            raw_text="Python developer",
            contact_info=ContactInfo(name="John Doe"),
            skills=["Python"],
            experience=[],
            education=[],
            embedding=np.random.rand(384)  # Mock embedding
        )
        
        job_desc = JobDescription(
            title="Python Developer",
            description="Python programming",
            required_skills=["Python"],
            embedding=np.random.rand(384)  # Mock embedding
        )
        
        scores = self.engine.calculate_hybrid_score(resume, job_desc)
        
        assert isinstance(scores, dict)
        assert all(0.0 <= score <= 1.0 for score in scores.values())
    
    def test_rank_candidates_empty_list(self):
        """Test ranking with empty candidate list."""
        job_desc = JobDescription(
            title="Test Job",
            description="Test description",
            required_skills=["Python"]
        )
        
        results = self.engine.rank_candidates([], job_desc)
        
        assert results == []
    
    def test_rank_candidates_single_candidate(self):
        """Test ranking with single candidate."""
        resume = ResumeData(
            file_name="test.pdf",
            raw_text="Python developer",
            contact_info=ContactInfo(name="John Doe"),
            skills=["Python"],
            experience=[],
            education=[]
        )
        
        job_desc = JobDescription(
            title="Python Developer",
            description="Python programming",
            required_skills=["Python"]
        )
        
        results = self.engine.rank_candidates([resume], job_desc)
        
        assert len(results) == 1
        assert isinstance(results[0], RankedCandidate)
        assert results[0].rank == 1
        assert results[0].resume == resume
    
    def test_rank_candidates_multiple_candidates(self):
        """Test ranking with multiple candidates."""
        resumes = [
            ResumeData(
                file_name="resume1.pdf",
                raw_text="Expert Python developer with machine learning",
                contact_info=ContactInfo(name="Alice"),
                skills=["Python", "Machine Learning", "TensorFlow"],
                experience=[],
                education=[]
            ),
            ResumeData(
                file_name="resume2.pdf",
                raw_text="Junior Python developer",
                contact_info=ContactInfo(name="Bob"),
                skills=["Python"],
                experience=[],
                education=[]
            ),
            ResumeData(
                file_name="resume3.pdf",
                raw_text="Java developer",
                contact_info=ContactInfo(name="Charlie"),
                skills=["Java"],
                experience=[],
                education=[]
            )
        ]
        
        job_desc = JobDescription(
            title="ML Engineer",
            description="Python and machine learning required",
            required_skills=["Python", "Machine Learning"]
        )
        
        results = self.engine.rank_candidates(resumes, job_desc)
        
        assert len(results) == 3
        
        # Check ranking order (should be sorted by score)
        for i in range(len(results) - 1):
            assert results[i].hybrid_score >= results[i + 1].hybrid_score
        
        # Check rank assignments
        for i, candidate in enumerate(results):
            assert candidate.rank == i + 1
        
        # Alice should rank higher than Bob (more skills)
        # Bob should rank higher than Charlie (has Python)
        alice_rank = next(c.rank for c in results if c.resume.contact_info.name == "Alice")
        bob_rank = next(c.rank for c in results if c.resume.contact_info.name == "Bob")
        charlie_rank = next(c.rank for c in results if c.resume.contact_info.name == "Charlie")
        
        assert alice_rank < bob_rank < charlie_rank
    
    def test_explain_ranking(self):
        """Test ranking explanation generation."""
        candidate = RankedCandidate(
            resume=ResumeData(
                file_name="test.pdf",
                raw_text="Python developer",
                contact_info=ContactInfo(name="John Doe"),
                skills=["Python", "Machine Learning"],
                experience=[],
                education=[]
            ),
            semantic_score=0.8,
            skill_score=0.9,
            hybrid_score=0.83,
            rank=1
        )
        
        job_desc = JobDescription(
            title="ML Engineer",
            description="Python and ML required",
            required_skills=["Python", "Machine Learning"]
        )
        
        explanation = self.engine.explain_ranking(candidate, job_desc)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "83%" in explanation or "83.0%" in explanation  # Should mention the score
        assert "#1" in explanation  # Should mention the rank
    
    def test_update_weights(self):
        """Test updating scoring weights."""
        # Valid weight update
        self.engine.update_weights(0.6, 0.4)
        assert self.engine.semantic_weight == 0.6
        assert self.engine.skill_weight == 0.4
        
        # Invalid weight updates
        with pytest.raises(ValueError):
            self.engine.update_weights(0.8, 0.3)  # Sum != 1
        
        with pytest.raises(ValueError):
            self.engine.update_weights(-0.1, 1.1)  # Negative weight
    
    def test_get_top_candidates(self):
        """Test getting top N candidates."""
        candidates = [
            RankedCandidate(
                resume=ResumeData(contact_info=ContactInfo(name=f"Candidate {i}")),
                semantic_score=0.8 - i * 0.1,
                skill_score=0.7 - i * 0.1,
                hybrid_score=0.75 - i * 0.1,
                rank=i + 1
            )
            for i in range(10)
        ]
        
        top_3 = self.engine.get_top_candidates(candidates, 3)
        
        assert len(top_3) == 3
        assert all(candidate.rank <= 3 for candidate in top_3)
    
    def test_get_ranking_stats(self):
        """Test ranking statistics calculation."""
        candidates = [
            RankedCandidate(
                resume=ResumeData(contact_info=ContactInfo(name=f"Candidate {i}")),
                semantic_score=0.8 - i * 0.1,
                skill_score=0.7 - i * 0.1,
                hybrid_score=0.75 - i * 0.1,
                rank=i + 1
            )
            for i in range(5)
        ]
        
        stats = self.engine.get_ranking_stats(candidates)
        
        assert isinstance(stats, dict)
        assert 'total_candidates' in stats
        assert 'hybrid_scores' in stats
        assert 'semantic_scores' in stats
        assert 'skill_scores' in stats
        assert 'weights' in stats
        
        assert stats['total_candidates'] == 5
        assert 'mean' in stats['hybrid_scores']
        assert 'median' in stats['hybrid_scores']
        assert 'std' in stats['hybrid_scores']


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator for testing."""
    class MockEmbeddingGenerator:
        def __init__(self):
            self.embedding_dimension = 384
        
        def encode_resume(self, resume_data):
            # Return mock embedding
            embedding = np.random.rand(self.embedding_dimension)
            resume_data.embedding = embedding
            return embedding
        
        def encode_job_description(self, job_desc):
            # Return mock embedding
            embedding = np.random.rand(self.embedding_dimension)
            job_desc.embedding = embedding
            return embedding
        
        def cosine_similarity(self, vec1, vec2):
            # Return mock similarity based on some logic
            return np.random.rand()
    
    return MockEmbeddingGenerator()


def test_ranking_with_mock_embeddings(mock_embedding_generator):
    """Test ranking with mock embedding generator."""
    engine = RankingEngine(
        semantic_weight=0.7,
        skill_weight=0.3,
        embedding_generator=mock_embedding_generator
    )
    
    resumes = [
        ResumeData(
            file_name="resume1.pdf",
            raw_text="Python developer",
            contact_info=ContactInfo(name="Alice"),
            skills=["Python", "Machine Learning"],
            experience=[],
            education=[]
        ),
        ResumeData(
            file_name="resume2.pdf",
            raw_text="Java developer",
            contact_info=ContactInfo(name="Bob"),
            skills=["Java"],
            experience=[],
            education=[]
        )
    ]
    
    job_desc = JobDescription(
        title="Python Developer",
        description="Python programming",
        required_skills=["Python"]
    )
    
    results = engine.rank_candidates(resumes, job_desc)
    
    assert len(results) == 2
    assert all(isinstance(candidate, RankedCandidate) for candidate in results)
    assert all(0.0 <= candidate.hybrid_score <= 1.0 for candidate in results)