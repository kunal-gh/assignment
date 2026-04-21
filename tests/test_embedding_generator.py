"""Tests for embedding generator functionality."""

import pytest
import numpy as np

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.models.resume import ResumeData, ContactInfo
from src.models.job import JobDescription


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a small model for testing to avoid long download times
        try:
            self.generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        except Exception:
            # If model loading fails, skip these tests
            pytest.skip("Embedding model not available for testing")
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        assert self.generator is not None
        assert self.generator.model is not None
        assert self.generator.embedding_dimension > 0
    
    def test_encode_text(self):
        """Test basic text encoding."""
        text = "This is a test sentence for embedding generation."
        embedding = self.generator.encode_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (self.generator.embedding_dimension,)
        assert not np.any(np.isnan(embedding))
        assert not np.any(np.isinf(embedding))
    
    def test_encode_empty_text(self):
        """Test encoding empty text."""
        embedding = self.generator.encode_text("")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (self.generator.embedding_dimension,)
        # Should return zero vector for empty text
        assert np.allclose(embedding, np.zeros(self.generator.embedding_dimension))
    
    def test_encode_resume(self):
        """Test resume encoding."""
        resume_data = ResumeData(
            file_name="test.pdf",
            raw_text="Software engineer with Python experience",
            contact_info=ContactInfo(name="John Doe"),
            skills=["Python", "Machine Learning"],
            experience=[],
            education=[]
        )
        
        embedding = self.generator.encode_resume(resume_data)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (self.generator.embedding_dimension,)
        assert not np.any(np.isnan(embedding))
        assert resume_data.embedding is not None
    
    def test_encode_job_description(self):
        """Test job description encoding."""
        job_desc = JobDescription(
            title="Software Engineer",
            description="Looking for a Python developer with ML experience",
            required_skills=["Python", "Machine Learning"]
        )
        
        embedding = self.generator.encode_job_description(job_desc)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (self.generator.embedding_dimension,)
        assert not np.any(np.isnan(embedding))
        assert job_desc.embedding is not None
    
    def test_batch_encode(self):
        """Test batch encoding functionality."""
        texts = [
            "First test sentence",
            "Second test sentence", 
            "Third test sentence"
        ]
        
        embeddings = self.generator.batch_encode(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(texts), self.generator.embedding_dimension)
        assert not np.any(np.isnan(embeddings))
        assert not np.any(np.isinf(embeddings))
    
    def test_batch_encode_empty_list(self):
        """Test batch encoding with empty list."""
        embeddings = self.generator.batch_encode([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, self.generator.embedding_dimension)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        text1 = "Python programming language"
        text2 = "Python software development"
        text3 = "Java programming language"
        
        emb1 = self.generator.encode_text(text1)
        emb2 = self.generator.encode_text(text2)
        emb3 = self.generator.encode_text(text3)
        
        # Similar texts should have higher similarity
        sim_12 = self.generator.cosine_similarity(emb1, emb2)
        sim_13 = self.generator.cosine_similarity(emb1, emb3)
        
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0
        assert sim_12 > sim_13  # Python-Python should be more similar than Python-Java
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        text = "Test sentence"
        embedding = self.generator.encode_text(text)
        
        similarity = self.generator.cosine_similarity(embedding, embedding)
        
        assert abs(similarity - 1.0) < 1e-6  # Should be very close to 1.0
    
    def test_cosine_similarity_dimension_mismatch(self):
        """Test cosine similarity with mismatched dimensions."""
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(768)  # Different dimension
        
        similarity = self.generator.cosine_similarity(emb1, emb2)
        
        assert similarity == 0.0  # Should return 0 for dimension mismatch
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.generator.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'embedding_dimension' in info
        assert 'max_sequence_length' in info
        assert 'cache_enabled' in info
        assert 'model_loaded' in info
        
        assert info['model_loaded'] is True
        assert info['embedding_dimension'] == self.generator.embedding_dimension


@pytest.fixture
def sample_resumes():
    """Sample resume data for testing."""
    return [
        ResumeData(
            file_name="resume1.pdf",
            raw_text="Python developer with machine learning experience",
            contact_info=ContactInfo(name="Alice Smith"),
            skills=["Python", "Machine Learning", "TensorFlow"],
            experience=[],
            education=[]
        ),
        ResumeData(
            file_name="resume2.pdf", 
            raw_text="Java developer with web development background",
            contact_info=ContactInfo(name="Bob Johnson"),
            skills=["Java", "Spring", "React"],
            experience=[],
            education=[]
        )
    ]


def test_batch_encode_resumes(sample_resumes):
    """Test batch encoding of resumes."""
    try:
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    except Exception:
        pytest.skip("Embedding model not available for testing")
    
    embeddings = generator.batch_encode_resumes(sample_resumes)
    
    assert len(embeddings) == len(sample_resumes)
    
    for i, embedding in enumerate(embeddings):
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (generator.embedding_dimension,)
        assert sample_resumes[i].embedding is not None