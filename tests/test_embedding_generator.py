"""Tests for embedding generator functionality."""

import numpy as np
import pytest

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.models.job import JobDescription
from src.models.resume import ContactInfo, Education, Experience, ResumeData

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator():
    """Module-scoped generator to avoid reloading the model for every test."""
    try:
        return EmbeddingGenerator(model_name="all-MiniLM-L6-v2", use_cache=False)
    except Exception:
        pytest.skip("Embedding model not available for testing")


@pytest.fixture
def simple_resume():
    return ResumeData(
        file_name="test.pdf",
        raw_text="Software engineer with Python experience",
        contact_info=ContactInfo(name="John Doe"),
        skills=["Python", "Machine Learning"],
        experience=[],
        education=[],
    )


@pytest.fixture
def full_resume():
    return ResumeData(
        file_name="full.pdf",
        raw_text="Experienced backend engineer",
        contact_info=ContactInfo(name="Jane Smith", email="jane@example.com"),
        skills=["Python", "Django", "PostgreSQL", "Docker"],
        experience=[
            Experience(
                title="Senior Software Engineer",
                company="Acme Corp",
                start_date="2020-01",
                end_date="2023-12",
                description="Built scalable REST APIs and microservices",
                skills_used=["Python", "FastAPI", "Docker"],
                is_current=False,
            ),
            Experience(
                title="Software Engineer",
                company="Startup Inc",
                start_date="2018-06",
                end_date="2019-12",
                description="Developed Django web applications",
                skills_used=["Python", "Django"],
                is_current=False,
            ),
        ],
        education=[
            Education(
                degree="B.Sc. Computer Science",
                institution="State University",
                graduation_date="2018",
                major="Computer Science",
                relevant_coursework=["Algorithms", "Databases"],
            )
        ],
    )


@pytest.fixture
def sample_resumes():
    return [
        ResumeData(
            file_name="resume1.pdf",
            raw_text="Python developer with machine learning experience",
            contact_info=ContactInfo(name="Alice Smith"),
            skills=["Python", "Machine Learning", "TensorFlow"],
            experience=[],
            education=[],
        ),
        ResumeData(
            file_name="resume2.pdf",
            raw_text="Java developer with web development background",
            contact_info=ContactInfo(name="Bob Johnson"),
            skills=["Java", "Spring", "React"],
            experience=[],
            education=[],
        ),
    ]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_generator_initialises(self, generator):
        assert generator is not None
        assert generator.model is not None
        assert generator.embedding_dimension > 0

    def test_model_info_keys(self, generator):
        info = generator.get_model_info()
        for key in (
            "model_name",
            "embedding_dimension",
            "max_sequence_length",
            "cache_enabled",
            "model_loaded",
            "chunk_size",
            "chunk_overlap",
        ):
            assert key in info
        assert info["model_loaded"] is True


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessText:
    def test_strips_whitespace(self, generator):
        assert generator._preprocess_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self, generator):
        result = generator._preprocess_text("hello   world")
        assert "  " not in result

    def test_removes_urls(self, generator):
        result = generator._preprocess_text("Visit https://example.com for details")
        assert "https" not in result
        assert "example.com" not in result

    def test_normalises_unicode_dashes(self, generator):
        result = generator._preprocess_text("2020\u20132023")
        assert "\u2013" not in result
        assert "-" in result

    def test_empty_string_returns_empty(self, generator):
        assert generator._preprocess_text("") == ""

    def test_none_like_empty_returns_empty(self, generator):
        assert generator._preprocess_text("   ") == ""


# ---------------------------------------------------------------------------
# Resume text construction
# ---------------------------------------------------------------------------


class TestBuildResumeText:
    def test_includes_skills(self, generator, simple_resume):
        text = generator._build_resume_text(simple_resume)
        assert "Python" in text
        assert "Machine Learning" in text

    def test_includes_raw_text(self, generator, simple_resume):
        text = generator._build_resume_text(simple_resume)
        assert "Software engineer" in text

    def test_includes_experience(self, generator, full_resume):
        text = generator._build_resume_text(full_resume)
        assert "Senior Software Engineer" in text
        assert "Acme Corp" in text

    def test_includes_education(self, generator, full_resume):
        text = generator._build_resume_text(full_resume)
        assert "Computer Science" in text
        assert "State University" in text

    def test_empty_resume_returns_string(self, generator):
        empty = ResumeData(file_name="empty.pdf")
        text = generator._build_resume_text(empty)
        assert isinstance(text, str)

    def test_experience_with_current_job(self, generator):
        resume = ResumeData(
            file_name="current.pdf",
            skills=[],
            experience=[
                Experience(
                    title="Engineer",
                    company="Corp",
                    start_date="2022-01",
                    is_current=True,
                )
            ],
        )
        text = generator._build_resume_text(resume)
        assert "Present" in text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_text_not_chunked(self, generator):
        text = "short text"
        chunks = generator._chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_chunked(self, generator):
        # Create text longer than chunk_size words
        words = ["word"] * (generator.chunk_size + 50)
        text = " ".join(words)
        chunks = generator._chunk_text(text)
        assert len(chunks) > 1

    def test_chunks_overlap(self, generator):
        words = [f"w{i}" for i in range(generator.chunk_size + generator.chunk_overlap + 10)]
        text = " ".join(words)
        chunks = generator._chunk_text(text)
        if len(chunks) >= 2:
            # Last words of chunk 0 should appear at start of chunk 1
            end_of_first = chunks[0].split()[-generator.chunk_overlap :]
            start_of_second = chunks[1].split()[: generator.chunk_overlap]
            assert end_of_first == start_of_second

    def test_all_words_covered(self, generator):
        words = [f"w{i}" for i in range(generator.chunk_size * 2 + 5)]
        text = " ".join(words)
        chunks = generator._chunk_text(text)
        # Every word in the original text should appear in at least one chunk
        all_chunk_words = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        for word in words:
            assert word in all_chunk_words


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


class TestEncodeText:
    def test_returns_numpy_array(self, generator):
        emb = generator.encode_text("Hello world")
        assert isinstance(emb, np.ndarray)

    def test_correct_dimension(self, generator):
        emb = generator.encode_text("Hello world")
        assert emb.shape == (generator.embedding_dimension,)

    def test_no_nan_or_inf(self, generator):
        emb = generator.encode_text("Hello world")
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))

    def test_empty_text_returns_zeros(self, generator):
        emb = generator.encode_text("")
        assert np.allclose(emb, np.zeros(generator.embedding_dimension))

    def test_whitespace_only_returns_zeros(self, generator):
        emb = generator.encode_text("   ")
        assert np.allclose(emb, np.zeros(generator.embedding_dimension))

    def test_long_text_handled(self, generator):
        long_text = " ".join(["word"] * (generator.chunk_size * 3))
        emb = generator.encode_text(long_text)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (generator.embedding_dimension,)
        assert not np.any(np.isnan(emb))


class TestEncodeResume:
    def test_returns_numpy_array(self, generator, simple_resume):
        emb = generator.encode_resume(simple_resume)
        assert isinstance(emb, np.ndarray)

    def test_correct_dimension(self, generator, simple_resume):
        emb = generator.encode_resume(simple_resume)
        assert emb.shape == (generator.embedding_dimension,)

    def test_stores_embedding_on_resume(self, generator, simple_resume):
        generator.encode_resume(simple_resume)
        assert simple_resume.embedding is not None
        assert isinstance(simple_resume.embedding, np.ndarray)

    def test_no_nan_or_inf(self, generator, full_resume):
        emb = generator.encode_resume(full_resume)
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))

    def test_different_resumes_different_embeddings(self, generator, sample_resumes):
        emb1 = generator.encode_resume(sample_resumes[0])
        emb2 = generator.encode_resume(sample_resumes[1])
        # Python ML resume vs Java web resume should not be identical
        assert not np.allclose(emb1, emb2)

    def test_same_resume_same_embedding(self, generator, simple_resume):
        emb1 = generator.encode_resume(simple_resume)
        emb2 = generator.encode_resume(simple_resume)
        assert np.allclose(emb1, emb2)


class TestJobDescriptionGetCombinedText:
    """Tests for JobDescription.get_combined_text()."""

    def test_includes_title(self):
        job = JobDescription(title="Software Engineer", description="", required_skills=["Python"])
        text = job.get_combined_text()
        assert "Software Engineer" in text

    def test_includes_description(self):
        job = JobDescription(
            title="Engineer",
            description="Build scalable systems",
            required_skills=["Python"],
        )
        text = job.get_combined_text()
        assert "Build scalable systems" in text

    def test_includes_required_skills(self):
        job = JobDescription(
            title="Engineer",
            description="Role",
            required_skills=["Python", "Docker"],
        )
        text = job.get_combined_text()
        assert "Python" in text
        assert "Docker" in text

    def test_includes_preferred_skills(self):
        job = JobDescription(
            title="Engineer",
            description="Role",
            required_skills=["Python"],
            preferred_skills=["Kubernetes", "AWS"],
        )
        text = job.get_combined_text()
        assert "Kubernetes" in text
        assert "AWS" in text

    def test_includes_experience_level(self):
        job = JobDescription(
            title="Engineer",
            description="Role",
            required_skills=["Python"],
            experience_level="senior",
        )
        text = job.get_combined_text()
        assert "senior" in text

    def test_empty_optional_fields_omitted(self):
        job = JobDescription(
            title="Engineer",
            description="Role",
            required_skills=["Python"],
            preferred_skills=[],
        )
        text = job.get_combined_text()
        assert "Preferred Skills" not in text

    def test_returns_string(self):
        job = JobDescription(title="Engineer", description="Role", required_skills=["Python"])
        assert isinstance(job.get_combined_text(), str)

    def test_all_fields_combined(self):
        job = JobDescription(
            title="ML Engineer",
            description="Machine learning role",
            required_skills=["Python", "TensorFlow"],
            preferred_skills=["PyTorch"],
            experience_level="mid",
        )
        text = job.get_combined_text()
        assert "ML Engineer" in text
        assert "Machine learning role" in text
        assert "TensorFlow" in text
        assert "PyTorch" in text
        assert "mid" in text


class TestEncodeJobDescription:
    def test_returns_numpy_array(self, generator):
        job = JobDescription(
            title="Software Engineer",
            description="Looking for a Python developer",
            required_skills=["Python"],
        )
        emb = generator.encode_job_description(job)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (generator.embedding_dimension,)

    def test_stores_embedding_on_job(self, generator):
        job = JobDescription(
            title="Data Scientist",
            description="ML role",
            required_skills=["Python", "TensorFlow"],
        )
        generator.encode_job_description(job)
        assert job.embedding is not None

    def test_no_nan_or_inf(self, generator):
        job = JobDescription(
            title="Backend Engineer",
            description="Build REST APIs with Python and FastAPI",
            required_skills=["Python", "FastAPI", "PostgreSQL"],
            preferred_skills=["Docker", "AWS"],
            experience_level="mid",
        )
        emb = generator.encode_job_description(job)
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))

    def test_correct_dimension(self, generator):
        job = JobDescription(
            title="DevOps Engineer",
            description="CI/CD and infrastructure automation",
            required_skills=["Docker", "Kubernetes"],
        )
        emb = generator.encode_job_description(job)
        assert emb.shape == (generator.embedding_dimension,)

    def test_embedding_stored_on_object(self, generator):
        job = JobDescription(
            title="Frontend Developer",
            description="React and TypeScript development",
            required_skills=["React", "TypeScript"],
        )
        emb = generator.encode_job_description(job)
        assert job.embedding is not None
        assert np.allclose(job.embedding, emb)

    def test_different_jobs_different_embeddings(self, generator):
        job_ml = JobDescription(
            title="ML Engineer",
            description="Deep learning and model training",
            required_skills=["Python", "TensorFlow", "PyTorch"],
        )
        job_web = JobDescription(
            title="Frontend Developer",
            description="Web UI development with React",
            required_skills=["React", "JavaScript", "CSS"],
        )
        emb_ml = generator.encode_job_description(job_ml)
        emb_web = generator.encode_job_description(job_web)
        assert not np.allclose(emb_ml, emb_web)

    def test_same_job_same_embedding(self, generator):
        job = JobDescription(
            title="Python Developer",
            description="Backend Python development",
            required_skills=["Python", "Django"],
        )
        emb1 = generator.encode_job_description(job)
        emb2 = generator.encode_job_description(job)
        assert np.allclose(emb1, emb2)

    def test_job_with_only_required_skills(self, generator):
        job = JobDescription(
            title="Engineer",
            description="Engineering role",
            required_skills=["Python"],
            preferred_skills=[],
        )
        emb = generator.encode_job_description(job)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (generator.embedding_dimension,)

    def test_job_with_all_experience_levels(self, generator):
        for level in ["entry", "mid", "senior", "executive"]:
            job = JobDescription(
                title="Engineer",
                description="Engineering role",
                required_skills=["Python"],
                experience_level=level,
            )
            emb = generator.encode_job_description(job)
            assert isinstance(emb, np.ndarray)
            assert not np.any(np.isnan(emb))

    def test_similar_jobs_higher_similarity_than_dissimilar(self, generator):
        job_python1 = JobDescription(
            title="Python Backend Engineer",
            description="Python REST API development",
            required_skills=["Python", "FastAPI"],
        )
        job_python2 = JobDescription(
            title="Python Developer",
            description="Python web services",
            required_skills=["Python", "Django"],
        )
        job_design = JobDescription(
            title="UX Designer",
            description="User experience and interface design",
            required_skills=["Figma", "Sketch"],
        )
        emb1 = generator.encode_job_description(job_python1)
        emb2 = generator.encode_job_description(job_python2)
        emb3 = generator.encode_job_description(job_design)

        sim_similar = generator.cosine_similarity(emb1, emb2)
        sim_dissimilar = generator.cosine_similarity(emb1, emb3)
        assert sim_similar > sim_dissimilar


class TestJobDescriptionEmbeddingCaching:
    """Tests for caching behavior in encode_job_description."""

    def test_cache_hit_returns_same_embedding(self, tmp_path):
        """Second call with same job should return cached embedding."""
        try:
            gen = EmbeddingGenerator(
                model_name="all-MiniLM-L6-v2",
                cache_dir=str(tmp_path),
                use_cache=True,
            )
        except Exception:
            pytest.skip("Embedding model not available for testing")

        job = JobDescription(
            title="Software Engineer",
            description="Python backend development",
            required_skills=["Python", "FastAPI"],
        )
        emb1 = gen.encode_job_description(job)

        # Reset embedding on object to force re-fetch from cache
        job.embedding = None
        emb2 = gen.encode_job_description(job)

        assert np.allclose(emb1, emb2)

    def test_cache_stores_embedding(self, tmp_path):
        """After encoding, cache should contain the embedding."""
        try:
            gen = EmbeddingGenerator(
                model_name="all-MiniLM-L6-v2",
                cache_dir=str(tmp_path),
                use_cache=True,
            )
        except Exception:
            pytest.skip("Embedding model not available for testing")

        job = JobDescription(
            title="Data Engineer",
            description="Data pipeline development",
            required_skills=["Python", "Spark"],
        )
        gen.encode_job_description(job)

        stats = gen.get_cache_stats()
        assert stats["statistics"]["stores"] >= 1

    def test_cache_hit_increments_hit_count(self, tmp_path):
        """Repeated calls should increment cache hit counter."""
        try:
            gen = EmbeddingGenerator(
                model_name="all-MiniLM-L6-v2",
                cache_dir=str(tmp_path),
                use_cache=True,
            )
        except Exception:
            pytest.skip("Embedding model not available for testing")

        job = JobDescription(
            title="Cloud Architect",
            description="AWS cloud infrastructure design",
            required_skills=["AWS", "Terraform"],
        )
        gen.encode_job_description(job)  # first call - miss + store
        gen.encode_job_description(job)  # second call - hit

        stats = gen.get_cache_stats()
        assert stats["statistics"]["hits"] >= 1

    def test_no_cache_mode_still_works(self, tmp_path):
        """Generator with use_cache=False should still produce valid embeddings."""
        try:
            gen = EmbeddingGenerator(
                model_name="all-MiniLM-L6-v2",
                use_cache=False,
            )
        except Exception:
            pytest.skip("Embedding model not available for testing")

        job = JobDescription(
            title="Security Engineer",
            description="Application security and penetration testing",
            required_skills=["Python", "Security"],
        )
        emb = gen.encode_job_description(job)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (gen.embedding_dimension,)
        assert not np.any(np.isnan(emb))


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------


class TestBatchEncode:
    def test_correct_shape(self, generator):
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embs = generator.batch_encode(texts)
        assert embs.shape == (len(texts), generator.embedding_dimension)

    def test_empty_list(self, generator):
        embs = generator.batch_encode([])
        assert embs.shape == (0, generator.embedding_dimension)

    def test_no_nan_or_inf(self, generator):
        texts = ["Hello", "World", "Python"]
        embs = generator.batch_encode(texts)
        assert not np.any(np.isnan(embs))
        assert not np.any(np.isinf(embs))

    def test_batch_encode_resumes(self, generator, sample_resumes):
        embeddings = generator.batch_encode_resumes(sample_resumes)
        assert len(embeddings) == len(sample_resumes)
        for i, emb in enumerate(embeddings):
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (generator.embedding_dimension,)
            assert sample_resumes[i].embedding is not None


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self, generator):
        emb = generator.encode_text("Python developer")
        sim = generator.cosine_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_similar_texts_higher_than_dissimilar(self, generator):
        emb_py1 = generator.encode_text("Python programming language")
        emb_py2 = generator.encode_text("Python software development")
        emb_java = generator.encode_text("Java enterprise applications")

        sim_py = generator.cosine_similarity(emb_py1, emb_py2)
        sim_diff = generator.cosine_similarity(emb_py1, emb_java)

        assert sim_py > sim_diff

    def test_score_in_range(self, generator):
        emb1 = generator.encode_text("Machine learning engineer")
        emb2 = generator.encode_text("Frontend web developer")
        sim = generator.cosine_similarity(emb1, emb2)
        assert 0.0 <= sim <= 1.0

    def test_dimension_mismatch_returns_zero(self, generator):
        emb1 = np.random.rand(384)
        emb2 = np.random.rand(768)
        assert generator.cosine_similarity(emb1, emb2) == 0.0

    def test_zero_vector_returns_zero(self, generator):
        zero = np.zeros(generator.embedding_dimension)
        emb = generator.encode_text("Some text")
        assert generator.cosine_similarity(zero, emb) == 0.0
