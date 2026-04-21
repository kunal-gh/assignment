"""Unit tests for SimilaritySearchEngine."""

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.job import JobDescription
from src.models.resume import ContactInfo, ResumeData
from src.ranking.similarity_search import SearchResult, SimilaritySearchEngine

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DIM = 8  # small dimension keeps tests fast


def _unit_vec(seed: int, dim: int = DIM) -> np.ndarray:
    """Return a deterministic unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_resume(candidate_id: str, embedding: np.ndarray, skills: List[str] = None) -> ResumeData:
    resume = ResumeData(
        candidate_id=candidate_id,
        file_name=f"{candidate_id}.pdf",
        raw_text=f"Resume for {candidate_id}",
        contact_info=ContactInfo(name=candidate_id),
        skills=skills or [],
        embedding=embedding,
    )
    return resume


def _make_job(embedding: np.ndarray = None) -> JobDescription:
    job = JobDescription(
        title="Software Engineer",
        description="Python developer with ML experience",
        required_skills=["python", "machine learning"],
    )
    job.embedding = embedding
    return job


def _mock_generator(dim: int = DIM) -> MagicMock:
    """Return a mock EmbeddingGenerator that returns deterministic embeddings."""
    gen = MagicMock()
    gen.model_name = "mock-model"
    gen.embedding_dimension = dim

    def encode_job(job_desc):
        v = _unit_vec(0, dim)
        job_desc.embedding = v
        return v

    def batch_encode_resumes(resumes):
        for i, r in enumerate(resumes):
            v = _unit_vec(i + 1, dim)
            r.embedding = v
        return [r.embedding for r in resumes]

    gen.encode_job_description.side_effect = encode_job
    gen.batch_encode_resumes.side_effect = batch_encode_resumes
    return gen


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_accepts_custom_generator(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        assert engine.embedding_generator is gen

    def test_creates_default_generator_when_none(self):
        """Verify a real EmbeddingGenerator is created when none is supplied.

        We patch the constructor to avoid loading a real model in CI.
        """
        with patch(
            "src.ranking.similarity_search.EmbeddingGenerator"
        ) as MockGen:
            mock_instance = MagicMock()
            mock_instance.model_name = "all-MiniLM-L6-v2"
            MockGen.return_value = mock_instance

            engine = SimilaritySearchEngine()
            MockGen.assert_called_once_with(model_name="all-MiniLM-L6-v2")
            assert engine.embedding_generator is mock_instance


# ---------------------------------------------------------------------------
# search – basic behaviour
# ---------------------------------------------------------------------------


class TestSearch:
    def _engine(self, dim: int = DIM) -> SimilaritySearchEngine:
        return SimilaritySearchEngine(embedding_generator=_mock_generator(dim))

    def test_empty_resumes_returns_empty_list(self):
        engine = self._engine()
        job = _make_job()
        results = engine.search(job, [], top_k=5)
        assert results == []

    def test_returns_search_results(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_limits_results(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_resumes_returns_all(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=100)
        assert len(results) == 3

    def test_invalid_top_k_raises(self):
        engine = self._engine()
        job = _make_job(_unit_vec(0))
        resumes = [_make_resume("r0", _unit_vec(1))]
        with pytest.raises(ValueError):
            engine.search(job, resumes, top_k=0)

    def test_results_sorted_by_descending_score(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=5)
        scores = [r.similarity_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_from_one(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(4)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=4)
        assert [r.rank for r in results] == list(range(1, 5))

    def test_scores_clamped_to_unit_interval(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=3)
        for r in results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_identical_query_and_resume_scores_near_one(self):
        """When the job embedding equals a resume embedding, score ≈ 1."""
        engine = self._engine()
        vec = _unit_vec(42)
        resume = _make_resume("perfect", vec)
        job = _make_job(vec)
        # Prevent the mock from overwriting the pre-set embedding
        engine.embedding_generator.encode_job_description.side_effect = None
        engine.embedding_generator.encode_job_description.return_value = vec
        results = engine.search(job, [resume], top_k=1)
        assert len(results) == 1
        assert results[0].similarity_score == pytest.approx(1.0, abs=1e-5)

    def test_result_candidate_ids_match_input(self):
        engine = self._engine()
        ids = ["alice", "bob", "carol"]
        resumes = [_make_resume(cid, _unit_vec(i)) for i, cid in enumerate(ids)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=3)
        returned_ids = {r.candidate_id for r in results}
        assert returned_ids == set(ids)

    def test_result_resume_objects_match_input(self):
        engine = self._engine()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        job = _make_job(_unit_vec(0))
        results = engine.search(job, resumes, top_k=3)
        input_ids = {r.candidate_id for r in resumes}
        result_ids = {r.candidate_id for r in results}
        assert result_ids == input_ids


# ---------------------------------------------------------------------------
# Embedding generation integration
# ---------------------------------------------------------------------------


class TestEmbeddingGeneration:
    def test_generates_job_embedding_when_missing(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        job = _make_job(embedding=None)  # no embedding yet
        resumes = [_make_resume("r0", _unit_vec(1))]
        engine.search(job, resumes, top_k=1)
        gen.encode_job_description.assert_called_once_with(job)

    def test_reuses_existing_job_embedding(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        job = _make_job(embedding=_unit_vec(0))  # already has embedding
        resumes = [_make_resume("r0", _unit_vec(1))]
        engine.search(job, resumes, top_k=1)
        gen.encode_job_description.assert_not_called()

    def test_generates_resume_embeddings_when_missing(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        job = _make_job(_unit_vec(0))
        resumes = [
            ResumeData(candidate_id="r0", file_name="r0.pdf", raw_text="text"),
            ResumeData(candidate_id="r1", file_name="r1.pdf", raw_text="text"),
        ]
        # embeddings are None – engine should call batch_encode_resumes
        engine.search(job, resumes, top_k=2)
        gen.batch_encode_resumes.assert_called_once()

    def test_skips_batch_encode_when_all_embeddings_present(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        job = _make_job(_unit_vec(0))
        resumes = [_make_resume(f"r{i}", _unit_vec(i + 1)) for i in range(3)]
        engine.search(job, resumes, top_k=3)
        gen.batch_encode_resumes.assert_not_called()


# ---------------------------------------------------------------------------
# search_by_text
# ---------------------------------------------------------------------------


class TestSearchByText:
    def test_search_by_text_returns_results(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        results = engine.search_by_text("Python developer", resumes, top_k=3)
        assert len(results) == 3

    def test_search_by_text_calls_encode_job(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        resumes = [_make_resume("r0", _unit_vec(1))]
        engine.search_by_text("some query", resumes, top_k=1)
        gen.encode_job_description.assert_called_once()

    def test_search_by_text_empty_resumes(self):
        gen = _mock_generator()
        engine = SimilaritySearchEngine(embedding_generator=gen)
        results = engine.search_by_text("query", [], top_k=5)
        assert results == []
