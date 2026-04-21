"""Unit tests for BatchProcessor."""

from typing import List
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from src.models.job import JobDescription
from src.models.resume import ContactInfo, ResumeData
from src.ranking.batch_processor import BatchProcessor, BatchResult, DEFAULT_BATCH_CHUNK_SIZE
from src.ranking.similarity_search import SearchResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DIM = 8  # small dimension keeps tests fast


def _unit_vec(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_resume(
    candidate_id: str,
    embedding: np.ndarray = None,
    skills: List[str] = None,
) -> ResumeData:
    return ResumeData(
        candidate_id=candidate_id,
        file_name=f"{candidate_id}.pdf",
        raw_text=f"Resume for {candidate_id}",
        contact_info=ContactInfo(name=candidate_id),
        skills=skills or [],
        embedding=embedding,
    )


def _make_job(embedding: np.ndarray = None) -> JobDescription:
    job = JobDescription(
        title="Software Engineer",
        description="Python developer with ML experience",
        required_skills=["python", "machine learning"],
    )
    job.embedding = embedding
    return job


def _mock_generator(dim: int = DIM) -> MagicMock:
    """Return a mock EmbeddingGenerator with deterministic behaviour."""
    gen = MagicMock()
    gen.model_name = "mock-model"
    gen.embedding_dimension = dim

    def encode_job(job_desc):
        v = _unit_vec(0, dim)
        job_desc.embedding = v
        return v

    def batch_encode_resumes(resumes):
        for i, r in enumerate(resumes):
            if r.embedding is None:
                v = _unit_vec(hash(r.candidate_id) % 1000, dim)
                r.embedding = v
        return [r.embedding for r in resumes]

    def encode_resume(resume):
        v = _unit_vec(hash(resume.candidate_id) % 1000, dim)
        resume.embedding = v
        return v

    gen.encode_job_description.side_effect = encode_job
    gen.batch_encode_resumes.side_effect = batch_encode_resumes
    gen.encode_resume.side_effect = encode_resume
    return gen


def _make_processor(chunk_size: int = 10, dim: int = DIM) -> BatchProcessor:
    return BatchProcessor(
        embedding_generator=_mock_generator(dim),
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_chunk_size(self):
        proc = _make_processor()
        assert proc.chunk_size == 10

    def test_custom_chunk_size(self):
        proc = _make_processor(chunk_size=25)
        assert proc.chunk_size == 25

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError):
            BatchProcessor(embedding_generator=_mock_generator(), chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError):
            BatchProcessor(embedding_generator=_mock_generator(), chunk_size=-5)

    def test_accepts_custom_generator(self):
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=5)
        assert proc.embedding_generator is gen

    def test_default_chunk_size_constant(self):
        assert DEFAULT_BATCH_CHUNK_SIZE == 50


# ---------------------------------------------------------------------------
# BatchResult dataclass
# ---------------------------------------------------------------------------


class TestBatchResult:
    def test_success_rate_all_processed(self):
        br = BatchResult(total_processed=10, total_failed=0)
        assert br.success_rate == pytest.approx(1.0)

    def test_success_rate_partial(self):
        br = BatchResult(total_processed=8, total_failed=2)
        assert br.success_rate == pytest.approx(0.8)

    def test_success_rate_all_failed(self):
        br = BatchResult(total_processed=0, total_failed=5)
        assert br.success_rate == pytest.approx(0.0)

    def test_success_rate_empty(self):
        br = BatchResult()
        assert br.success_rate == pytest.approx(0.0)

    def test_defaults(self):
        br = BatchResult()
        assert br.results == []
        assert br.total_processed == 0
        assert br.total_failed == 0
        assert br.failed_ids == []


# ---------------------------------------------------------------------------
# process_batch – basic behaviour
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_empty_resumes_returns_empty_batch_result(self):
        proc = _make_processor()
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, [], top_k=5)
        assert isinstance(result, BatchResult)
        assert result.results == []
        assert result.total_processed == 0

    def test_returns_batch_result(self):
        proc = _make_processor()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=5)
        assert isinstance(result, BatchResult)

    def test_top_k_limits_results(self):
        proc = _make_processor(chunk_size=10)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(20)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=5)
        assert len(result.results) == 5

    def test_top_k_larger_than_resumes_returns_all(self):
        proc = _make_processor()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=100)
        assert len(result.results) == 3

    def test_invalid_top_k_raises(self):
        proc = _make_processor()
        job = _make_job(_unit_vec(0))
        resumes = [_make_resume("r0", _unit_vec(1))]
        with pytest.raises(ValueError):
            proc.process_batch(job, resumes, top_k=0)

    def test_results_sorted_by_descending_score(self):
        proc = _make_processor(chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(10)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=10)
        scores = [r.similarity_score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential_from_one(self):
        proc = _make_processor(chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(6)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=6)
        assert [r.rank for r in result.results] == list(range(1, 7))

    def test_scores_in_unit_interval(self):
        proc = _make_processor(chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(8)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=8)
        for r in result.results:
            assert 0.0 <= r.similarity_score <= 1.0

    def test_total_processed_count(self):
        proc = _make_processor(chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(12)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=12)
        assert result.total_processed == 12

    def test_total_failed_zero_on_success(self):
        proc = _make_processor()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=5)
        assert result.total_failed == 0
        assert result.failed_ids == []


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------


class TestChunking:
    def test_split_into_chunks_even(self):
        items = list(range(10))
        chunks = BatchProcessor._split_into_chunks(items, 5)
        assert chunks == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_split_into_chunks_uneven(self):
        items = list(range(7))
        chunks = BatchProcessor._split_into_chunks(items, 3)
        assert chunks == [[0, 1, 2], [3, 4, 5], [6]]

    def test_split_into_chunks_smaller_than_chunk_size(self):
        items = list(range(3))
        chunks = BatchProcessor._split_into_chunks(items, 10)
        assert chunks == [[0, 1, 2]]

    def test_split_into_chunks_empty(self):
        assert BatchProcessor._split_into_chunks([], 5) == []

    def test_processes_in_multiple_chunks(self):
        """Verify batch_encode_resumes is called once per chunk."""
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=3)
        # 7 resumes → 3 chunks (3, 3, 1); all missing embeddings
        resumes = [_make_resume(f"r{i}") for i in range(7)]
        job = _make_job(_unit_vec(0))
        proc.process_batch(job, resumes, top_k=7)
        # batch_encode_resumes should be called 3 times (once per chunk)
        assert gen.batch_encode_resumes.call_count == 3

    def test_skips_batch_encode_when_embeddings_present(self):
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        job = _make_job(_unit_vec(0))
        proc.process_batch(job, resumes, top_k=5)
        gen.batch_encode_resumes.assert_not_called()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_callback_called_once_per_chunk(self):
        proc = _make_processor(chunk_size=3)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(7)]
        job = _make_job(_unit_vec(0))
        calls = []
        proc.process_batch(job, resumes, top_k=7, progress_callback=lambda p, t: calls.append((p, t)))
        # 3 chunks → 3 callback calls
        assert len(calls) == 3

    def test_callback_receives_total(self):
        proc = _make_processor(chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(10)]
        job = _make_job(_unit_vec(0))
        totals = []
        proc.process_batch(job, resumes, top_k=10, progress_callback=lambda p, t: totals.append(t))
        assert all(t == 10 for t in totals)

    def test_callback_final_processed_equals_total(self):
        proc = _make_processor(chunk_size=4)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(8)]
        job = _make_job(_unit_vec(0))
        calls = []
        proc.process_batch(job, resumes, top_k=8, progress_callback=lambda p, t: calls.append(p))
        assert calls[-1] == 8

    def test_no_callback_does_not_raise(self):
        proc = _make_processor()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(3)]
        job = _make_job(_unit_vec(0))
        # Should not raise
        proc.process_batch(job, resumes, top_k=3, progress_callback=None)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_failed_resumes_recorded_in_failed_ids(self):
        """Resumes that fail embedding should be recorded as failed."""
        gen = _mock_generator()
        call_count = [0]

        def flaky_batch_encode(resumes):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated embedding failure")
            for i, r in enumerate(resumes):
                r.embedding = _unit_vec(i)

        gen.batch_encode_resumes.side_effect = flaky_batch_encode
        # Individual fallback also fails for the first chunk
        gen.encode_resume.side_effect = RuntimeError("Individual failure")

        proc = BatchProcessor(embedding_generator=gen, chunk_size=3)
        resumes = [_make_resume(f"r{i}") for i in range(6)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=6)

        # First chunk (3 resumes) all fail; second chunk succeeds
        assert result.total_failed == 3
        assert len(result.failed_ids) == 3

    def test_processing_continues_after_chunk_failure(self):
        """A failed chunk should not stop subsequent chunks from processing."""
        gen = _mock_generator()
        call_count = [0]

        def flaky_batch_encode(resumes):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First chunk fails")
            for i, r in enumerate(resumes):
                r.embedding = _unit_vec(i)

        gen.batch_encode_resumes.side_effect = flaky_batch_encode
        gen.encode_resume.side_effect = RuntimeError("Individual failure")

        proc = BatchProcessor(embedding_generator=gen, chunk_size=3)
        resumes = [_make_resume(f"r{i}") for i in range(6)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=6)

        # Second chunk (3 resumes) should succeed
        assert result.total_processed == 3

    def test_partial_failure_success_rate(self):
        gen = _mock_generator()
        call_count = [0]

        def flaky_batch_encode(resumes):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Chunk 1 fails")
            for i, r in enumerate(resumes):
                r.embedding = _unit_vec(i)

        gen.batch_encode_resumes.side_effect = flaky_batch_encode
        gen.encode_resume.side_effect = RuntimeError("Individual failure")

        proc = BatchProcessor(embedding_generator=gen, chunk_size=5)
        resumes = [_make_resume(f"r{i}") for i in range(10)]
        job = _make_job(_unit_vec(0))
        result = proc.process_batch(job, resumes, top_k=10)

        assert result.success_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# generate_batch_embeddings
# ---------------------------------------------------------------------------


class TestGenerateBatchEmbeddings:
    def test_generates_embeddings_for_missing(self):
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=5)
        resumes = [_make_resume(f"r{i}") for i in range(5)]
        count = proc.generate_batch_embeddings(resumes)
        assert count == 5
        assert all(r.embedding is not None for r in resumes)

    def test_skips_resumes_with_existing_embeddings(self):
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=5)
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        count = proc.generate_batch_embeddings(resumes)
        assert count == 0
        gen.batch_encode_resumes.assert_not_called()

    def test_mixed_existing_and_missing(self):
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=10)
        resumes = [
            _make_resume("r0", _unit_vec(0)),  # already has embedding
            _make_resume("r1"),                 # missing
            _make_resume("r2", _unit_vec(2)),  # already has embedding
            _make_resume("r3"),                 # missing
        ]
        count = proc.generate_batch_embeddings(resumes)
        assert count == 2

    def test_progress_callback_called(self):
        gen = _mock_generator()
        proc = BatchProcessor(embedding_generator=gen, chunk_size=3)
        resumes = [_make_resume(f"r{i}") for i in range(6)]
        calls = []
        proc.generate_batch_embeddings(resumes, progress_callback=lambda p, t: calls.append((p, t)))
        assert len(calls) == 2  # 2 chunks

    def test_chunk_failure_fills_zero_vectors(self):
        gen = _mock_generator()
        gen.batch_encode_resumes.side_effect = RuntimeError("Embedding failure")
        proc = BatchProcessor(embedding_generator=gen, chunk_size=5)
        resumes = [_make_resume(f"r{i}") for i in range(3)]
        proc.generate_batch_embeddings(resumes)
        # All resumes should have a zero-vector fallback
        for r in resumes:
            assert r.embedding is not None
            assert r.embedding.shape == (DIM,)

    def test_returns_zero_for_empty_list(self):
        proc = _make_processor()
        count = proc.generate_batch_embeddings([])
        assert count == 0

    def test_returns_zero_when_all_embeddings_present(self):
        proc = _make_processor()
        resumes = [_make_resume(f"r{i}", _unit_vec(i)) for i in range(5)]
        count = proc.generate_batch_embeddings(resumes)
        assert count == 0
