"""Unit tests for VectorStoreManager."""

import os
import tempfile

import numpy as np
import pytest

from src.embeddings.vector_store import VectorStoreManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for fast tests


def _random_embedding(dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _random_embeddings(n: int, dim: int = DIM, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_dimension(self):
        vsm = VectorStoreManager(dimension=DIM)
        assert vsm.dimension == DIM

    def test_size_starts_at_zero(self):
        vsm = VectorStoreManager(dimension=DIM)
        assert vsm.size == 0

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError):
            VectorStoreManager(dimension=0)

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError):
            VectorStoreManager(dimension=-1)


# ---------------------------------------------------------------------------
# add_embeddings
# ---------------------------------------------------------------------------


class TestAddEmbeddings:
    def test_add_single_embedding(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["doc1"], _random_embeddings(1))
        assert vsm.size == 1

    def test_add_multiple_embeddings(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a", "b", "c"], _random_embeddings(3))
        assert vsm.size == 3

    def test_add_empty_list_is_noop(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings([], np.empty((0, DIM), dtype=np.float32))
        assert vsm.size == 0

    def test_wrong_dimension_raises(self):
        vsm = VectorStoreManager(dimension=DIM)
        bad = _random_embeddings(1, dim=DIM + 1)
        with pytest.raises(ValueError, match="dimension"):
            vsm.add_embeddings(["x"], bad)

    def test_mismatched_ids_and_embeddings_raises(self):
        vsm = VectorStoreManager(dimension=DIM)
        with pytest.raises(ValueError):
            vsm.add_embeddings(["a", "b"], _random_embeddings(3))

    def test_duplicate_id_raises(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["doc1"], _random_embeddings(1))
        with pytest.raises(ValueError, match="Duplicate"):
            vsm.add_embeddings(["doc1"], _random_embeddings(1, seed=99))

    def test_accepts_1d_embedding(self):
        vsm = VectorStoreManager(dimension=DIM)
        vec = _random_embedding()
        vsm.add_embeddings(["doc1"], vec)
        assert vsm.size == 1

    def test_accepts_float64_input(self):
        vsm = VectorStoreManager(dimension=DIM)
        vec = _random_embeddings(1).astype(np.float64)
        vsm.add_embeddings(["doc1"], vec)
        assert vsm.size == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_search_empty_index_returns_empty(self):
        vsm = VectorStoreManager(dimension=DIM)
        results = vsm.search(_random_embedding(), k=5)
        assert results == []

    def test_search_returns_correct_number(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a", "b", "c"], _random_embeddings(3))
        results = vsm.search(_random_embedding(), k=2)
        assert len(results) == 2

    def test_search_returns_all_when_k_exceeds_size(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a", "b"], _random_embeddings(2))
        results = vsm.search(_random_embedding(), k=100)
        assert len(results) == 2

    def test_search_result_format(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["doc1"], _random_embeddings(1))
        results = vsm.search(_random_embedding(), k=1)
        assert len(results) == 1
        doc_id, score = results[0]
        assert isinstance(doc_id, str)
        assert isinstance(score, float)

    def test_search_scores_in_descending_order(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a", "b", "c", "d"], _random_embeddings(4))
        results = vsm.search(_random_embedding(), k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_identical_query_returns_score_near_one(self):
        vsm = VectorStoreManager(dimension=DIM)
        vec = _random_embedding()
        vsm.add_embeddings(["doc1"], vec.reshape(1, -1))
        results = vsm.search(vec, k=1)
        assert len(results) == 1
        _, score = results[0]
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_search_returns_correct_ids(self):
        vsm = VectorStoreManager(dimension=DIM)
        vecs = _random_embeddings(3)
        vsm.add_embeddings(["alpha", "beta", "gamma"], vecs)
        results = vsm.search(_random_embedding(), k=3)
        returned_ids = {doc_id for doc_id, _ in results}
        assert returned_ids == {"alpha", "beta", "gamma"}

    def test_invalid_k_raises(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a"], _random_embeddings(1))
        with pytest.raises(ValueError):
            vsm.search(_random_embedding(), k=0)


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        vsm = VectorStoreManager(dimension=DIM)
        vecs = _random_embeddings(3)
        vsm.add_embeddings(["x", "y", "z"], vecs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "test_index")
            vsm.save_index(base)

            vsm2 = VectorStoreManager(dimension=DIM)
            vsm2.load_index(base)

            assert vsm2.size == 3
            assert set(vsm2._id_map.values()) == {"x", "y", "z"}

    def test_search_after_load_gives_same_results(self):
        vsm = VectorStoreManager(dimension=DIM)
        vecs = _random_embeddings(5)
        vsm.add_embeddings([f"doc{i}" for i in range(5)], vecs)
        query = _random_embedding(seed=7)
        original_results = vsm.search(query, k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "idx")
            vsm.save_index(base)

            vsm2 = VectorStoreManager(dimension=DIM)
            vsm2.load_index(base)
            loaded_results = vsm2.search(query, k=3)

        assert [doc_id for doc_id, _ in original_results] == [
            doc_id for doc_id, _ in loaded_results
        ]

    def test_load_missing_index_raises(self):
        vsm = VectorStoreManager(dimension=DIM)
        with pytest.raises(FileNotFoundError):
            vsm.load_index("/nonexistent/path/index")

    def test_load_dimension_mismatch_raises(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a"], _random_embeddings(1))

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "idx")
            vsm.save_index(base)

            vsm2 = VectorStoreManager(dimension=DIM + 4)
            with pytest.raises(ValueError, match="Dimension mismatch"):
                vsm2.load_index(base)

    def test_save_creates_parent_directories(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a"], _random_embeddings(1))

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "nested", "dir", "idx")
            vsm.save_index(base)
            assert os.path.exists(base + ".index")
            assert os.path.exists(base + ".meta.json")


# ---------------------------------------------------------------------------
# Metadata management
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_id_map_populated_after_add(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["resume_1", "resume_2"], _random_embeddings(2))
        assert set(vsm._id_map.values()) == {"resume_1", "resume_2"}

    def test_reverse_id_map_populated(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["r1"], _random_embeddings(1))
        assert "r1" in vsm._reverse_id_map

    def test_get_embedding_returns_vector(self):
        vsm = VectorStoreManager(dimension=DIM)
        vec = _random_embedding()
        vsm.add_embeddings(["doc1"], vec.reshape(1, -1))
        retrieved = vsm.get_embedding("doc1")
        assert retrieved is not None
        assert retrieved.shape == (DIM,)

    def test_get_embedding_unknown_id_returns_none(self):
        vsm = VectorStoreManager(dimension=DIM)
        assert vsm.get_embedding("nonexistent") is None

    def test_clear_resets_metadata(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a", "b"], _random_embeddings(2))
        vsm.clear()
        assert vsm.size == 0
        assert len(vsm._id_map) == 0
        assert len(vsm._reverse_id_map) == 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self):
        vsm = VectorStoreManager(dimension=DIM)
        stats = vsm.get_stats()
        assert stats["dimension"] == DIM
        assert stats["total_vectors"] == 0
        assert "index_type" in stats

    def test_stats_total_vectors_updates(self):
        vsm = VectorStoreManager(dimension=DIM)
        vsm.add_embeddings(["a", "b"], _random_embeddings(2))
        assert vsm.get_stats()["total_vectors"] == 2
