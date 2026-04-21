"""Unit tests for CacheManager (task 2.2.4)."""

import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings.cache_manager import CacheManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_embedding(dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """CacheManager with a temporary disk directory and no Redis."""
    return CacheManager(cache_dir=str(tmp_path), max_memory_cache=5)


@pytest.fixture
def embedding():
    return make_embedding()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_cache_dir(self, tmp_path):
        subdir = tmp_path / "sub" / "dir"
        cm = CacheManager(cache_dir=str(subdir))
        assert subdir.exists()

    def test_redis_disabled_by_default(self, cache):
        assert cache._redis_available is False
        assert cache._redis is None

    def test_stats_initialised_to_zero(self, cache):
        stats = cache.get_stats()["statistics"]
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["stores"] == 0

    def test_redis_stats_in_get_stats(self, cache):
        stats = cache.get_stats()
        assert "redis_cache" in stats
        assert stats["redis_cache"]["enabled"] is False


# ---------------------------------------------------------------------------
# Memory cache
# ---------------------------------------------------------------------------

class TestMemoryCache:
    def test_store_and_retrieve(self, cache, embedding):
        cache.store_embedding("key1", embedding)
        result = cache.get_embedding("key1")
        assert result is not None
        assert np.allclose(result, embedding)

    def test_miss_returns_none(self, cache):
        assert cache.get_embedding("nonexistent") is None

    def test_miss_increments_miss_counter(self, cache):
        cache.get_embedding("nonexistent")
        assert cache.get_stats()["statistics"]["misses"] == 1

    def test_hit_increments_hit_counter(self, cache, embedding):
        cache.store_embedding("k", embedding)
        cache.get_embedding("k")
        assert cache.get_stats()["statistics"]["hits"] == 1

    def test_store_increments_store_counter(self, cache, embedding):
        cache.store_embedding("k", embedding)
        assert cache.get_stats()["statistics"]["stores"] == 1

    def test_lru_eviction_when_full(self, tmp_path):
        cm = CacheManager(cache_dir=str(tmp_path), max_memory_cache=3)
        for i in range(3):
            cm.store_embedding(f"key{i}", make_embedding(seed=i))
        # Access key0 and key1 to make key2 the LRU
        cm.get_embedding("key0")
        cm.get_embedding("key1")
        # Adding a 4th entry should evict key2 (LRU)
        cm.store_embedding("key3", make_embedding(seed=3))
        assert len(cm._memory_cache) <= 3
        assert cm.get_stats()["statistics"]["memory_evictions"] >= 1

    def test_stored_embedding_is_a_copy(self, cache, embedding):
        cache.store_embedding("k", embedding)
        embedding[:] = 0  # mutate original
        result = cache.get_embedding("k")
        assert not np.allclose(result, 0)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

class TestDiskCache:
    def test_disk_file_created(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("disk_key", embedding)
        pkl_files = list(tmp_path.glob("embedding_*.pkl"))
        assert len(pkl_files) == 1

    def test_disk_hit_after_memory_cleared(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("dk", embedding)
        # Wipe memory cache to force disk lookup
        cm._memory_cache.clear()
        cm._cache_access_times.clear()
        result = cm.get_embedding("dk")
        assert result is not None
        assert np.allclose(result, embedding)

    def test_disk_hit_promotes_to_memory(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("dk", embedding)
        cm._memory_cache.clear()
        cm._cache_access_times.clear()
        cm.get_embedding("dk")
        assert "dk" in cm._memory_cache

    def test_metadata_file_created(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("meta_key", embedding)
        assert (tmp_path / "cache_metadata.json").exists()

    def test_expired_disk_file_returns_none(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("exp_key", embedding)
        cm._memory_cache.clear()
        cm._cache_access_times.clear()
        # Patch _is_cache_expired to always return True
        with patch.object(cm, "_is_cache_expired", return_value=True):
            result = cm.get_embedding("exp_key")
        assert result is None


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

class TestCacheInvalidation:
    def test_clear_cache_empties_memory(self, cache, embedding):
        cache.store_embedding("k", embedding)
        cache.clear_cache()
        assert len(cache._memory_cache) == 0

    def test_clear_cache_removes_disk_files(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("k", embedding)
        cm.clear_cache()
        assert list(tmp_path.glob("embedding_*.pkl")) == []

    def test_clear_cache_resets_metadata(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("k", embedding)
        cm.clear_cache()
        assert cm._metadata == {}

    def test_remove_cache_entry_from_memory(self, cache, embedding):
        cache.store_embedding("k", embedding)
        removed = cache.remove_cache_entry("k")
        assert removed is True
        assert "k" not in cache._memory_cache

    def test_remove_cache_entry_from_disk(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("k", embedding)
        cm.remove_cache_entry("k")
        assert list(tmp_path.glob("embedding_*.pkl")) == []

    def test_remove_nonexistent_entry_returns_false(self, cache):
        assert cache.remove_cache_entry("ghost") is False

    def test_cleanup_expired_removes_old_files(self, tmp_path, embedding):
        cm = CacheManager(cache_dir=str(tmp_path))
        cm.store_embedding("old", embedding)
        with patch.object(cm, "_is_cache_expired", return_value=True):
            cm.cleanup_expired()
        assert list(tmp_path.glob("embedding_*.pkl")) == []


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_roundtrip_float32(self):
        arr = make_embedding(dim=384)
        data = CacheManager._serialize_embedding(arr)
        result = CacheManager._deserialize_embedding(data)
        assert result is not None
        assert np.allclose(arr, result)
        assert result.dtype == arr.dtype

    def test_roundtrip_float64(self):
        arr = np.random.rand(128).astype(np.float64)
        data = CacheManager._serialize_embedding(arr)
        result = CacheManager._deserialize_embedding(data)
        assert result is not None
        assert np.allclose(arr, result)

    def test_deserialize_invalid_bytes_returns_none(self):
        result = CacheManager._deserialize_embedding(b"not valid pickle data")
        assert result is None

    def test_deserialize_non_array_returns_none(self):
        bad_data = pickle.dumps({"array": "not_an_array", "dtype": "float32", "shape": (8,)})
        result = CacheManager._deserialize_embedding(bad_data)
        assert result is None


# ---------------------------------------------------------------------------
# Redis integration (mocked)
# ---------------------------------------------------------------------------

class TestRedisIntegration:
    """Tests for Redis layer using a mock Redis client."""

    def _make_cache_with_mock_redis(self, tmp_path):
        """Return a CacheManager wired to a mock Redis client."""
        cm = CacheManager(cache_dir=str(tmp_path))
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        cm._redis = mock_redis
        cm._redis_available = True
        return cm, mock_redis

    def test_store_calls_redis_setex(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        emb = make_embedding()
        cm.store_embedding("rk", emb)
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == f"{CacheManager.REDIS_KEY_PREFIX}rk"
        assert args[1] == cm.redis_ttl

    def test_get_checks_redis_on_memory_miss(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        emb = make_embedding()
        mock_redis.get.return_value = CacheManager._serialize_embedding(emb)
        result = cm.get_embedding("rk")
        assert result is not None
        assert np.allclose(result, emb)
        mock_redis.get.assert_called_once_with(f"{CacheManager.REDIS_KEY_PREFIX}rk")

    def test_redis_hit_increments_redis_hit_counter(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        emb = make_embedding()
        mock_redis.get.return_value = CacheManager._serialize_embedding(emb)
        cm.get_embedding("rk")
        assert cm._cache_stats["redis_hits"] == 1

    def test_redis_miss_falls_through_to_disk(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        emb = make_embedding()
        # Pre-store on disk only
        cm._redis_available = False
        cm.store_embedding("dk", emb)
        cm._memory_cache.clear()
        cm._cache_access_times.clear()
        # Re-enable Redis but make it return None (miss)
        cm._redis_available = True
        mock_redis.get.return_value = None
        result = cm.get_embedding("dk")
        assert result is not None
        assert np.allclose(result, emb)

    def test_redis_error_falls_through_gracefully(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        mock_redis.get.side_effect = Exception("connection refused")
        # Should not raise; should return None (disk miss too)
        result = cm.get_embedding("err_key")
        assert result is None
        assert cm._cache_stats["redis_errors"] >= 1

    def test_remove_entry_calls_redis_delete(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        emb = make_embedding()
        cm.store_embedding("del_key", emb)
        cm.remove_cache_entry("del_key")
        mock_redis.delete.assert_called()

    def test_clear_cache_flushes_redis(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        mock_redis.keys.return_value = [b"embedding:k1"]
        cm.clear_cache()
        mock_redis.keys.assert_called_once()
        mock_redis.delete.assert_called()

    def test_redis_disabled_when_connection_fails(self, tmp_path):
        with patch("src.embeddings.cache_manager.REDIS_AVAILABLE", True):
            with patch("redis.Redis.from_url") as mock_from_url:
                mock_client = MagicMock()
                mock_client.ping.side_effect = Exception("refused")
                mock_from_url.return_value = mock_client
                cm = CacheManager(
                    cache_dir=str(tmp_path),
                    redis_url="redis://localhost:6379/0",
                )
        assert cm._redis_available is False

    def test_redis_stats_reported_in_get_stats(self, tmp_path):
        cm, mock_redis = self._make_cache_with_mock_redis(tmp_path)
        emb = make_embedding()
        mock_redis.get.return_value = CacheManager._serialize_embedding(emb)
        cm.get_embedding("k")
        stats = cm.get_stats()
        assert stats["redis_cache"]["enabled"] is True
        assert stats["redis_cache"]["hits"] == 1

    def test_redis_ttl_used_in_setex(self, tmp_path):
        custom_ttl = 3600
        cm = CacheManager(cache_dir=str(tmp_path), redis_ttl=custom_ttl)
        mock_redis = MagicMock()
        cm._redis = mock_redis
        cm._redis_available = True
        cm.store_embedding("ttl_key", make_embedding())
        args = mock_redis.setex.call_args[0]
        assert args[1] == custom_ttl


# ---------------------------------------------------------------------------
# get_stats completeness
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_stats_structure(self, cache, embedding):
        cache.store_embedding("k", embedding)
        stats = cache.get_stats()
        assert "memory_cache" in stats
        assert "redis_cache" in stats
        assert "disk_cache" in stats
        assert "statistics" in stats

    def test_hit_rate_calculation(self, cache, embedding):
        cache.store_embedding("k", embedding)
        cache.get_embedding("k")   # hit
        cache.get_embedding("x")   # miss
        stats = cache.get_stats()["statistics"]
        assert stats["hit_rate_percent"] == 50.0

    def test_zero_requests_hit_rate_is_zero(self, cache):
        stats = cache.get_stats()["statistics"]
        assert stats["hit_rate_percent"] == 0.0

    def test_memory_cache_entry_count(self, cache, embedding):
        cache.store_embedding("k1", embedding)
        cache.store_embedding("k2", make_embedding(seed=1))
        stats = cache.get_stats()["memory_cache"]
        assert stats["entries"] == 2
