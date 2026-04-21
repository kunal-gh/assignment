"""Cache management for embeddings."""

import hashlib
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of embeddings using Redis (primary), memory, and disk (fallback)."""

    # Default TTL for Redis cache entries (7 days)
    DEFAULT_REDIS_TTL = 60 * 60 * 24 * 7
    # Prefix for all Redis keys managed by this class
    REDIS_KEY_PREFIX = "embedding:"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_cache: int = 1000,
        redis_url: Optional[str] = None,
        redis_ttl: int = DEFAULT_REDIS_TTL,
    ):
        """
        Initialize cache manager.

        The cache hierarchy is:
          1. In-memory LRU cache (fastest)
          2. Redis cache (fast, shared across processes, optional)
          3. Disk cache (persistent fallback)

        Args:
            cache_dir: Directory for disk cache
            max_memory_cache: Maximum number of embeddings to keep in memory
            redis_url: Redis connection URL (e.g. "redis://localhost:6379/0").
                       If None, Redis is disabled and only memory/disk caches are used.
            redis_ttl: Time-to-live in seconds for Redis cache entries.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "resume-screener" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_cache = max_memory_cache
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        self._cache_stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "memory_evictions": 0,
            "redis_hits": 0,
            "redis_stores": 0,
            "redis_errors": 0,
        }

        # Redis setup
        self.redis_ttl = redis_ttl
        self._redis: Optional[Any] = None
        self._redis_available = False
        if redis_url:
            self._connect_redis(redis_url)

        # Load disk cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()

        logger.info(
            f"CacheManager initialized | cache_dir={self.cache_dir} "
            f"| redis={'enabled' if self._redis_available else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # Redis connection helpers
    # ------------------------------------------------------------------

    def _connect_redis(self, redis_url: str) -> None:
        """Attempt to connect to Redis; silently disable on failure."""
        if not REDIS_AVAILABLE:
            logger.warning("redis package is not installed. Redis caching is disabled. " "Install it with: pip install redis")
            return
        try:
            client = redis.Redis.from_url(redis_url, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            self._redis_available = True
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as exc:
            logger.warning(f"Could not connect to Redis ({exc}). " "Falling back to memory/disk cache only.")
            self._redis = None
            self._redis_available = False

    # ------------------------------------------------------------------
    # Numpy ↔ bytes serialisation for Redis
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialise a numpy array to bytes for Redis storage."""
        return pickle.dumps({"array": embedding, "dtype": str(embedding.dtype), "shape": embedding.shape})

    @staticmethod
    def _deserialize_embedding(data: bytes) -> Optional[np.ndarray]:
        """Deserialise bytes from Redis back to a numpy array."""
        try:
            obj = pickle.loads(data)  # nosec B301
            arr = obj["array"]
            if not isinstance(arr, np.ndarray):
                return None
            return arr
        except Exception as exc:
            logger.warning(f"Failed to deserialise embedding from Redis: {exc}")
            return None

    def get_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.

        Lookup order: memory → Redis → disk.
        On a Redis or disk hit the result is promoted to the memory cache.

        Args:
            cache_key: Unique key for the embedding

        Returns:
            Cached embedding array or None if not found (cache miss)
        """
        # 1. Memory cache
        if cache_key in self._memory_cache:
            self._cache_access_times[cache_key] = datetime.now()
            self._cache_stats["hits"] += 1
            logger.debug(f"Memory cache hit: {cache_key}")
            return self._memory_cache[cache_key]

        # 2. Redis cache
        if self._redis_available:
            redis_result = self._get_from_redis(cache_key)
            if redis_result is not None:
                self._store_in_memory(cache_key, redis_result)
                self._cache_stats["hits"] += 1
                self._cache_stats["redis_hits"] += 1
                logger.debug(f"Redis cache hit: {cache_key}")
                return redis_result

        # 3. Disk cache
        disk_result = self._load_from_disk(cache_key)
        if disk_result is not None:
            self._store_in_memory(cache_key, disk_result)
            # Backfill Redis so future lookups are faster
            if self._redis_available:
                self._store_in_redis(cache_key, disk_result)
            self._cache_stats["hits"] += 1
            logger.debug(f"Disk cache hit: {cache_key}")
            return disk_result

        # Cache miss
        self._cache_stats["misses"] += 1
        logger.debug(f"Cache miss: {cache_key}")
        return None

    def store_embedding(self, cache_key: str, embedding: np.ndarray) -> None:
        """
        Store embedding in all available cache layers.

        Args:
            cache_key: Unique key for the embedding
            embedding: Embedding array to store
        """
        try:
            self._store_in_memory(cache_key, embedding)

            if self._redis_available:
                self._store_in_redis(cache_key, embedding)

            self._store_on_disk(cache_key, embedding)

            self._cache_stats["stores"] += 1
            logger.debug(f"Stored embedding: {cache_key}")

        except Exception as exc:
            logger.error(f"Failed to store embedding ({cache_key}): {exc}")

    def _store_in_memory(self, cache_key: str, embedding: np.ndarray):
        """Store embedding in memory cache with LRU eviction."""
        # Check if we need to evict old entries
        if len(self._memory_cache) >= self.max_memory_cache:
            self._evict_lru_entries()

        # Store the embedding
        self._memory_cache[cache_key] = embedding.copy()
        self._cache_access_times[cache_key] = datetime.now()

    def _evict_lru_entries(self):
        """Evict least recently used entries from memory cache."""
        if not self._cache_access_times:
            return

        # Find the least recently used entry
        lru_key = min(self._cache_access_times.keys(), key=lambda k: self._cache_access_times[k])

        # Remove from memory cache
        if lru_key in self._memory_cache:
            del self._memory_cache[lru_key]
        if lru_key in self._cache_access_times:
            del self._cache_access_times[lru_key]

        self._cache_stats["memory_evictions"] += 1
        logger.debug(f"Evicted LRU entry from memory cache: {lru_key}")

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    def _redis_key(self, cache_key: str) -> str:
        """Return the namespaced Redis key for a given cache key."""
        return f"{self.REDIS_KEY_PREFIX}{cache_key}"

    def _get_from_redis(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve an embedding from Redis; returns None on miss or error."""
        try:
            data = self._redis.get(self._redis_key(cache_key))
            if data is None:
                return None
            return self._deserialize_embedding(data)
        except Exception as exc:
            self._cache_stats["redis_errors"] += 1
            logger.warning(f"Redis GET error for key {cache_key}: {exc}")
            return None

    def _store_in_redis(self, cache_key: str, embedding: np.ndarray) -> None:
        """Store an embedding in Redis with TTL; silently handles errors."""
        try:
            data = self._serialize_embedding(embedding)
            self._redis.setex(self._redis_key(cache_key), self.redis_ttl, data)
            self._cache_stats["redis_stores"] += 1
            logger.debug(f"Stored in Redis (TTL={self.redis_ttl}s): {cache_key}")
        except Exception as exc:
            self._cache_stats["redis_errors"] += 1
            logger.warning(f"Redis SET error for key {cache_key}: {exc}")

    def _delete_from_redis(self, cache_key: str) -> None:
        """Delete a single key from Redis; silently handles errors."""
        if not self._redis_available:
            return
        try:
            self._redis.delete(self._redis_key(cache_key))
        except Exception as exc:
            self._cache_stats["redis_errors"] += 1
            logger.warning(f"Redis DEL error for key {cache_key}: {exc}")

    def _flush_redis(self) -> None:
        """Delete all embedding keys managed by this instance from Redis."""
        if not self._redis_available:
            return
        try:
            pattern = f"{self.REDIS_KEY_PREFIX}*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
            logger.debug(f"Flushed {len(keys)} keys from Redis")
        except Exception as exc:
            self._cache_stats["redis_errors"] += 1
            logger.warning(f"Redis flush error: {exc}")

    def _store_on_disk(self, cache_key: str, embedding: np.ndarray):
        """Store embedding on disk."""
        try:
            # Create filename from cache key
            filename = self._get_cache_filename(cache_key)
            filepath = self.cache_dir / filename

            # Store embedding as pickle file
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "embedding": embedding,
                        "timestamp": datetime.now().isoformat(),
                        "shape": embedding.shape,
                        "dtype": str(embedding.dtype),
                    },
                    f,
                )

            # Update metadata
            self._update_metadata(cache_key, filepath)

        except Exception as e:
            logger.error(f"Failed to store embedding on disk: {str(e)}")

    def _load_from_disk(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk."""
        try:
            filename = self._get_cache_filename(cache_key)
            filepath = self.cache_dir / filename

            if not filepath.exists():
                return None

            # Check if file is too old (optional TTL)
            if self._is_cache_expired(filepath):
                logger.debug(f"Cache file expired: {filepath}")
                filepath.unlink()  # Delete expired file
                return None

            # Load embedding
            with open(filepath, "rb") as f:
                data = pickle.load(f)  # nosec B301

            embedding = data["embedding"]

            # Validate embedding
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Invalid embedding type in cache: {type(embedding)}")
                return None

            return embedding

        except Exception as e:
            logger.error(f"Failed to load embedding from disk: {str(e)}")
            return None

    def _get_cache_filename(self, cache_key: str) -> str:
        """Generate filename for cache key."""
        # Use hash to create safe filename
        hash_obj = hashlib.sha256(cache_key.encode("utf-8"))
        return f"embedding_{hash_obj.hexdigest()[:16]}.pkl"

    def _is_cache_expired(self, filepath: Path, ttl_days: int = 30) -> bool:
        """Check if cache file is expired."""
        try:
            file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
            expiry_time = datetime.now() - timedelta(days=ttl_days)
            return file_time < expiry_time
        except Exception:
            return True  # Consider expired if we can't check

    def _load_metadata(self):
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {str(e)}")
            self._metadata = {}

    def _update_metadata(self, cache_key: str, filepath: Path):
        """Update cache metadata."""
        try:
            self._metadata[cache_key] = {
                "filename": filepath.name,
                "created": datetime.now().isoformat(),
                "size": filepath.stat().st_size,
            }

            # Save metadata
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update cache metadata: {str(e)}")

    def clear_cache(self):
        """Clear all cached embeddings from all layers."""
        try:
            # Clear memory cache
            self._memory_cache.clear()
            self._cache_access_times.clear()

            # Clear Redis cache
            self._flush_redis()

            # Clear disk cache
            for file_path in self.cache_dir.glob("embedding_*.pkl"):
                file_path.unlink()

            # Clear metadata
            self._metadata.clear()
            if self.metadata_file.exists():
                self.metadata_file.unlink()

            logger.info("Cleared all cached embeddings")

        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")

    def cleanup_expired(self, ttl_days: int = 30):
        """Remove expired cache files."""
        try:
            removed_count = 0

            for file_path in self.cache_dir.glob("embedding_*.pkl"):
                if self._is_cache_expired(file_path, ttl_days):
                    file_path.unlink()
                    removed_count += 1

            # Clean up metadata for removed files
            valid_files = {f.name for f in self.cache_dir.glob("embedding_*.pkl")}
            self._metadata = {k: v for k, v in self._metadata.items() if v.get("filename") in valid_files}

            # Save updated metadata
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)

            logger.info(f"Cleaned up {removed_count} expired cache files")

        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics across all layers."""
        try:
            memory_size = sum(embedding.nbytes for embedding in self._memory_cache.values())

            disk_size = 0
            disk_files = 0
            for file_path in self.cache_dir.glob("embedding_*.pkl"):
                disk_size += file_path.stat().st_size
                disk_files += 1

            total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
            hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0

            return {
                "memory_cache": {
                    "entries": len(self._memory_cache),
                    "max_entries": self.max_memory_cache,
                    "size_bytes": memory_size,
                    "size_mb": round(memory_size / (1024 * 1024), 2),
                },
                "redis_cache": {
                    "enabled": self._redis_available,
                    "ttl_seconds": self.redis_ttl,
                    "hits": self._cache_stats["redis_hits"],
                    "stores": self._cache_stats["redis_stores"],
                    "errors": self._cache_stats["redis_errors"],
                },
                "disk_cache": {
                    "files": disk_files,
                    "size_bytes": disk_size,
                    "size_mb": round(disk_size / (1024 * 1024), 2),
                    "cache_dir": str(self.cache_dir),
                },
                "statistics": {
                    "hits": self._cache_stats["hits"],
                    "misses": self._cache_stats["misses"],
                    "stores": self._cache_stats["stores"],
                    "memory_evictions": self._cache_stats["memory_evictions"],
                    "hit_rate_percent": round(hit_rate, 2),
                },
            }

        except Exception as exc:
            logger.error(f"Failed to get cache stats: {exc}")
            return {"error": str(exc)}

    def get_cache_keys(self) -> list:
        """Get list of all cache keys."""
        keys = set()

        # Add memory cache keys
        keys.update(self._memory_cache.keys())

        # Add disk cache keys from metadata
        keys.update(self._metadata.keys())

        return list(keys)

    def remove_cache_entry(self, cache_key: str) -> bool:
        """Remove a specific cache entry from all layers."""
        try:
            removed = False

            # Remove from memory cache
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
                removed = True

            if cache_key in self._cache_access_times:
                del self._cache_access_times[cache_key]

            # Remove from Redis
            self._delete_from_redis(cache_key)

            # Remove from disk cache
            if cache_key in self._metadata:
                filename = self._metadata[cache_key]["filename"]
                filepath = self.cache_dir / filename
                if filepath.exists():
                    filepath.unlink()
                    removed = True

                del self._metadata[cache_key]

                # Update metadata file
                with open(self.metadata_file, "w") as f:
                    json.dump(self._metadata, f, indent=2)

            if removed:
                logger.debug(f"Removed cache entry: {cache_key}")

            return removed

        except Exception as exc:
            logger.error(f"Failed to remove cache entry: {exc}")
            return False
