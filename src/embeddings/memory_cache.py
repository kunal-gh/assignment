"""In-memory cache for embeddings (Redis-free alternative)."""

import hashlib
import logging
import pickle
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MemoryCache:
    """
    Simple in-memory LRU cache for embeddings.

    Replaces Redis for 100% free deployment.
    Uses OrderedDict for LRU eviction policy.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: dict = {}
        self.hits = 0
        self.misses = 0

        logger.info(f"MemoryCache initialized: max_size={max_size}, ttl={ttl}s")

    def _generate_key(self, text: str, model_name: str = "") -> str:
        """Generate cache key from text and model name."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model_name: str = "") -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            text: Text to look up
            model_name: Model name for key generation

        Returns:
            Cached embedding or None if not found/expired
        """
        key = self._generate_key(text, model_name)

        # Check if key exists
        if key not in self.cache:
            self.misses += 1
            return None

        # Check if expired
        if key in self.timestamps:
            age = time.time() - self.timestamps[key]
            if age > self.ttl:
                # Expired, remove it
                del self.cache[key]
                del self.timestamps[key]
                self.misses += 1
                return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1

        # Deserialize and return
        try:
            embedding = pickle.loads(self.cache[key])  # nosec B301
            return embedding
        except Exception as e:
            logger.error(f"Cache deserialization error: {str(e)}")
            del self.cache[key]
            if key in self.timestamps:
                del self.timestamps[key]
            self.misses += 1
            return None

    def set(self, text: str, embedding: np.ndarray, model_name: str = "") -> bool:
        """
        Store embedding in cache.

        Args:
            text: Text key
            embedding: Embedding vector to cache
            model_name: Model name for key generation

        Returns:
            True if successfully cached
        """
        key = self._generate_key(text, model_name)

        try:
            # Serialize embedding
            serialized = pickle.dumps(embedding)

            # Check size limit
            if len(self.cache) >= self.max_size:
                # Remove oldest item (LRU eviction)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.timestamps:
                    del self.timestamps[oldest_key]
                logger.debug(f"Cache full, evicted oldest item")

            # Store in cache
            self.cache[key] = serialized
            self.timestamps[key] = time.time()

            return True

        except Exception as e:
            logger.error(f"Cache serialization error: {str(e)}")
            return False

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.timestamps.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl": self.ttl,
        }

    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self.cache)

    def __contains__(self, text: str) -> bool:
        """Check if text is in cache."""
        key = self._generate_key(text)
        return key in self.cache


# Global cache instance
_global_cache: Optional[MemoryCache] = None


def get_cache() -> MemoryCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MemoryCache(max_size=1000, ttl=3600)
    return _global_cache


def clear_cache():
    """Clear global cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
