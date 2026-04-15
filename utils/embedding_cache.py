"""Thread-safe LRU cache for ArcFace face embeddings.

Caches embeddings keyed by the SHA-256 hash of the raw image bytes so that
repeated calls with the same image avoid redundant model inference.

Usage::

    cache = EmbeddingCache(max_size=128)
    key = image_array.tobytes()

    embedding = cache.get(key)
    if embedding is None:
        embedding = recognizer.get_embedding(image)
        cache.put(key, embedding)

    info = cache.cache_info()
    print(f"hits={info['hits']} misses={info['misses']} size={info['size']}")
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """LRU cache for face embeddings backed by an OrderedDict.

    Keys are SHA-256 hex digests of the raw image bytes (obtained via
    ``numpy_array.tobytes()``).  Values are 1-D float32 numpy arrays.

    Args:
        max_size: Maximum number of embeddings to store before evicting
                  the least-recently-used entry (default: 128).
    """

    def __init__(self, max_size: int = 128) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Look up the cached embedding for *image_bytes*.

        Moves the entry to the *most-recently-used* position on a hit.

        Args:
            image_bytes: Raw image bytes — typically ``numpy_array.tobytes()``.

        Returns:
            A copy of the cached embedding array, or ``None`` on a miss.
        """
        key = self._hash(image_bytes)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key].copy()
            self._misses += 1
            return None

    def put(self, image_bytes: bytes, embedding: np.ndarray) -> None:
        """Store an embedding in the cache.

        If the cache is full the least-recently-used entry is evicted first.

        Args:
            image_bytes: Raw image bytes used to derive the cache key.
            embedding:   1-D float32 numpy array to store.
        """
        key = self._hash(image_bytes)
        with self._lock:
            if key in self._cache:
                # Refresh existing entry
                self._cache.move_to_end(key)
                self._cache[key] = embedding.copy()
                return
            # Evict LRU entry if at capacity
            if len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"[EmbeddingCache] Evicted LRU entry key={evicted_key[:12]}…")
            self._cache[key] = embedding.copy()

    def clear(self) -> None:
        """Remove all entries and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
        logger.debug("[EmbeddingCache] Cache cleared.")

    def cache_info(self) -> Dict[str, int]:
        """Return hit/miss statistics and current cache size.

        Returns:
            Dictionary with keys 'hits', 'misses', 'size', 'max_size'.
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "max_size": self._max_size,
            }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of entries currently stored in the cache."""
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        info = self.cache_info()
        return (
            f"EmbeddingCache("
            f"size={info['size']}/{info['max_size']}, "
            f"hits={info['hits']}, "
            f"misses={info['misses']})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(image_bytes: bytes) -> str:
        """Return the SHA-256 hex digest of *image_bytes*.

        Args:
            image_bytes: Raw bytes to hash.

        Returns:
            64-character hex string.
        """
        return hashlib.sha256(image_bytes).hexdigest()
