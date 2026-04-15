"""Unit tests for utils.embedding_cache.EmbeddingCache.

No model loading is required — tests use random numpy arrays as mock
embeddings.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from utils.embedding_cache import EmbeddingCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_embedding(dim: int = 512) -> np.ndarray:
    """Return a random float32 embedding vector."""
    return np.random.rand(dim).astype(np.float32)


def _rand_image_bytes(shape: tuple = (64, 64, 3)) -> bytes:
    """Return random bytes simulating raw image data."""
    return np.random.randint(0, 256, shape, dtype=np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbeddingCacheBasic:
    """Basic get/put/miss behaviour."""

    def test_miss_on_empty_cache(self):
        """A fresh cache returns None for any lookup."""
        cache = EmbeddingCache(max_size=10)
        result = cache.get(_rand_image_bytes())
        assert result is None

    def test_put_then_get_returns_same_values(self):
        """After put(), get() should return an embedding with equal values."""
        cache = EmbeddingCache(max_size=10)
        img_bytes = _rand_image_bytes()
        emb = _rand_embedding()
        cache.put(img_bytes, emb)
        retrieved = cache.get(img_bytes)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, emb)

    def test_get_returns_copy_not_reference(self):
        """Mutating the returned embedding must not corrupt the cached value."""
        cache = EmbeddingCache(max_size=10)
        img_bytes = _rand_image_bytes()
        emb = _rand_embedding()
        cache.put(img_bytes, emb)
        retrieved = cache.get(img_bytes)
        assert retrieved is not None
        retrieved[:] = 0.0
        # Cache should still hold original values
        second = cache.get(img_bytes)
        assert second is not None
        np.testing.assert_array_almost_equal(second, emb)

    def test_different_images_have_independent_entries(self):
        """Two different image byte sequences should map to different keys."""
        cache = EmbeddingCache(max_size=10)
        img_a = _rand_image_bytes()
        img_b = _rand_image_bytes()
        emb_a = _rand_embedding()
        emb_b = _rand_embedding()
        cache.put(img_a, emb_a)
        cache.put(img_b, emb_b)
        assert len(cache) == 2
        np.testing.assert_array_almost_equal(cache.get(img_a), emb_a)
        np.testing.assert_array_almost_equal(cache.get(img_b), emb_b)


class TestEmbeddingCacheHitMiss:
    """Hit and miss counter tracking."""

    def test_hit_miss_counters(self):
        """cache_info() should track hits and misses accurately."""
        cache = EmbeddingCache(max_size=10)
        img_bytes = _rand_image_bytes()
        emb = _rand_embedding()

        # Two misses
        cache.get(img_bytes)
        cache.get(img_bytes)
        # One put, then two hits
        cache.put(img_bytes, emb)
        cache.get(img_bytes)
        cache.get(img_bytes)

        info = cache.cache_info()
        assert info["hits"] == 2
        assert info["misses"] == 2
        assert info["size"] == 1

    def test_clear_resets_counters(self):
        """clear() should reset hit/miss counters and remove all entries."""
        cache = EmbeddingCache(max_size=10)
        img_bytes = _rand_image_bytes()
        cache.put(img_bytes, _rand_embedding())
        cache.get(img_bytes)  # hit
        cache.clear()
        info = cache.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["size"] == 0
        assert len(cache) == 0


class TestEmbeddingCacheLRUEviction:
    """LRU eviction behaviour when max_size is reached."""

    def test_lru_eviction_removes_oldest_entry(self):
        """When the cache is full, the LRU entry should be evicted."""
        cache = EmbeddingCache(max_size=3)
        images = [_rand_image_bytes() for _ in range(3)]
        embeddings = [_rand_embedding() for _ in range(3)]

        for img, emb in zip(images, embeddings):
            cache.put(img, emb)

        assert len(cache) == 3

        # Access images[1] and images[2] to make images[0] the LRU
        cache.get(images[1])
        cache.get(images[2])

        # Adding a 4th entry should evict images[0]
        new_img = _rand_image_bytes()
        new_emb = _rand_embedding()
        cache.put(new_img, new_emb)

        assert len(cache) == 3
        assert cache.get(images[0]) is None, "LRU entry should have been evicted"
        assert cache.get(images[1]) is not None
        assert cache.get(images[2]) is not None
        assert cache.get(new_img) is not None

    def test_max_size_one(self):
        """A cache with max_size=1 should only ever hold one entry."""
        cache = EmbeddingCache(max_size=1)
        img_a = _rand_image_bytes()
        img_b = _rand_image_bytes()
        cache.put(img_a, _rand_embedding())
        cache.put(img_b, _rand_embedding())
        assert len(cache) == 1
        assert cache.get(img_a) is None

    def test_invalid_max_size_raises(self):
        """max_size < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            EmbeddingCache(max_size=0)


class TestEmbeddingCacheThreadSafety:
    """Thread-safety verification under concurrent access."""

    def test_concurrent_puts_do_not_exceed_max_size(self):
        """Concurrent puts from multiple threads must not exceed max_size."""
        max_size = 20
        cache = EmbeddingCache(max_size=max_size)
        errors: list = []

        def worker():
            try:
                for _ in range(50):
                    cache.put(_rand_image_bytes(), _rand_embedding())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert len(cache) <= max_size

    def test_concurrent_get_and_put(self):
        """Concurrent reads and writes must not raise exceptions."""
        cache = EmbeddingCache(max_size=50)
        shared_img = _rand_image_bytes()
        shared_emb = _rand_embedding()
        cache.put(shared_img, shared_emb)
        errors: list = []

        def reader():
            try:
                for _ in range(100):
                    cache.get(shared_img)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def writer():
            try:
                for _ in range(100):
                    cache.put(_rand_image_bytes(), _rand_embedding())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = (
            [threading.Thread(target=reader) for _ in range(5)]
            + [threading.Thread(target=writer) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"


class TestEmbeddingCacheDunder:
    """__len__ and __repr__ behaviour."""

    def test_len_reflects_cache_size(self):
        cache = EmbeddingCache(max_size=5)
        assert len(cache) == 0
        cache.put(_rand_image_bytes(), _rand_embedding())
        assert len(cache) == 1

    def test_repr_contains_key_info(self):
        cache = EmbeddingCache(max_size=8)
        r = repr(cache)
        assert "EmbeddingCache" in r
        assert "8" in r  # max_size
