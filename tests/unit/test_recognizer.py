# Unit tests for:
#   - FaceAttribute dataclass
#   - FaceEmbedding dataclass
#   - FaceMatch dataclass
#   - RecognitionResult dataclass
#   - cosine_similarity / cosine_similarity_matrix utilities
#   - average_embeddings utility
#   - BaseRecognizer abstract class
#   - FaceIdentity dataclass
#   - FaceDatabase  (register, search, persistence, management)
#
# All tests are pure unit tests — no real models or GPU required.

from __future__ import annotations

import pickle
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.recognizer.base_recognizer import (
    BaseRecognizer,
    FaceAttribute,
    FaceEmbedding,
    FaceMatch,
    RecognitionResult,
    average_embeddings,
    cosine_similarity,
    cosine_similarity_matrix,
)
from core.recognizer.face_database import FaceDatabase, FaceIdentity, SearchResult


def _rand_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    """Return a random unit-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_embedding(
    dim: int = 512,
    seed: int = 0,
    face_index: int = 0,
    source: Optional[str] = None,
) -> FaceEmbedding:
    """Create a FaceEmbedding with a deterministic random vector."""
    return FaceEmbedding(
        vector=_rand_vec(dim, seed),
        face_index=face_index,
        source_path=source,
    )


def _near_vec(base: np.ndarray, noise: float = 0.05) -> np.ndarray:
    """Return a vector close to *base* (simulates same person, different photo)."""
    rng = np.random.default_rng(42)
    v = base + rng.standard_normal(base.shape).astype(np.float32) * noise
    return v / np.linalg.norm(v)


def _far_vec(dim: int = 512, seed: int = 99) -> np.ndarray:
    """Return a vector orthogonal / dissimilar to common ones."""
    return _rand_vec(dim, seed)


@pytest.fixture
def vec_alice() -> np.ndarray:
    return _rand_vec(512, seed=1)


@pytest.fixture
def vec_bob() -> np.ndarray:
    return _rand_vec(512, seed=2)


@pytest.fixture
def vec_charlie() -> np.ndarray:
    return _rand_vec(512, seed=3)


@pytest.fixture
def emb_alice(vec_alice) -> FaceEmbedding:
    return FaceEmbedding(vector=vec_alice, face_index=0, source_path="alice.jpg")


@pytest.fixture
def emb_bob(vec_bob) -> FaceEmbedding:
    return FaceEmbedding(vector=vec_bob, face_index=1, source_path="bob.jpg")


@pytest.fixture
def emb_charlie(vec_charlie) -> FaceEmbedding:
    return FaceEmbedding(vector=vec_charlie, face_index=2, source_path="charlie.jpg")


@pytest.fixture
def populated_db(vec_alice, vec_bob, vec_charlie) -> FaceDatabase:
    """A FaceDatabase pre-loaded with Alice, Bob, Charlie."""
    db = FaceDatabase(similarity_threshold=0.45)
    db.register("Alice", FaceEmbedding(vector=vec_alice))
    db.register("Bob", FaceEmbedding(vector=vec_bob))
    db.register("Charlie", FaceEmbedding(vector=vec_charlie))
    return db


@pytest.fixture
def empty_db() -> FaceDatabase:
    return FaceDatabase(similarity_threshold=0.45)


class TestFaceAttribute:

    def test_default_fields_none(self):
        fa = FaceAttribute()
        assert fa.age is None
        assert fa.gender is None
        assert fa.gender_score is None
        assert fa.embedding_norm is None

    def test_is_male_true(self):
        fa = FaceAttribute(gender="M")
        assert fa.is_male is True
        assert fa.is_female is False

    def test_is_female_true(self):
        fa = FaceAttribute(gender="F")
        assert fa.is_female is True
        assert fa.is_male is False

    def test_is_male_none_when_gender_none(self):
        fa = FaceAttribute()
        assert fa.is_male is None
        assert fa.is_female is None

    def test_gender_case_insensitive(self):
        fa = FaceAttribute(gender="m")
        assert fa.is_male is True

    def test_repr_with_age_gender(self):
        fa = FaceAttribute(age=25.0, gender="F", gender_score=0.92)
        r = repr(fa)
        assert "age=25.0" in r
        assert "gender=F" in r

    def test_repr_unknown(self):
        fa = FaceAttribute()
        assert "unknown" in repr(fa)

    def test_full_construction(self):
        fa = FaceAttribute(age=30.5, gender="M", gender_score=0.88, embedding_norm=1.0)
        assert fa.age == pytest.approx(30.5)
        assert fa.gender_score == pytest.approx(0.88)
        assert fa.embedding_norm == pytest.approx(1.0)


class TestFaceEmbedding:

    def test_dim(self, emb_alice):
        assert emb_alice.dim == 512

    def test_norm_unit_vector(self, emb_alice):
        assert emb_alice.norm == pytest.approx(1.0, abs=1e-5)

    def test_is_normalised(self, emb_alice):
        assert emb_alice.is_normalised is True

    def test_not_normalised(self):
        v = np.ones(512, dtype=np.float32)  # large norm
        emb = FaceEmbedding(vector=v)
        assert emb.is_normalised is False

    def test_normalise_returns_unit(self):
        v = np.ones(512, dtype=np.float32)
        emb = FaceEmbedding(vector=v)
        normalised = emb.normalise()
        assert normalised.norm == pytest.approx(1.0, abs=1e-5)

    def test_normalise_zero_vector_returns_self(self):
        v = np.zeros(512, dtype=np.float32)
        emb = FaceEmbedding(vector=v)
        result = emb.normalise()
        assert result is emb

    def test_normalise_preserves_metadata(self, emb_alice):
        normalised = emb_alice.normalise()
        assert normalised.face_index == emb_alice.face_index
        assert normalised.source_path == emb_alice.source_path

    def test_cosine_similarity_self(self, emb_alice):
        sim = emb_alice.cosine_similarity(emb_alice)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_different(self, emb_alice, emb_bob):
        sim = emb_alice.cosine_similarity(emb_bob)
        # Random unit vectors in 512-D are nearly orthogonal
        assert -1.0 <= sim <= 1.0
        assert sim < 0.5  # should be low for random vectors

    def test_cosine_similarity_symmetric(self, emb_alice, emb_bob):
        assert emb_alice.cosine_similarity(emb_bob) == pytest.approx(
            emb_bob.cosine_similarity(emb_alice), abs=1e-6
        )

    def test_euclidean_distance_self(self, emb_alice):
        dist = emb_alice.euclidean_distance(emb_alice)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_euclidean_distance_positive(self, emb_alice, emb_bob):
        dist = emb_alice.euclidean_distance(emb_bob)
        assert dist > 0.0

    def test_euclidean_distance_symmetric(self, emb_alice, emb_bob):
        assert emb_alice.euclidean_distance(emb_bob) == pytest.approx(
            emb_bob.euclidean_distance(emb_alice), abs=1e-6
        )

    def test_as_list(self, emb_alice):
        lst = emb_alice.as_list()
        assert isinstance(lst, list)
        assert len(lst) == 512
        assert isinstance(lst[0], float)

    def test_face_index_stored(self):
        emb = FaceEmbedding(vector=_rand_vec(), face_index=3)
        assert emb.face_index == 3

    def test_source_path_stored(self):
        emb = FaceEmbedding(vector=_rand_vec(), source_path="test.png")
        assert emb.source_path == "test.png"

    def test_landmarks_default_none(self, emb_alice):
        assert emb_alice.landmarks is None

    def test_bbox_default_none(self, emb_alice):
        assert emb_alice.bbox is None

    def test_repr_contains_dim(self, emb_alice):
        r = repr(emb_alice)
        assert "512" in r
        assert "FaceEmbedding" in r

    def test_attributes_stored(self):
        attr = FaceAttribute(age=28.0, gender="F")
        emb = FaceEmbedding(vector=_rand_vec(), attributes=attr)
        assert emb.attributes is not None
        assert emb.attributes.age == pytest.approx(28.0)


class TestFaceMatch:

    def test_known_match(self):
        m = FaceMatch(
            identity="Alice",
            similarity=0.82,
            distance=0.12,
            is_known=True,
            threshold=0.45,
        )
        assert m.is_known is True
        assert m.identity == "Alice"
        assert m.label == "Alice"

    def test_unknown_match(self):
        m = FaceMatch(
            identity=None,
            similarity=0.30,
            distance=0.80,
            is_known=False,
            threshold=0.45,
        )
        assert m.is_known is False
        assert m.identity is None
        assert m.label == "Unknown"

    def test_label_unknown_when_identity_none(self):
        m = FaceMatch(identity=None, similarity=0.2, distance=1.0)
        assert m.label == "Unknown"

    def test_label_unknown_when_not_known(self):
        m = FaceMatch(identity="Alice", similarity=0.2, distance=1.0, is_known=False)
        assert m.label == "Unknown"

    def test_confidence_pct(self):
        m = FaceMatch(identity="Bob", similarity=0.75, distance=0.5, is_known=True)
        assert m.confidence_pct == pytest.approx(75.0, abs=0.01)

    def test_face_index_default_zero(self):
        m = FaceMatch(identity="X", similarity=0.9, distance=0.1)
        assert m.face_index == 0

    def test_face_index_set(self):
        m = FaceMatch(identity="X", similarity=0.9, distance=0.1, face_index=2)
        assert m.face_index == 2

    def test_repr_contains_key_info(self):
        m = FaceMatch(identity="Alice", similarity=0.82, distance=0.12, is_known=True)
        r = repr(m)
        assert "Alice" in r
        assert "0.8200" in r
        assert "FaceMatch" in r


class TestRecognitionResult:

    @pytest.fixture
    def result_with_matches(self) -> RecognitionResult:
        matches = [
            FaceMatch(
                identity="Alice", similarity=0.85, distance=0.10, face_index=0, is_known=True
            ),
            FaceMatch(identity=None, similarity=0.30, distance=0.90, face_index=1, is_known=False),
            FaceMatch(
                identity="Charlie", similarity=0.72, distance=0.25, face_index=2, is_known=True
            ),
        ]
        return RecognitionResult(
            matches=matches,
            image_width=640,
            image_height=480,
            inference_time_ms=20.0,
        )

    def test_num_faces(self, result_with_matches):
        assert result_with_matches.num_faces == 3

    def test_known_faces(self, result_with_matches):
        known = result_with_matches.known_faces
        assert len(known) == 2
        names = [m.identity for m in known]
        assert "Alice" in names
        assert "Charlie" in names

    def test_unknown_faces(self, result_with_matches):
        unknown = result_with_matches.unknown_faces
        assert len(unknown) == 1
        assert unknown[0].identity is None

    def test_identities_list(self, result_with_matches):
        ids = result_with_matches.identities
        assert "Alice" in ids
        assert "Unknown" in ids
        assert "Charlie" in ids

    def test_is_empty_false(self, result_with_matches):
        assert result_with_matches.is_empty is False

    def test_is_empty_true(self):
        r = RecognitionResult(matches=[], image_width=640, image_height=480)
        assert r.is_empty is True

    def test_get_match_valid(self, result_with_matches):
        m = result_with_matches.get_match(face_index=1)
        assert m is not None
        assert m.face_index == 1

    def test_get_match_not_found(self, result_with_matches):
        m = result_with_matches.get_match(face_index=99)
        assert m is None

    def test_repr(self, result_with_matches):
        r = repr(result_with_matches)
        assert "RecognitionResult" in r
        assert "num_faces=3" in r


class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = _rand_vec(seed=10)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = _rand_vec(seed=11)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        a = _rand_vec(seed=12)
        b = _rand_vec(seed=13)
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a), abs=1e-6)

    def test_result_in_range(self):
        for seed in range(20):
            a = _rand_vec(seed=seed)
            b = _rand_vec(seed=seed + 100)
            sim = cosine_similarity(a, b)
            assert -1.0 <= sim <= 1.0

    def test_shape_mismatch_raises(self):
        a = np.ones(512, dtype=np.float32)
        b = np.ones(256, dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity(a, b)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(512, dtype=np.float32)
        b = _rand_vec(seed=5)
        assert cosine_similarity(a, b) == pytest.approx(0.0)


class TestCosineSimilarityMatrix:

    def test_shape(self):
        Q = np.random.randn(3, 512).astype(np.float32)
        G = np.random.randn(5, 512).astype(np.float32)
        M = cosine_similarity_matrix(Q, G)
        assert M.shape == (3, 5)

    def test_self_similarity_diagonal(self):
        V = np.array([_rand_vec(seed=i) for i in range(4)])
        M = cosine_similarity_matrix(V, V)
        np.testing.assert_allclose(np.diag(M), np.ones(4), atol=1e-5)

    def test_values_in_range(self):
        Q = np.random.randn(4, 128).astype(np.float32)
        G = np.random.randn(6, 128).astype(np.float32)
        M = cosine_similarity_matrix(Q, G)
        assert M.min() >= -1.0 - 1e-5
        assert M.max() <= 1.0 + 1e-5

    def test_1d_input_treated_as_single_row(self):
        q = _rand_vec(seed=0)
        G = np.array([_rand_vec(seed=i) for i in range(3)])
        M = cosine_similarity_matrix(q, G)
        assert M.shape == (1, 3)

    def test_dim_mismatch_raises(self):
        Q = np.random.randn(2, 512).astype(np.float32)
        G = np.random.randn(2, 256).astype(np.float32)
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity_matrix(Q, G)


class TestAverageEmbeddings:

    def test_single_embedding_returns_normalised(self):
        v = np.ones(512, dtype=np.float32)
        result = average_embeddings([v])
        assert result.shape == (512,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_identical_embeddings_returns_same(self):
        v = _rand_vec(seed=1)
        result = average_embeddings([v, v, v])
        np.testing.assert_allclose(result, v, atol=1e-5)

    def test_average_is_normalised(self):
        vecs = [_rand_vec(seed=i) for i in range(5)]
        result = average_embeddings(vecs)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)

    def test_result_dtype_float32(self):
        vecs = [_rand_vec(seed=i) for i in range(3)]
        result = average_embeddings(vecs)
        assert result.dtype == np.float32

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            average_embeddings([])

    def test_different_shape_raises(self):
        a = np.ones(512, dtype=np.float32)
        b = np.ones(256, dtype=np.float32)
        with pytest.raises(Exception):
            average_embeddings([a, b])


class ConcreteRecognizer(BaseRecognizer):
    """Minimal concrete subclass for testing the abstract base."""

    def load_model(self) -> None:
        self._model = object()
        self._is_loaded = True

    def get_embedding(
        self,
        image: np.ndarray,
        bbox=None,
        landmarks=None,
    ) -> Optional[FaceEmbedding]:
        self._require_loaded()
        v = _rand_vec(seed=int(image.sum()) % 100)
        return FaceEmbedding(vector=v)


class TestBaseRecognizer:

    @pytest.fixture
    def recognizer(self) -> ConcreteRecognizer:
        return ConcreteRecognizer(
            model_pack="test_pack",
            similarity_threshold=0.50,
        )

    def test_initial_not_loaded(self, recognizer):
        assert recognizer.is_loaded is False

    def test_load_model_sets_flag(self, recognizer):
        recognizer.load_model()
        assert recognizer.is_loaded is True

    def test_threshold_stored(self, recognizer):
        assert recognizer.similarity_threshold == pytest.approx(0.50)

    def test_model_name(self, recognizer):
        assert recognizer.model_name == "test_pack"

    def test_require_loaded_raises_when_not_loaded(self, recognizer):
        with pytest.raises(RuntimeError, match="not loaded"):
            recognizer._require_loaded()

    def test_require_loaded_ok_after_load(self, recognizer):
        recognizer.load_model()
        recognizer._require_loaded()  # Should not raise

    def test_get_embedding_returns_result(self, recognizer):
        recognizer.load_model()
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        emb = recognizer.get_embedding(img)
        assert isinstance(emb, FaceEmbedding)
        assert emb.dim == 512

    def test_get_embeddings_batch(self, recognizer):
        recognizer.load_model()
        imgs = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(3)]
        results = recognizer.get_embeddings_batch(imgs)
        assert len(results) == 3
        assert all(isinstance(r, FaceEmbedding) for r in results)

    def test_get_embeddings_batch_requires_loaded(self, recognizer):
        imgs = [np.zeros((112, 112, 3), dtype=np.uint8)]
        with pytest.raises(RuntimeError):
            recognizer.get_embeddings_batch(imgs)

    def test_compare_self_returns_one(self, recognizer):
        recognizer.load_model()
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        emb = recognizer.get_embedding(img)
        assert recognizer.compare(emb, emb) == pytest.approx(1.0, abs=1e-5)

    def test_is_same_person_above_threshold(self, recognizer, emb_alice):
        recognizer.load_model()
        assert recognizer.is_same_person(emb_alice, emb_alice) is True

    def test_is_same_person_below_threshold(self, recognizer, emb_alice, emb_bob):
        recognizer.load_model()
        # Random vectors are typically far apart
        recognizer.similarity_threshold = 0.9999
        result = recognizer.is_same_person(emb_alice, emb_bob)
        assert isinstance(result, bool)

    def test_release_clears_model(self, recognizer):
        recognizer.load_model()
        recognizer.release()
        assert recognizer.is_loaded is False
        assert recognizer._model is None

    def test_context_manager_loads_on_enter(self, recognizer):
        with recognizer:
            assert recognizer.is_loaded is True

    def test_context_manager_releases_on_exit(self, recognizer):
        with recognizer:
            pass
        assert recognizer.is_loaded is False

    def test_context_manager_releases_on_exception(self, recognizer):
        try:
            with recognizer:
                raise ValueError("oops")
        except ValueError:
            pass
        assert recognizer.is_loaded is False

    def test_validate_image_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            BaseRecognizer._validate_image(None)

    def test_validate_image_wrong_type_raises(self):
        with pytest.raises(ValueError):
            BaseRecognizer._validate_image("not an array")

    def test_validate_image_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BaseRecognizer._validate_image(np.array([]))

    def test_repr(self, recognizer):
        r = repr(recognizer)
        assert "test_pack" in r
        assert "not loaded" in r


class TestFaceIdentity:

    def test_default_empty_embeddings(self):
        fi = FaceIdentity(name="Test")
        assert fi.num_embeddings == 0
        assert fi.mean_embedding is None

    def test_add_embedding(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        fi.add_embedding(vec_alice)
        assert fi.num_embeddings == 1

    def test_add_embedding_normalises(self):
        fi = FaceIdentity(name="Test")
        v = np.ones(512, dtype=np.float32)  # not unit-norm
        fi.add_embedding(v)
        stored = fi.embeddings[0]
        assert np.linalg.norm(stored) == pytest.approx(1.0, abs=1e-5)

    def test_mean_embedding_single_shot(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        fi.add_embedding(vec_alice)
        mean = fi.mean_embedding
        assert mean is not None
        assert mean.shape == (512,)
        assert np.linalg.norm(mean) == pytest.approx(1.0, abs=1e-5)

    def test_mean_embedding_multi_shot(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        noisy = _near_vec(vec_alice, noise=0.05)
        fi.add_embedding(vec_alice)
        fi.add_embedding(noisy)
        mean = fi.mean_embedding
        assert mean is not None
        assert np.linalg.norm(mean) == pytest.approx(1.0, abs=1e-5)

    def test_best_similarity_self(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        fi.add_embedding(vec_alice)
        sim = fi.best_similarity(vec_alice)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_best_similarity_empty_returns_minus_one(self):
        fi = FaceIdentity(name="Empty")
        assert fi.best_similarity(_rand_vec()) == pytest.approx(-1.0)

    def test_mean_similarity_self(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        fi.add_embedding(vec_alice)
        sim = fi.mean_similarity(vec_alice)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_mean_similarity_empty_returns_minus_one(self):
        fi = FaceIdentity(name="Empty")
        assert fi.mean_similarity(_rand_vec()) == pytest.approx(-1.0)

    def test_best_beats_mean_for_multi_shot(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        # Add alice + very different vector
        fi.add_embedding(vec_alice)
        far = _far_vec(seed=77)
        fi.add_embedding(far)
        best = fi.best_similarity(vec_alice)
        mean = fi.mean_similarity(vec_alice)
        # best should be >= mean for a good query matching one of the shots
        assert best >= mean - 0.01

    def test_identity_id_auto_generated(self):
        fi = FaceIdentity(name="Test")
        assert len(fi.identity_id) == 36  # UUID4 string length
        assert "-" in fi.identity_id

    def test_created_at_set(self):
        before = time.time()
        fi = FaceIdentity(name="Test")
        after = time.time()
        assert before <= fi.created_at <= after

    def test_updated_at_changes_on_add(self, vec_alice):
        fi = FaceIdentity(name="Alice")
        t0 = fi.updated_at
        fi.add_embedding(vec_alice)
        # updated_at should be >= the creation time (monotonic)
        assert fi.updated_at >= t0

    def test_repr(self):
        fi = FaceIdentity(name="Alice")
        r = repr(fi)
        assert "Alice" in r
        assert "FaceIdentity" in r


class TestFaceDatabaseRegister:

    def test_register_new_identity(self, empty_db, emb_alice):
        identity = empty_db.register("Alice", emb_alice)
        assert identity.name == "Alice"
        assert empty_db.count == 1

    def test_register_multiple_identities(self, empty_db, emb_alice, emb_bob):
        empty_db.register("Alice", emb_alice)
        empty_db.register("Bob", emb_bob)
        assert empty_db.count == 2

    def test_register_same_name_appends_shot(self, empty_db, vec_alice):
        emb1 = FaceEmbedding(vector=vec_alice)
        emb2 = FaceEmbedding(vector=_near_vec(vec_alice, 0.05))
        empty_db.register("Alice", emb1)
        empty_db.register("Alice", emb2)
        assert empty_db.count == 1
        identity = empty_db.get_identity("Alice")
        assert identity is not None
        assert identity.num_embeddings == 2

    def test_register_with_overwrite(self, empty_db, vec_alice):
        emb1 = FaceEmbedding(vector=vec_alice)
        emb2 = FaceEmbedding(vector=_near_vec(vec_alice, 0.1))
        empty_db.register("Alice", emb1)
        empty_db.register("Alice", emb2, overwrite=True)
        identity = empty_db.get_identity("Alice")
        assert identity is not None
        assert identity.num_embeddings == 1

    def test_register_raw_numpy_array(self, empty_db, vec_alice):
        empty_db.register("Alice", vec_alice)
        assert empty_db.count == 1

    def test_register_empty_name_raises(self, empty_db, emb_alice):
        with pytest.raises(ValueError, match="empty"):
            empty_db.register("", emb_alice)

    def test_register_whitespace_name_raises(self, empty_db, emb_alice):
        with pytest.raises(ValueError, match="empty"):
            empty_db.register("   ", emb_alice)

    def test_register_many(self, empty_db, vec_alice):
        vecs = [FaceEmbedding(vector=_near_vec(vec_alice, 0.05 * i)) for i in range(4)]
        identity = empty_db.register_many("Alice", vecs)
        assert identity.num_embeddings == 4

    def test_register_many_empty_raises(self, empty_db):
        with pytest.raises(ValueError, match="empty"):
            empty_db.register_many("Alice", [])

    def test_max_shots_cap(self, empty_db, vec_alice):
        db = FaceDatabase(similarity_threshold=0.45, max_embeddings_per_identity=3)
        for i in range(6):
            v = _near_vec(vec_alice, noise=0.01 * i)
            db.register("Alice", FaceEmbedding(vector=v))
        identity = db.get_identity("Alice")
        assert identity.num_embeddings == 3


class TestFaceDatabaseSearch:

    def test_search_known_identity(self, populated_db, vec_alice):
        # Query with alice's own vector — should match perfectly
        match = populated_db.search(FaceEmbedding(vector=vec_alice))
        assert match.is_known is True
        assert match.identity == "Alice"
        assert match.similarity == pytest.approx(1.0, abs=1e-4)

    def test_search_near_vector_matches(self, populated_db, vec_alice):
        near = _near_vec(vec_alice, noise=0.01)
        match = populated_db.search(FaceEmbedding(vector=near), threshold=0.60)
        assert match.is_known is True
        assert match.identity == "Alice"

    def test_search_unknown_returns_none_identity(self, populated_db):
        far = _far_vec(seed=999)
        match = populated_db.search(FaceEmbedding(vector=far), threshold=0.9999)
        assert match.is_known is False
        assert match.identity is None
        assert match.label == "Unknown"

    def test_search_empty_db_returns_unknown(self, empty_db, vec_alice):
        match = empty_db.search(FaceEmbedding(vector=vec_alice))
        assert match.is_known is False
        assert match.identity is None

    def test_search_raw_numpy_array(self, populated_db, vec_alice):
        match = populated_db.search(vec_alice)
        assert match.is_known is True
        assert match.identity == "Alice"

    def test_search_threshold_override(self, populated_db, vec_alice):
        # With threshold=1.0 nothing should match
        match = populated_db.search(FaceEmbedding(vector=vec_alice), threshold=1.0001)
        assert match.is_known is False

    def test_search_returns_face_index(self, populated_db, vec_alice):
        emb = FaceEmbedding(vector=vec_alice, face_index=7)
        match = populated_db.search(emb)
        assert match.face_index == 7

    def test_search_similarity_range(self, populated_db, vec_alice):
        match = populated_db.search(FaceEmbedding(vector=vec_alice))
        assert 0.0 <= match.similarity <= 1.0

    def test_search_distance_non_negative(self, populated_db, vec_alice):
        match = populated_db.search(FaceEmbedding(vector=vec_alice))
        assert match.distance >= 0.0

    def test_search_extended_returns_search_result(self, populated_db, vec_alice):
        result = populated_db.search_extended(FaceEmbedding(vector=vec_alice))
        assert isinstance(result, SearchResult)
        assert result.best_match.is_known is True

    def test_search_extended_all_matches_sorted(self, populated_db, vec_alice):
        result = populated_db.search_extended(FaceEmbedding(vector=vec_alice), top_k=3)
        sims = [sim for _, sim in result.all_matches]
        assert sims == sorted(sims, reverse=True)

    def test_search_extended_top_k_capped(self, populated_db, vec_alice):
        result = populated_db.search_extended(FaceEmbedding(vector=vec_alice), top_k=2)
        assert len(result.all_matches) <= 2

    def test_search_batch_returns_list(self, populated_db, vec_alice, vec_bob):
        queries = [
            FaceEmbedding(vector=vec_alice, face_index=0),
            FaceEmbedding(vector=vec_bob, face_index=1),
        ]
        results = populated_db.search_batch(queries)
        assert len(results) == 2
        assert results[0].identity == "Alice"
        assert results[1].identity == "Bob"

    def test_search_batch_empty_list(self, populated_db):
        results = populated_db.search_batch([])
        assert results == []

    def test_search_batch_empty_db(self, empty_db, vec_alice):
        results = empty_db.search_batch([FaceEmbedding(vector=vec_alice)])
        assert len(results) == 1
        assert results[0].is_known is False

    def test_search_mean_strategy(self, vec_alice):
        db = FaceDatabase(similarity_threshold=0.45, strategy="mean")
        noisy = _near_vec(vec_alice, noise=0.05)
        db.register("Alice", FaceEmbedding(vector=vec_alice))
        db.register("Alice", FaceEmbedding(vector=noisy))
        match = db.search(FaceEmbedding(vector=vec_alice))
        assert match.is_known is True
        assert match.identity == "Alice"

    def test_search_best_strategy_picks_closest_shot(self, vec_alice, vec_bob):
        db = FaceDatabase(similarity_threshold=0.45, strategy="best")
        db.register("Alice", FaceEmbedding(vector=vec_alice))
        db.register("Alice", FaceEmbedding(vector=vec_bob))  # 2nd shot is different
        # Querying with alice's exact vector should still match via best strategy
        match = db.search(FaceEmbedding(vector=vec_alice))
        assert match.identity == "Alice"
        assert match.similarity == pytest.approx(1.0, abs=1e-4)


class TestFaceDatabaseManagement:

    def test_remove_existing_identity(self, populated_db):
        result = populated_db.remove("Alice")
        assert result is True
        assert populated_db.count == 2
        assert not populated_db.has_identity("Alice")

    def test_remove_nonexistent_returns_false(self, empty_db):
        result = empty_db.remove("Nobody")
        assert result is False

    def test_rename_identity(self, populated_db):
        result = populated_db.rename("Alice", "Alicia")
        assert result is True
        assert populated_db.has_identity("Alicia")
        assert not populated_db.has_identity("Alice")

    def test_rename_preserves_embeddings(self, populated_db):
        identity_before = populated_db.get_identity("Alice")
        shots_before = identity_before.num_embeddings
        populated_db.rename("Alice", "Alicia")
        identity_after = populated_db.get_identity("Alicia")
        assert identity_after.num_embeddings == shots_before

    def test_rename_nonexistent_returns_false(self, populated_db):
        result = populated_db.rename("Ghost", "Spirit")
        assert result is False

    def test_rename_to_existing_name_raises(self, populated_db):
        with pytest.raises(ValueError, match="already exists"):
            populated_db.rename("Alice", "Bob")

    def test_rename_same_name_ok(self, populated_db):
        result = populated_db.rename("Alice", "Alice")
        assert result is True

    def test_has_identity_true(self, populated_db):
        assert populated_db.has_identity("Alice") is True

    def test_has_identity_false(self, populated_db):
        assert populated_db.has_identity("Nobody") is False

    def test_contains_operator(self, populated_db):
        assert "Alice" in populated_db
        assert "Nobody" not in populated_db

    def test_list_identities_sorted(self, populated_db):
        names = populated_db.list_identities()
        assert names == sorted(names)

    def test_list_identities_includes_all(self, populated_db):
        names = populated_db.list_identities()
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    def test_clear_removes_all(self, populated_db):
        populated_db.clear()
        assert populated_db.count == 0
        assert populated_db.is_empty is True

    def test_count_property(self, populated_db):
        assert populated_db.count == 3

    def test_total_embeddings(self, populated_db):
        assert populated_db.total_embeddings == 3

    def test_is_empty_false(self, populated_db):
        assert populated_db.is_empty is False

    def test_is_empty_true(self, empty_db):
        assert empty_db.is_empty is True

    def test_len_operator(self, populated_db):
        assert len(populated_db) == 3

    def test_get_identity_returns_none_for_missing(self, empty_db):
        assert empty_db.get_identity("Ghost") is None

    def test_stats_returns_dict(self, populated_db):
        s = populated_db.stats()
        assert isinstance(s, dict)
        assert s["count"] == 3
        assert s["total_embeddings"] == 3
        assert "threshold" in s
        assert "strategy" in s

    def test_stats_empty_db(self, empty_db):
        s = empty_db.stats()
        assert s["count"] == 0
        assert s["total_embeddings"] == 0

    def test_threshold_setter(self, empty_db):
        empty_db.similarity_threshold = 0.60
        assert empty_db.similarity_threshold == pytest.approx(0.60)

    def test_threshold_setter_out_of_range_raises(self, empty_db):
        with pytest.raises(ValueError):
            empty_db.similarity_threshold = 1.5

    def test_strategy_setter(self, empty_db):
        empty_db.strategy = "mean"
        assert empty_db.strategy == "mean"

    def test_strategy_setter_invalid_raises(self, empty_db):
        with pytest.raises(ValueError):
            empty_db.strategy = "invalid_strategy"

    def test_repr(self, populated_db):
        r = repr(populated_db)
        assert "FaceDatabase" in r
        assert "count=3" in r


class TestFaceDatabasePersistence:

    def test_save_and_load_roundtrip(self, populated_db, tmp_path):
        path = tmp_path / "face_db.pkl"
        populated_db.save(path)
        loaded = FaceDatabase.load(path)
        assert loaded.count == populated_db.count
        assert set(loaded.list_identities()) == set(populated_db.list_identities())

    def test_save_creates_file(self, populated_db, tmp_path):
        path = tmp_path / "face_db.pkl"
        populated_db.save(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_creates_parent_dirs(self, populated_db, tmp_path):
        path = tmp_path / "deep" / "nested" / "face_db.pkl"
        populated_db.save(path)
        assert path.exists()

    def test_load_embeddings_preserved(self, populated_db, vec_alice, tmp_path):
        path = tmp_path / "face_db.pkl"
        populated_db.save(path)
        loaded = FaceDatabase.load(path)
        # Search with alice's vector — should still match after roundtrip
        match = loaded.search(FaceEmbedding(vector=vec_alice))
        assert match.is_known is True
        assert match.identity == "Alice"

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FaceDatabase.load(tmp_path / "nonexistent.pkl")

    def test_load_corrupted_raises(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"not a valid pickle file!!!")
        with pytest.raises(Exception):
            FaceDatabase.load(path)

    def test_save_preserves_threshold(self, tmp_path):
        db = FaceDatabase(similarity_threshold=0.72)
        path = tmp_path / "db.pkl"
        db.save(path)
        loaded = FaceDatabase.load(path)
        assert loaded.similarity_threshold == pytest.approx(0.72)

    def test_save_preserves_strategy(self, tmp_path):
        db = FaceDatabase(strategy="mean")
        path = tmp_path / "db.pkl"
        db.save(path)
        loaded = FaceDatabase.load(path)
        assert loaded.strategy == "mean"

    def test_to_dict_and_from_dict_roundtrip(self, populated_db, vec_alice):
        d = populated_db.to_dict()
        assert "identities" in d
        assert d["count"] == 3

        restored = FaceDatabase.from_dict(d)
        assert restored.count == 3
        match = restored.search(FaceEmbedding(vector=vec_alice))
        assert match.identity == "Alice"

    def test_to_dict_embeddings_are_lists(self, populated_db):
        d = populated_db.to_dict()
        for rec in d["identities"]:
            for emb_list in rec["embeddings"]:
                assert isinstance(emb_list, list)

    def test_save_if_changed_saves_when_count_differs(self, empty_db, vec_alice, tmp_path):
        path = tmp_path / "db.pkl"
        empty_db.register("Alice", FaceEmbedding(vector=vec_alice))
        saved = empty_db.save_if_changed(path, last_count=0)
        assert saved is True
        assert path.exists()

    def test_save_if_changed_skips_when_count_same(self, populated_db, tmp_path):
        path = tmp_path / "db.pkl"
        saved = populated_db.save_if_changed(path, last_count=populated_db.count)
        assert saved is False
        assert not path.exists()


class TestFaceDatabaseThreadSafety:

    def test_concurrent_register(self, empty_db):
        """Multiple threads can register without data corruption."""
        errors = []

        def register_worker(name: str, seed: int) -> None:
            try:
                v = _rand_vec(seed=seed)
                empty_db.register(name, FaceEmbedding(vector=v))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_worker, args=(f"person_{i}", i)) for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert empty_db.count == 20

    def test_concurrent_search(self, populated_db, vec_alice):
        """Multiple threads can search concurrently."""
        errors = []
        results = []

        def search_worker() -> None:
            try:
                match = populated_db.search(FaceEmbedding(vector=vec_alice))
                results.append(match)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=search_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        # All threads should have found Alice
        assert all(r.identity == "Alice" for r in results)
