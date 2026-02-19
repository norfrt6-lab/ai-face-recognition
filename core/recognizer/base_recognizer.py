# ============================================================
# AI Face Recognition & Face Swap
# core/recognizer/base_recognizer.py
# ============================================================
# Defines the abstract contract that ALL face recognizers must
# implement, plus shared data-types used throughout the pipeline.
#
# Hierarchy:
#   BaseRecognizer  (abstract)
#       └── InsightFaceRecognizer
#       └── <any future recognizer>
#
# Key data types:
#   FaceEmbedding  — 512-dim ArcFace vector + metadata
#   FaceMatch      — identity result from a database lookup
#   FaceAttribute  — optional age / gender / expression info
# ============================================================

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# Data Types
# ============================================================

@dataclass
class FaceAttribute:
    """
    Optional demographic / expression attributes predicted by the
    face analysis model.

    Not all recognizers populate every field — check for None before use.

    Attributes:
        age:        Estimated age in years (float), or None.
        gender:     'M' | 'F' | None.
        gender_score: Confidence score for the gender prediction [0, 1].
        embedding_norm: L2 norm of the raw embedding vector (quality proxy).
    """

    age: Optional[float] = None
    gender: Optional[str] = None          # 'M' | 'F'
    gender_score: Optional[float] = None  # confidence [0, 1]
    embedding_norm: Optional[float] = None

    @property
    def is_male(self) -> Optional[bool]:
        if self.gender is None:
            return None
        return self.gender.upper() == "M"

    @property
    def is_female(self) -> Optional[bool]:
        if self.gender is None:
            return None
        return self.gender.upper() == "F"

    def __repr__(self) -> str:
        parts = []
        if self.age is not None:
            parts.append(f"age={self.age:.1f}")
        if self.gender is not None:
            score = f"/{self.gender_score:.2f}" if self.gender_score else ""
            parts.append(f"gender={self.gender}{score}")
        return f"FaceAttribute({', '.join(parts) if parts else 'unknown'})"


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceEmbedding:
    """
    A single face embedding vector extracted from one face image/crop.

    The embedding is the primary representation used for recognition:
    faces of the same person have embeddings with high cosine similarity,
    while faces of different people have low similarity.

    Attributes:
        vector:       Normalised embedding array of shape (D,) where D is
                      the embedding dimension (typically 512 for ArcFace).
        face_index:   Which face in the source image this came from
                      (aligned with DetectionResult.faces[i]).
        source_path:  Optional path / identifier for the source image.
        attributes:   Optional demographic / quality attributes.
        bbox:         Optional (x1, y1, x2, y2) face bounding box in the
                      source image (pixel coords).
        landmarks:    Optional (5, 2) float32 facial landmark array.
    """

    vector: np.ndarray                       # shape (D,) float32
    face_index: int = 0
    source_path: Optional[str] = None
    attributes: Optional[FaceAttribute] = None
    bbox: Optional[Tuple[int, int, int, int]] = field(default=None, repr=False)
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Embedding properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vector."""
        return int(self.vector.shape[0])

    @property
    def norm(self) -> float:
        """L2 norm of the embedding vector."""
        return float(np.linalg.norm(self.vector))

    @property
    def is_normalised(self, tol: float = 1e-3) -> bool:
        """True if the embedding vector has unit L2 norm (± tol)."""
        return abs(self.norm - 1.0) < tol

    def normalise(self) -> "FaceEmbedding":
        """
        Return a new FaceEmbedding with a unit-normalised vector.

        If the vector is already normalised (or has zero norm) it is
        returned as-is to avoid division-by-zero.

        Returns:
            New FaceEmbedding with L2-normalised vector.
        """
        n = self.norm
        if n < 1e-10:
            return self
        return FaceEmbedding(
            vector=self.vector / n,
            face_index=self.face_index,
            source_path=self.source_path,
            attributes=self.attributes,
            bbox=self.bbox,
            landmarks=self.landmarks,
        )

    def cosine_similarity(self, other: "FaceEmbedding") -> float:
        """
        Compute the cosine similarity between this and another embedding.

        Both vectors are normalised before the dot product so the result
        is independent of their magnitudes.

        Args:
            other: The embedding to compare against.

        Returns:
            Cosine similarity in [-1.0, 1.0].
            1.0 = identical direction (same person).
            0.0 = orthogonal (unrelated).
           -1.0 = opposite (rare; indicates very different faces).
        """
        a = self.vector / (np.linalg.norm(self.vector) + 1e-10)
        b = other.vector / (np.linalg.norm(other.vector) + 1e-10)
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def euclidean_distance(self, other: "FaceEmbedding") -> float:
        """
        Compute the Euclidean (L2) distance between two embeddings.

        Useful as an alternative metric to cosine similarity.
        Lower distance = more similar faces.

        Args:
            other: The embedding to compare against.

        Returns:
            Non-negative float Euclidean distance.
        """
        return float(np.linalg.norm(self.vector - other.vector))

    def as_list(self) -> List[float]:
        """Return the embedding vector as a Python list of floats."""
        return self.vector.tolist()

    def __repr__(self) -> str:
        attr_str = f", {self.attributes}" if self.attributes else ""
        path_str = f", src={self.source_path!r}" if self.source_path else ""
        return (
            f"FaceEmbedding(idx={self.face_index}, "
            f"dim={self.dim}, "
            f"norm={self.norm:.4f}"
            f"{attr_str}"
            f"{path_str})"
        )


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceMatch:
    """
    The result of searching a FaceDatabase for an identity.

    Attributes:
        identity:    The matched identity label (name / ID string).
                     None if no match was found above the threshold.
        similarity:  Cosine similarity score of the best match [0, 1].
                     Higher = more confident the same person.
        distance:    Euclidean distance to the matched embedding.
                     Lower = more similar.
        face_index:  Which face in the query image was matched.
        embedding:   The query FaceEmbedding that produced this match.
        matched_embedding: The stored embedding that was closest.
        is_known:    True if similarity ≥ threshold (identity confirmed).
        threshold:   The similarity threshold used for the decision.
    """

    identity: Optional[str]
    similarity: float
    distance: float
    face_index: int = 0
    embedding: Optional[FaceEmbedding] = field(default=None, repr=False)
    matched_embedding: Optional[FaceEmbedding] = field(default=None, repr=False)
    is_known: bool = False
    threshold: float = 0.45

    @property
    def confidence_pct(self) -> float:
        """Similarity expressed as a percentage [0, 100]."""
        return round(self.similarity * 100, 2)

    @property
    def label(self) -> str:
        """
        Human-readable identity label.

        Returns the identity name if known, or 'Unknown' otherwise.
        """
        return self.identity if self.is_known and self.identity else "Unknown"

    def __repr__(self) -> str:
        return (
            f"FaceMatch("
            f"identity={self.identity!r}, "
            f"similarity={self.similarity:.4f}, "
            f"is_known={self.is_known}, "
            f"face_idx={self.face_index})"
        )


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RecognitionResult:
    """
    The complete output of a single recognition call on one image.

    Contains one FaceMatch per detected face, plus timing metadata.

    Attributes:
        matches:          List of FaceMatch objects (one per face detected).
        image_width:      Width of the source image in pixels.
        image_height:     Height of the source image in pixels.
        inference_time_ms: Total wall-clock recognition time in ms.
        frame_index:      Optional frame number for video pipelines.
        metadata:         Optional free-form dict for extra info.
    """

    matches: List[FaceMatch]
    image_width: int
    image_height: int
    inference_time_ms: float = 0.0
    frame_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def num_faces(self) -> int:
        """Number of faces that were processed."""
        return len(self.matches)

    @property
    def known_faces(self) -> List[FaceMatch]:
        """Return only the matches where is_known=True."""
        return [m for m in self.matches if m.is_known]

    @property
    def unknown_faces(self) -> List[FaceMatch]:
        """Return only the matches where is_known=False."""
        return [m for m in self.matches if not m.is_known]

    @property
    def identities(self) -> List[str]:
        """Return label strings for all matches (including 'Unknown')."""
        return [m.label for m in self.matches]

    @property
    def is_empty(self) -> bool:
        """True if no faces were found in the image."""
        return len(self.matches) == 0

    def get_match(self, face_index: int) -> Optional[FaceMatch]:
        """
        Get the match for a specific face index.

        Args:
            face_index: Zero-based face index.

        Returns:
            FaceMatch or None.
        """
        for m in self.matches:
            if m.face_index == face_index:
                return m
        return None

    def __repr__(self) -> str:
        return (
            f"RecognitionResult("
            f"num_faces={self.num_faces}, "
            f"known={len(self.known_faces)}, "
            f"unknown={len(self.unknown_faces)}, "
            f"inference={self.inference_time_ms:.1f}ms)"
        )


# ============================================================
# Cosine similarity utilities (module-level, no class needed)
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Args:
        a: First vector (any length, float).
        b: Second vector (same length as *a*).

    Returns:
        Cosine similarity in [-1.0, 1.0].

    Raises:
        ValueError: If vectors have different shapes.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: a={a.shape}, b={b.shape}"
        )
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def cosine_similarity_matrix(
    queries: np.ndarray,
    gallery: np.ndarray,
) -> np.ndarray:
    """
    Compute a cosine similarity matrix between query and gallery embeddings.

    Args:
        queries: (N, D) array of query embeddings.
        gallery: (M, D) array of gallery (database) embeddings.

    Returns:
        (N, M) float32 similarity matrix.
        result[i, j] = cosine_similarity(queries[i], gallery[j])

    Raises:
        ValueError: If embedding dimensions do not match.
    """
    if queries.ndim == 1:
        queries = queries[np.newaxis, :]
    if gallery.ndim == 1:
        gallery = gallery[np.newaxis, :]

    if queries.shape[1] != gallery.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"queries={queries.shape[1]}, gallery={gallery.shape[1]}"
        )

    # L2-normalise rows
    q_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)
    g_norm = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-10)

    return np.clip(q_norm @ g_norm.T, -1.0, 1.0).astype(np.float32)


def average_embeddings(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute the mean of a list of embedding vectors and L2-normalise the result.

    Used when multiple photos of the same person are registered — averaging
    improves robustness against pose / lighting variation.

    Args:
        embeddings: List of (D,) float32 arrays, all same dimension.

    Returns:
        L2-normalised mean embedding of shape (D,).

    Raises:
        ValueError: If the list is empty or shapes are inconsistent.
    """
    if not embeddings:
        raise ValueError("Cannot average an empty list of embeddings.")

    stacked = np.stack(embeddings, axis=0)   # (N, D)
    mean    = stacked.mean(axis=0)            # (D,)
    norm    = np.linalg.norm(mean)
    if norm < 1e-10:
        return mean
    return (mean / norm).astype(np.float32)


# ============================================================
# Abstract Base Recognizer
# ============================================================

class BaseRecognizer(ABC):
    """
    Abstract base class for all face recognizers.

    Subclasses must implement:
        - ``load_model()``               — load weights into memory
        - ``get_embedding(image, bbox)`` — extract a single embedding

    Optional overrides:
        - ``get_embeddings_batch()``     — process multiple faces at once
        - ``get_attributes()``           — predict age/gender/etc.
        - ``release()``                  — free GPU/model memory

    Usage::

        recognizer = InsightFaceRecognizer(model_pack="buffalo_l")
        recognizer.load_model()

        embedding = recognizer.get_embedding(image)
        print(embedding)

    Context-manager usage (auto load + release)::

        with InsightFaceRecognizer(...) as rec:
            embedding = rec.get_embedding(image)
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        model_root: str = "models",
        similarity_threshold: float = 0.45,
        embedding_dim: int = 512,
        providers: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            model_pack:           InsightFace model pack name
                                  ('buffalo_l' | 'buffalo_m' | 'buffalo_s').
            model_root:           Root directory for InsightFace model files.
            similarity_threshold: Cosine similarity threshold for positive
                                  identity match [0.0, 1.0].
            embedding_dim:        Expected embedding vector dimension.
                                  ArcFace = 512.
            providers:            ONNX Runtime execution providers in
                                  priority order.
        """
        self.model_pack = model_pack
        self.model_root = model_root
        self.similarity_threshold = float(similarity_threshold)
        self.embedding_dim = int(embedding_dim)
        self.providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self._model = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the recognition model weights into memory.

        Must:
          - Populate ``self._model``
          - Set ``self._is_loaded = True``
          - Raise ``RuntimeError`` on failure.
        """

    @abstractmethod
    def get_embedding(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[FaceEmbedding]:
        """
        Extract a face embedding from an image.

        If *bbox* is provided, crop the face region before embedding.
        If *landmarks* are provided, align the face before embedding.
        If neither is provided, the recognizer should detect the most
        prominent face and embed that.

        Args:
            image:     BGR numpy array (H, W, 3).
            bbox:      Optional (x1, y1, x2, y2) face region.
            landmarks: Optional (5, 2) float32 landmark array.

        Returns:
            FaceEmbedding, or None if no face could be processed.

        Raises:
            RuntimeError: If the model has not been loaded.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — subclasses MAY override for efficiency
    # ------------------------------------------------------------------

    def get_embeddings_batch(
        self,
        images: List[np.ndarray],
        bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    ) -> List[Optional[FaceEmbedding]]:
        """
        Extract embeddings for a list of images.

        The default calls ``get_embedding()`` sequentially.
        Subclasses can override with a batched implementation.

        Args:
            images: List of BGR numpy arrays.
            bboxes: Optional list of bounding boxes aligned with *images*.
                    None entries mean "detect face automatically".

        Returns:
            List of FaceEmbedding (or None for failed images), same length
            as *images*.
        """
        self._require_loaded()
        bboxes = bboxes or [None] * len(images)
        return [
            self.get_embedding(img, bbox=bb)
            for img, bb in zip(images, bboxes)
        ]

    def compare(
        self,
        embedding_a: FaceEmbedding,
        embedding_b: FaceEmbedding,
    ) -> float:
        """
        Compute the cosine similarity between two FaceEmbeddings.

        Args:
            embedding_a: First face embedding.
            embedding_b: Second face embedding.

        Returns:
            Cosine similarity in [-1.0, 1.0].
        """
        return cosine_similarity(embedding_a.vector, embedding_b.vector)

    def is_same_person(
        self,
        embedding_a: FaceEmbedding,
        embedding_b: FaceEmbedding,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Decide whether two embeddings belong to the same person.

        Args:
            embedding_a:  First embedding.
            embedding_b:  Second embedding.
            threshold:    Override the instance-level threshold.
                          Defaults to ``self.similarity_threshold``.

        Returns:
            True if cosine similarity >= threshold.
        """
        t = threshold if threshold is not None else self.similarity_threshold
        return self.compare(embedding_a, embedding_b) >= t

    def release(self) -> None:
        """
        Release model resources.

        The default implementation clears ``self._model`` and resets
        ``self._is_loaded``. Subclasses should call ``super().release()``
        after their own cleanup.
        """
        self._model = None
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if ``load_model()`` has been called successfully."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        return self.model_pack

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseRecognizer":
        if not self._is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise RuntimeError if model is not loaded."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} model is not loaded. "
                "Call load_model() first or use as a context manager."
            )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Raise ValueError for obviously invalid images."""
        if image is None:
            raise ValueError("Image is None.")
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Expected numpy ndarray, got {type(image).__name__}."
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2-D or 3-D array, got shape {image.shape}."
            )
        if image.size == 0:
            raise ValueError("Image array is empty (zero size).")

    @staticmethod
    def _timer() -> float:
        """Return current time in milliseconds."""
        return time.perf_counter() * 1000.0

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_pack!r}, "
            f"threshold={self.similarity_threshold}, "
            f"status={status})"
        )
