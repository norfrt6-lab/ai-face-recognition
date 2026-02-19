# ============================================================
# AI Face Recognition & Face Swap
# core/recognizer/face_database.py
# ============================================================
# Persistent face identity store with cosine similarity search.
#
# Features:
#   - Register one or many embeddings per identity (multi-shot)
#   - Cosine similarity search against all registered identities
#   - Averaged / multi-embedding comparison strategies
#   - Pickle-based persistence (save / load)
#   - Thread-safe read/write operations
#   - Identity management (add, remove, rename, list, count)
#   - Bulk import / export helpers
#   - Statistics and diagnostics
# ============================================================

from __future__ import annotations

import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from core.recognizer.base_recognizer import (
    FaceEmbedding,
    FaceMatch,
    average_embeddings,
    cosine_similarity,
    cosine_similarity_matrix,
)


# ============================================================
# Data Types
# ============================================================

@dataclass
class FaceIdentity:
    """
    A single registered identity in the face database.

    Each identity stores one or more embeddings (multi-shot
    registration) to improve robustness under pose / lighting
    variation.

    Attributes:
        name:           Human-readable label (e.g. "Alice", "person_001").
        embeddings:     List of raw 512-dim embedding vectors for this
                        person.  At least one must be present.
        identity_id:    Unique UUID string assigned at creation.
        metadata:       Optional free-form dict (e.g. source paths, dates).
        created_at:     Unix timestamp of first registration.
        updated_at:     Unix timestamp of most recent update.
    """

    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    identity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def num_embeddings(self) -> int:
        """Number of registered embedding vectors for this identity."""
        return len(self.embeddings)

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        """
        L2-normalised mean of all registered embeddings.

        Returns None if no embeddings are stored.
        Averaging multiple shots of the same person improves
        robustness against pose / expression / lighting variation.
        """
        if not self.embeddings:
            return None
        return average_embeddings(self.embeddings)

    def add_embedding(self, vector: np.ndarray) -> None:
        """
        Append a new embedding vector for this identity.

        The vector is L2-normalised before storage.

        Args:
            vector: (D,) float32 embedding array.
        """
        norm = np.linalg.norm(vector)
        normalised = vector / norm if norm > 1e-10 else vector
        self.embeddings.append(normalised.astype(np.float32))
        self.updated_at = time.time()

    def best_similarity(self, query: np.ndarray) -> float:
        """
        Return the highest cosine similarity between *query* and any
        stored embedding (max-pool strategy).

        Args:
            query: (D,) float32 query embedding vector.

        Returns:
            Best cosine similarity in [-1.0, 1.0], or -1.0 if no
            embeddings are stored.
        """
        if not self.embeddings:
            return -1.0
        sims = [cosine_similarity(query, e) for e in self.embeddings]
        return float(max(sims))

    def mean_similarity(self, query: np.ndarray) -> float:
        """
        Return the cosine similarity between *query* and the mean
        embedding of this identity.

        Args:
            query: (D,) float32 query embedding vector.

        Returns:
            Cosine similarity, or -1.0 if no embeddings are stored.
        """
        mean = self.mean_embedding
        if mean is None:
            return -1.0
        return cosine_similarity(query, mean)

    def __repr__(self) -> str:
        return (
            f"FaceIdentity("
            f"name={self.name!r}, "
            f"embeddings={self.num_embeddings}, "
            f"id={self.identity_id[:8]}...)"
        )


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    Extended result from a database search, including all ranked matches.

    Attributes:
        best_match:     The closest FaceMatch found.
        all_matches:    All identities ranked by similarity (best first).
        query_embedding: The query FaceEmbedding used for the search.
        search_time_ms: Wall-clock search time in milliseconds.
        strategy:       The comparison strategy used ('best' | 'mean').
    """

    best_match: FaceMatch
    all_matches: List[Tuple[str, float]]   # [(identity_name, similarity), ...]
    query_embedding: Optional[FaceEmbedding] = None
    search_time_ms: float = 0.0
    strategy: str = "best"

    @property
    def top_n(self) -> List[Tuple[str, float]]:
        """Alias for all_matches (already sorted best-first)."""
        return self.all_matches

    def __repr__(self) -> str:
        return (
            f"SearchResult("
            f"best={self.best_match.identity!r}, "
            f"similarity={self.best_match.similarity:.4f}, "
            f"candidates={len(self.all_matches)}, "
            f"time={self.search_time_ms:.1f}ms)"
        )


# ============================================================
# FaceDatabase
# ============================================================

class FaceDatabase:
    """
    Thread-safe face identity store with cosine similarity search.

    Supports:
        - Single-shot and multi-shot identity registration
        - Cosine similarity search (best-match or mean-embedding)
        - Vectorised search using numpy matrix operations (fast)
        - Pickle-based persistence (save / load)
        - Identity management (add, remove, rename, update, clear)
        - Bulk import / export (dict / list formats)
        - Statistics and diagnostics

    Quick usage::

        db = FaceDatabase()
        db.register("Alice", alice_embedding)
        db.register("Bob",   bob_embedding)
        db.save("cache/face_db.pkl")

        match = db.search(query_embedding, threshold=0.45)
        print(match.label)   # "Alice" | "Unknown"

    Multi-shot registration (recommended)::

        db.register("Alice", emb1)
        db.register("Alice", emb2)   # Second shot — same name
        db.register("Alice", emb3)   # Third shot
        # Search now uses the best similarity across all three shots
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        strategy: str = "best",
        max_embeddings_per_identity: int = 10,
    ) -> None:
        """
        Args:
            similarity_threshold:       Minimum cosine similarity required
                                        to accept a match [0.0, 1.0].
            strategy:                   How to compare a query against a
                                        multi-shot identity:
                                        'best' — max similarity across
                                                 all stored shots (default).
                                        'mean' — similarity to mean
                                                 embedding.
            max_embeddings_per_identity: Cap on stored shots per person.
                                        When exceeded, oldest is dropped.
        """
        self._threshold = float(similarity_threshold)
        self._strategy = strategy
        self._max_shots = max_embeddings_per_identity

        # name → FaceIdentity
        self._identities: Dict[str, FaceIdentity] = {}

        # Thread safety
        self._lock = threading.RLock()

        logger.debug(
            f"FaceDatabase created | "
            f"threshold={self._threshold} | "
            f"strategy={self._strategy}"
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        embedding: FaceEmbedding | np.ndarray,
        *,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> FaceIdentity:
        """
        Register a face embedding under the given identity name.

        If *name* already exists, the embedding is appended (multi-shot).
        If *overwrite* is True, any existing embeddings are cleared first.

        Args:
            name:       Identity label (e.g. person's name / employee ID).
            embedding:  FaceEmbedding object or raw (D,) numpy array.
            metadata:   Optional dict to attach to the identity record
                        (e.g. {'source': 'photo_2024.jpg'}).
            overwrite:  If True, replace all existing embeddings for
                        *name* with just this one.

        Returns:
            The FaceIdentity record (created or updated).

        Raises:
            ValueError: If *name* is empty or the embedding vector is invalid.
        """
        name = name.strip()
        if not name:
            raise ValueError("Identity name must not be empty.")

        vector = self._extract_vector(embedding)
        self._validate_vector(vector)

        with self._lock:
            if name not in self._identities or overwrite:
                identity = FaceIdentity(
                    name=name,
                    metadata=metadata or {},
                )
                self._identities[name] = identity
                logger.info(f"Registered new identity: {name!r}")
            else:
                identity = self._identities[name]
                logger.debug(
                    f"Adding shot #{identity.num_embeddings + 1} "
                    f"to identity: {name!r}"
                )

            if overwrite:
                identity.embeddings.clear()

            # Cap shots per identity
            while identity.num_embeddings >= self._max_shots:
                identity.embeddings.pop(0)   # Drop oldest shot

            identity.add_embedding(vector)

            if metadata:
                identity.metadata.update(metadata)

        logger.debug(
            f"Identity '{name}': "
            f"{identity.num_embeddings} shot(s) stored."
        )
        return identity

    def register_many(
        self,
        name: str,
        embeddings: List[FaceEmbedding | np.ndarray],
        *,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> FaceIdentity:
        """
        Register multiple embeddings for the same identity at once.

        Args:
            name:        Identity label.
            embeddings:  List of FaceEmbedding objects or numpy arrays.
            metadata:    Optional metadata dict.
            overwrite:   If True, replace existing embeddings.

        Returns:
            The updated FaceIdentity record.
        """
        if not embeddings:
            raise ValueError("embeddings list must not be empty.")

        identity = None
        for i, emb in enumerate(embeddings):
            identity = self.register(
                name,
                emb,
                metadata=metadata,
                overwrite=(overwrite and i == 0),   # overwrite only on first
            )
        return identity  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Search / recognition
    # ------------------------------------------------------------------

    def search(
        self,
        query: FaceEmbedding | np.ndarray,
        *,
        threshold: Optional[float] = None,
        top_k: int = 1,
        strategy: Optional[str] = None,
    ) -> FaceMatch:
        """
        Search the database for the identity closest to *query*.

        Args:
            query:     FaceEmbedding or raw (D,) numpy array to search.
            threshold: Override the instance-level similarity threshold.
            top_k:     Number of candidates to consider (not returned —
                       only the best is returned in the FaceMatch).
                       Set > 1 to improve accuracy with large databases.
            strategy:  Override the instance-level comparison strategy
                       for this call. One of:
                       - ``"mean"`` — compare against averaged embedding
                         (fast, default for backward compatibility).
                       - ``"best"`` — compare against all individual
                         embeddings, take maximum similarity (more
                         accurate for multi-shot identities).

        Returns:
            FaceMatch with the best identity found.
            ``is_known=False`` and ``identity=None`` if no match
            exceeds the threshold or the database is empty.
        """
        t0 = time.perf_counter() * 1000.0
        threshold = threshold if threshold is not None else self._threshold
        face_index = 0

        if strategy is not None and strategy not in ("best", "mean"):
            raise ValueError(f"strategy must be 'best' or 'mean', got {strategy!r}.")

        # Extract query vector
        query_vec = self._extract_vector(query)
        self._validate_vector(query_vec)

        # Normalise query
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-10:
            query_vec = query_vec / q_norm

        face_index = (
            query.face_index
            if isinstance(query, FaceEmbedding)
            else 0
        )

        with self._lock:
            if not self._identities:
                return FaceMatch(
                    identity=None,
                    similarity=0.0,
                    distance=float("inf"),
                    face_index=face_index,
                    is_known=False,
                    threshold=threshold,
                )

            best_name, best_sim = self._find_best(query_vec, strategy=strategy)

        best_dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - best_sim))))

        is_known = best_sim >= threshold
        identity_name = best_name if is_known else None

        elapsed = time.perf_counter() * 1000.0 - t0

        match = FaceMatch(
            identity=identity_name,
            similarity=best_sim,
            distance=best_dist,
            face_index=face_index,
            embedding=query if isinstance(query, FaceEmbedding) else None,
            is_known=is_known,
            threshold=threshold,
        )

        logger.debug(
            f"Search: best={best_name!r}, "
            f"sim={best_sim:.4f}, "
            f"known={is_known}, "
            f"time={elapsed:.1f}ms"
        )

        return match

    def search_extended(
        self,
        query: FaceEmbedding | np.ndarray,
        *,
        threshold: Optional[float] = None,
        top_k: int = 5,
    ) -> SearchResult:
        """
        Search the database and return a full ranked list of candidates.

        Args:
            query:     FaceEmbedding or raw numpy array.
            threshold: Override instance-level threshold.
            top_k:     Maximum number of ranked candidates to return.

        Returns:
            SearchResult with best_match and all_matches ranked list.
        """
        t0 = time.perf_counter() * 1000.0
        threshold = threshold if threshold is not None else self._threshold

        query_vec = self._extract_vector(query)
        self._validate_vector(query_vec)

        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-10:
            query_vec = query_vec / q_norm

        face_index = (
            query.face_index if isinstance(query, FaceEmbedding) else 0
        )

        with self._lock:
            ranked = self._rank_all(query_vec)

        # Keep top_k
        ranked_top = ranked[:top_k]

        elapsed = time.perf_counter() * 1000.0 - t0

        if not ranked:
            best_match = FaceMatch(
                identity=None,
                similarity=0.0,
                distance=float("inf"),
                face_index=face_index,
                is_known=False,
                threshold=threshold,
            )
        else:
            best_name, best_sim = ranked[0]
            best_dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - best_sim))))
            is_known  = best_sim >= threshold
            best_match = FaceMatch(
                identity=best_name if is_known else None,
                similarity=best_sim,
                distance=best_dist,
                face_index=face_index,
                embedding=query if isinstance(query, FaceEmbedding) else None,
                is_known=is_known,
                threshold=threshold,
            )

        return SearchResult(
            best_match=best_match,
            all_matches=ranked_top,
            query_embedding=query if isinstance(query, FaceEmbedding) else None,
            search_time_ms=elapsed,
            strategy=self._strategy,
        )

    def search_batch(
        self,
        queries: List[FaceEmbedding | np.ndarray],
        *,
        threshold: Optional[float] = None,
    ) -> List[FaceMatch]:
        """
        Search the database for a list of query embeddings.

        Uses vectorised matrix multiplication for efficiency when the
        database is large.

        Args:
            queries:   List of FaceEmbedding or numpy arrays.
            threshold: Override instance-level threshold.

        Returns:
            List of FaceMatch objects, one per query, in the same order.
        """
        if not queries:
            return []

        threshold = threshold if threshold is not None else self._threshold

        with self._lock:
            if not self._identities:
                return [
                    FaceMatch(
                        identity=None, similarity=0.0,
                        distance=float("inf"),
                        face_index=i, is_known=False,
                        threshold=threshold,
                    )
                    for i in range(len(queries))
                ]

            # Build gallery matrix: (M, D)
            names, gallery = self._build_gallery_matrix()

        # Build query matrix: (N, D)
        vectors = []
        face_indices = []
        for q in queries:
            v = self._extract_vector(q)
            n = np.linalg.norm(v)
            vectors.append(v / n if n > 1e-10 else v)
            face_indices.append(
                q.face_index if isinstance(q, FaceEmbedding) else 0
            )

        Q = np.stack(vectors, axis=0).astype(np.float32)   # (N, D)
        G = gallery.astype(np.float32)                      # (M, D)

        # (N, M) similarity matrix
        sim_matrix = cosine_similarity_matrix(Q, G)

        results: List[FaceMatch] = []
        for i in range(len(queries)):
            sims = sim_matrix[i]              # (M,)
            best_j = int(np.argmax(sims))
            best_sim = float(sims[best_j])
            best_name = names[best_j]

            # Resolve per-identity best score using stored strategy
            # (gallery is mean embeddings; re-check with strategy if needed)
            if self._strategy == "best":
                with self._lock:
                    identity = self._identities.get(best_name)
                if identity:
                    best_sim = identity.best_similarity(vectors[i])

            best_dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - best_sim))))
            is_known  = best_sim >= threshold

            results.append(FaceMatch(
                identity=best_name if is_known else None,
                similarity=best_sim,
                distance=best_dist,
                face_index=face_indices[i],
                embedding=queries[i] if isinstance(queries[i], FaceEmbedding) else None,
                is_known=is_known,
                threshold=threshold,
            ))

        return results

    # ------------------------------------------------------------------
    # Identity management
    # ------------------------------------------------------------------

    def remove(self, name: str) -> bool:
        """
        Remove an identity from the database.

        Args:
            name: Identity label to remove.

        Returns:
            True if the identity existed and was removed, False otherwise.
        """
        with self._lock:
            if name in self._identities:
                del self._identities[name]
                logger.info(f"Removed identity: {name!r}")
                return True
            logger.warning(f"Identity not found for removal: {name!r}")
            return False

    def rename(self, old_name: str, new_name: str) -> bool:
        """
        Rename an existing identity.

        Args:
            old_name: Current identity label.
            new_name: New identity label.

        Returns:
            True on success, False if *old_name* was not found.

        Raises:
            ValueError: If *new_name* already exists in the database.
        """
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("New name must not be empty.")

        with self._lock:
            if old_name not in self._identities:
                logger.warning(f"Rename failed: '{old_name}' not found.")
                return False
            if new_name in self._identities and new_name != old_name:
                raise ValueError(
                    f"Cannot rename '{old_name}' → '{new_name}': "
                    f"'{new_name}' already exists."
                )
            identity = self._identities.pop(old_name)
            identity.name = new_name
            identity.updated_at = time.time()
            self._identities[new_name] = identity

        logger.info(f"Renamed identity: {old_name!r} → {new_name!r}")
        return True

    def get_identity(self, name: str) -> Optional[FaceIdentity]:
        """
        Retrieve a FaceIdentity by name.

        Args:
            name: Identity label.

        Returns:
            FaceIdentity or None if not found.
        """
        with self._lock:
            return self._identities.get(name)

    def has_identity(self, name: str) -> bool:
        """Return True if *name* is registered in the database."""
        with self._lock:
            return name in self._identities

    def list_identities(self) -> List[str]:
        """
        Return a sorted list of all registered identity names.

        Returns:
            Alphabetically sorted list of name strings.
        """
        with self._lock:
            return sorted(self._identities.keys())

    def clear(self) -> None:
        """Remove ALL registered identities from the database."""
        with self._lock:
            count = len(self._identities)
            self._identities.clear()
        logger.warning(f"FaceDatabase cleared ({count} identities removed).")

    @property
    def count(self) -> int:
        """Number of registered identities."""
        with self._lock:
            return len(self._identities)

    @property
    def total_embeddings(self) -> int:
        """Total number of embedding vectors stored across all identities."""
        with self._lock:
            return sum(
                identity.num_embeddings
                for identity in self._identities.values()
            )

    @property
    def is_empty(self) -> bool:
        """True if no identities are registered."""
        return self.count == 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """
        Serialize the database to a pickle file.

        The file contains a dict with metadata and all FaceIdentity
        objects. The format is forward-compatible: unknown keys are
        ignored on load.

        Args:
            path: Output file path (e.g. 'cache/face_db.pkl').

        Returns:
            Resolved Path to the saved file.

        Raises:
            IOError: If the file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version":              "1.0",
            "saved_at":             time.time(),
            "similarity_threshold": self._threshold,
            "strategy":             self._strategy,
            "max_shots":            self._max_shots,
            "identities":           dict(self._identities),
            "count":                self.count,
            "total_embeddings":     self.total_embeddings,
        }

        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"FaceDatabase saved → {path} "
            f"({self.count} identities, "
            f"{self.total_embeddings} embeddings)"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FaceDatabase":
        """
        Deserialize a FaceDatabase from a pickle file.

        Args:
            path: Path to the saved pickle file.

        Returns:
            Loaded FaceDatabase instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError:        If the file format is unrecognised.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"FaceDatabase file not found: {path}"
            )

        with open(path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict) or "identities" not in payload:
            raise ValueError(
                f"Unrecognised FaceDatabase format in: {path}"
            )

        db = cls(
            similarity_threshold=payload.get("similarity_threshold", 0.45),
            strategy=payload.get("strategy", "best"),
            max_embeddings_per_identity=payload.get("max_shots", 10),
        )
        db._identities = payload["identities"]

        logger.info(
            f"FaceDatabase loaded ← {path} "
            f"({db.count} identities, "
            f"{db.total_embeddings} embeddings)"
        )
        return db

    def save_if_changed(
        self,
        path: str | Path,
        *,
        last_count: Optional[int] = None,
    ) -> bool:
        """
        Save the database only if the number of identities has changed.

        Useful for auto-save loops that don't want to write to disk
        on every frame.

        Args:
            path:       Output file path.
            last_count: The count at the last save point.
                        If None, always saves.

        Returns:
            True if the database was saved, False if it was unchanged.
        """
        if last_count is None or self.count != last_count:
            self.save(path)
            return True
        return False

    # ------------------------------------------------------------------
    # Bulk import / export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Export the database as a plain Python dict.

        The embedding vectors are converted to Python lists for JSON
        serialisability.

        Returns:
            Dict with keys: 'identities' (list of identity records).
        """
        with self._lock:
            records = []
            for name, identity in self._identities.items():
                records.append({
                    "name":         identity.name,
                    "identity_id":  identity.identity_id,
                    "num_shots":    identity.num_embeddings,
                    "embeddings":   [e.tolist() for e in identity.embeddings],
                    "metadata":     identity.metadata,
                    "created_at":   identity.created_at,
                    "updated_at":   identity.updated_at,
                })
        return {
            "version":   "1.0",
            "count":     self.count,
            "identities": records,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        similarity_threshold: float = 0.45,
        strategy: str = "best",
    ) -> "FaceDatabase":
        """
        Reconstruct a FaceDatabase from a plain dict (from ``to_dict()``).

        Args:
            data:                 Dict in the format produced by ``to_dict()``.
            similarity_threshold: Threshold to apply to the new instance.
            strategy:             Comparison strategy for the new instance.

        Returns:
            Populated FaceDatabase instance.
        """
        db = cls(
            similarity_threshold=similarity_threshold,
            strategy=strategy,
        )
        for rec in data.get("identities", []):
            identity = FaceIdentity(
                name=rec["name"],
                identity_id=rec.get("identity_id", str(uuid.uuid4())),
                metadata=rec.get("metadata", {}),
                created_at=rec.get("created_at", time.time()),
                updated_at=rec.get("updated_at", time.time()),
            )
            for vec_list in rec.get("embeddings", []):
                identity.embeddings.append(
                    np.array(vec_list, dtype=np.float32)
                )
            with db._lock:
                db._identities[rec["name"]] = identity
        return db

    # ------------------------------------------------------------------
    # Statistics & diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Return a summary statistics dict for the database.

        Returns:
            Dict with keys: count, total_embeddings, avg_shots_per_identity,
            min_shots, max_shots, identities (list of names), threshold,
            strategy.
        """
        with self._lock:
            if not self._identities:
                return {
                    "count":                   0,
                    "total_embeddings":        0,
                    "avg_shots_per_identity":  0.0,
                    "min_shots":               0,
                    "max_shots":               0,
                    "identities":              [],
                    "threshold":               self._threshold,
                    "strategy":                self._strategy,
                }

            shot_counts = [i.num_embeddings for i in self._identities.values()]
            return {
                "count":                   self.count,
                "total_embeddings":        self.total_embeddings,
                "avg_shots_per_identity":  round(sum(shot_counts) / len(shot_counts), 2),
                "min_shots":               min(shot_counts),
                "max_shots":               max(shot_counts),
                "identities":              self.list_identities(),
                "threshold":               self._threshold,
                "strategy":                self._strategy,
            }

    def print_stats(self) -> None:
        """Pretty-print database statistics to the logger."""
        s = self.stats()
        logger.info("=" * 50)
        logger.info("FaceDatabase Statistics")
        logger.info("=" * 50)
        logger.info(f"  Registered identities : {s['count']}")
        logger.info(f"  Total embeddings      : {s['total_embeddings']}")
        logger.info(f"  Avg shots / identity  : {s['avg_shots_per_identity']}")
        logger.info(f"  Min shots             : {s['min_shots']}")
        logger.info(f"  Max shots             : {s['max_shots']}")
        logger.info(f"  Similarity threshold  : {s['threshold']}")
        logger.info(f"  Comparison strategy   : {s['strategy']}")
        logger.info(f"  Identities            : {', '.join(s['identities'])}")
        logger.info("=" * 50)

    # ------------------------------------------------------------------
    # Properties / configuration
    # ------------------------------------------------------------------

    @property
    def similarity_threshold(self) -> float:
        """The current similarity threshold."""
        return self._threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        """Update the similarity threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {value}.")
        self._threshold = float(value)

    @property
    def strategy(self) -> str:
        """The current comparison strategy ('best' | 'mean')."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: str) -> None:
        """Update the comparison strategy."""
        if value not in ("best", "mean"):
            raise ValueError(f"Strategy must be 'best' or 'mean', got {value!r}.")
        self._strategy = value

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_best(
        self, query_vec: np.ndarray, strategy: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Find the identity with the highest similarity to *query_vec*.

        Called with ``self._lock`` already held.

        Args:
            query_vec: L2-normalised (D,) float32 query vector.
            strategy:  Override comparison strategy (``None`` = use instance default).

        Returns:
            (identity_name, best_similarity) tuple.
        """
        effective_strategy = strategy or self._strategy
        best_name = ""
        best_sim  = -1.0

        for name, identity in self._identities.items():
            if effective_strategy == "mean":
                sim = identity.mean_similarity(query_vec)
            else:
                sim = identity.best_similarity(query_vec)

            if sim > best_sim:
                best_sim  = sim
                best_name = name

        return best_name, float(best_sim)

    def _rank_all(
        self, query_vec: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Rank all identities by similarity to *query_vec*, best first.

        Called with ``self._lock`` already held.

        Args:
            query_vec: L2-normalised (D,) float32 query vector.

        Returns:
            List of (name, similarity) tuples, sorted descending.
        """
        scores: List[Tuple[str, float]] = []
        for name, identity in self._identities.items():
            if self._strategy == "mean":
                sim = identity.mean_similarity(query_vec)
            else:
                sim = identity.best_similarity(query_vec)
            scores.append((name, float(sim)))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _build_gallery_matrix(self) -> Tuple[List[str], np.ndarray]:
        """
        Build a (M, D) numpy matrix of mean embeddings for fast batch search.

        Called with ``self._lock`` already held.

        Returns:
            (names_list, gallery_matrix) where gallery_matrix[i] is the
            mean embedding for names_list[i].
        """
        names: List[str] = []
        vectors: List[np.ndarray] = []

        for name, identity in self._identities.items():
            mean = identity.mean_embedding
            if mean is not None:
                names.append(name)
                vectors.append(mean)

        if not vectors:
            return [], np.zeros((0, self._get_dim()), dtype=np.float32)

        return names, np.stack(vectors, axis=0).astype(np.float32)

    def _get_dim(self) -> int:
        """Return embedding dimension from the first stored vector, or 512."""
        for identity in self._identities.values():
            if identity.embeddings:
                return int(identity.embeddings[0].shape[0])
        return 512

    @staticmethod
    def _extract_vector(
        embedding: "FaceEmbedding | np.ndarray",
    ) -> np.ndarray:
        """
        Extract the raw numpy vector from a FaceEmbedding or ndarray.

        Args:
            embedding: FaceEmbedding object or raw numpy array.

        Returns:
            (D,) float32 numpy array.
        """
        if isinstance(embedding, FaceEmbedding):
            return embedding.vector.astype(np.float32)
        if isinstance(embedding, np.ndarray):
            return embedding.flatten().astype(np.float32)
        raise TypeError(
            f"Expected FaceEmbedding or np.ndarray, got {type(embedding).__name__}."
        )

    @staticmethod
    def _validate_vector(vector: np.ndarray) -> None:
        """
        Raise ValueError for obviously invalid embedding vectors.

        Args:
            vector: Candidate embedding array.
        """
        if vector.ndim != 1:
            raise ValueError(
                f"Embedding must be a 1-D vector, got shape {vector.shape}."
            )
        if vector.size == 0:
            raise ValueError("Embedding vector is empty.")
        if not np.isfinite(vector).all():
            raise ValueError("Embedding vector contains NaN or Inf values.")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered identities."""
        return self.count

    def __contains__(self, name: str) -> bool:
        """Support ``'Alice' in db`` syntax."""
        return self.has_identity(name)

    def __repr__(self) -> str:
        return (
            f"FaceDatabase("
            f"count={self.count}, "
            f"total_embeddings={self.total_embeddings}, "
            f"threshold={self._threshold}, "
            f"strategy={self._strategy!r})"
        )
