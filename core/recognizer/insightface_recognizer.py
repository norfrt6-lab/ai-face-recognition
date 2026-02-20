# InsightFace-based face recognizer.
#
# Uses the InsightFace FaceAnalysis pipeline (buffalo_l model pack)
# to extract 512-dimensional ArcFace embeddings, 5-point landmarks,
# and optional demographic attributes (age, gender).
#
# Supports two usage modes:
#   1. Full pipeline mode  — InsightFace detects + embeds internally
#   2. Embedding-only mode — Accepts a pre-detected face crop/bbox
#                            from the YOLOv8 detector and embeds only
#
# Key design decisions:
#   - Thread-safe model loading (threading.Lock)
#   - Automatic ONNX execution provider selection
#   - Graceful fallback to CPU if CUDA is unavailable
#   - All outputs normalised to unit L2 norm

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from core.recognizer.base_recognizer import (
    BaseRecognizer,
    FaceAttribute,
    FaceEmbedding,
    FaceMatch,
    RecognitionResult,
    cosine_similarity,
)
from core.swapper.base_swapper import ARCFACE_REF_112, norm_crop
from utils.image_utils import normalise_channels


class InsightFaceRecognizer(BaseRecognizer):
    """
    InsightFace ArcFace face recognizer.

    Wraps InsightFace's ``FaceAnalysis`` pipeline and translates its
    output into the standardised ``FaceEmbedding`` / ``FaceMatch``
    types used across this pipeline.

    The buffalo_l model pack provides:
        - ``det_10g``     — RetinaFace face detector (internal)
        - ``w600k_r50``   — ArcFace ResNet-50 recognition (512-dim)
        - ``genderage``   — Age and gender estimator
        - ``2d106det``    — 106-point landmark detector

    Quick usage::

        rec = InsightFaceRecognizer(model_pack="buffalo_l")
        rec.load_model()

        embedding = rec.get_embedding(image)
        print(embedding)

    Context-manager usage::

        with InsightFaceRecognizer() as rec:
            emb = rec.get_embedding(image)
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        model_root: str = "models",
        similarity_threshold: float = 0.45,
        embedding_dim: int = 512,
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (640, 640),
        det_score_thresh: float = 0.5,
        ctx_id: int = 0,
    ) -> None:
        """
        Args:
            model_pack:          InsightFace model pack name.
                                 'buffalo_l' (best quality) |
                                 'buffalo_m' | 'buffalo_s' (faster).
            model_root:          Root directory for InsightFace downloads.
                                 Passed as ``root`` to FaceAnalysis.
            similarity_threshold: Cosine similarity threshold for a
                                  positive identity match [0.0, 1.0].
            embedding_dim:       ArcFace embedding size (512 for buffalo_l).
            providers:           ONNX Runtime execution providers in
                                 priority order.
                                 Default: CUDA → CPU.
            det_size:            (width, height) resolution used by the
                                 InsightFace internal detector.
            det_score_thresh:    Minimum detection confidence for the
                                 InsightFace internal detector.
            ctx_id:              GPU device index for InsightFace.
                                 -1 = CPU, 0 = first GPU.
        """
        super().__init__(
            model_pack=model_pack,
            model_root=model_root,
            similarity_threshold=similarity_threshold,
            embedding_dim=embedding_dim,
            providers=providers,
        )

        self.det_size = det_size
        self.det_score_thresh = det_score_thresh
        self.ctx_id = ctx_id

        self._load_lock = threading.Lock()
        self._app = None           # insightface.app.FaceAnalysis instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the InsightFace FaceAnalysis model pack.

        - Downloads the model pack automatically on first use if not
          present in ``model_root``.
        - Selects CUDA automatically if available.
        - Sets ``self._is_loaded = True`` on success.

        Raises:
            RuntimeError: If insightface is not installed or loading fails.
        """
        with self._load_lock:
            if self._is_loaded:
                logger.debug(
                    f"{self.__class__.__name__} already loaded — skipping."
                )
                return

            logger.info(
                f"Loading InsightFace recognizer | "
                f"pack={self.model_pack} | "
                f"root={self.model_root} | "
                f"providers={self.providers}"
            )

            t0 = self._timer()

            try:
                from insightface.app import FaceAnalysis  # noqa: PLC0415

                # Resolve the effective ctx_id based on available providers
                effective_ctx = self._resolve_ctx_id()

                self._app = FaceAnalysis(
                    name=self.model_pack,
                    root=str(self.model_root),
                    providers=self._resolve_providers(),
                )
                self._app.prepare(
                    ctx_id=effective_ctx,
                    det_size=self.det_size,
                    det_thresh=self.det_score_thresh,
                )

                self._model = self._app          # keep base class ref
                self._is_loaded = True

            except ImportError as exc:
                raise RuntimeError(
                    "insightface is not installed. "
                    "Run: pip install insightface>=0.7.3"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load InsightFace model pack "
                    f"'{self.model_pack}': {exc}"
                ) from exc

            elapsed = self._timer() - t0
            logger.success(
                f"InsightFace recognizer ready in {elapsed:.0f} ms | "
                f"pack={self.model_pack}"
            )

    # ------------------------------------------------------------------
    # Core embedding extraction
    # ------------------------------------------------------------------

    def get_embedding(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[FaceEmbedding]:
        """
        Extract a 512-dim ArcFace embedding from a face in *image*.

        Behaviour:
          - If *bbox* is provided, the face region is cropped + padded
            before being passed to InsightFace.
          - If *landmarks* are provided, they are used for precise face
            alignment (affine warp to canonical 112×112 crop).
          - If neither is given, InsightFace runs its own internal
            detector and returns the most confident face embedding.
          - Returns None if no face is found or inference fails.

        Args:
            image:     BGR numpy array (H, W, 3).
            bbox:      Optional (x1, y1, x2, y2) face bounding box.
            landmarks: Optional (5, 2) float32 landmark array.

        Returns:
            FaceEmbedding with normalised 512-dim vector,
            or None on failure.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._require_loaded()
        self._validate_image(image)

        t0 = self._timer()

        try:
            if landmarks is not None and landmarks.shape == (5, 2):
                return self._embed_aligned(
                    image, landmarks=landmarks, t0=t0
                )

            if bbox is not None:
                return self._embed_from_bbox(image, bbox=bbox, t0=t0)

            return self._embed_full_image(image, t0=t0)

        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.warning(f"get_embedding failed: {exc}")
            return None

    def get_all_embeddings(
        self,
        image: np.ndarray,
    ) -> List[FaceEmbedding]:
        """
        Extract embeddings for ALL faces detected in *image*.

        Uses InsightFace's internal detector to find every face, then
        returns one FaceEmbedding per face, sorted by detection score
        descending.

        Args:
            image: BGR numpy array (H, W, 3).

        Returns:
            List of FaceEmbedding (may be empty if no faces found).

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._require_loaded()
        self._validate_image(image)

        bgr = self._ensure_bgr(image)

        try:
            faces = self._app.get(bgr)
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.warning(f"InsightFace.get() failed: {exc}")
            return []

        if not faces:
            return []

        # Sort by detection score descending
        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)

        embeddings: List[FaceEmbedding] = []
        for idx, face in enumerate(faces):
            emb = self._face_to_embedding(face, face_index=idx)
            if emb is not None:
                embeddings.append(emb)

        return embeddings

    def get_embeddings_batch(
        self,
        images: List[np.ndarray],
        bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    ) -> List[Optional[FaceEmbedding]]:
        """
        Extract one embedding per image from a list.

        Calls ``get_embedding()`` for each image individually.
        (InsightFace does not natively support true image batching.)

        Args:
            images: List of BGR numpy arrays.
            bboxes: Optional aligned list of bounding boxes.

        Returns:
            List of FaceEmbedding (or None for failed images).
        """
        self._require_loaded()
        bboxes = bboxes or [None] * len(images)
        return [
            self.get_embedding(img, bbox=bb)
            for img, bb in zip(images, bboxes)
        ]

    # ------------------------------------------------------------------
    # Attribute prediction
    # ------------------------------------------------------------------

    def get_attributes(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[FaceAttribute]:
        """
        Predict demographic attributes (age, gender) for a face.

        Args:
            image: BGR numpy array (H, W, 3).
            bbox:  Optional face bounding box.  If None, InsightFace
                   detects the best face internally.

        Returns:
            FaceAttribute with age and gender fields populated,
            or None if no face was found.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._require_loaded()
        self._validate_image(image)

        bgr = self._ensure_bgr(image)

        if bbox is not None:
            bgr = self._crop_padded(bgr, bbox, pad_frac=0.20)

        try:
            faces = self._app.get(bgr)
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.warning(f"get_attributes InsightFace.get() failed: {exc}")
            return None

        if not faces:
            return None

        face = max(faces, key=lambda f: f.det_score)
        return self._extract_attributes(face)

    # ------------------------------------------------------------------
    # Full recognition pipeline (detect + embed + match)
    # ------------------------------------------------------------------

    def recognize(
        self,
        image: np.ndarray,
        database: "FaceDatabase",  # type: ignore[name-defined]  # forward ref
        *,
        frame_index: Optional[int] = None,
    ) -> RecognitionResult:
        """
        Run the full recognition pipeline on *image*:
          1. Detect all faces (InsightFace internal detector)
          2. Extract embeddings for each face
          3. Search the database for the closest match

        Args:
            image:       BGR numpy array (H, W, 3).
            database:    FaceDatabase to search for identity matches.
            frame_index: Optional frame number for video pipelines.

        Returns:
            RecognitionResult with one FaceMatch per detected face.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._require_loaded()
        self._validate_image(image)

        t0 = self._timer()
        h, w = image.shape[:2]

        embeddings = self.get_all_embeddings(image)

        matches: List[FaceMatch] = []
        for emb in embeddings:
            match = database.search(emb, threshold=self.similarity_threshold)
            matches.append(match)

        elapsed = self._timer() - t0

        return RecognitionResult(
            matches=matches,
            image_width=w,
            image_height=h,
            inference_time_ms=elapsed,
            frame_index=frame_index,
            metadata={
                "model_pack": self.model_pack,
                "threshold": self.similarity_threshold,
            },
        )

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free InsightFace model resources and CUDA memory."""
        if self._app is not None:
            try:
                del self._app
                self._app = None
            except Exception:
                pass

            # Clear CUDA cache if running on GPU
            if self.ctx_id >= 0:
                try:
                    import torch  # noqa: PLC0415
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("CUDA cache cleared after InsightFace release.")
                except Exception:
                    pass

        super().release()
        logger.info("InsightFaceRecognizer released.")

    # ------------------------------------------------------------------
    # Model info / diagnostics
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """
        Return a dict with model metadata.

        Returns:
            Dict with keys: model_pack, model_root, det_size,
            similarity_threshold, embedding_dim, providers, is_loaded.
        """
        return {
            "model_pack":           self.model_pack,
            "model_root":           str(self.model_root),
            "det_size":             self.det_size,
            "det_score_thresh":     self.det_score_thresh,
            "similarity_threshold": self.similarity_threshold,
            "embedding_dim":        self.embedding_dim,
            "providers":            self.providers,
            "ctx_id":               self.ctx_id,
            "is_loaded":            self._is_loaded,
        }

    def warmup(self, iterations: int = 2) -> float:
        """
        Run dummy inference passes to warm up the model.

        Args:
            iterations: Number of warm-up passes.

        Returns:
            Average time per pass in milliseconds.
        """
        self._require_loaded()
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        times = []
        for _ in range(iterations):
            t0 = self._timer()
            self.get_embedding(dummy)
            times.append(self._timer() - t0)
        avg = sum(times) / len(times)
        logger.info(f"InsightFace warmup done ({iterations} iters) — avg {avg:.1f} ms")
        return avg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_full_image(
        self,
        image: np.ndarray,
        *,
        t0: float,
    ) -> Optional[FaceEmbedding]:
        """
        Run InsightFace on the full image and return the most confident
        face embedding.
        """
        bgr = self._ensure_bgr(image)

        try:
            faces = self._app.get(bgr)
        except Exception as exc:
            logger.warning(f"InsightFace.get() failed: {exc}")
            return None

        if not faces:
            logger.debug("No faces detected by InsightFace internal detector.")
            return None

        # Pick most confident face
        best_face = max(faces, key=lambda f: f.det_score)
        return self._face_to_embedding(best_face, face_index=0)

    def _embed_from_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        t0: float,
    ) -> Optional[FaceEmbedding]:
        """
        Crop the region defined by *bbox* (with padding) and run
        InsightFace inside the crop.

        We pad the crop so InsightFace's internal detector can
        reliably find the face.
        """
        bgr = self._ensure_bgr(image)
        crop = self._crop_padded(bgr, bbox, pad_frac=0.20)

        try:
            faces = self._app.get(crop)
        except Exception as exc:
            logger.warning(f"InsightFace on bbox crop failed: {exc}")
            return None

        if not faces:
            # Fallback: try wider crop
            logger.debug("No face in tight crop, retrying with wider padding.")
            crop = self._crop_padded(bgr, bbox, pad_frac=0.40)
            try:
                faces = self._app.get(crop)
            except Exception:
                return None

        if not faces:
            return None

        best_face = max(faces, key=lambda f: f.det_score)
        emb = self._face_to_embedding(best_face, face_index=0)

        # Attach original bbox to the embedding
        if emb is not None:
            emb.bbox = bbox

        return emb

    def _embed_aligned(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        *,
        t0: float,
    ) -> Optional[FaceEmbedding]:
        """
        Align the face using 5-point landmarks and then extract the
        embedding from the aligned 112×112 crop.

        Uses the ArcFace canonical reference points for alignment.
        """
        bgr = self._ensure_bgr(image)

        try:
            aligned = self._align_face(bgr, landmarks, output_size=112)
        except Exception as exc:
            logger.warning(f"Face alignment failed: {exc}")
            # Fallback: run InsightFace on the raw image
            return self._embed_full_image(bgr, t0=t0)

        try:
            faces = self._app.get(aligned)
        except Exception as exc:
            logger.warning(f"InsightFace on aligned crop failed: {exc}")
            return None

        if not faces:
            return None

        best_face = max(faces, key=lambda f: f.det_score)
        emb = self._face_to_embedding(best_face, face_index=0)
        if emb is not None:
            emb.landmarks = landmarks

        return emb

    def _face_to_embedding(
        self,
        face,                         # insightface Face object
        face_index: int = 0,
    ) -> Optional[FaceEmbedding]:
        """
        Convert an InsightFace ``Face`` object to our ``FaceEmbedding``.

        InsightFace ``Face`` attributes we use:
            face.normed_embedding  — L2-normalised 512-dim vector
            face.embedding         — raw (un-normalised) embedding
            face.bbox              — (x1, y1, x2, y2) float array
            face.kps               — (5, 2) float keypoints
            face.age               — estimated age (float)
            face.gender            — 0 = Female, 1 = Male (int)
            face.det_score         — detection confidence

        Returns:
            FaceEmbedding or None if embedding is unavailable.
        """
        # Prefer normed_embedding; fall back to raw normalised manually
        raw_emb = getattr(face, "normed_embedding", None)
        if raw_emb is None:
            raw_emb = getattr(face, "embedding", None)
        if raw_emb is None:
            logger.debug("InsightFace Face object has no embedding.")
            return None

        vec = np.array(raw_emb, dtype=np.float32).flatten()
        # Ensure unit norm
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        else:
            logger.warning("Embedding norm near zero — degenerate extraction.")
            return None

        bbox_raw = getattr(face, "bbox", None)
        bbox = None
        if bbox_raw is not None:
            try:
                b = np.array(bbox_raw).flatten()
                bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            except Exception:
                bbox = None

        kps = getattr(face, "kps", None)
        landmarks = None
        if kps is not None:
            try:
                landmarks = np.array(kps, dtype=np.float32).reshape(5, 2)
            except Exception:
                landmarks = None

        attributes = self._extract_attributes(face)

        return FaceEmbedding(
            vector=vec,
            face_index=face_index,
            attributes=attributes,
            bbox=bbox,
            landmarks=landmarks,
        )

    @staticmethod
    def _extract_attributes(face) -> Optional[FaceAttribute]:
        """
        Extract age and gender from an InsightFace Face object.

        InsightFace encodes gender as:  0 = Female, 1 = Male.
        We normalise to 'F' / 'M' strings.
        """
        age    = getattr(face, "age",    None)
        gender = getattr(face, "gender", None)

        if age is None and gender is None:
            return None

        gender_str: Optional[str] = None
        gender_score: Optional[float] = None

        if gender is not None:
            try:
                g = int(gender)
                gender_str   = "M" if g == 1 else "F"
                gender_score = 1.0   # InsightFace doesn't expose soft score
            except (ValueError, TypeError):
                gender_str = None

        emb_norm = None
        raw_emb = getattr(face, "embedding", None)
        if raw_emb is not None:
            try:
                emb_norm = float(np.linalg.norm(raw_emb))
            except Exception:
                emb_norm = None

        return FaceAttribute(
            age=float(age) if age is not None else None,
            gender=gender_str,
            gender_score=gender_score,
            embedding_norm=emb_norm,
        )

    @staticmethod
    def _align_face(
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: int = 112,
    ) -> np.ndarray:
        """Affine-align a face to the ArcFace canonical crop using shared norm_crop."""
        crop, M = norm_crop(image, landmarks, output_size=output_size)
        if crop is None:
            raise ValueError("Could not estimate affine transform from landmarks.")
        return crop

    @staticmethod
    def _crop_padded(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        pad_frac: float = 0.20,
    ) -> np.ndarray:
        """
        Crop *image* at *bbox* with fractional padding, clamped to
        image bounds.

        Args:
            image:    BGR source image.
            bbox:     (x1, y1, x2, y2) pixel coordinates.
            pad_frac: Fractional padding added to each side.

        Returns:
            Cropped BGR ndarray.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        px = int(bw * pad_frac)
        py = int(bh * pad_frac)

        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px)
        cy2 = min(h, y2 + py)

        return image[cy1:cy2, cx1:cx2].copy()

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image * 255 if image.max() <= 1.0 else image,
                            0, 255).astype(np.uint8)
        return normalise_channels(image)

    def _resolve_ctx_id(self) -> int:
        """
        Resolve the effective InsightFace ctx_id based on available
        hardware and the configured providers.

        Returns:
            int — 0 (or self.ctx_id) for GPU, -1 for CPU.
        """
        if "CUDAExecutionProvider" not in self.providers:
            return -1

        try:
            import torch  # noqa: PLC0415
            if not torch.cuda.is_available():
                logger.debug(
                    "CUDA provider requested but torch.cuda not available "
                    "— falling back to CPU."
                )
                return -1
        except ImportError:
            return -1

        return max(self.ctx_id, 0)

    def _resolve_providers(self) -> List[str]:
        """
        Return the ONNX RT providers list, falling back to CPU-only
        if CUDA is not available.
        """
        try:
            import onnxruntime as ort  # noqa: PLC0415
            available = ort.get_available_providers()
            resolved = [p for p in self.providers if p in available]
            if not resolved:
                resolved = ["CPUExecutionProvider"]
            logger.debug(f"ONNX providers resolved: {resolved}")
            return resolved
        except ImportError:
            return ["CPUExecutionProvider"]

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"InsightFaceRecognizer("
            f"pack={self.model_pack!r}, "
            f"det_size={self.det_size}, "
            f"threshold={self.similarity_threshold}, "
            f"status={status})"
        )
