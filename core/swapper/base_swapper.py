# ============================================================
# AI Face Recognition & Face Swap
# core/swapper/base_swapper.py
# ============================================================
# Defines the abstract contract that ALL face swap engines must
# implement, plus shared data-types used throughout the pipeline.
#
# Hierarchy:
#   BaseSwapper  (abstract)
#       └── InSwapper        (inswapper_128.onnx)
#       └── <any future swapper>
#
# Key data types:
#   SwapRequest   — everything needed to perform one swap operation
#   SwapResult    — the output of a swap (frame + metadata)
#   BlendMode     — how the swapped face is composited back
# ============================================================

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector.base_detector import DetectionResult, FaceBox
from core.recognizer.base_recognizer import FaceEmbedding


# ============================================================
# Enumerations
# ============================================================

class BlendMode(Enum):
    """
    How the swapped face patch is composited back into the target frame.

    ALPHA         — Simple per-pixel alpha blend using a feathered ellipse mask.
                    Fast but may leave visible seams at the face boundary.

    POISSON       — OpenCV seamlessClone (Poisson blending).
                    Produces photorealistic seam-free results but is slower.

    MASKED_ALPHA  — Alpha blend using a custom skin-aware mask.
                    Better boundary handling than plain ALPHA.
    """
    ALPHA        = auto()
    POISSON      = auto()
    MASKED_ALPHA = auto()


class SwapStatus(Enum):
    """
    Status of a swap operation.

    SUCCESS         — Face was swapped successfully.
    NO_SOURCE_FACE  — Could not extract source face embedding.
    NO_TARGET_FACE  — No face detected in the target image/frame.
    INFERENCE_ERROR — ONNX Runtime inference raised an exception.
    ALIGN_ERROR     — Face alignment (affine warp) failed.
    BLEND_ERROR     — Paste-back or blending operation failed.
    MODEL_NOT_LOADED— Swap attempted before load_model() was called.
    """
    SUCCESS          = "success"
    NO_SOURCE_FACE   = "no_source_face"
    NO_TARGET_FACE   = "no_target_face"
    INFERENCE_ERROR  = "inference_error"
    ALIGN_ERROR      = "align_error"
    BLEND_ERROR      = "blend_error"
    MODEL_NOT_LOADED = "model_not_loaded"


# ============================================================
# Data Types
# ============================================================

@dataclass
class SwapRequest:
    """
    Everything needed to perform a single face-swap operation.

    Attributes:
        source_embedding:   ArcFace embedding of the source identity
                            (whose face will be copied INTO the target).
        target_image:       BGR numpy array — the frame/image to modify.
        target_face:        The specific face in *target_image* that will
                            be replaced.  Must carry valid 5-point landmarks
                            for alignment; if landmarks are missing the
                            swapper will estimate them from the bounding box.
        source_face_index:  Index of the source face (informational only).
        target_face_index:  Index of the target face (informational only).
        blend_mode:         How the swapped patch is composited back.
        blend_alpha:        Alpha weight for ALPHA blend mode [0.0, 1.0].
                            1.0 = fully swapped, 0.0 = original unchanged.
        mask_feather:       Gaussian blur radius for mask edge softening
                            (pixels).  Larger = softer boundary.
        enhance_after_swap: Hint to the pipeline to run face enhancement
                            (GFPGAN / CodeFormer) on the swapped result.
        metadata:           Optional free-form dict for debugging / tracing.
    """

    source_embedding:   FaceEmbedding
    target_image:       np.ndarray
    target_face:        FaceBox
    source_face_index:  int = 0
    target_face_index:  int = 0
    blend_mode:         BlendMode = BlendMode.POISSON
    blend_alpha:        float = 1.0
    mask_feather:       int = 20
    enhance_after_swap: bool = False
    metadata:           dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def target_has_landmarks(self) -> bool:
        """True if the target face carries 5-point landmark data."""
        return self.target_face.has_landmarks

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """(height, width, channels) of the target image."""
        return self.target_image.shape  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"SwapRequest("
            f"src_idx={self.source_face_index}, "
            f"tgt_idx={self.target_face_index}, "
            f"blend={self.blend_mode.name}, "
            f"alpha={self.blend_alpha}, "
            f"feather={self.mask_feather}px)"
        )


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SwapResult:
    """
    The complete output of a single face-swap operation.

    Attributes:
        output_image:       BGR numpy array — target frame with the face
                            swapped in.  Equal to the original frame if
                            swap failed (check *status*).
        status:             ``SwapStatus`` enum indicating success / failure.
        target_face:        The FaceBox that was replaced (or attempted).
        swap_time_ms:       Wall-clock time for the entire swap call (ms).
        inference_time_ms:  ONNX Runtime inference time only (ms).
        align_time_ms:      Face alignment time (ms).
        blend_time_ms:      Paste-back / blending time (ms).
        error:              Exception message string if status != SUCCESS.
        intermediate:       Optional dict with debug arrays
                            (aligned_face, swapped_patch, mask, …).
    """

    output_image:       np.ndarray
    status:             SwapStatus
    target_face:        FaceBox
    swap_time_ms:       float = 0.0
    inference_time_ms:  float = 0.0
    align_time_ms:      float = 0.0
    blend_time_ms:      float = 0.0
    error:              Optional[str] = None
    intermediate:       Optional[dict] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def success(self) -> bool:
        """True if the swap completed without errors."""
        return self.status == SwapStatus.SUCCESS

    @property
    def total_time_ms(self) -> float:
        """Alias for swap_time_ms (wall-clock total)."""
        return self.swap_time_ms

    def __repr__(self) -> str:
        if self.success:
            return (
                f"SwapResult("
                f"status=SUCCESS, "
                f"total={self.swap_time_ms:.1f}ms, "
                f"inference={self.inference_time_ms:.1f}ms)"
            )
        return (
            f"SwapResult("
            f"status={self.status.value}, "
            f"error={self.error!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchSwapResult:
    """
    The result of swapping multiple faces in a single frame (or batch of frames).

    Attributes:
        output_image:   Final BGR frame with all requested swaps applied.
        swap_results:   Individual SwapResult for each face that was processed.
        total_time_ms:  Total wall-clock time for all swaps.
        frame_index:    Optional frame number (video pipelines).
    """

    output_image:  np.ndarray
    swap_results:  List[SwapResult]
    total_time_ms: float = 0.0
    frame_index:   Optional[int] = None

    @property
    def num_swapped(self) -> int:
        """Number of faces successfully swapped."""
        return sum(1 for r in self.swap_results if r.success)

    @property
    def num_failed(self) -> int:
        """Number of faces that failed to swap."""
        return len(self.swap_results) - self.num_swapped

    @property
    def all_success(self) -> bool:
        """True if every face swap succeeded."""
        return all(r.success for r in self.swap_results)

    def __repr__(self) -> str:
        return (
            f"BatchSwapResult("
            f"swapped={self.num_swapped}/{len(self.swap_results)}, "
            f"total={self.total_time_ms:.1f}ms, "
            f"frame={self.frame_index})"
        )


# ============================================================
# Face Alignment Utilities
# (shared across all swapper implementations)
# ============================================================

# ArcFace canonical 5-point reference for 112 × 112
_ARCFACE_REF_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def get_reference_points(output_size: int = 128) -> np.ndarray:
    """
    Return the ArcFace canonical 5-point reference landmarks scaled to
    *output_size* × *output_size*.

    The inswapper_128 model expects a 128 × 128 aligned crop, so call
    this with ``output_size=128``.

    Args:
        output_size: Target square crop resolution in pixels.

    Returns:
        (5, 2) float32 array of reference landmark positions.
    """
    scale = output_size / 112.0
    return (_ARCFACE_REF_112 * scale).astype(np.float32)


def estimate_norm(
    landmarks: np.ndarray,
    output_size: int = 128,
) -> Optional[np.ndarray]:
    """
    Compute the 2 × 3 affine matrix that maps *landmarks* to the
    ArcFace canonical reference grid at *output_size* × *output_size*.

    Uses ``cv2.estimateAffinePartial2D`` with LMEDS (robust to outliers).

    Args:
        landmarks:   (5, 2) float32 facial keypoints in source image space.
                     Order: left_eye, right_eye, nose, left_mouth, right_mouth.
        output_size: Square output resolution (default 128 for inswapper).

    Returns:
        (2, 3) float32 affine matrix, or None if estimation fails.
    """
    ref = get_reference_points(output_size)
    src = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)
    return M


def norm_crop(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 128,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Affine-align a face to a canonical *output_size* × *output_size* crop.

    Args:
        image:       BGR source frame (H, W, 3).
        landmarks:   (5, 2) float32 facial keypoints.
        output_size: Square target resolution (128 for inswapper_128).

    Returns:
        Tuple of (aligned_crop, affine_matrix_2x3).
        Both are None if the affine estimation failed.
    """
    M = estimate_norm(landmarks, output_size)
    if M is None:
        return None, None

    aligned = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, M


def estimate_landmarks_from_bbox(bbox: FaceBox) -> np.ndarray:
    """
    Estimate rough 5-point landmarks from a bounding box alone.

    Used as a fallback when no landmark data is available.
    The estimates are based on typical face proportions.

    Args:
        bbox: FaceBox with (x1, y1, x2, y2) coordinates.

    Returns:
        (5, 2) float32 approximate landmarks:
        [left_eye, right_eye, nose, left_mouth, right_mouth].
    """
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    w = x2 - x1
    h = y2 - y1

    # Typical facial proportions as fractions of bbox
    lm = np.array(
        [
            [x1 + w * 0.30, y1 + h * 0.37],   # left eye
            [x1 + w * 0.70, y1 + h * 0.37],   # right eye
            [x1 + w * 0.50, y1 + h * 0.55],   # nose tip
            [x1 + w * 0.35, y1 + h * 0.75],   # left mouth corner
            [x1 + w * 0.65, y1 + h * 0.75],   # right mouth corner
        ],
        dtype=np.float32,
    )
    return lm


def paste_back(
    original: np.ndarray,
    swapped_crop: np.ndarray,
    affine_matrix: np.ndarray,
    blend_mask: Optional[np.ndarray] = None,
    feather: int = 20,
) -> np.ndarray:
    """
    Warp *swapped_crop* back into *original* using the inverse of
    *affine_matrix*, then blend using *blend_mask*.

    Args:
        original:       BGR target frame (H, W, 3).
        swapped_crop:   BGR swapped face crop (N, N, 3) where N matches
                        the output_size used during alignment.
        affine_matrix:  (2, 3) float32 forward affine matrix (face → canonical).
        blend_mask:     Optional (N, N) uint8 mask for the crop.
                        Defaults to a feathered ellipse covering the crop.
        feather:        Blur radius for default mask feathering.

    Returns:
        BGR frame with the swapped face pasted back.
    """
    H, W = original.shape[:2]
    crop_size = swapped_crop.shape[0]

    # ── Build the default mask if none provided ─────────────────────
    if blend_mask is None:
        blend_mask = _make_crop_mask(crop_size, feather)

    # ── Invert affine: canonical → source image space ───────────────
    inv_M = cv2.invertAffineTransform(affine_matrix)

    # ── Warp swapped crop back to full-frame space ───────────────────
    warped_face = cv2.warpAffine(
        swapped_crop,
        inv_M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # ── Warp the mask into the same space ───────────────────────────
    warped_mask = cv2.warpAffine(
        blend_mask,
        inv_M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    # ── Alpha blend ─────────────────────────────────────────────────
    alpha = warped_mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]   # (H, W, 1)

    result = (
        warped_face.astype(np.float32) * alpha
        + original.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def paste_back_poisson(
    original: np.ndarray,
    swapped_crop: np.ndarray,
    affine_matrix: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Paste the swapped face back using Poisson (seamlessClone) blending.

    More photorealistic than plain alpha blending but slower.
    Falls back to alpha blending if seamlessClone raises an error.

    Args:
        original:       BGR target frame (H, W, 3).
        swapped_crop:   BGR swapped face crop.
        affine_matrix:  (2, 3) forward affine matrix.
        mask:           Optional crop-space uint8 mask.

    Returns:
        BGR frame with Poisson-blended swapped face.
    """
    H, W = original.shape[:2]
    crop_size = swapped_crop.shape[0]

    if mask is None:
        mask = _make_crop_mask(crop_size, feather=0)   # hard mask for Poisson

    inv_M = cv2.invertAffineTransform(affine_matrix)

    # Warp crop and mask back to full frame space
    warped_face = cv2.warpAffine(
        swapped_crop, inv_M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )

    # Find the centre of the mask region for seamlessClone
    ys, xs = np.where(warped_mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        # Mask is empty — fall back to alpha
        return paste_back(original, swapped_crop, affine_matrix)

    cx = int(xs.mean())
    cy = int(ys.mean())

    try:
        result = cv2.seamlessClone(
            warped_face,
            original,
            warped_mask,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
    except cv2.error:
        # Fallback to alpha blend if Poisson fails
        result = paste_back(original, swapped_crop, affine_matrix)

    return result


def _make_crop_mask(size: int, feather: int = 20) -> np.ndarray:
    """
    Create a soft elliptical mask for a square crop of *size* × *size*.

    Args:
        size:    Square canvas dimension in pixels.
        feather: Gaussian blur radius for edge softening.

    Returns:
        (size, size) uint8 mask — white inside the ellipse, feathered edges.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    ax = int(size * 0.45)
    ay = int(size * 0.48)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    if feather > 0:
        ksize = max(3, feather * 2 + 1)
        if ksize % 2 == 0:
            ksize += 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


# ============================================================
# Abstract Base Swapper
# ============================================================

class BaseSwapper(ABC):
    """
    Abstract base class for all face swap engines.

    Subclasses must implement:
        - ``load_model()``           — load ONNX / PyTorch weights
        - ``swap(request)``          — perform one face swap

    Optional overrides:
        - ``swap_all(...)``          — swap every face in a frame
        - ``release()``              — free GPU / memory resources

    Context-manager usage (auto load + release)::

        with InSwapper(model_path="models/inswapper_128.onnx") as swapper:
            result = swapper.swap(request)

    Or load manually::

        swapper = InSwapper(model_path="models/inswapper_128.onnx")
        swapper.load_model()
        result = swapper.swap(request)
        swapper.release()
    """

    def __init__(
        self,
        model_path: str = "models/inswapper_128.onnx",
        providers: Optional[List[str]] = None,
        blend_mode: BlendMode = BlendMode.POISSON,
        blend_alpha: float = 1.0,
        mask_feather: int = 20,
        input_size: int = 128,
    ) -> None:
        """
        Args:
            model_path:   Path to the ONNX model file.
            providers:    ONNX Runtime execution providers in priority order.
                          Defaults to [CUDAExecutionProvider, CPUExecutionProvider].
            blend_mode:   Default compositing mode for paste-back.
            blend_alpha:  Default alpha weight [0.0, 1.0].
            mask_feather: Default Gaussian blur radius for mask edges (px).
            input_size:   Square input resolution expected by the model (128).
        """
        self.model_path   = model_path
        self.providers    = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.blend_mode   = blend_mode
        self.blend_alpha  = float(blend_alpha)
        self.mask_feather = int(mask_feather)
        self.input_size   = int(input_size)

        self._model      = None
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the swap model weights from ``self.model_path``.

        Must:
          - Populate ``self._model``
          - Set ``self._is_loaded = True``
          - Raise ``FileNotFoundError`` if the model file is missing.
          - Raise ``RuntimeError`` on any other loading failure.
        """

    @abstractmethod
    def swap(self, request: SwapRequest) -> SwapResult:
        """
        Swap the source face identity into the target face region.

        Args:
            request: ``SwapRequest`` describing what to swap and how.

        Returns:
            ``SwapResult`` — always returns an object; check ``.success``
            or ``.status`` to determine whether the swap succeeded.

        Raises:
            RuntimeError: If the model has not been loaded.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — subclasses MAY override
    # ------------------------------------------------------------------

    def swap_all(
        self,
        source_embedding: FaceEmbedding,
        target_image: np.ndarray,
        target_detection: DetectionResult,
        *,
        blend_mode: Optional[BlendMode] = None,
        blend_alpha: Optional[float] = None,
        mask_feather: Optional[int] = None,
        max_faces: Optional[int] = None,
    ) -> BatchSwapResult:
        """
        Swap *source_embedding* into every detected face in *target_image*.

        Applies the swaps sequentially, passing the updated frame from
        each step to the next so that multiple faces in one image are all
        replaced.

        Args:
            source_embedding:    ArcFace embedding of the donor face.
            target_image:        BGR frame to modify.
            target_detection:    ``DetectionResult`` containing all target faces.
            blend_mode:          Override default blend mode for this call.
            blend_alpha:         Override default alpha for this call.
            mask_feather:        Override default feather for this call.
            max_faces:           Cap on how many faces to swap (None = all).

        Returns:
            ``BatchSwapResult`` with the final composited frame and
            individual ``SwapResult`` objects for each face.
        """
        self._require_loaded()

        t0      = self._timer()
        current = target_image.copy()
        results: List[SwapResult] = []

        faces = target_detection.faces
        if max_faces is not None:
            faces = faces[:max_faces]

        for face in faces:
            req = SwapRequest(
                source_embedding=source_embedding,
                target_image=current,
                target_face=face,
                source_face_index=0,
                target_face_index=face.face_index,
                blend_mode=blend_mode   or self.blend_mode,
                blend_alpha=blend_alpha if blend_alpha is not None else self.blend_alpha,
                mask_feather=mask_feather if mask_feather is not None else self.mask_feather,
            )
            result = self.swap(req)
            results.append(result)

            # Use the updated frame for the next iteration
            if result.success:
                current = result.output_image

        return BatchSwapResult(
            output_image=current,
            swap_results=results,
            total_time_ms=self._timer() - t0,
        )

    def release(self) -> None:
        """
        Release model resources.

        The default clears ``self._model`` and resets ``self._is_loaded``.
        Subclasses should call ``super().release()`` after their cleanup.
        """
        self._model      = None
        self._is_loaded  = False

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
        import os
        return os.path.basename(self.model_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseSwapper":
        """Auto-load the model when used as a context manager."""
        if not self._is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Auto-release resources on context manager exit."""
        self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise RuntimeError if the model is not loaded yet."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} model is not loaded. "
                "Call load_model() first or use as a context manager."
            )

    def _get_landmarks(self, face: FaceBox) -> np.ndarray:
        """
        Return 5-point landmarks for *face*, estimating from bbox if
        landmark data is missing.

        Args:
            face: FaceBox (may or may not have .landmarks).

        Returns:
            (5, 2) float32 landmark array.
        """
        if face.has_landmarks and face.landmarks is not None:
            return face.landmarks.astype(np.float32)
        return estimate_landmarks_from_bbox(face)

    def _validate_image(self, image: np.ndarray) -> None:
        """Raise ValueError for obviously invalid images."""
        if image is None:
            raise ValueError("Image is None.")
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Expected numpy ndarray, got {type(image).__name__}."
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected BGR (H, W, 3) array, got shape {image.shape}."
            )
        if image.size == 0:
            raise ValueError("Image array is empty.")

    def _make_failed_result(
        self,
        status: SwapStatus,
        target_image: np.ndarray,
        target_face: FaceBox,
        error: str,
        t0: float,
    ) -> SwapResult:
        """
        Convenience factory for a failed SwapResult.

        Returns the original (unmodified) image so the pipeline can
        continue even when one face fails.

        Args:
            status:       Failure status enum.
            target_image: The unmodified original frame.
            target_face:  The face that failed to swap.
            error:        Error description string.
            t0:           Start timestamp from ``self._timer()``.

        Returns:
            SwapResult with success=False and output_image=target_image.
        """
        return SwapResult(
            output_image=target_image,
            status=status,
            target_face=target_face,
            swap_time_ms=self._timer() - t0,
            error=error,
        )

    @staticmethod
    def _timer() -> float:
        """Return current time in milliseconds (monotonic clock)."""
        return time.perf_counter() * 1000.0

    @staticmethod
    def _resolve_providers(requested: List[str]) -> List[str]:
        """
        Filter *requested* providers to those actually available in the
        current ONNX Runtime installation.

        Falls back to CPU-only if nothing matches.

        Args:
            requested: Ordered list of desired provider strings.

        Returns:
            Filtered list of available providers.
        """
        try:
            import onnxruntime as ort  # noqa: PLC0415
            available = set(ort.get_available_providers())
            resolved  = [p for p in requested if p in available]
            return resolved or ["CPUExecutionProvider"]
        except ImportError:
            return ["CPUExecutionProvider"]

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"blend={self.blend_mode.name}, "
            f"status={status})"
        )
