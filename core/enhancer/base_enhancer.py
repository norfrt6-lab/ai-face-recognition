# ============================================================
# AI Face Recognition & Face Swap
# core/enhancer/base_enhancer.py
# ============================================================
# Defines the abstract contract that ALL face enhancement engines
# must implement, plus shared data-types used throughout the pipeline.
#
# Hierarchy:
#   BaseEnhancer  (abstract)
#       └── GFPGANEnhancer      (GFPGAN v1.4)
#       └── CodeFormerEnhancer  (CodeFormer)
#       └── <any future enhancer>
#
# Key data types:
#   EnhancementRequest — everything needed to run one enhancement pass
#   EnhancementResult  — the output of enhancement (frame + metadata)
#   EnhancerBackend    — enum of supported backends
#   EnhancementStatus  — success / failure enum
# ============================================================

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector.base_detector import FaceBox


# ============================================================
# Enumerations
# ============================================================

class EnhancerBackend(Enum):
    """
    Supported face enhancement backends.

    GFPGAN      — GFPGAN v1.4: fast, good general-purpose restoration.
    CODEFORMER  — CodeFormer: higher fidelity control via fidelity_weight.
    NONE        — Pass-through (no enhancement applied).
    """
    GFPGAN     = "gfpgan"
    CODEFORMER = "codeformer"
    NONE       = "none"


class EnhancementStatus(Enum):
    """
    Status of a face enhancement operation.

    SUCCESS           — Enhancement completed successfully.
    NO_FACE_DETECTED  — No face found in the input crop.
    INFERENCE_ERROR   — Model raised an exception during inference.
    MODEL_NOT_LOADED  — enhance() called before load_model().
    INVALID_INPUT     — Input image failed validation.
    DISABLED          — Backend is set to NONE (pass-through).
    """
    SUCCESS          = "success"
    NO_FACE_DETECTED = "no_face_detected"
    INFERENCE_ERROR  = "inference_error"
    MODEL_NOT_LOADED = "model_not_loaded"
    INVALID_INPUT    = "invalid_input"
    DISABLED         = "disabled"


# ============================================================
# Data Types
# ============================================================

@dataclass
class EnhancementRequest:
    """
    Everything needed to perform a single face enhancement operation.

    Attributes:
        image:              BGR numpy array — the face crop or full frame
                            to enhance.  When ``full_frame=False`` the
                            image is expected to be a tight face crop;
                            when ``full_frame=True`` the enhancer will
                            locate faces internally and paste back.
        face_boxes:         Optional list of pre-detected FaceBox objects.
                            When provided, the enhancer uses these
                            bounding boxes instead of running its own
                            face detector.
        fidelity_weight:    Balance between restoration quality and
                            identity fidelity [0.0, 1.0].
                            0.0 = maximum enhancement (may alter identity).
                            1.0 = preserve original details most closely.
                            Only meaningful for CodeFormer.
        upscale:            Integer upscale factor (1, 2, or 4).
        only_center_face:   When True, enhance only the most central
                            detected face and leave others untouched.
        paste_back:         When True, paste enhanced face(s) back into
                            the original full-resolution image.
        full_frame:         When True, *image* is a full scene frame;
                            the enhancer detects and enhances all faces.
                            When False, *image* is already a face crop.
        metadata:           Optional free-form dict for debugging / tracing.
    """

    image:            np.ndarray
    face_boxes:       Optional[List[FaceBox]] = None
    fidelity_weight:  float = 0.5
    upscale:          int   = 2
    only_center_face: bool  = False
    paste_back:       bool  = True
    full_frame:       bool  = True
    metadata:         dict  = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def image_shape(self) -> Tuple[int, int, int]:
        """(height, width, channels) of the input image."""
        return self.image.shape  # type: ignore[return-value]

    @property
    def has_face_boxes(self) -> bool:
        """True if pre-detected face boxes were provided."""
        return self.face_boxes is not None and len(self.face_boxes) > 0

    def __repr__(self) -> str:
        h, w = self.image.shape[:2]
        return (
            f"EnhancementRequest("
            f"image={w}x{h}, "
            f"full_frame={self.full_frame}, "
            f"upscale={self.upscale}x, "
            f"fidelity={self.fidelity_weight:.2f}, "
            f"faces={'pre-detected' if self.has_face_boxes else 'auto-detect'})"
        )


# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EnhancementResult:
    """
    The complete output of a single face enhancement operation.

    Attributes:
        output_image:       BGR numpy array — enhanced image.
                            Equal to the input image if enhancement
                            failed or was disabled (check *status*).
        status:             ``EnhancementStatus`` indicating success /
                            failure reason.
        backend:            Which backend produced this result.
        num_faces_enhanced: Number of faces that were enhanced.
        enhance_time_ms:    Total wall-clock time for the call (ms).
        inference_time_ms:  Model inference time only (ms).
        upscale_factor:     Actual upscale factor applied (may differ
                            from requested if model has constraints).
        error:              Exception / error description string.
                            None on success.
        face_crops:         Optional list of enhanced face crops
                            (before paste-back), keyed by face index.
    """

    output_image:       np.ndarray
    status:             EnhancementStatus
    backend:            EnhancerBackend
    num_faces_enhanced: int   = 0
    enhance_time_ms:    float = 0.0
    inference_time_ms:  float = 0.0
    upscale_factor:     int   = 1
    error:              Optional[str] = None
    face_crops:         Optional[List[np.ndarray]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def success(self) -> bool:
        """True if enhancement completed without errors."""
        return self.status == EnhancementStatus.SUCCESS

    @property
    def is_passthrough(self) -> bool:
        """True if no enhancement was applied (backend=NONE or DISABLED)."""
        return self.status == EnhancementStatus.DISABLED

    def __repr__(self) -> str:
        if self.success:
            return (
                f"EnhancementResult("
                f"status=SUCCESS, "
                f"backend={self.backend.value}, "
                f"faces={self.num_faces_enhanced}, "
                f"upscale={self.upscale_factor}x, "
                f"total={self.enhance_time_ms:.1f}ms)"
            )
        return (
            f"EnhancementResult("
            f"status={self.status.value}, "
            f"backend={self.backend.value}, "
            f"error={self.error!r})"
        )


# ============================================================
# Shared Utilities
# ============================================================

def pad_image_for_enhancement(
    image: np.ndarray,
    min_size: int = 128,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad *image* to at least *min_size* × *min_size* if it is too small.

    Some enhancement models require a minimum input resolution.

    Args:
        image:    BGR (H, W, 3) input image.
        min_size: Minimum side length in pixels.

    Returns:
        Tuple of (padded_image, (pad_top, pad_bottom, pad_left, pad_right)).
        Padding values are zero if the image was already large enough.
    """
    h, w = image.shape[:2]
    pad_top = pad_bottom = pad_left = pad_right = 0

    if h < min_size:
        total = min_size - h
        pad_top    = total // 2
        pad_bottom = total - pad_top

    if w < min_size:
        total = min_size - w
        pad_left  = total // 2
        pad_right = total - pad_left

    if pad_top or pad_bottom or pad_left or pad_right:
        image = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )

    return image, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(
    image: np.ndarray,
    padding: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Remove padding applied by ``pad_image_for_enhancement``.

    Args:
        image:   Padded BGR image.
        padding: (pad_top, pad_bottom, pad_left, pad_right) tuple.

    Returns:
        Cropped image with padding removed.
    """
    pad_top, pad_bottom, pad_left, pad_right = padding
    h, w = image.shape[:2]
    y1 = pad_top
    y2 = h - pad_bottom if pad_bottom else h
    x1 = pad_left
    x2 = w - pad_right  if pad_right  else w
    return image[y1:y2, x1:x2]


def find_center_face(face_boxes: List[FaceBox], image_w: int, image_h: int) -> Optional[FaceBox]:
    """
    Return the FaceBox whose centre is closest to the image centre.

    Args:
        face_boxes: List of detected FaceBox objects.
        image_w:    Image width in pixels.
        image_h:    Image height in pixels.

    Returns:
        The most central FaceBox, or None if the list is empty.
    """
    if not face_boxes:
        return None

    img_cx = image_w / 2.0
    img_cy = image_h / 2.0

    def _dist(fb: FaceBox) -> float:
        cx, cy = fb.center
        return (cx - img_cx) ** 2 + (cy - img_cy) ** 2

    return min(face_boxes, key=_dist)


# ============================================================
# Abstract Base Enhancer
# ============================================================

class BaseEnhancer(ABC):
    """
    Abstract base class for all face enhancement engines.

    Subclasses must implement:
        - ``load_model()``          — load weights from disk
        - ``enhance(request)``      — perform one enhancement pass

    Optional overrides:
        - ``release()``             — free GPU / memory resources

    Context-manager usage (auto load + release)::

        with GFPGANEnhancer(model_path="models/GFPGANv1.4.pth") as enh:
            result = enh.enhance(request)

    Or load manually::

        enh = GFPGANEnhancer(model_path="models/GFPGANv1.4.pth")
        enh.load_model()
        result = enh.enhance(request)
        enh.release()
    """

    def __init__(
        self,
        model_path: str,
        backend: EnhancerBackend,
        upscale: int = 2,
        only_center_face: bool = False,
        paste_back: bool = True,
        device: str = "auto",
    ) -> None:
        """
        Args:
            model_path:        Path to the model weights file.
            backend:           Which ``EnhancerBackend`` this instance represents.
            upscale:           Default upscale factor (1, 2, or 4).
            only_center_face:  Default for center-face-only mode.
            paste_back:        Default paste-back behaviour.
            device:            Inference device: 'auto' | 'cpu' | 'cuda' | 'cuda:N'.
        """
        self.model_path       = model_path
        self.backend          = backend
        self.upscale          = int(upscale)
        self.only_center_face = bool(only_center_face)
        self.paste_back       = bool(paste_back)
        self.device           = device

        self._model      = None
        self._is_loaded: bool = False

        # Cumulative statistics
        self._total_calls:     int   = 0
        self._total_inference: float = 0.0

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the enhancement model weights from ``self.model_path``.

        Must:
          - Populate ``self._model``
          - Set ``self._is_loaded = True``
          - Raise ``FileNotFoundError`` if the weights file is missing.
          - Raise ``RuntimeError`` on any other loading failure.
        """

    @abstractmethod
    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        """
        Enhance face(s) in the input image.

        Args:
            request: ``EnhancementRequest`` describing input + parameters.

        Returns:
            ``EnhancementResult`` — always returns an object; check
            ``.success`` or ``.status`` to determine outcome.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — subclasses MAY override
    # ------------------------------------------------------------------

    def enhance_image(
        self,
        image: np.ndarray,
        *,
        fidelity_weight: Optional[float] = None,
        upscale: Optional[int] = None,
        only_center_face: Optional[bool] = None,
        paste_back: Optional[bool] = None,
    ) -> EnhancementResult:
        """
        Convenience wrapper: enhance a raw BGR image with optional overrides.

        Args:
            image:            BGR numpy array to enhance.
            fidelity_weight:  Override default fidelity_weight.
            upscale:          Override default upscale factor.
            only_center_face: Override default center-face mode.
            paste_back:       Override default paste-back behaviour.

        Returns:
            ``EnhancementResult``.
        """
        request = EnhancementRequest(
            image=image,
            fidelity_weight=fidelity_weight if fidelity_weight is not None else 0.5,
            upscale=upscale if upscale is not None else self.upscale,
            only_center_face=only_center_face if only_center_face is not None else self.only_center_face,
            paste_back=paste_back if paste_back is not None else self.paste_back,
            full_frame=True,
        )
        return self.enhance(request)

    def release(self) -> None:
        """
        Release model resources.

        The default clears ``self._model`` and resets ``self._is_loaded``.
        Subclasses should call ``super().release()`` after their cleanup.
        """
        self._model     = None
        self._is_loaded = False

    def reset_stats(self) -> None:
        """Reset cumulative inference statistics."""
        self._total_calls     = 0
        self._total_inference = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True if ``load_model()`` has been called successfully."""
        return self._is_loaded

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (file basename)."""
        import os
        return os.path.basename(self.model_path)

    @property
    def avg_inference_ms(self) -> float:
        """Average model inference time per call in milliseconds."""
        if self._total_calls == 0:
            return 0.0
        return self._total_inference / self._total_calls

    @property
    def total_calls(self) -> int:
        """Total number of ``enhance()`` calls since the model was loaded."""
        return self._total_calls

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseEnhancer":
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

    def _resolve_device(self) -> str:
        """
        Resolve 'auto' device to 'cuda' or 'cpu' based on availability.

        Returns:
            Resolved device string: 'cuda' | 'cuda:N' | 'cpu'.
        """
        if self.device != "auto":
            return self.device

        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        return "cpu"

    def _make_failed_result(
        self,
        status: EnhancementStatus,
        image: np.ndarray,
        error: str,
        t0: float,
    ) -> EnhancementResult:
        """
        Convenience factory for a failed EnhancementResult.

        Returns the original (unmodified) image so the pipeline can
        continue even when enhancement fails.

        Args:
            status: Failure status enum.
            image:  The unmodified original image.
            error:  Error description string.
            t0:     Start timestamp from ``self._timer()``.

        Returns:
            EnhancementResult with success=False and output_image=image.
        """
        return EnhancementResult(
            output_image=image,
            status=status,
            backend=self.backend,
            enhance_time_ms=self._timer() - t0,
            error=error,
        )

    @staticmethod
    def _timer() -> float:
        """Return current time in milliseconds (monotonic clock)."""
        return time.perf_counter() * 1000.0

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"backend={self.backend.value}, "
            f"model={self.model_name!r}, "
            f"upscale={self.upscale}x, "
            f"device={self.device!r}, "
            f"status={status})"
        )
