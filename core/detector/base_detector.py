# Defines the abstract contract that ALL face detectors must
# implement, plus shared data-types used throughout the pipeline.
#
# Hierarchy:
#   BaseDetector  (abstract)
#       └── YOLOFaceDetector
#       └── <any future detector>

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FaceBox:
    """
    A single detected face bounding box.

    All coordinates are in absolute pixel space of the source image
    (not normalised 0-1 values).

    Attributes:
        x1:          Left edge of the bounding box (pixels).
        y1:          Top edge of the bounding box (pixels).
        x2:          Right edge of the bounding box (pixels).
        y2:          Bottom edge of the bounding box (pixels).
        confidence:  Detection confidence score in [0.0, 1.0].
        face_index:  Zero-based index of this face within the detection
                     result (sorted by confidence descending by default).
        landmarks:   Optional (5, 2) float32 array of facial keypoints
                     [left_eye, right_eye, nose, left_mouth, right_mouth].
                     Not all detectors supply landmarks.
        track_id:    Optional object-tracking ID (for video pipelines).
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    face_index: int = 0
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)
    track_id: Optional[int] = None

    # ------------------------------------------------------------------
    # Derived geometry properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Bounding box width in pixels."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Bounding box height in pixels."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """Bounding box area in pixels²."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Centre point (cx, cy) of the bounding box."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def aspect_ratio(self) -> float:
        """Width-to-height ratio (> 1 = wider than tall)."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) as a plain tuple."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def as_xywh(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) format."""
        return self.x1, self.y1, self.width, self.height

    @property
    def has_landmarks(self) -> bool:
        """True if 5-point landmark data is available."""
        return self.landmarks is not None and self.landmarks.shape == (5, 2)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def scale(self, sx: float, sy: float) -> "FaceBox":
        """
        Return a new FaceBox with coordinates scaled by (sx, sy).

        Useful when images are resized before detection and you need to
        map boxes back to the original resolution.

        Args:
            sx: Horizontal scale factor (new_width / old_width).
            sy: Vertical scale factor   (new_height / old_height).

        Returns:
            New FaceBox with scaled coordinates.
        """
        new_lm = None
        if self.landmarks is not None:
            new_lm = self.landmarks * np.array([sx, sy], dtype=np.float32)

        return FaceBox(
            x1=int(self.x1 * sx),
            y1=int(self.y1 * sy),
            x2=int(self.x2 * sx),
            y2=int(self.y2 * sy),
            confidence=self.confidence,
            face_index=self.face_index,
            landmarks=new_lm,
            track_id=self.track_id,
        )

    def pad(
        self,
        px: int = 0,
        py: int = 0,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> "FaceBox":
        """
        Return a new FaceBox expanded by *px* pixels horizontally and
        *py* pixels vertically, optionally clamped to image bounds.

        Args:
            px:    Horizontal padding (added to each side).
            py:    Vertical padding (added to each side).
            img_w: Image width used to clamp x2.  None = no clamp.
            img_h: Image height used to clamp y2.  None = no clamp.

        Returns:
            New padded FaceBox.
        """
        x1 = max(0, self.x1 - px)
        y1 = max(0, self.y1 - py)
        x2 = self.x2 + px if img_w is None else min(img_w, self.x2 + px)
        y2 = self.y2 + py if img_h is None else min(img_h, self.y2 + py)

        return FaceBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=self.confidence,
            face_index=self.face_index,
            landmarks=self.landmarks,
            track_id=self.track_id,
        )

    def pad_fractional(
        self,
        frac: float = 0.1,
        img_w: Optional[int] = None,
        img_h: Optional[int] = None,
    ) -> "FaceBox":
        """
        Expand the box by *frac* × its own dimensions on each side.

        E.g. frac=0.1 adds 10 % of the box width left/right and
        10 % of the box height top/bottom.

        Args:
            frac:  Fractional expansion (0.1 = 10 %).
            img_w: Optional image width clamp.
            img_h: Optional image height clamp.

        Returns:
            New FaceBox with fractional padding.
        """
        return self.pad(
            px=int(self.width * frac),
            py=int(self.height * frac),
            img_w=img_w,
            img_h=img_h,
        )

    def clamp(self, img_w: int, img_h: int) -> "FaceBox":
        """
        Clamp all coordinates to be within [0, img_w] × [0, img_h].

        Args:
            img_w: Image width.
            img_h: Image height.

        Returns:
            New FaceBox with clamped coordinates.
        """
        return FaceBox(
            x1=max(0, min(img_w, self.x1)),
            y1=max(0, min(img_h, self.y1)),
            x2=max(0, min(img_w, self.x2)),
            y2=max(0, min(img_h, self.y2)),
            confidence=self.confidence,
            face_index=self.face_index,
            landmarks=self.landmarks,
            track_id=self.track_id,
        )

    def iou(self, other: "FaceBox") -> float:
        """
        Compute the Intersection-over-Union with another FaceBox.

        Args:
            other: The second bounding box.

        Returns:
            IoU score in [0.0, 1.0].
        """
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop this face region from *image*.

        Args:
            image: Source BGR numpy array (H, W, 3).

        Returns:
            Cropped numpy array containing only the face region.
        """
        h, w = image.shape[:2]
        x1 = max(0, self.x1)
        y1 = max(0, self.y1)
        x2 = min(w, self.x2)
        y2 = min(h, self.y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=image.dtype)
        return image[y1:y2, x1:x2].copy()

    def __repr__(self) -> str:
        lm_str = f", landmarks={'yes' if self.has_landmarks else 'no'}"
        return (
            f"FaceBox(idx={self.face_index}, "
            f"bbox=[{self.x1},{self.y1},{self.x2},{self.y2}], "
            f"conf={self.confidence:.3f}, "
            f"size={self.width}×{self.height}"
            f"{lm_str})"
        )


# ── convenience constructor ──────────────────────────────────


def face_box_from_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    face_index: int = 0,
    landmarks: Optional[np.ndarray] = None,
) -> FaceBox:
    """
    Create a FaceBox from float coordinates, rounding to int.

    Args:
        x1, y1, x2, y2: Bounding box corners (float, pixel space).
        confidence:      Detection confidence [0, 1].
        face_index:      Sorted position in the result list.
        landmarks:       Optional (5, 2) float32 landmark array.

    Returns:
        FaceBox instance.
    """
    return FaceBox(
        x1=int(round(x1)),
        y1=int(round(y1)),
        x2=int(round(x2)),
        y2=int(round(y2)),
        confidence=float(confidence),
        face_index=face_index,
        landmarks=landmarks,
    )


@dataclass
class DetectionResult:
    """
    The complete output of a single detection call on one image/frame.

    Attributes:
        faces:             List of detected FaceBox objects, sorted by
                           confidence (highest first).
        image_width:       Width of the source image in pixels.
        image_height:      Height of the source image in pixels.
        inference_time_ms: Wall-clock time for the detector inference
                           (does NOT include pre/post-processing).
        frame_index:       Optional frame number (for video pipelines).
        metadata:          Optional free-form dict for extra info
                           (e.g. model name, device, input scale).
    """

    faces: List[FaceBox]
    image_width: int
    image_height: int
    inference_time_ms: float = 0.0
    frame_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_faces(self) -> int:
        """Number of detected faces."""
        return len(self.faces)

    @property
    def is_empty(self) -> bool:
        """True if no faces were detected."""
        return len(self.faces) == 0

    @property
    def best_face(self) -> Optional[FaceBox]:
        """
        Return the face with the highest confidence score,
        or None if no faces were detected.
        """
        if not self.faces:
            return None
        return max(self.faces, key=lambda f: f.confidence)

    @property
    def bboxes(self) -> List[Tuple[int, int, int, int]]:
        """Return all bounding boxes as (x1, y1, x2, y2) tuples."""
        return [f.as_tuple for f in self.faces]

    @property
    def confidences(self) -> List[float]:
        """Return all confidence scores."""
        return [f.confidence for f in self.faces]

    @property
    def landmarks_list(self) -> List[Optional[np.ndarray]]:
        """Return all landmark arrays (may contain None entries)."""
        return [f.landmarks for f in self.faces]

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """
        Return a new DetectionResult containing only faces whose
        confidence is >= *threshold*.

        Args:
            threshold: Minimum confidence score [0.0, 1.0].

        Returns:
            Filtered DetectionResult (same metadata, new faces list).
        """
        filtered = []
        for i, f in enumerate(f for f in self.faces if f.confidence >= threshold):
            from copy import copy  # noqa: PLC0415

            fc = copy(f)
            fc.face_index = i
            filtered.append(fc)
        return DetectionResult(
            faces=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
            frame_index=self.frame_index,
            metadata=self.metadata,
        )

    def filter_by_min_size(self, min_px: int) -> "DetectionResult":
        """
        Return a new DetectionResult keeping only faces where both
        width and height are at least *min_px* pixels.

        Args:
            min_px: Minimum dimension in pixels.

        Returns:
            Filtered DetectionResult.
        """
        filtered = []
        for i, f in enumerate(f for f in self.faces if f.width >= min_px and f.height >= min_px):
            from copy import copy  # noqa: PLC0415

            fc = copy(f)
            fc.face_index = i
            filtered.append(fc)
        return DetectionResult(
            faces=filtered,
            image_width=self.image_width,
            image_height=self.image_height,
            inference_time_ms=self.inference_time_ms,
            frame_index=self.frame_index,
            metadata=self.metadata,
        )

    def get_face(self, index: int) -> Optional[FaceBox]:
        """
        Get a face by its index.

        Args:
            index: Zero-based face index.

        Returns:
            FaceBox or None if index is out of range.
        """
        if 0 <= index < len(self.faces):
            return self.faces[index]
        return None

    def __repr__(self) -> str:
        return (
            f"DetectionResult("
            f"num_faces={self.num_faces}, "
            f"image={self.image_width}×{self.image_height}, "
            f"inference={self.inference_time_ms:.1f}ms"
            f")"
        )


class BaseDetector(ABC):
    """
    Abstract base class for all face detectors.

    Subclasses must implement:
        - ``load_model()``   — load weights into memory
        - ``detect(image)``  — run inference and return DetectionResult

    Optional overrides:
        - ``detect_batch()`` — process multiple frames at once
        - ``release()``      — free GPU/model memory

    Usage::

        detector = YOLOFaceDetector(model_path="models/yolov8n-face.pt")
        detector.load_model()

        result = detector.detect(image)
        for face in result.faces:
            print(face)

    Or using the context manager (auto load + release)::

        with YOLOFaceDetector(...) as detector:
            result = detector.detect(image)
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_faces: int = 20,
        device: str = "auto",
    ) -> None:
        """
        Args:
            model_path:            Path to the model weights file.
            confidence_threshold:  Minimum confidence to accept a detection.
            iou_threshold:         NMS IoU threshold for overlapping boxes.
            max_faces:             Maximum number of faces to return per image.
            device:                Inference device — 'auto' | 'cpu' |
                                   'cuda' | 'cuda:0' | 'mps'.
        """
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_faces = int(max_faces)
        self.device = self._resolve_device(device)

        self._model = None  # Set by load_model()
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the model weights into memory (CPU or GPU).

        This method must:
          - Populate ``self._model``
          - Set ``self._is_loaded = True``
          - Raise ``RuntimeError`` if loading fails.

        It is safe to call this multiple times; implementations should
        guard against re-loading an already-loaded model.
        """

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        *,
        frame_index: Optional[int] = None,
    ) -> DetectionResult:
        """
        Run face detection on a single image.

        Args:
            image:       BGR numpy array of shape (H, W, 3).
            frame_index: Optional frame number for video pipelines.

        Returns:
            DetectionResult containing all detected FaceBox objects,
            sorted by confidence descending.

        Raises:
            RuntimeError:  If the model has not been loaded yet.
            ValueError:    If the image is invalid or empty.
        """

    # ------------------------------------------------------------------
    # Concrete helpers — subclasses MAY override for efficiency
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        images: List[np.ndarray],
        *,
        show_progress: bool = False,
    ) -> List[DetectionResult]:
        """
        Run detection on a list of images.

        The default implementation calls ``detect()`` sequentially.
        Subclasses can override this with a true batched inference call
        for better GPU utilisation.

        Args:
            images:        List of BGR numpy arrays.
            show_progress: Show a tqdm progress bar.

        Returns:
            List of DetectionResult objects, one per input image.
        """
        self._require_loaded()
        results: List[DetectionResult] = []

        iterable = images
        if show_progress:
            from tqdm import tqdm

            iterable = tqdm(images, desc="Detecting faces", unit="img")

        for idx, img in enumerate(iterable):
            results.append(self.detect(img, frame_index=idx))

        return results

    def release(self) -> None:
        """
        Release model resources (GPU memory, file handles, etc.).

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
        """Human-readable model identifier (override in subclasses)."""
        import os

        return os.path.basename(self.model_path)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseDetector":
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
        """
        Guard helper — raise RuntimeError if model is not loaded.

        Call at the top of ``detect()`` and related methods.
        """
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} model is not loaded. "
                "Call load_model() or use the detector as a context manager "
                "(with YOLOFaceDetector(...) as det: ...)."
            )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Resolve 'auto' to the best available device.

        'auto' logic:
          1. CUDA if torch reports CUDA available
          2. MPS  if on Apple Silicon and MPS is available
          3. CPU  as fallback

        Args:
            device: Device string from config.

        Returns:
            Resolved device string ('cpu', 'cuda', 'cuda:0', 'mps', …).
        """
        if device.lower() != "auto":
            return device.lower()

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """
        Raise ValueError for obviously invalid images.

        Args:
            image: Input array to validate.

        Raises:
            ValueError: If the image is None, empty, or wrong shape.
        """
        if image is None:
            raise ValueError("Image is None.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy ndarray, got {type(image).__name__}.")
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected 2-D or 3-D array, got shape {image.shape}.")
        if image.size == 0:
            raise ValueError("Image array is empty (zero size).")

    @staticmethod
    def _timer() -> float:
        """Return the current time in milliseconds (monotonic clock)."""
        return time.perf_counter() * 1000.0

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"device={self.device!r}, "
            f"conf={self.confidence_threshold}, "
            f"status={status})"
        )
