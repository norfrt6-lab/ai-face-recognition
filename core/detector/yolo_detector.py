# YOLOv8-based face detector implementation.
#
# Uses the Ultralytics YOLOv8 model fine-tuned on face detection
# (yolov8n-face.pt / yolov8s-face.pt / etc.) to detect faces in
# images, video frames, and webcam streams.
#
# Key features:
#   - Auto device selection (CUDA > MPS > CPU)
#   - Half-precision (FP16) inference for GPU speedup
#   - Letterbox pre-processing + coordinate unscaling
#   - Batch inference support
#   - Face crop extraction
#   - Bounding-box + landmark visualisation
#   - Thread-safe model loading guard

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from core.detector.base_detector import (
    BaseDetector,
    DetectionResult,
    FaceBox,
    face_box_from_xyxy,
)
from utils.image_utils import normalise_channels


class YOLOFaceDetector(BaseDetector):
    """
    YOLOv8 face detector.

    Wraps the Ultralytics YOLO model and translates its output
    into the standardised ``DetectionResult`` / ``FaceBox`` types
    used across this pipeline.

    Supported input types for ``detect()``:
        - ``np.ndarray``  BGR image  (H, W, 3)
        - file path       ``str`` or ``pathlib.Path``
        - raw bytes       ``bytes``  (JPEG / PNG encoded)
        - PIL Image       ``PIL.Image.Image``

    Quick usage::

        detector = YOLOFaceDetector(
            model_path="models/yolov8n-face.pt",
            confidence_threshold=0.5,
            device="auto",
        )
        detector.load_model()

        result = detector.detect(frame)
        for face in result.faces:
            print(face)

    Context-manager usage (auto load + release)::

        with YOLOFaceDetector(model_path="models/yolov8n-face.pt") as det:
            result = det.detect(frame)
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_faces: int = 20,
        device: str = "auto",
        input_size: int = 640,
        half_precision: bool = False,
        min_face_size: int = 20,
        sort_by: str = "confidence",
    ) -> None:
        """
        Args:
            model_path:            Path to YOLOv8 face model weights (.pt).
                                   Can be a local path or an Ultralytics Hub
                                   model name (e.g. 'yolov8n.pt').
            confidence_threshold:  Minimum confidence to keep a detection
                                   [0.0, 1.0].  Default 0.5.
            iou_threshold:         NMS IoU threshold [0.0, 1.0].  Default 0.45.
            max_faces:             Maximum faces to return per image.
            device:                Inference device.
                                   'auto' selects CUDA > MPS > CPU.
            input_size:            Square resolution fed to the YOLO model.
                                   Must be a multiple of 32 (e.g. 320, 640).
            half_precision:        Use FP16 weights for faster GPU inference.
                                   Ignored on CPU.
            min_face_size:         Discard detections smaller than this many
                                   pixels in width OR height.
            sort_by:               How to sort detected faces in the result.
                                   'confidence' (default) | 'area' | 'left_to_right'.
        """
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_faces=max_faces,
            device=device,
        )

        self.input_size = input_size
        self.half_precision = half_precision and device != "cpu"
        self.min_face_size = min_face_size
        self.sort_by = sort_by

        # Thread-safety: prevent concurrent load_model() calls
        self._load_lock = threading.Lock()

        # Cached model info (set after load)
        self._model_info: dict = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the YOLOv8 model weights from ``self.model_path``.

        - Downloads the model automatically if using an Ultralytics Hub
          name and the file is not present locally.
        - Moves the model to the target device.
        - Optionally converts to FP16 (GPU only).

        Raises:
            FileNotFoundError: If ``model_path`` is a local path that
                               does not exist.
            RuntimeError:      If Ultralytics YOLO raises any error during
                               loading.
        """
        with self._load_lock:
            if self._is_loaded:
                logger.debug(
                    f"{self.__class__.__name__} already loaded — skipping."
                )
                return

            model_path = Path(self.model_path)

            # Validate local path exists (skip check for hub names like 'yolov8n.pt')
            if (
                not model_path.exists()
                and "/" not in self.model_path
                and os.sep not in self.model_path
            ):
                # Ultralytics Hub name — YOLO will auto-download
                logger.info(
                    f"Model path not found locally — will attempt "
                    f"Ultralytics auto-download: {self.model_path}"
                )
            elif not model_path.exists():
                raise FileNotFoundError(
                    f"YOLOv8 model file not found: {model_path.resolve()}\n"
                    f"Run: python utils/download_models.py --model yolov8n-face"
                )

            logger.info(
                f"Loading YOLOv8 face detector | "
                f"model={self.model_path} | "
                f"device={self.device} | "
                f"fp16={self.half_precision}"
            )

            t0 = self._timer()

            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)

                # Move to target device
                # Ultralytics handles device selection internally during predict();
                # we still store the device so it is passed on every call.
                if self.half_precision and self.device != "cpu":
                    self._model.model.half()

                self._is_loaded = True

            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. "
                    "Run: pip install ultralytics>=8.2.0"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load YOLOv8 model from {self.model_path}: {exc}"
                ) from exc

            elapsed = self._timer() - t0
            logger.success(
                f"YOLOv8 face detector ready in {elapsed:.0f} ms | "
                f"device={self.device}"
            )

    # ------------------------------------------------------------------
    # Single-image detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: Union[np.ndarray, str, Path, bytes],
        *,
        frame_index: Optional[int] = None,
    ) -> DetectionResult:
        """
        Detect faces in a single image.

        Args:
            image:       BGR numpy array  — or a file path / raw bytes /
                         PIL Image that will be decoded automatically.
            frame_index: Optional frame number (for video pipelines).

        Returns:
            ``DetectionResult`` with detected ``FaceBox`` objects sorted
            by the strategy configured at construction time.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError:   If the image cannot be decoded.
        """
        self._require_loaded()

        bgr = self._to_bgr(image)
        self._validate_image(bgr)

        h, w = bgr.shape[:2]
        t0 = self._timer()

        try:
            results = self._model.predict(
                source=bgr,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_faces,
                imgsz=self.input_size,
                device=self.device,
                half=self.half_precision,
                verbose=False,          # Suppress Ultralytics console spam
                save=False,
                stream=False,
            )
        except Exception as exc:
            logger.error(f"YOLO inference error: {exc}")
            return DetectionResult(
                faces=[],
                image_width=w,
                image_height=h,
                inference_time_ms=self._timer() - t0,
                frame_index=frame_index,
                metadata={"error": str(exc)},
            )

        inference_ms = self._timer() - t0

        faces = self._parse_results(results, src_w=w, src_h=h)

        faces = [
            f for f in faces
            if f.width >= self.min_face_size
            and f.height >= self.min_face_size
        ]

        faces = self._sort_faces(faces, strategy=self.sort_by)

        # Re-assign face_index after sort + filter
        for i, face in enumerate(faces):
            face.face_index = i

        detection = DetectionResult(
            faces=faces,
            image_width=w,
            image_height=h,
            inference_time_ms=inference_ms,
            frame_index=frame_index,
            metadata={
                "model": self.model_path,
                "device": self.device,
                "input_size": self.input_size,
                "conf_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
            },
        )

        logger.debug(
            f"Detected {detection.num_faces} face(s) | "
            f"{w}×{h} px | "
            f"{inference_ms:.1f} ms"
        )

        return detection

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        *,
        show_progress: bool = False,
    ) -> List[DetectionResult]:
        """
        Detect faces in a list of images using YOLO's native batch mode.

        This is more GPU-efficient than calling ``detect()`` in a loop
        because the YOLO model processes all images in a single forward
        pass (up to the GPU memory limit).

        Args:
            images:        List of BGR numpy arrays, file paths, or bytes.
            show_progress: Show a tqdm progress bar.

        Returns:
            List of ``DetectionResult``, one per input image,
            in the same order as the input list.
        """
        self._require_loaded()

        if not images:
            return []

        # Decode all inputs to BGR ndarrays
        bgr_list = [self._to_bgr(img) for img in images]
        shapes   = [(img.shape[1], img.shape[0]) for img in bgr_list]  # (w, h)

        t0 = self._timer()

        try:
            batch_results = self._model.predict(
                source=bgr_list,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_faces,
                imgsz=self.input_size,
                device=self.device,
                half=self.half_precision,
                verbose=False,
                save=False,
                stream=False,
            )
        except Exception as exc:
            logger.error(f"YOLO batch inference error: {exc}")
            # Fall back to sequential
            logger.warning("Falling back to sequential detection.")
            return super().detect_batch(images, show_progress=show_progress)

        total_ms = self._timer() - t0
        per_ms   = total_ms / len(images)

        detection_results: List[DetectionResult] = []

        iterable = zip(batch_results, shapes)
        if show_progress:
            from tqdm import tqdm
            iterable = tqdm(
                list(iterable), desc="Processing detections", unit="img"
            )

        for idx, (yolo_res, (w, h)) in enumerate(iterable):
            faces = self._parse_single_result(yolo_res, src_w=w, src_h=h)
            faces = [
                f for f in faces
                if f.width >= self.min_face_size
                and f.height >= self.min_face_size
            ]
            faces = self._sort_faces(faces, strategy=self.sort_by)
            for i, face in enumerate(faces):
                face.face_index = i

            detection_results.append(
                DetectionResult(
                    faces=faces,
                    image_width=w,
                    image_height=h,
                    inference_time_ms=per_ms,
                    frame_index=idx,
                    metadata={"model": self.model_path, "device": self.device},
                )
            )

        logger.debug(
            f"Batch of {len(images)} images | "
            f"total={total_ms:.0f} ms | "
            f"avg={per_ms:.1f} ms/img"
        )

        return detection_results

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def detect_and_crop(
        self,
        image: np.ndarray,
        *,
        pad_fraction: float = 0.0,
    ) -> Tuple[DetectionResult, List[np.ndarray]]:
        """
        Detect faces and return both the result AND cropped face patches.

        Args:
            image:        BGR source image.
            pad_fraction: Fractional padding to add around each face crop
                          (0.1 = 10 % extra on each side).

        Returns:
            Tuple of (DetectionResult, List[crop_ndarray]).
            The list is aligned with ``result.faces``.
        """
        result = self.detect(image)
        crops: List[np.ndarray] = []
        h, w = image.shape[:2]

        for face in result.faces:
            box = face
            if pad_fraction > 0.0:
                box = face.pad_fractional(pad_fraction, img_w=w, img_h=h)
            crops.append(box.crop(image))

        return result, crops

    def detect_largest(
        self,
        image: np.ndarray,
    ) -> Optional[FaceBox]:
        """
        Detect faces and return only the largest (by area) bounding box,
        or None if no faces found.

        Args:
            image: BGR source image.

        Returns:
            The FaceBox with the largest area, or None.
        """
        result = self.detect(image)
        if result.is_empty:
            return None
        return max(result.faces, key=lambda f: f.area)

    def detect_most_confident(
        self,
        image: np.ndarray,
    ) -> Optional[FaceBox]:
        """
        Return only the most confidently-detected face, or None.

        Args:
            image: BGR source image.

        Returns:
            FaceBox with the highest confidence, or None.
        """
        result = self.detect(image)
        return result.best_face

    def is_face_present(
        self,
        image: np.ndarray,
        min_confidence: Optional[float] = None,
    ) -> bool:
        """
        Quick check: does this image contain at least one face?

        Args:
            image:          BGR source image.
            min_confidence: Optional confidence override.

        Returns:
            True if at least one face is detected.
        """
        result = self.detect(image)
        if min_confidence is not None:
            return any(f.confidence >= min_confidence for f in result.faces)
        return not result.is_empty

    def count_faces(self, image: np.ndarray) -> int:
        """
        Return the number of faces detected in an image.

        Args:
            image: BGR source image.

        Returns:
            Integer face count.
        """
        return self.detect(image).num_faces

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        image: np.ndarray,
        result: DetectionResult,
        *,
        box_color: Tuple[int, int, int] = (0, 230, 0),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        box_thickness: int = 2,
        font_scale: float = 0.55,
        show_confidence: bool = True,
        show_index: bool = True,
        show_landmarks: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 165, 255),
        landmark_radius: int = 3,
    ) -> np.ndarray:
        """
        Draw detection results onto a copy of the source image.

        Args:
            image:            BGR source image (not modified in place).
            result:           DetectionResult from ``detect()``.
            box_color:        BGR colour for bounding boxes.
            text_color:       BGR colour for text labels.
            box_thickness:    Thickness of the bounding-box rectangle.
            font_scale:       OpenCV font scale for labels.
            show_confidence:  Include confidence score in labels.
            show_index:       Include face index in labels.
            show_landmarks:   Draw 5-point facial landmarks if available.
            landmark_color:   BGR colour for landmark dots.
            landmark_radius:  Radius of landmark circles in pixels.

        Returns:
            New BGR image with detections drawn.
        """
        vis = image.copy()

        for face in result.faces:
            x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2

            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thickness)

            parts = []
            if show_index:
                parts.append(f"#{face.face_index}")
            if show_confidence:
                parts.append(f"{face.confidence:.2f}")
            label = " ".join(parts)

            if label:
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                bg_y1 = max(y1 - th - 8, 0)
                # Label background
                cv2.rectangle(
                    vis,
                    (x1, bg_y1),
                    (x1 + tw + 6, y1),
                    box_color,
                    -1,
                )
                cv2.putText(
                    vis, label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

            if show_landmarks and face.has_landmarks:
                for x, y in face.landmarks.astype(int):
                    cv2.circle(vis, (x, y), landmark_radius, landmark_color, -1, cv2.LINE_AA)

        summary = f"Faces: {result.num_faces}  |  {result.inference_time_ms:.0f} ms"
        cv2.putText(
            vis, summary,
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return vis

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """
        Return a dict with model metadata.

        Returns:
            Dict with keys: name, path, device, input_size,
            conf_threshold, iou_threshold, max_faces, is_loaded.
        """
        return {
            "name":             self.model_name,
            "path":             str(self.model_path),
            "device":           self.device,
            "input_size":       self.input_size,
            "conf_threshold":   self.confidence_threshold,
            "iou_threshold":    self.iou_threshold,
            "max_faces":        self.max_faces,
            "half_precision":   self.half_precision,
            "min_face_size":    self.min_face_size,
            "sort_by":          self.sort_by,
            "is_loaded":        self._is_loaded,
        }

    def warmup(self, iterations: int = 3) -> float:
        """
        Run a few dummy inference passes to warm up the GPU/model.

        This can reduce latency spikes on the first real inference call.

        Args:
            iterations: Number of dummy forward passes to run.

        Returns:
            Average inference time per pass in milliseconds.
        """
        self._require_loaded()
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        times = []
        for _ in range(iterations):
            t0 = self._timer()
            self.detect(dummy)
            times.append(self._timer() - t0)
        avg = sum(times) / len(times)
        logger.info(f"Warmup done ({iterations} iters) — avg {avg:.1f} ms/iter")
        return avg

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def release(self) -> None:
        """
        Free the YOLO model from memory.

        Clears CUDA cache if the model was running on GPU.
        """
        if self._model is not None:
            try:
                del self._model
            except (AttributeError, TypeError):
                pass

            if "cuda" in self.device:
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared.")
                except ImportError:
                    pass

        super().release()
        logger.info("YOLOFaceDetector released.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_results(
        self,
        yolo_results: list,
        *,
        src_w: int,
        src_h: int,
    ) -> List[FaceBox]:
        """
        Parse the full list returned by ``model.predict()`` into FaceBoxes.

        ``model.predict()`` on a single image returns a list with one
        element; we take the first.
        """
        if not yolo_results:
            return []
        return self._parse_single_result(yolo_results[0], src_w=src_w, src_h=src_h)

    def _parse_single_result(
        self,
        yolo_result,
        *,
        src_w: int,
        src_h: int,
    ) -> List[FaceBox]:
        """
        Convert a single Ultralytics ``Results`` object into a list of
        ``FaceBox`` instances.

        Ultralytics stores boxes in ``result.boxes``:
            - ``result.boxes.xyxy``  — (N, 4) tensor in pixel coords
            - ``result.boxes.conf``  — (N,)   confidence scores
            - ``result.boxes.cls``   — (N,)   class indices (all 0 for face models)

        Keypoints (if the model supports them) are in ``result.keypoints``.
        """
        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Move tensors to CPU + numpy
        try:
            xyxy  = boxes.xyxy.cpu().numpy()    # shape (N, 4)
            confs = boxes.conf.cpu().numpy()     # shape (N,)
        except (AttributeError, IndexError, RuntimeError) as exc:
            logger.warning(f"Failed to parse YOLO boxes: {exc}")
            return []

        # Optional keypoints (5-point facial landmarks)
        landmarks_array: Optional[np.ndarray] = None
        try:
            if yolo_result.keypoints is not None:
                kp_data = yolo_result.keypoints.xy.cpu().numpy()   # (N, 5, 2)
                landmarks_array = kp_data
        except (AttributeError, RuntimeError):
            landmarks_array = None

        face_boxes: List[FaceBox] = []

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])

            # Clamp to image bounds
            x1 = max(0.0, min(float(src_w), x1))
            y1 = max(0.0, min(float(src_h), y1))
            x2 = max(0.0, min(float(src_w), x2))
            y2 = max(0.0, min(float(src_h), y2))

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract per-face landmarks if available
            lm = None
            if landmarks_array is not None and i < len(landmarks_array):
                lm = landmarks_array[i].astype(np.float32)  # (5, 2)
                # If all zeros, landmarks were not detected
                if lm.sum() == 0:
                    lm = None

            face_boxes.append(
                face_box_from_xyxy(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=conf,
                    face_index=i,
                    landmarks=lm,
                )
            )

        return face_boxes

    @staticmethod
    def _sort_faces(
        faces: List[FaceBox],
        strategy: str = "confidence",
    ) -> List[FaceBox]:
        """
        Sort a list of FaceBox objects by a given strategy.

        Args:
            faces:    List of FaceBox instances.
            strategy: Sort key:
                      'confidence'    — highest confidence first (default)
                      'area'          — largest face first
                      'left_to_right' — smallest x1 first
                      'top_to_bottom' — smallest y1 first

        Returns:
            Sorted list (new list, original not modified).
        """
        if not faces:
            return faces

        if strategy == "confidence":
            return sorted(faces, key=lambda f: f.confidence, reverse=True)
        elif strategy == "area":
            return sorted(faces, key=lambda f: f.area, reverse=True)
        elif strategy == "left_to_right":
            return sorted(faces, key=lambda f: f.x1)
        elif strategy == "top_to_bottom":
            return sorted(faces, key=lambda f: f.y1)
        else:
            logger.warning(
                f"Unknown sort strategy {strategy!r} — using 'confidence'."
            )
            return sorted(faces, key=lambda f: f.confidence, reverse=True)

    @staticmethod
    def _to_bgr(
        source: Union[np.ndarray, str, Path, bytes, "PIL.Image.Image"],
    ) -> np.ndarray:
        """
        Decode various input formats to a BGR numpy array.

        Supported types:
            - ``np.ndarray``          — returned as-is (copy on 4-channel)
            - ``str`` / ``Path``      — read from disk with OpenCV
            - ``bytes``               — decoded with cv2.imdecode
            - ``PIL.Image.Image``     — converted via numpy

        Args:
            source: Input image in any supported format.

        Returns:
            BGR uint8 numpy array of shape (H, W, 3).

        Raises:
            ValueError: If decoding fails or the type is unsupported.
        """
        if isinstance(source, np.ndarray):
            return normalise_channels(source)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"OpenCV could not decode: {path}")
            return img

        if isinstance(source, bytes):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from bytes.")
            return img

        # PIL.Image.Image (lazy import)
        try:
            from PIL import Image as PILImage
            if isinstance(source, PILImage.Image):
                rgb = np.array(source.convert("RGB"), dtype=np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        raise TypeError(
            f"Unsupported image type: {type(source).__name__}. "
            "Expected: np.ndarray | str | Path | bytes | PIL.Image"
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"YOLOFaceDetector("
            f"model={os.path.basename(self.model_path)!r}, "
            f"device={self.device!r}, "
            f"conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold}, "
            f"max_faces={self.max_faces}, "
            f"input_size={self.input_size}, "
            f"status={status})"
        )
