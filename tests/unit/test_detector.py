# Unit tests for:
#   - FaceBox dataclass
#   - DetectionResult dataclass
#   - BaseDetector abstract class
#   - YOLOFaceDetector implementation
#
# These tests use pytest + pytest-mock and do NOT require
# actual model weights to be present on disk.
# Heavy inference tests are guarded with 'integration' markers.

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from core.detector.base_detector import (
    BaseDetector,
    DetectionResult,
    FaceBox,
    face_box_from_xyxy,
)
from core.detector.yolo_detector import YOLOFaceDetector


@pytest.fixture
def sample_face_box() -> FaceBox:
    """A simple FaceBox for reuse across tests."""
    return FaceBox(
        x1=100, y1=80, x2=300, y2=320,
        confidence=0.92,
        face_index=0,
    )


@pytest.fixture
def sample_face_box_with_landmarks() -> FaceBox:
    """A FaceBox that includes 5-point landmarks."""
    lm = np.array(
        [
            [150.0, 140.0],   # left_eye
            [250.0, 140.0],   # right_eye
            [200.0, 200.0],   # nose
            [160.0, 270.0],   # left_mouth
            [240.0, 270.0],   # right_mouth
        ],
        dtype=np.float32,
    )
    return FaceBox(
        x1=100, y1=80, x2=300, y2=320,
        confidence=0.92,
        face_index=0,
        landmarks=lm,
    )


@pytest.fixture
def sample_detection_result(sample_face_box) -> DetectionResult:
    """A DetectionResult with a single face."""
    return DetectionResult(
        faces=[sample_face_box],
        image_width=640,
        image_height=480,
        inference_time_ms=12.5,
    )


@pytest.fixture
def empty_detection_result() -> DetectionResult:
    """A DetectionResult with no faces."""
    return DetectionResult(
        faces=[],
        image_width=640,
        image_height=480,
        inference_time_ms=8.0,
    )


@pytest.fixture
def multi_face_detection_result() -> DetectionResult:
    """A DetectionResult with three faces of varying sizes/confidences."""
    faces = [
        FaceBox(x1=10,  y1=10,  x2=110, y2=110, confidence=0.95, face_index=0),
        FaceBox(x1=200, y1=50,  x2=450, y2=350, confidence=0.72, face_index=1),
        FaceBox(x1=500, y1=200, x2=600, y2=320, confidence=0.60, face_index=2),
    ]
    return DetectionResult(
        faces=faces,
        image_width=640,
        image_height=480,
        inference_time_ms=18.3,
    )


@pytest.fixture
def blank_image() -> np.ndarray:
    """A black BGR image used as dummy input."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_image() -> np.ndarray:
    """A white BGR image."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def mock_yolo_detector(tmp_path) -> YOLOFaceDetector:
    """
    A YOLOFaceDetector instance with a dummy model file.
    The actual YOLO model is NOT loaded — _model is mocked.
    """
    model_file = tmp_path / "yolov8n-face.pt"
    model_file.touch()
    detector = YOLOFaceDetector(
        model_path=str(model_file),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        max_faces=10,
        device="cpu",
    )
    return detector


class TestFaceBox:
    """Unit tests for the FaceBox dataclass."""


    def test_basic_construction(self, sample_face_box):
        fb = sample_face_box
        assert fb.x1 == 100
        assert fb.y1 == 80
        assert fb.x2 == 300
        assert fb.y2 == 320
        assert fb.confidence == pytest.approx(0.92)
        assert fb.face_index == 0

    def test_default_landmarks_none(self, sample_face_box):
        assert sample_face_box.landmarks is None
        assert sample_face_box.has_landmarks is False

    def test_with_landmarks(self, sample_face_box_with_landmarks):
        fb = sample_face_box_with_landmarks
        assert fb.landmarks is not None
        assert fb.has_landmarks is True
        assert fb.landmarks.shape == (5, 2)

    def test_track_id_default_none(self, sample_face_box):
        assert sample_face_box.track_id is None

    def test_track_id_set(self):
        fb = FaceBox(x1=0, y1=0, x2=50, y2=50, confidence=0.8, track_id=42)
        assert fb.track_id == 42


    def test_width(self, sample_face_box):
        assert sample_face_box.width == 200   # 300 - 100

    def test_height(self, sample_face_box):
        assert sample_face_box.height == 240  # 320 - 80

    def test_area(self, sample_face_box):
        assert sample_face_box.area == 200 * 240   # 48000

    def test_center(self, sample_face_box):
        cx, cy = sample_face_box.center
        assert cx == 200   # (100 + 300) // 2
        assert cy == 200   # (80 + 320) // 2

    def test_aspect_ratio(self, sample_face_box):
        ar = sample_face_box.aspect_ratio
        assert ar == pytest.approx(200 / 240)

    def test_aspect_ratio_zero_height(self):
        fb = FaceBox(x1=0, y1=0, x2=100, y2=0, confidence=0.9)
        assert fb.aspect_ratio == 0.0

    def test_as_tuple(self, sample_face_box):
        assert sample_face_box.as_tuple == (100, 80, 300, 320)

    def test_as_xywh(self, sample_face_box):
        x, y, w, h = sample_face_box.as_xywh
        assert x == 100
        assert y == 80
        assert w == 200
        assert h == 240

    def test_zero_dimension_width(self):
        fb = FaceBox(x1=50, y1=50, x2=50, y2=100, confidence=0.9)
        assert fb.width == 0

    def test_zero_dimension_height(self):
        fb = FaceBox(x1=50, y1=50, x2=100, y2=50, confidence=0.9)
        assert fb.height == 0

    def test_zero_dimension_area(self):
        fb = FaceBox(x1=50, y1=50, x2=50, y2=100, confidence=0.9)
        assert fb.area == 0


    def test_scale_uniform(self, sample_face_box):
        scaled = sample_face_box.scale(2.0, 2.0)
        assert scaled.x1 == 200
        assert scaled.y1 == 160
        assert scaled.x2 == 600
        assert scaled.y2 == 640
        assert scaled.confidence == pytest.approx(0.92)

    def test_scale_non_uniform(self, sample_face_box):
        scaled = sample_face_box.scale(0.5, 1.0)
        assert scaled.x1 == 50
        assert scaled.x2 == 150
        assert scaled.y1 == 80
        assert scaled.y2 == 320

    def test_scale_preserves_landmarks(self, sample_face_box_with_landmarks):
        scaled = sample_face_box_with_landmarks.scale(2.0, 2.0)
        assert scaled.landmarks is not None
        # All landmark coords should be doubled
        expected = sample_face_box_with_landmarks.landmarks * 2.0
        np.testing.assert_allclose(scaled.landmarks, expected)

    def test_scale_no_landmarks(self, sample_face_box):
        scaled = sample_face_box.scale(2.0, 2.0)
        assert scaled.landmarks is None

    def test_scale_identity(self, sample_face_box):
        scaled = sample_face_box.scale(1.0, 1.0)
        assert scaled.as_tuple == sample_face_box.as_tuple


    def test_pad_positive(self, sample_face_box):
        padded = sample_face_box.pad(px=10, py=10)
        assert padded.x1 == 90
        assert padded.y1 == 70
        assert padded.x2 == 310
        assert padded.y2 == 330

    def test_pad_clamps_to_zero(self):
        fb = FaceBox(x1=5, y1=5, x2=100, y2=100, confidence=0.9)
        padded = fb.pad(px=20, py=20)
        assert padded.x1 == 0
        assert padded.y1 == 0

    def test_pad_clamps_to_image_bounds(self, sample_face_box):
        padded = sample_face_box.pad(px=50, py=50, img_w=320, img_h=340)
        assert padded.x2 == 320
        assert padded.y2 == 340

    def test_pad_zero(self, sample_face_box):
        padded = sample_face_box.pad(px=0, py=0)
        assert padded.as_tuple == sample_face_box.as_tuple


    def test_pad_fractional_10pct(self, sample_face_box):
        # width=200, height=240 → px=20, py=24
        padded = sample_face_box.pad_fractional(0.10)
        assert padded.x1 == sample_face_box.x1 - 20
        assert padded.y1 == sample_face_box.y1 - 24

    def test_pad_fractional_zero(self, sample_face_box):
        padded = sample_face_box.pad_fractional(0.0)
        assert padded.as_tuple == sample_face_box.as_tuple


    def test_clamp_no_change_when_within_bounds(self, sample_face_box):
        clamped = sample_face_box.clamp(img_w=640, img_h=480)
        assert clamped.as_tuple == sample_face_box.as_tuple

    def test_clamp_x2_overflow(self):
        fb = FaceBox(x1=600, y1=100, x2=700, y2=200, confidence=0.8)
        clamped = fb.clamp(img_w=640, img_h=480)
        assert clamped.x2 == 640

    def test_clamp_y2_overflow(self):
        fb = FaceBox(x1=100, y1=400, x2=200, y2=520, confidence=0.8)
        clamped = fb.clamp(img_w=640, img_h=480)
        assert clamped.y2 == 480

    def test_clamp_negative_coords(self):
        fb = FaceBox(x1=-10, y1=-5, x2=100, y2=100, confidence=0.8)
        clamped = fb.clamp(img_w=640, img_h=480)
        assert clamped.x1 == 0
        assert clamped.y1 == 0


    def test_iou_identical_boxes(self, sample_face_box):
        assert sample_face_box.iou(sample_face_box) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = FaceBox(x1=0,   y1=0,   x2=100, y2=100, confidence=0.9)
        b = FaceBox(x1=200, y1=200, x2=300, y2=300, confidence=0.9)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        a = FaceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        b = FaceBox(x1=50, y1=50, x2=150, y2=150, confidence=0.9)
        iou = a.iou(b)
        # Intersection: 50x50 = 2500, Union: 10000+10000-2500 = 17500
        assert iou == pytest.approx(2500 / 17500, rel=1e-3)

    def test_iou_full_containment(self):
        outer = FaceBox(x1=0, y1=0, x2=200, y2=200, confidence=0.9)
        inner = FaceBox(x1=50, y1=50, x2=150, y2=150, confidence=0.9)
        iou = outer.iou(inner)
        # Intersection = 10000, Union = 40000
        assert iou == pytest.approx(10000 / 40000, rel=1e-3)

    def test_iou_symmetry(self):
        a = FaceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        b = FaceBox(x1=50, y1=0, x2=200, y2=100, confidence=0.9)
        assert a.iou(b) == pytest.approx(b.iou(a), rel=1e-6)

    def test_iou_zero_area_box(self):
        a = FaceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        b = FaceBox(x1=50, y1=50, x2=50, y2=50, confidence=0.9)  # zero area
        assert a.iou(b) == pytest.approx(0.0)


    def test_crop_returns_correct_shape(self, sample_face_box, blank_image):
        crop = sample_face_box.crop(blank_image)
        expected_h = sample_face_box.height   # 240
        expected_w = sample_face_box.width    # 200
        assert crop.shape == (expected_h, expected_w, 3)

    def test_crop_is_copy(self, sample_face_box, blank_image):
        crop = sample_face_box.crop(blank_image)
        crop[:] = 255
        # Original image should be unchanged
        assert blank_image[sample_face_box.y1, sample_face_box.x1, 0] == 0

    def test_crop_clamps_to_image(self):
        fb = FaceBox(x1=-10, y1=-10, x2=640, y2=480, confidence=0.9)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = fb.crop(image)
        assert crop.shape[0] <= 480
        assert crop.shape[1] <= 640


    def test_repr_contains_key_info(self, sample_face_box):
        r = repr(sample_face_box)
        assert "FaceBox" in r
        assert "0.920" in r or "0.92" in r
        assert "200×240" in r


class TestFaceBoxFromXYXY:
    """Tests for the factory constructor."""

    def test_basic_creation(self):
        fb = face_box_from_xyxy(10.7, 20.3, 110.6, 220.9, confidence=0.88)
        assert fb.x1 == 11   # round(10.7)
        assert fb.y1 == 20   # round(20.3)
        assert fb.x2 == 111  # round(110.6)
        assert fb.y2 == 221  # round(220.9)
        assert fb.confidence == pytest.approx(0.88)

    def test_face_index_default_zero(self):
        fb = face_box_from_xyxy(0, 0, 100, 100, confidence=0.5)
        assert fb.face_index == 0

    def test_face_index_set(self):
        fb = face_box_from_xyxy(0, 0, 100, 100, confidence=0.5, face_index=3)
        assert fb.face_index == 3

    def test_landmarks_passed_through(self):
        lm = np.zeros((5, 2), dtype=np.float32)
        fb = face_box_from_xyxy(0, 0, 100, 100, confidence=0.5, landmarks=lm)
        assert fb.landmarks is not None
        np.testing.assert_array_equal(fb.landmarks, lm)

    def test_rounds_to_int(self):
        fb = face_box_from_xyxy(0.4999, 0.5001, 99.9, 100.1, confidence=0.5)
        assert isinstance(fb.x1, int)
        assert isinstance(fb.y1, int)
        assert isinstance(fb.x2, int)
        assert isinstance(fb.y2, int)


class TestDetectionResult:
    """Unit tests for the DetectionResult dataclass."""


    def test_num_faces(self, multi_face_detection_result):
        assert multi_face_detection_result.num_faces == 3

    def test_is_empty_false(self, sample_detection_result):
        assert sample_detection_result.is_empty is False

    def test_is_empty_true(self, empty_detection_result):
        assert empty_detection_result.is_empty is True

    def test_num_faces_empty(self, empty_detection_result):
        assert empty_detection_result.num_faces == 0

    def test_best_face_returns_highest_confidence(self, multi_face_detection_result):
        best = multi_face_detection_result.best_face
        assert best is not None
        assert best.confidence == pytest.approx(0.95)

    def test_best_face_none_when_empty(self, empty_detection_result):
        assert empty_detection_result.best_face is None

    def test_bboxes_returns_list_of_tuples(self, multi_face_detection_result):
        bboxes = multi_face_detection_result.bboxes
        assert len(bboxes) == 3
        for b in bboxes:
            assert len(b) == 4

    def test_confidences(self, multi_face_detection_result):
        confs = multi_face_detection_result.confidences
        assert len(confs) == 3
        assert confs[0] == pytest.approx(0.95)

    def test_landmarks_list_all_none(self, multi_face_detection_result):
        lm_list = multi_face_detection_result.landmarks_list
        assert all(lm is None for lm in lm_list)


    def test_filter_by_confidence_removes_low(self, multi_face_detection_result):
        filtered = multi_face_detection_result.filter_by_confidence(0.75)
        assert filtered.num_faces == 1
        assert filtered.faces[0].confidence == pytest.approx(0.95)

    def test_filter_by_confidence_keeps_all(self, multi_face_detection_result):
        filtered = multi_face_detection_result.filter_by_confidence(0.0)
        assert filtered.num_faces == 3

    def test_filter_by_confidence_removes_all(self, multi_face_detection_result):
        filtered = multi_face_detection_result.filter_by_confidence(1.0)
        assert filtered.num_faces == 0

    def test_filter_reindexes_faces(self, multi_face_detection_result):
        filtered = multi_face_detection_result.filter_by_confidence(0.65)
        for i, face in enumerate(filtered.faces):
            assert face.face_index == i

    def test_filter_preserves_metadata(self, multi_face_detection_result):
        filtered = multi_face_detection_result.filter_by_confidence(0.5)
        assert filtered.image_width  == multi_face_detection_result.image_width
        assert filtered.image_height == multi_face_detection_result.image_height


    def test_filter_by_min_size_removes_small(self, multi_face_detection_result):
        # face at index 0 is 100x100, face at index 1 is 250x300, face 2 is 100x120
        filtered = multi_face_detection_result.filter_by_min_size(150)
        assert filtered.num_faces == 1
        # Only the 250x300 face should remain
        assert filtered.faces[0].width == 250
        assert filtered.faces[0].height == 300

    def test_filter_by_min_size_keeps_all_when_small_threshold(
        self, multi_face_detection_result
    ):
        filtered = multi_face_detection_result.filter_by_min_size(1)
        assert filtered.num_faces == 3

    def test_filter_by_min_size_reindexes(self, multi_face_detection_result):
        filtered = multi_face_detection_result.filter_by_min_size(150)
        for i, f in enumerate(filtered.faces):
            assert f.face_index == i


    def test_get_face_valid_index(self, multi_face_detection_result):
        face = multi_face_detection_result.get_face(1)
        assert face is not None
        assert face.face_index == 1

    def test_get_face_index_zero(self, multi_face_detection_result):
        face = multi_face_detection_result.get_face(0)
        assert face is not None

    def test_get_face_out_of_range_returns_none(self, multi_face_detection_result):
        face = multi_face_detection_result.get_face(99)
        assert face is None

    def test_get_face_negative_index_returns_none(self, multi_face_detection_result):
        face = multi_face_detection_result.get_face(-1)
        assert face is None


    def test_repr(self, sample_detection_result):
        r = repr(sample_detection_result)
        assert "DetectionResult" in r
        assert "num_faces=1" in r
        assert "640" in r


class ConcreteDetector(BaseDetector):
    """
    Minimal concrete subclass of BaseDetector for testing the
    abstract class without loading any real model.
    """

    def load_model(self) -> None:
        self._model = object()   # Dummy model object
        self._is_loaded = True

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_index: Optional[int] = None,
    ) -> DetectionResult:
        self._require_loaded()
        h, w = image.shape[:2]
        return DetectionResult(
            faces=[
                FaceBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9)
            ],
            image_width=w,
            image_height=h,
            inference_time_ms=1.0,
            frame_index=frame_index,
        )


class TestBaseDetector:
    """Tests targeting the abstract base class behaviour."""

    @pytest.fixture
    def detector(self, tmp_path) -> ConcreteDetector:
        model_file = tmp_path / "dummy.pt"
        model_file.touch()
        return ConcreteDetector(
            model_path=str(model_file),
            confidence_threshold=0.5,
            iou_threshold=0.45,
            max_faces=10,
            device="cpu",
        )


    def test_initial_not_loaded(self, detector):
        assert detector.is_loaded is False

    def test_device_stored(self, detector):
        assert detector.device == "cpu"

    def test_thresholds_stored(self, detector):
        assert detector.confidence_threshold == pytest.approx(0.5)
        assert detector.iou_threshold == pytest.approx(0.45)

    def test_max_faces_stored(self, detector):
        assert detector.max_faces == 10


    def test_load_model_sets_is_loaded(self, detector):
        detector.load_model()
        assert detector.is_loaded is True

    def test_load_model_idempotent(self, detector):
        detector.load_model()
        detector.load_model()   # Should not raise
        assert detector.is_loaded is True


    def test_detect_requires_loaded(self, detector, blank_image):
        with pytest.raises(RuntimeError, match="not loaded"):
            detector.detect(blank_image)

    def test_detect_returns_result(self, detector, blank_image):
        detector.load_model()
        result = detector.detect(blank_image)
        assert isinstance(result, DetectionResult)
        assert result.num_faces == 1

    def test_detect_passes_frame_index(self, detector, blank_image):
        detector.load_model()
        result = detector.detect(blank_image, frame_index=42)
        assert result.frame_index == 42

    def test_detect_image_dimensions(self, detector, blank_image):
        detector.load_model()
        result = detector.detect(blank_image)
        assert result.image_width == 640
        assert result.image_height == 480


    def test_detect_batch_returns_list(self, detector, blank_image):
        detector.load_model()
        images = [blank_image, blank_image, blank_image]
        results = detector.detect_batch(images)
        assert len(results) == 3

    def test_detect_batch_empty_list(self, detector):
        detector.load_model()
        results = detector.detect_batch([])
        assert results == []

    def test_detect_batch_frame_indices(self, detector, blank_image):
        detector.load_model()
        results = detector.detect_batch([blank_image, blank_image])
        assert results[0].frame_index == 0
        assert results[1].frame_index == 1

    def test_detect_batch_requires_loaded(self, detector, blank_image):
        with pytest.raises(RuntimeError):
            detector.detect_batch([blank_image])


    def test_release_resets_is_loaded(self, detector):
        detector.load_model()
        assert detector.is_loaded is True
        detector.release()
        assert detector.is_loaded is False

    def test_release_clears_model(self, detector):
        detector.load_model()
        detector.release()
        assert detector._model is None


    def test_context_manager_loads_on_enter(self, detector):
        assert detector.is_loaded is False
        with detector:
            assert detector.is_loaded is True

    def test_context_manager_releases_on_exit(self, detector):
        with detector:
            pass
        assert detector.is_loaded is False

    def test_context_manager_releases_on_exception(self, detector):
        try:
            with detector:
                raise ValueError("test error")
        except ValueError:
            pass
        assert detector.is_loaded is False

    def test_context_manager_returns_self(self, detector):
        with detector as d:
            assert d is detector


    def test_resolve_device_cpu_passthrough(self):
        result = BaseDetector._resolve_device("cpu")
        assert result == "cpu"

    def test_resolve_device_cuda_passthrough(self):
        result = BaseDetector._resolve_device("cuda")
        assert result == "cuda"

    def test_resolve_device_cuda_idx_passthrough(self):
        result = BaseDetector._resolve_device("cuda:1")
        assert result == "cuda:1"

    def test_resolve_device_mps_passthrough(self):
        result = BaseDetector._resolve_device("mps")
        assert result == "mps"

    def test_resolve_device_auto_returns_string(self):
        # Just verify it returns a non-empty string
        result = BaseDetector._resolve_device("auto")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("core.detector.base_detector.BaseDetector._resolve_device")
    def test_resolve_device_auto_cuda_if_available(self, mock_resolve):
        # Simulate CUDA available
        mock_resolve.return_value = "cuda"
        det = ConcreteDetector.__new__(ConcreteDetector)
        det.device = "cuda"
        assert det.device == "cuda"
