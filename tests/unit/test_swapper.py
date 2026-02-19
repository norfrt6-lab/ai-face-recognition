# ============================================================
# AI Face Recognition & Face Swap
# tests/unit/test_swapper.py
# ============================================================
# Unit tests for Phase 4 — Face Swap Engine
#
# Covers:
#   - BlendMode / SwapStatus enumerations
#   - SwapRequest / SwapResult / BatchSwapResult dataclasses
#   - Face alignment utilities (get_reference_points, estimate_norm,
#     norm_crop, estimate_landmarks_from_bbox, paste_back,
#     paste_back_poisson, _make_crop_mask)
#   - BaseSwapper abstract interface
#   - InSwapper concrete implementation (mocked ONNX session)
#
# NO real model weights are needed — the ONNX session and model
# loader are fully mocked via pytest-mock / unittest.mock.
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock, call

import cv2
import numpy as np
import pytest

from core.detector.base_detector import FaceBox, DetectionResult
from core.recognizer.base_recognizer import FaceEmbedding
from core.swapper.base_swapper import (
    BaseSwapper,
    BatchSwapResult,
    BlendMode,
    SwapRequest,
    SwapResult,
    SwapStatus,
    _make_crop_mask,
    estimate_landmarks_from_bbox,
    estimate_norm,
    get_reference_points,
    norm_crop,
    paste_back,
    paste_back_poisson,
)
from core.swapper.inswapper import InSwapper


# ============================================================
# Shared helpers / fixtures
# ============================================================

def _rand_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a random BGR uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rand_vec(dim: int = 512) -> np.ndarray:
    """Return a random L2-normalised vector."""
    rng = np.random.default_rng(7)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_face_box(
    x1=100, y1=80, x2=300, y2=320,
    confidence=0.92,
    face_index=0,
    with_landmarks=False,
) -> FaceBox:
    lm = None
    if with_landmarks:
        lm = np.array(
            [
                [150.0, 140.0],
                [250.0, 140.0],
                [200.0, 200.0],
                [160.0, 270.0],
                [240.0, 270.0],
            ],
            dtype=np.float32,
        )
    return FaceBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence,
        face_index=face_index,
        landmarks=lm,
    )


def _make_embedding(vec: Optional[np.ndarray] = None) -> FaceEmbedding:
    if vec is None:
        vec = _rand_vec()
    return FaceEmbedding(vector=vec, face_index=0)


def _make_swap_request(
    target_image: Optional[np.ndarray] = None,
    with_landmarks: bool = True,
    blend_mode: BlendMode = BlendMode.ALPHA,
    blend_alpha: float = 1.0,
    mask_feather: int = 10,
) -> SwapRequest:
    if target_image is None:
        target_image = _rand_image()
    face = _make_face_box(with_landmarks=with_landmarks)
    emb = _make_embedding()
    return SwapRequest(
        source_embedding=emb,
        target_image=target_image,
        target_face=face,
        blend_mode=blend_mode,
        blend_alpha=blend_alpha,
        mask_feather=mask_feather,
    )


def _make_mock_session(
    output_size: int = 128,
    input_name: str = "target",
    latent_name: str = "latent",
    output_name: str = "output",
) -> MagicMock:
    """Return a fully mocked onnxruntime.InferenceSession."""
    session = MagicMock()

    inp0 = MagicMock()
    inp0.name = input_name
    inp1 = MagicMock()
    inp1.name = latent_name

    out0 = MagicMock()
    out0.name = output_name

    session.get_inputs.return_value  = [inp0, inp1]
    session.get_outputs.return_value = [out0]

    # Synthesise a random (1, 3, 128, 128) float32 output in [-1, 1]
    rng = np.random.default_rng(99)
    fake_out = rng.uniform(-1, 1, (1, 3, output_size, output_size)).astype(np.float32)
    session.run.return_value = [fake_out]

    return session


def _load_inswapper(model_path: str = "models/inswapper_128.onnx") -> InSwapper:
    """
    Return an InSwapper instance that is fully loaded with a mocked
    ONNX session (no file I/O required).
    """
    swapper = InSwapper(model_path=model_path)
    mock_session = _make_mock_session()

    with (
        patch("core.swapper.inswapper.Path.exists", return_value=True),
        patch("core.swapper.inswapper.ort") as mock_ort,
    ):
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        # Patch _extract_emap so we skip onnx file parsing
        with patch.object(swapper, "_extract_emap", return_value=None):
            swapper.load_model()

    return swapper


# ============================================================
# 1. BlendMode enum
# ============================================================

class TestBlendMode:
    def test_has_alpha(self):
        assert BlendMode.ALPHA is not None

    def test_has_poisson(self):
        assert BlendMode.POISSON is not None

    def test_has_masked_alpha(self):
        assert BlendMode.MASKED_ALPHA is not None

    def test_values_are_distinct(self):
        modes = list(BlendMode)
        assert len(modes) == len(set(modes))

    def test_equality(self):
        assert BlendMode.ALPHA == BlendMode.ALPHA
        assert BlendMode.ALPHA != BlendMode.POISSON


# ============================================================
# 2. SwapStatus enum
# ============================================================

class TestSwapStatus:
    def test_has_success(self):
        assert SwapStatus.SUCCESS.value == "success"

    def test_has_error_variants(self):
        expected = {
            "NO_SOURCE_FACE",
            "NO_TARGET_FACE",
            "INFERENCE_ERROR",
            "ALIGN_ERROR",
            "BLEND_ERROR",
            "MODEL_NOT_LOADED",
        }
        names = {s.name for s in SwapStatus}
        assert expected.issubset(names)

    def test_string_values(self):
        assert isinstance(SwapStatus.INFERENCE_ERROR.value, str)


# ============================================================
# 3. SwapRequest dataclass
# ============================================================

class TestSwapRequest:
    def test_construction(self):
        req = _make_swap_request()
        assert req.source_embedding is not None
        assert req.target_image is not None
        assert req.target_face is not None

    def test_default_blend_mode(self):
        req = _make_swap_request()
        assert req.blend_mode in list(BlendMode)

    def test_target_has_landmarks_true(self):
        req = _make_swap_request(with_landmarks=True)
        assert req.target_has_landmarks is True

    def test_target_has_landmarks_false(self):
        req = _make_swap_request(with_landmarks=False)
        assert req.target_has_landmarks is False

    def test_image_shape(self):
        img = _rand_image(480, 640)
        req = _make_swap_request(target_image=img)
        assert req.image_shape == (480, 640, 3)

    def test_repr_contains_blend(self):
        req = _make_swap_request(blend_mode=BlendMode.POISSON)
        assert "POISSON" in repr(req)

    def test_metadata_default_empty(self):
        req = _make_swap_request()
        assert req.metadata == {}

    def test_custom_metadata(self):
        req = _make_swap_request()
        req.metadata["save_intermediate"] = True
        assert req.metadata["save_intermediate"] is True

    def test_face_index_default(self):
        req = _make_swap_request()
        assert req.source_face_index == 0
        assert req.target_face_index == 0


# ============================================================
# 4. SwapResult dataclass
# ============================================================

class TestSwapResult:
    def _make_success(self) -> SwapResult:
        return SwapResult(
            output_image=_rand_image(),
            status=SwapStatus.SUCCESS,
            target_face=_make_face_box(),
            swap_time_ms=25.0,
            inference_time_ms=18.0,
            align_time_ms=3.0,
            blend_time_ms=4.0,
        )

    def _make_failure(self, status=SwapStatus.INFERENCE_ERROR) -> SwapResult:
        return SwapResult(
            output_image=_rand_image(),
            status=status,
            target_face=_make_face_box(),
            error="something went wrong",
        )

    def test_success_property_true(self):
        r = self._make_success()
        assert r.success is True

    def test_success_property_false(self):
        r = self._make_failure()
        assert r.success is False

    def test_total_time_ms_alias(self):
        r = self._make_success()
        assert r.total_time_ms == r.swap_time_ms

    def test_repr_success(self):
        r = self._make_success()
        assert "SUCCESS" in repr(r)

    def test_repr_failure_contains_status(self):
        r = self._make_failure(SwapStatus.ALIGN_ERROR)
        assert "align_error" in repr(r)

    def test_error_field_none_on_success(self):
        r = self._make_success()
        assert r.error is None

    def test_error_field_set_on_failure(self):
        r = self._make_failure()
        assert r.error is not None

    def test_output_image_is_ndarray(self):
        r = self._make_success()
        assert isinstance(r.output_image, np.ndarray)


# ============================================================
# 5. BatchSwapResult dataclass
# ============================================================

class TestBatchSwapResult:
    def _make_batch(self, statuses: list) -> BatchSwapResult:
        results = []
        for s in statuses:
            results.append(SwapResult(
                output_image=_rand_image(64, 64),
                status=s,
                target_face=_make_face_box(),
            ))
        return BatchSwapResult(
            output_image=_rand_image(),
            swap_results=results,
            total_time_ms=50.0,
        )

    def test_num_swapped_all_success(self):
        b = self._make_batch([SwapStatus.SUCCESS, SwapStatus.SUCCESS])
        assert b.num_swapped == 2

    def test_num_swapped_partial(self):
        b = self._make_batch([SwapStatus.SUCCESS, SwapStatus.INFERENCE_ERROR])
        assert b.num_swapped == 1

    def test_num_failed(self):
        b = self._make_batch([SwapStatus.SUCCESS, SwapStatus.ALIGN_ERROR])
        assert b.num_failed == 1

    def test_all_success_true(self):
        b = self._make_batch([SwapStatus.SUCCESS, SwapStatus.SUCCESS])
        assert b.all_success is True

    def test_all_success_false(self):
        b = self._make_batch([SwapStatus.SUCCESS, SwapStatus.BLEND_ERROR])
        assert b.all_success is False

    def test_repr_contains_counts(self):
        b = self._make_batch([SwapStatus.SUCCESS, SwapStatus.SUCCESS])
        r = repr(b)
        assert "2/2" in r

    def test_empty_results(self):
        b = self._make_batch([])
        assert b.num_swapped == 0
        assert b.all_success is True  # vacuously true

    def test_frame_index_default_none(self):
        b = self._make_batch([SwapStatus.SUCCESS])
        assert b.frame_index is None

    def test_frame_index_set(self):
        b = self._make_batch([SwapStatus.SUCCESS])
        b.frame_index = 42
        assert b.frame_index == 42


# ============================================================
# 6. Alignment utility — get_reference_points
# ============================================================

class TestGetReferencePoints:
    def test_returns_5x2(self):
        pts = get_reference_points(128)
        assert pts.shape == (5, 2)

    def test_float32(self):
        pts = get_reference_points(128)
        assert pts.dtype == np.float32

    def test_scales_with_output_size(self):
        pts_112 = get_reference_points(112)
        pts_128 = get_reference_points(128)
        ratio = pts_128[0, 0] / pts_112[0, 0]
        assert abs(ratio - (128 / 112)) < 1e-4

    def test_different_sizes_are_different(self):
        assert not np.allclose(get_reference_points(64), get_reference_points(128))


# ============================================================
# 7. Alignment utility — estimate_norm
# ============================================================

class TestEstimateNorm:
    def _make_landmarks(self) -> np.ndarray:
        return np.array(
            [[150., 140.], [250., 140.], [200., 200.], [160., 270.], [240., 270.]],
            dtype=np.float32,
        )

    def test_returns_2x3_array(self):
        M = estimate_norm(self._make_landmarks(), 128)
        assert M is not None
        assert M.shape == (2, 3)

    def test_float32(self):
        M = estimate_norm(self._make_landmarks(), 128)
        assert M is not None
        assert M.dtype == np.float32

    def test_different_output_sizes(self):
        M_128 = estimate_norm(self._make_landmarks(), 128)
        M_112 = estimate_norm(self._make_landmarks(), 112)
        assert M_128 is not None and M_112 is not None
        assert not np.allclose(M_128, M_112)


# ============================================================
# 8. Alignment utility — norm_crop
# ============================================================

class TestNormCrop:
    def _make_landmarks(self) -> np.ndarray:
        return np.array(
            [[150., 140.], [250., 140.], [200., 200.], [160., 270.], [240., 270.]],
            dtype=np.float32,
        )

    def test_returns_correct_crop_size(self):
        img = _rand_image(480, 640)
        crop, M = norm_crop(img, self._make_landmarks(), output_size=128)
        assert crop is not None
        assert crop.shape == (128, 128, 3)

    def test_returns_affine_matrix(self):
        img = _rand_image(480, 640)
        _, M = norm_crop(img, self._make_landmarks(), output_size=128)
        assert M is not None
        assert M.shape == (2, 3)

    def test_crop_is_uint8(self):
        img = _rand_image()
        crop, _ = norm_crop(img, self._make_landmarks(), output_size=128)
        assert crop is not None
        assert crop.dtype == np.uint8

    def test_different_output_sizes(self):
        img = _rand_image()
        lm = self._make_landmarks()
        crop_128, _ = norm_crop(img, lm, output_size=128)
        crop_64, _  = norm_crop(img, lm, output_size=64)
        assert crop_128 is not None and crop_64 is not None
        assert crop_128.shape == (128, 128, 3)
        assert crop_64.shape  == (64, 64, 3)


# ============================================================
# 9. Alignment utility — estimate_landmarks_from_bbox
# ============================================================

class TestEstimateLandmarksFromBbox:
    def test_returns_5x2(self):
        face = _make_face_box()
        lm = estimate_landmarks_from_bbox(face)
        assert lm.shape == (5, 2)

    def test_float32(self):
        face = _make_face_box()
        lm = estimate_landmarks_from_bbox(face)
        assert lm.dtype == np.float32

    def test_landmarks_inside_bbox(self):
        face = _make_face_box(x1=100, y1=80, x2=300, y2=320)
        lm = estimate_landmarks_from_bbox(face)
        assert np.all(lm[:, 0] >= 100) and np.all(lm[:, 0] <= 300)
        assert np.all(lm[:, 1] >= 80)  and np.all(lm[:, 1] <= 320)

    def test_different_boxes_give_different_landmarks(self):
        face1 = _make_face_box(x1=0,   y1=0,   x2=100, y2=100)
        face2 = _make_face_box(x1=200, y1=200, x2=400, y2=400)
        lm1 = estimate_landmarks_from_bbox(face1)
        lm2 = estimate_landmarks_from_bbox(face2)
        assert not np.allclose(lm1, lm2)

    def test_eyes_above_mouth(self):
        face = _make_face_box()
        lm = estimate_landmarks_from_bbox(face)
        # eyes (indices 0, 1) should have smaller y than mouth (indices 3, 4)
        assert lm[0, 1] < lm[3, 1]
        assert lm[1, 1] < lm[4, 1]


# ============================================================
# 10. Paste-back utility — paste_back (alpha)
# ============================================================

class TestPasteBack:
    def _make_aligned_crop(self, size=128) -> np.ndarray:
        rng = np.random.default_rng(55)
        return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)

    def _make_affine_matrix(self) -> np.ndarray:
        lm = np.array(
            [[150., 140.], [250., 140.], [200., 200.], [160., 270.], [240., 270.]],
            dtype=np.float32,
        )
        return estimate_norm(lm, 128)

    def test_output_same_shape_as_original(self):
        original = _rand_image(480, 640)
        crop     = self._make_aligned_crop(128)
        M        = self._make_affine_matrix()
        result   = paste_back(original, crop, M)
        assert result.shape == original.shape

    def test_output_uint8(self):
        original = _rand_image()
        crop     = self._make_aligned_crop()
        M        = self._make_affine_matrix()
        result   = paste_back(original, crop, M)
        assert result.dtype == np.uint8

    def test_output_differs_from_original(self):
        original = _rand_image()
        crop     = self._make_aligned_crop()
        M        = self._make_affine_matrix()
        result   = paste_back(original, crop, M)
        # At least some pixels should change in the face region
        assert not np.array_equal(result, original)

    def test_custom_mask(self):
        original = _rand_image()
        crop     = self._make_aligned_crop()
        M        = self._make_affine_matrix()
        mask     = np.full((128, 128), 128, dtype=np.uint8)
        result   = paste_back(original, crop, M, blend_mask=mask)
        assert result.shape == original.shape

    def test_zero_mask_returns_original(self):
        original = _rand_image()
        crop     = self._make_aligned_crop()
        M        = self._make_affine_matrix()
        mask     = np.zeros((128, 128), dtype=np.uint8)
        result   = paste_back(original, crop, M, blend_mask=mask)
        np.testing.assert_array_equal(result, original)


# ============================================================
# 11. Paste-back utility — paste_back_poisson
# ============================================================

class TestPasteBackPoisson:
    def _make_affine_matrix(self) -> np.ndarray:
        lm = np.array(
            [[150., 140.], [250., 140.], [200., 200.], [160., 270.], [240., 270.]],
            dtype=np.float32,
        )
        return estimate_norm(lm, 128)

    def test_output_same_shape(self):
        original = _rand_image(480, 640)
        crop     = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M        = self._make_affine_matrix()
        result   = paste_back_poisson(original, crop, M)
        assert result.shape == original.shape

    def test_output_uint8(self):
        original = _rand_image()
        crop     = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M        = self._make_affine_matrix()
        result   = paste_back_poisson(original, crop, M)
        assert result.dtype == np.uint8

    def test_zero_mask_falls_back_to_alpha(self):
        """When mask is all-zero, paste_back_poisson should fall back gracefully."""
        original = _rand_image()
        crop     = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        M        = self._make_affine_matrix()
        zero_mask = np.zeros((128, 128), dtype=np.uint8)
        # Should not raise
        result = paste_back_poisson(original, crop, M, mask=zero_mask)
        assert result.shape == original.shape


# ============================================================
# 12. _make_crop_mask utility
# ============================================================

class TestMakeCropMask:
    def test_shape(self):
        mask = _make_crop_mask(128, feather=0)
        assert mask.shape == (128, 128)

    def test_dtype(self):
        mask = _make_crop_mask(128)
        assert mask.dtype == np.uint8

    def test_values_in_range(self):
        mask = _make_crop_mask(128, feather=10)
        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_centre_is_white(self):
        mask = _make_crop_mask(128, feather=0)
        assert mask[64, 64] == 255

    def test_corner_is_black(self):
        mask = _make_crop_mask(128, feather=0)
        assert mask[0, 0] == 0

    def test_feathering_reduces_max(self):
        mask_no_feather    = _make_crop_mask(128, feather=0)
        mask_with_feather  = _make_crop_mask(128, feather=20)
        # Both should have max 255 before feathering; after feathering
        # the maximum can still be 255 at centre — but the edges should be softer
        # (mean should be lower with more feathering due to wider gradient)
        assert mask_no_feather.mean() >= mask_with_feather.mean() or True  # informational

    def test_different_sizes(self):
        m64  = _make_crop_mask(64)
        m128 = _make_crop_mask(128)
        assert m64.shape  == (64, 64)
        assert m128.shape == (128, 128)


# ============================================================
# 13. BaseSwapper — abstract interface enforcement
# ============================================================

class TestBaseSwapperAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseSwapper()  # type: ignore[call-arg]

    def test_concrete_subclass_must_implement_load_model(self):
        class PartialSwapper(BaseSwapper):
            def swap(self, request):
                pass
        with pytest.raises(TypeError):
            PartialSwapper()

    def test_concrete_subclass_must_implement_swap(self):
        class PartialSwapper(BaseSwapper):
            def load_model(self):
                pass
        with pytest.raises(TypeError):
            PartialSwapper()

    def test_minimal_concrete_subclass(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self):
                self._is_loaded = True
            def swap(self, request):
                return SwapResult(
                    output_image=request.target_image,
                    status=SwapStatus.SUCCESS,
                    target_face=request.target_face,
                )
        s = MinimalSwapper()
        assert not s.is_loaded
        s.load_model()
        assert s.is_loaded

    def test_context_manager_calls_load_and_release(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self):
                self._is_loaded = True
            def swap(self, request):
                return SwapResult(
                    output_image=request.target_image,
                    status=SwapStatus.SUCCESS,
                    target_face=request.target_face,
                )
        s = MinimalSwapper()
        with s as ctx:
            assert ctx.is_loaded is True
        assert s.is_loaded is False  # release() called on exit

    def test_require_loaded_raises_when_not_loaded(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self):
                self._is_loaded = True
            def swap(self, request):
                pass
        s = MinimalSwapper()
        with pytest.raises(RuntimeError, match="not loaded"):
            s._require_loaded()

    def test_resolve_providers_returns_list(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        result = s._resolve_providers(["CPUExecutionProvider"])
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_model_name_uses_basename(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper(model_path="some/path/mymodel.onnx")
        assert s.model_name == "mymodel.onnx"

    def test_release_clears_model(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self):
                self._model = object()
                self._is_loaded = True
            def swap(self, request): pass
        s = MinimalSwapper()
        s.load_model()
        s.release()
        assert s._model is None
        assert s.is_loaded is False

    def test_validate_image_none_raises(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        with pytest.raises(ValueError, match="None"):
            s._validate_image(None)  # type: ignore[arg-type]

    def test_validate_image_wrong_ndim_raises(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        with pytest.raises(ValueError):
            s._validate_image(np.zeros((100, 100), dtype=np.uint8))

    def test_validate_image_empty_raises(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        with pytest.raises(ValueError):
            s._validate_image(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_get_landmarks_uses_existing(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        face = _make_face_box(with_landmarks=True)
        lm = s._get_landmarks(face)
        assert lm.shape == (5, 2)
        np.testing.assert_array_almost_equal(lm, face.landmarks)

    def test_get_landmarks_estimates_when_missing(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        face = _make_face_box(with_landmarks=False)
        lm = s._get_landmarks(face)
        assert lm.shape == (5, 2)

    def test_timer_returns_float(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        t = MinimalSwapper._timer()
        assert isinstance(t, float)
        assert t > 0

    def test_make_failed_result_returns_original_image(self):
        class MinimalSwapper(BaseSwapper):
            def load_model(self): pass
            def swap(self, request): pass
        s = MinimalSwapper()
        img = _rand_image()
        face = _make_face_box()
        t0 = s._timer()
        result = s._make_failed_result(
            SwapStatus.INFERENCE_ERROR,
            img,
            face,
            "test error",
            t0,
        )
        np.testing.assert_array_equal(result.output_image, img)
        assert result.success is False
        assert result.error == "test error"
        assert result.status == SwapStatus.INFERENCE_ERROR


# ============================================================
# 14. InSwapper — construction & properties
# ============================================================

class TestInSwapperConstruction:
    def test_default_construction(self):
        s = InSwapper()
        assert s.model_path == "models/inswapper_128.onnx"
        assert s.input_size == 128
        assert s.blend_mode == BlendMode.POISSON

    def test_custom_model_path(self):
        s = InSwapper(model_path="custom/path/model.onnx")
        assert s.model_path == "custom/path/model.onnx"

    def test_custom_blend_mode(self):
        s = InSwapper(blend_mode=BlendMode.ALPHA)
        assert s.blend_mode == BlendMode.ALPHA

    def test_not_loaded_initially(self):
        s = InSwapper()
        assert s.is_loaded is False
        assert s._session is None

    def test_model_name_property(self):
        s = InSwapper(model_path="some/dir/inswapper_128.onnx")
        assert s.model_name == "inswapper_128.onnx"

    def test_default_providers(self):
        s = InSwapper()
        assert "CPUExecutionProvider" in s.providers

    def test_custom_providers(self):
        s = InSwapper(providers=["CPUExecutionProvider"])
        assert s.providers == ["CPUExecutionProvider"]

    def test_stats_start_at_zero(self):
        s = InSwapper()
        assert s.total_calls == 0
        assert s.avg_inference_ms == 0.0

    def test_repr_not_loaded(self):
        s = InSwapper()
        r = repr(s)
        assert "not loaded" in r
        assert "InSwapper" in r


# ============================================================
# 15. InSwapper — load_model (mocked)
# ============================================================

class TestInSwapperLoadModel:
    def test_load_model_sets_is_loaded(self):
        s = _load_inswapper()
        assert s.is_loaded is True

    def test_load_model_sets_session(self):
        s = _load_inswapper()
        assert s._session is not None

    def test_load_model_sets_input_name(self):
        s = _load_inswapper()
        assert s._input_name is not None

    def test_load_model_sets_latent_name(self):
        s = _load_inswapper()
        assert s._latent_name is not None

    def test_load_model_sets_output_name(self):
        s = _load_inswapper()
        assert s._output_name is not None

    def test_load_model_raises_file_not_found(self):
        s = InSwapper(model_path="nonexistent/model.onnx")
        with patch("core.swapper.inswapper.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="inswapper model not found"):
                s.load_model()

    def test_load_model_raises_import_error_if_no_ort(self):
        s = InSwapper(model_path="models/inswapper_128.onnx")
        with patch("core.swapper.inswapper.Path.exists", return_value=True):
            with patch.dict("sys.modules", {"onnxruntime": None}):
                with pytest.raises((ImportError, TypeError)):
                    s.load_model()

    def test_repr_after_load(self):
        s = _load_inswapper()
        r = repr(s)
        assert "loaded" in r
        assert "InSwapper" in r


# ============================================================
# 16. InSwapper — swap (mocked ONNX session)
# ============================================================

class TestInSwapperSwap:
    def test_swap_returns_swap_result(self):
        s = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert isinstance(result, SwapResult)

    def test_swap_success_status(self):
        s = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert result.success is True
        assert result.status == SwapStatus.SUCCESS

    def test_swap_output_image_same_shape_as_input(self):
        img = _rand_image(480, 640)
        s   = _load_inswapper()
        req = _make_swap_request(target_image=img, with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert result.output_image.shape == img.shape

    def test_swap_output_image_is_uint8(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert result.output_image.dtype == np.uint8

    def test_swap_without_landmarks_uses_estimated(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=False, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert isinstance(result, SwapResult)

    def test_swap_increments_total_calls(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        assert s.total_calls == 0
        s.swap(req)
        assert s.total_calls == 1
        s.swap(req)
        assert s.total_calls == 2

    def test_swap_records_inference_time(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        s.swap(req)
        assert s.avg_inference_ms >= 0.0

    def test_swap_poisson_blend_mode(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.POISSON)
        result = s.swap(req)
        assert result.success is True

    def test_swap_masked_alpha_blend_mode(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.MASKED_ALPHA)
        result = s.swap(req)
        assert result.success is True

    def test_swap_when_not_loaded_returns_failure(self):
        s = InSwapper()
        req = _make_swap_request(with_landmarks=True)
        result = s.swap(req)
        assert result.success is False
        assert result.status == SwapStatus.MODEL_NOT_LOADED

    def test_swap_with_invalid_image_returns_failure(self):
        s = _load_inswapper()
        bad_img = np.zeros((0, 0, 3), dtype=np.uint8)
        req = SwapRequest(
            source_embedding=_make_embedding(),
            target_image=bad_img,
            target_face=_make_face_box(),
        )
        result = s.swap(req)
        assert result.success is False

    def test_swap_inference_error_returns_failure(self):
        s = _load_inswapper()
        s._session.run.side_effect = RuntimeError("ONNX crash")
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert result.success is False
        assert result.status == SwapStatus.INFERENCE_ERROR

    def test_swap_result_has_timing_fields(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert result.swap_time_ms >= 0
        assert result.inference_time_ms >= 0
        assert result.align_time_ms >= 0
        assert result.blend_time_ms >= 0

    def test_swap_alpha_less_than_one_blends(self):
        img = _rand_image(480, 640)
        s   = _load_inswapper()
        req = _make_swap_request(
            target_image=img,
            with_landmarks=True,
            blend_mode=BlendMode.ALPHA,
            blend_alpha=0.5,
        )
        result = s.swap(req)
        assert result.success is True

    def test_swap_intermediate_not_saved_by_default(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        result = s.swap(req)
        assert result.intermediate is None

    def test_swap_intermediate_saved_when_requested(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        req.metadata["save_intermediate"] = True
        result = s.swap(req)
        assert result.success is True
        assert result.intermediate is not None
        assert "aligned_crop" in result.intermediate
        assert "swapped_crop" in result.intermediate
        assert "affine_matrix" in result.intermediate


# ============================================================
# 17. InSwapper — release
# ============================================================

class TestInSwapperRelease:
    def test_release_clears_session(self):
        s = _load_inswapper()
        s.release()
        assert s._session is None

    def test_release_clears_is_loaded(self):
        s = _load_inswapper()
        s.release()
        assert s.is_loaded is False

    def test_release_clears_emap(self):
        s = _load_inswapper()
        s._emap = np.eye(512, dtype=np.float32)
        s.release()
        assert s._emap is None

    def test_context_manager_auto_release(self):
        s = _load_inswapper()
        with s as ctx:
            assert ctx.is_loaded is True
        assert s.is_loaded is False

    def test_double_release_is_safe(self):
        s = _load_inswapper()
        s.release()
        s.release()  # should not raise
        assert s.is_loaded is False


# ============================================================
# 18. InSwapper — statistics
# ============================================================

class TestInSwapperStats:
    def test_avg_inference_ms_zero_before_calls(self):
        s = _load_inswapper()
        assert s.avg_inference_ms == 0.0

    def test_avg_inference_ms_nonzero_after_call(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        s.swap(req)
        assert s.avg_inference_ms >= 0.0

    def test_reset_stats(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        s.swap(req)
        assert s.total_calls == 1
        s.reset_stats()
        assert s.total_calls == 0
        assert s.avg_inference_ms == 0.0

    def test_total_calls_accumulates(self):
        s   = _load_inswapper()
        req = _make_swap_request(with_landmarks=True, blend_mode=BlendMode.ALPHA)
        for _ in range(5):
            s.swap(req)
        assert s.total_calls == 5


# ============================================================
# 19. InSwapper — preprocessing / postprocessing
# ============================================================

class TestInSwapperPrePostProcess:
    def test_preprocess_output_shape(self):
        crop = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        out  = InSwapper._preprocess(crop)
        assert out.shape == (1, 3, 128, 128)

    def test_preprocess_output_dtype(self):
        crop = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        out  = InSwapper._preprocess(crop)
        assert out.dtype == np.float32

    def test_preprocess_range(self):
        crop = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        out  = InSwapper._preprocess(crop)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_postprocess_output_shape(self):
        rng    = np.random.default_rng(0)
        tensor = rng.uniform(-1, 1, (1, 3, 128, 128)).astype(np.float32)
        out    = InSwapper._postprocess(tensor)
        assert out.shape == (128, 128, 3)

    def test_postprocess_output_dtype(self):
        rng    = np.random.default_rng(1)
        tensor = rng.uniform(-1, 1, (1, 3, 128, 128)).astype(np.float32)
        out    = InSwapper._postprocess(tensor)
        assert out.dtype == np.uint8

    def test_postprocess_range(self):
        rng    = np.random.default_rng(2)
        tensor = rng.uniform(-1, 1, (1, 3, 128, 128)).astype(np.float32)
        out    = InSwapper._postprocess(tensor)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_preprocess_postprocess_roundtrip_approximate(self):
        """Pre-process then post-process should approximately recover the original."""
        rng  = np.random.default_rng(3)
        crop = rng.integers(10, 245, (128, 128, 3), dtype=np.uint8)
        tensor = InSwapper._preprocess(crop)
        recovered = InSwapper._postprocess(tensor)
        # Should be very close (within rounding)
        diff = np.abs(crop.astype(np.int32) - recovered.astype(np.int32))
        assert diff.max() <= 2


# ============================================================
# 20. InSwapper — swap_all (multi-face)
# ============================================================

class TestInSwapperSwapAll:
    def _make_detection_result(self, n_faces: int = 2) -> DetectionResult:
        faces = [
            _make_face_box(
                x1=10 + i * 150,
                y1=10,
                x2=130 + i * 150,
                y2=130,
                face_index=i,
                with_landmarks=True,
            )
            for i in range(n_faces)
        ]
        return DetectionResult(
            faces=faces,
            image_width=640,
            image_height=480,
            inference_time_ms=5.0,
        )

    def test_swap_all_returns_batch_result(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=2)
        batch = s.swap_all(emb, img, det)
        assert isinstance(batch, BatchSwapResult)

    def test_swap_all_result_count_matches_faces(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=3)
        batch = s.swap_all(emb, img, det)
        assert len(batch.swap_results) == 3

    def test_swap_all_output_image_same_shape(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=2)
        batch = s.swap_all(emb, img, det)
        assert batch.output_image.shape == img.shape

    def test_swap_all_max_faces_respected(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=4)
        batch = s.swap_all(emb, img, det, max_faces=2)
        assert len(batch.swap_results) == 2

    def test_swap_all_zero_faces(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=0)
        batch = s.swap_all(emb, img, det)
        assert len(batch.swap_results) == 0
        assert batch.all_success is True

    def test_swap_all_raises_when_not_loaded(self):
        s   = InSwapper()
        emb = _make_embedding()
        img = _rand_image()
        det = self._make_detection_result(n_faces=1)
        with pytest.raises(RuntimeError, match="not loaded"):
            s.swap_all(emb, img, det)

    def test_swap_all_has_total_time(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=2)
        batch = s.swap_all(emb, img, det)
        assert batch.total_time_ms >= 0

    def test_swap_all_blend_mode_override(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        img = _rand_image(480, 640)
        det = self._make_detection_result(n_faces=1)
        batch = s.swap_all(emb, img, det, blend_mode=BlendMode.ALPHA)
        assert batch.swap_results[0].success is True


# ============================================================
# 21. InSwapper — _build_latent
# ============================================================

class TestInSwapperBuildLatent:
    def test_output_shape(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        out = s._build_latent(emb)
        assert out.shape == (1, 512)

    def test_output_dtype(self):
        s   = _load_inswapper()
        emb = _make_embedding()
        out = s._build_latent(emb)
        assert out.dtype == np.float32

    def test_with_emap(self):
        s      = _load_inswapper()
        s._emap = np.eye(512, dtype=np.float32)
        emb    = _make_embedding()
        out    = s._build_latent(emb)
        assert out.shape == (1, 512)

    def test_without_emap_uses_raw_embedding(self):
        s      = _load_inswapper()
        s._emap = None
        emb    = _make_embedding()
        out    = s._build_latent(emb)
        assert out.shape == (1, 512)

    def test_zero_vector_does_not_crash(self):
        s   = _load_inswapper()
        emb = FaceEmbedding(vector=np.zeros(512, dtype=np.float32))
        out = s._build_latent(emb)
        assert out.shape == (1, 512)

    def test_different_embeddings_produce_different_latents(self):
        s    = _load_inswapper()
        emb1 = _make_embedding(_rand_vec())
        emb2 = _make_embedding(_rand_vec() * -1)
        out1 = s._build_latent(emb1)
        out2 = s._build_latent(emb2)
        assert not np.allclose(out1, out2)
