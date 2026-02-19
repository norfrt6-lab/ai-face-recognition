# ============================================================
# AI Face Recognition & Face Swap
# tests/unit/test_enhancer.py
# ============================================================
# Unit tests for Phase 5 — Face Enhancement.
#
# Covers:
#   - EnhancerBackend / EnhancementStatus enumerations
#   - EnhancementRequest dataclass — construction, properties, repr
#   - EnhancementResult dataclass  — construction, properties, repr
#   - Utility functions: pad_image_for_enhancement, unpad_image,
#     find_center_face
#   - BaseEnhancer abstract interface (via a concrete stub)
#   - GFPGANEnhancer — construction, load_model error paths,
#     enhance() happy path and all failure branches (mocked
#     GFPGANer), release(), repr
#   - CodeFormerEnhancer — construction, load_model error paths,
#     enhance() happy path and all failure branches (mocked torch
#     + facexlib helpers), release(), repr
#
# NO real model weights are required — every third-party call
# (gfpgan, torch, basicsr, facexlib) is patched via unittest.mock.
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock, call

import cv2
import numpy as np
import pytest

from core.detector.base_detector import FaceBox
from core.enhancer.base_enhancer import (
    BaseEnhancer,
    EnhancementRequest,
    EnhancementResult,
    EnhancementStatus,
    EnhancerBackend,
    find_center_face,
    pad_image_for_enhancement,
    unpad_image,
)


# ============================================================
# Shared test helpers
# ============================================================

def _img(h: int = 128, w: int = 128) -> np.ndarray:
    """Return a random BGR uint8 test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _face_box(
    x1=20.0, y1=20.0, x2=100.0, y2=100.0,
    confidence=0.95,
    face_index=0,
) -> FaceBox:
    lm = np.array(
        [[40., 45.], [80., 45.], [60., 65.], [45., 85.], [75., 85.]],
        dtype=np.float32,
    )
    return FaceBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        confidence=confidence,
        face_index=face_index,
        landmarks=lm,
    )


def _make_request(**kwargs) -> EnhancementRequest:
    """Build an EnhancementRequest with sensible defaults."""
    defaults = dict(image=_img())
    defaults.update(kwargs)
    return EnhancementRequest(**defaults)


def _make_result(success: bool = True) -> EnhancementResult:
    return EnhancementResult(
        output_image=_img(),
        status=EnhancementStatus.SUCCESS if success else EnhancementStatus.INFERENCE_ERROR,
        backend=EnhancerBackend.GFPGAN,
        num_faces_enhanced=1 if success else 0,
        enhance_time_ms=30.0,
        inference_time_ms=25.0,
        upscale_factor=2,
        error=None if success else "mock error",
    )


# ============================================================
# Minimal concrete subclass of BaseEnhancer for abstract tests
# ============================================================

class _StubEnhancer(BaseEnhancer):
    """Minimal concrete enhancer for testing BaseEnhancer behaviour."""

    def __init__(self, model_path: str = "stub.pth", backend=EnhancerBackend.GFPGAN):
        super().__init__(
            model_path=model_path,
            backend=backend,
            upscale=2,
            only_center_face=False,
            paste_back=True,
            device="cpu",
        )
        self._load_called = False

    def load_model(self) -> None:
        self._load_called = True
        self._model = object()
        self._is_loaded = True

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        if not self._is_loaded:
            return self._make_failed_result(
                EnhancementStatus.MODEL_NOT_LOADED,
                request.image,
                "not loaded",
                self._timer(),
            )
        return EnhancementResult(
            output_image=request.image.copy(),
            status=EnhancementStatus.SUCCESS,
            backend=self.backend,
            num_faces_enhanced=1,
            enhance_time_ms=1.0,
            inference_time_ms=0.5,
            upscale_factor=self.upscale,
        )


# ============================================================
# 1. EnhancerBackend
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestEnhancerBackend:

    def test_gfpgan_value(self):
        assert EnhancerBackend.GFPGAN.value == "gfpgan"

    def test_codeformer_value(self):
        assert EnhancerBackend.CODEFORMER.value == "codeformer"

    def test_none_value(self):
        assert EnhancerBackend.NONE.value == "none"

    def test_all_members_present(self):
        members = {e.value for e in EnhancerBackend}
        assert {"gfpgan", "codeformer", "none"}.issubset(members)

    def test_from_string_gfpgan(self):
        assert EnhancerBackend("gfpgan") == EnhancerBackend.GFPGAN

    def test_from_string_codeformer(self):
        assert EnhancerBackend("codeformer") == EnhancerBackend.CODEFORMER


# ============================================================
# 2. EnhancementStatus
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestEnhancementStatus:

    def test_success_value(self):
        assert EnhancementStatus.SUCCESS.value == "success"

    def test_no_face_detected_value(self):
        assert EnhancementStatus.NO_FACE_DETECTED.value == "no_face_detected"

    def test_inference_error_value(self):
        assert EnhancementStatus.INFERENCE_ERROR.value == "inference_error"

    def test_model_not_loaded_value(self):
        assert EnhancementStatus.MODEL_NOT_LOADED.value == "model_not_loaded"

    def test_invalid_input_value(self):
        assert EnhancementStatus.INVALID_INPUT.value == "invalid_input"

    def test_disabled_value(self):
        assert EnhancementStatus.DISABLED.value == "disabled"

    def test_all_six_statuses_exist(self):
        assert len(EnhancementStatus) >= 6


# ============================================================
# 3. EnhancementRequest
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestEnhancementRequest:

    def test_image_stored(self):
        img = _img()
        req = EnhancementRequest(image=img)
        assert req.image is img

    def test_default_fidelity_weight(self):
        req = EnhancementRequest(image=_img())
        assert req.fidelity_weight == 0.5

    def test_default_upscale(self):
        req = EnhancementRequest(image=_img())
        assert req.upscale == 2

    def test_default_only_center_face_false(self):
        req = EnhancementRequest(image=_img())
        assert req.only_center_face is False

    def test_default_paste_back_true(self):
        req = EnhancementRequest(image=_img())
        assert req.paste_back is True

    def test_default_full_frame_true(self):
        req = EnhancementRequest(image=_img())
        assert req.full_frame is True

    def test_default_face_boxes_none(self):
        req = EnhancementRequest(image=_img())
        assert req.face_boxes is None

    def test_default_metadata_empty_dict(self):
        req = EnhancementRequest(image=_img())
        assert req.metadata == {}

    def test_image_shape_property(self):
        img = _img(64, 96)
        req = EnhancementRequest(image=img)
        assert req.image_shape == (64, 96, 3)

    def test_has_face_boxes_false_by_default(self):
        req = EnhancementRequest(image=_img())
        assert req.has_face_boxes is False

    def test_has_face_boxes_true_when_provided(self):
        req = EnhancementRequest(image=_img(), face_boxes=[_face_box()])
        assert req.has_face_boxes is True

    def test_has_face_boxes_false_for_empty_list(self):
        req = EnhancementRequest(image=_img(), face_boxes=[])
        assert req.has_face_boxes is False

    def test_custom_fidelity_weight(self):
        req = EnhancementRequest(image=_img(), fidelity_weight=0.8)
        assert req.fidelity_weight == 0.8

    def test_custom_upscale(self):
        req = EnhancementRequest(image=_img(), upscale=4)
        assert req.upscale == 4

    def test_repr_contains_image_dimensions(self):
        req = _make_request()
        r   = repr(req)
        assert "128" in r   # width
        assert "128" in r   # height

    def test_repr_contains_upscale(self):
        req = EnhancementRequest(image=_img(), upscale=4)
        assert "4" in repr(req)

    def test_repr_contains_full_frame(self):
        req = _make_request()
        assert "full_frame" in repr(req)

    def test_metadata_dict_is_mutable(self):
        req = _make_request()
        req.metadata["key"] = "value"
        assert req.metadata["key"] == "value"


# ============================================================
# 4. EnhancementResult
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestEnhancementResult:

    def test_output_image_stored(self):
        img = _img()
        r   = EnhancementResult(
            output_image=img,
            status=EnhancementStatus.SUCCESS,
            backend=EnhancerBackend.GFPGAN,
        )
        assert r.output_image is img

    def test_success_property_true_on_success(self):
        r = _make_result(success=True)
        assert r.success is True

    def test_success_property_false_on_failure(self):
        r = _make_result(success=False)
        assert r.success is False

    def test_success_false_for_model_not_loaded(self):
        r = EnhancementResult(
            output_image=_img(),
            status=EnhancementStatus.MODEL_NOT_LOADED,
            backend=EnhancerBackend.GFPGAN,
        )
        assert r.success is False

    def test_is_passthrough_false_for_success(self):
        r = _make_result(success=True)
        assert r.is_passthrough is False

    def test_is_passthrough_true_for_disabled(self):
        r = EnhancementResult(
            output_image=_img(),
            status=EnhancementStatus.DISABLED,
            backend=EnhancerBackend.NONE,
        )
        assert r.is_passthrough is True

    def test_default_num_faces_enhanced_zero(self):
        r = EnhancementResult(
            output_image=_img(),
            status=EnhancementStatus.SUCCESS,
            backend=EnhancerBackend.GFPGAN,
        )
        assert r.num_faces_enhanced == 0

    def test_default_error_none(self):
        r = _make_result(success=True)
        assert r.error is None

    def test_error_stored_on_failure(self):
        r = _make_result(success=False)
        assert r.error == "mock error"

    def test_enhance_time_ms_stored(self):
        r = _make_result()
        assert r.enhance_time_ms == 30.0

    def test_inference_time_ms_stored(self):
        r = _make_result()
        assert r.inference_time_ms == 25.0

    def test_upscale_factor_stored(self):
        r = _make_result()
        assert r.upscale_factor == 2

    def test_backend_stored(self):
        r = _make_result()
        assert r.backend == EnhancerBackend.GFPGAN

    def test_repr_success_contains_SUCCESS(self):
        r   = _make_result(success=True)
        rep = repr(r)
        assert "SUCCESS" in rep or "success" in rep.lower()

    def test_repr_failure_contains_status(self):
        r   = _make_result(success=False)
        rep = repr(r)
        assert "inference_error" in rep or "INFERENCE_ERROR" in rep.lower()

    def test_repr_contains_backend(self):
        r = _make_result()
        assert "gfpgan" in repr(r).lower()

    def test_face_crops_default_none(self):
        r = _make_result()
        assert r.face_crops is None


# ============================================================
# 5. pad_image_for_enhancement
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestPadImageForEnhancement:

    def test_no_padding_when_large_enough(self):
        img, padding = pad_image_for_enhancement(_img(256, 256), min_size=128)
        assert padding == (0, 0, 0, 0)
        assert img.shape == (256, 256, 3)

    def test_pads_small_image_height(self):
        img, padding = pad_image_for_enhancement(_img(64, 256), min_size=128)
        top, bottom, left, right = padding
        assert top + bottom > 0
        assert img.shape[0] >= 128

    def test_pads_small_image_width(self):
        img, padding = pad_image_for_enhancement(_img(256, 64), min_size=128)
        top, bottom, left, right = padding
        assert left + right > 0
        assert img.shape[1] >= 128

    def test_pads_both_dimensions(self):
        img, padding = pad_image_for_enhancement(_img(32, 32), min_size=128)
        assert img.shape[0] >= 128
        assert img.shape[1] >= 128

    def test_padding_tuple_length_is_four(self):
        _, padding = pad_image_for_enhancement(_img(64, 64), min_size=128)
        assert len(padding) == 4

    def test_exact_min_size_no_padding(self):
        img, padding = pad_image_for_enhancement(_img(128, 128), min_size=128)
        assert padding == (0, 0, 0, 0)

    def test_output_dtype_preserved(self):
        img, _ = pad_image_for_enhancement(_img(64, 64), min_size=128)
        assert img.dtype == np.uint8

    def test_output_channels_preserved(self):
        img, _ = pad_image_for_enhancement(_img(64, 64), min_size=128)
        assert img.shape[2] == 3

    def test_returns_ndarray(self):
        img, _ = pad_image_for_enhancement(_img(32, 32))
        assert isinstance(img, np.ndarray)

    def test_custom_min_size(self):
        img, _ = pad_image_for_enhancement(_img(64, 64), min_size=256)
        assert img.shape[0] >= 256
        assert img.shape[1] >= 256


# ============================================================
# 6. unpad_image
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestUnpadImage:

    def test_zero_padding_returns_same_shape(self):
        img = _img(256, 256)
        out = unpad_image(img, (0, 0, 0, 0))
        assert out.shape == img.shape

    def test_removes_top_padding(self):
        img = _img(200, 200)
        out = unpad_image(img, (20, 0, 0, 0))
        assert out.shape[0] == 180

    def test_removes_bottom_padding(self):
        img = _img(200, 200)
        out = unpad_image(img, (0, 20, 0, 0))
        assert out.shape[0] == 180

    def test_removes_left_padding(self):
        img = _img(200, 200)
        out = unpad_image(img, (0, 0, 30, 0))
        assert out.shape[1] == 170

    def test_removes_right_padding(self):
        img = _img(200, 200)
        out = unpad_image(img, (0, 0, 0, 30))
        assert out.shape[1] == 170

    def test_removes_all_padding(self):
        img = _img(160, 160)
        out = unpad_image(img, (10, 10, 15, 15))
        assert out.shape == (140, 130, 3)

    def test_roundtrip_pad_unpad(self):
        original = _img(80, 60)
        padded, padding = pad_image_for_enhancement(original, min_size=128)
        restored = unpad_image(padded, padding)
        assert restored.shape == original.shape

    def test_returns_ndarray(self):
        out = unpad_image(_img(128, 128), (0, 0, 0, 0))
        assert isinstance(out, np.ndarray)


# ============================================================
# 7. find_center_face
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestFindCenterFace:

    def test_returns_none_for_empty_list(self):
        assert find_center_face([], 640, 480) is None

    def test_single_face_returned(self):
        fb = _face_box()
        assert find_center_face([fb], 640, 480) is fb

    def test_picks_face_closest_to_center(self):
        # Face A is centred; face B is far to the right
        fa = FaceBox(x1=280, y1=200, x2=360, y2=280,
                     confidence=0.9, face_index=0)
        fb = FaceBox(x1=580, y1=200, x2=640, y2=280,
                     confidence=0.9, face_index=1)
        result = find_center_face([fa, fb], 640, 480)
        assert result is fa

    def test_picks_center_face_with_three_candidates(self):
        # Left / centre / right
        left   = FaceBox(x1=0,   y1=200, x2=60,  y2=280, confidence=0.9, face_index=0)
        centre = FaceBox(x1=300, y1=200, x2=360, y2=280, confidence=0.9, face_index=1)
        right  = FaceBox(x1=580, y1=200, x2=640, y2=280, confidence=0.9, face_index=2)
        result = find_center_face([left, centre, right], 640, 480)
        assert result is centre

    def test_returns_facebox_type(self):
        fb = _face_box()
        assert isinstance(find_center_face([fb], 200, 200), FaceBox)

    def test_image_center_calculation_uses_passed_dimensions(self):
        # Place face at (0, 0)→(10, 10) — closer to top-left than to (1000, 1000)
        far_face  = FaceBox(x1=980, y1=980, x2=1000, y2=1000,
                            confidence=0.9, face_index=0)
        near_face = FaceBox(x1=0,   y1=0,   x2=10,   y2=10,
                            confidence=0.9, face_index=1)
        # Image 20×20: center = (10, 10); near_face center=(5,5), far=(990,990)
        result = find_center_face([far_face, near_face], 20, 20)
        assert result is near_face


# ============================================================
# 8. BaseEnhancer (via _StubEnhancer)
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestBaseEnhancer:

    def test_initial_is_loaded_false(self):
        enh = _StubEnhancer()
        assert enh.is_loaded is False

    def test_model_name_from_path(self):
        enh = _StubEnhancer(model_path="models/stub.pth")
        assert enh.model_name == "stub.pth"

    def test_load_model_sets_is_loaded(self):
        enh = _StubEnhancer()
        enh.load_model()
        assert enh.is_loaded is True

    def test_load_model_called_flag(self):
        enh = _StubEnhancer()
        enh.load_model()
        assert enh._load_called is True

    def test_release_clears_is_loaded(self):
        enh = _StubEnhancer()
        enh.load_model()
        enh.release()
        assert enh.is_loaded is False

    def test_release_clears_model(self):
        enh = _StubEnhancer()
        enh.load_model()
        enh.release()
        assert enh._model is None

    def test_avg_inference_ms_zero_before_calls(self):
        enh = _StubEnhancer()
        assert enh.avg_inference_ms == 0.0

    def test_total_calls_zero_initially(self):
        enh = _StubEnhancer()
        assert enh.total_calls == 0

    def test_reset_stats_clears_counters(self):
        enh = _StubEnhancer()
        enh._total_calls     = 5
        enh._total_inference = 200.0
        enh.reset_stats()
        assert enh.total_calls == 0
        assert enh.avg_inference_ms == 0.0

    def test_enhance_image_convenience_wrapper(self):
        enh = _StubEnhancer()
        enh.load_model()
        img    = _img()
        result = enh.enhance_image(img)
        assert result.success is True

    def test_context_manager_auto_loads(self):
        enh = _StubEnhancer()
        with enh:
            assert enh.is_loaded is True

    def test_context_manager_auto_releases(self):
        enh = _StubEnhancer()
        with enh:
            pass
        assert enh.is_loaded is False

    def test_require_loaded_raises_when_not_loaded(self):
        enh = _StubEnhancer()
        with pytest.raises(RuntimeError, match="not loaded"):
            enh._require_loaded()

    def test_require_loaded_ok_when_loaded(self):
        enh = _StubEnhancer()
        enh.load_model()
        enh._require_loaded()   # should not raise

    def test_validate_image_raises_on_none(self):
        enh = _StubEnhancer()
        with pytest.raises(ValueError):
            enh._validate_image(None)

    def test_validate_image_raises_on_wrong_type(self):
        enh = _StubEnhancer()
        with pytest.raises(ValueError):
            enh._validate_image("not an array")

    def test_validate_image_raises_on_wrong_ndim(self):
        enh = _StubEnhancer()
        with pytest.raises(ValueError):
            enh._validate_image(np.zeros((64, 64), dtype=np.uint8))

    def test_validate_image_raises_on_empty(self):
        enh = _StubEnhancer()
        with pytest.raises(ValueError):
            enh._validate_image(np.zeros((0, 64, 3), dtype=np.uint8))

    def test_validate_image_ok_for_valid(self):
        enh = _StubEnhancer()
        enh._validate_image(_img())   # should not raise

    def test_resolve_device_auto_returns_string(self):
        enh = _StubEnhancer()
        d   = enh._resolve_device()
        assert d in ("cpu", "cuda") or d.startswith("cuda:")

    def test_resolve_device_explicit_cpu(self):
        enh = _StubEnhancer()
        enh.device = "cpu"
        assert enh._resolve_device() == "cpu"

    def test_make_failed_result_has_correct_status(self):
        enh = _StubEnhancer()
        t0  = enh._timer()
        r   = enh._make_failed_result(
            EnhancementStatus.INFERENCE_ERROR,
            _img(),
            "test error",
            t0,
        )
        assert r.status == EnhancementStatus.INFERENCE_ERROR

    def test_make_failed_result_has_error_message(self):
        enh = _StubEnhancer()
        r   = enh._make_failed_result(
            EnhancementStatus.INFERENCE_ERROR,
            _img(),
            "my error",
            enh._timer(),
        )
        assert r.error == "my error"

    def test_make_failed_result_returns_original_image(self):
        enh = _StubEnhancer()
        img = _img()
        r   = enh._make_failed_result(
            EnhancementStatus.INFERENCE_ERROR,
            img,
            "err",
            enh._timer(),
        )
        assert r.output_image is img

    def test_timer_returns_float(self):
        enh = _StubEnhancer()
        assert isinstance(enh._timer(), float)

    def test_timer_increases(self):
        enh = _StubEnhancer()
        t0  = enh._timer()
        time.sleep(0.005)
        assert enh._timer() > t0

    def test_repr_contains_backend(self):
        enh = _StubEnhancer()
        assert "gfpgan" in repr(enh).lower()

    def test_repr_contains_loaded_status(self):
        enh = _StubEnhancer()
        assert "not loaded" in repr(enh)
        enh.load_model()
        assert "loaded" in repr(enh)

    def test_backend_stored(self):
        enh = _StubEnhancer(backend=EnhancerBackend.CODEFORMER)
        assert enh.backend == EnhancerBackend.CODEFORMER

    def test_upscale_stored(self):
        enh = _StubEnhancer()
        assert enh.upscale == 2

    def test_device_stored(self):
        enh = _StubEnhancer()
        assert enh.device == "cpu"

    def test_enhance_before_load_returns_not_loaded_status(self):
        enh = _StubEnhancer()
        r   = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.MODEL_NOT_LOADED


# ============================================================
# 9. GFPGANEnhancer — construction
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestGFPGANEnhancerConstruction:

    def _make(self, **kwargs):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        defaults = dict(model_path="models/GFPGANv1.4.pth")
        defaults.update(kwargs)
        return GFPGANEnhancer(**defaults)

    def test_default_model_path(self):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        enh = GFPGANEnhancer()
        assert "GFPGANv1.4" in enh.model_path

    def test_backend_is_gfpgan(self):
        enh = self._make()
        assert enh.backend == EnhancerBackend.GFPGAN

    def test_arch_default(self):
        enh = self._make()
        assert enh.arch == "clean"

    def test_channel_multiplier_default(self):
        enh = self._make()
        assert enh.channel_multiplier == 2

    def test_upscale_default(self):
        enh = self._make()
        assert enh.upscale == 2

    def test_device_stored(self):
        enh = self._make(device="cpu")
        assert enh.device == "cpu"

    def test_not_loaded_initially(self):
        enh = self._make()
        assert enh.is_loaded is False

    def test_restorer_none_initially(self):
        enh = self._make()
        assert enh._restorer is None

    def test_custom_arch(self):
        enh = self._make(arch="RestoreFormer")
        assert enh.arch == "RestoreFormer"

    def test_bg_upsampler_name_stored(self):
        enh = self._make(bg_upsampler="realesrgan")
        assert enh.bg_upsampler_name == "realesrgan"

    def test_repr_contains_arch(self):
        enh = self._make()
        assert "clean" in repr(enh)

    def test_repr_contains_upscale(self):
        enh = self._make(upscale=4)
        assert "4" in repr(enh)

    def test_repr_not_loaded(self):
        enh = self._make()
        assert "not loaded" in repr(enh)


# ============================================================
# 10. GFPGANEnhancer — load_model error paths
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestGFPGANEnhancerLoadModel:

    def _make(self, **kwargs):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        defaults = dict(model_path="models/GFPGANv1.4.pth")
        defaults.update(kwargs)
        return GFPGANEnhancer(**defaults)

    def test_raises_file_not_found_when_model_missing(self):
        enh = self._make(model_path="nonexistent/GFPGANv1.4.pth")
        with pytest.raises(FileNotFoundError):
            enh.load_model()

    def test_raises_import_error_when_gfpgan_not_installed(self, tmp_path):
        model = tmp_path / "GFPGANv1.4.pth"
        model.write_bytes(b"dummy")
        enh = self._make(model_path=str(model))
        with patch.dict("sys.modules", {"gfpgan": None}):
            with pytest.raises((ImportError, Exception)):
                enh.load_model()

    def test_is_loaded_false_after_failed_load(self):
        enh = self._make(model_path="nonexistent.pth")
        try:
            enh.load_model()
        except Exception:
            pass
        assert enh.is_loaded is False


# ============================================================
# 11. GFPGANEnhancer — enhance() with mocked GFPGANer
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestGFPGANEnhancerEnhance:

    def _make_loaded(self, **kwargs):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        enh = GFPGANEnhancer(model_path="models/GFPGANv1.4.pth", **kwargs)
        # Manually inject a mock restorer so we skip real load_model
        enh._restorer  = MagicMock()
        enh._is_loaded = True
        return enh

    def _setup_restorer(self, enh, output_img=None, restored_faces=None):
        if output_img is None:
            output_img = _img(256, 256)
        if restored_faces is None:
            restored_faces = [_img(128, 128)]
        enh._restorer.enhance.return_value = (
            [_img(128, 128)],  # cropped_faces
            restored_faces,    # restored_faces
            output_img,        # output_img
        )

    def test_returns_enhancement_result(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        r = enh.enhance(_make_request())
        assert isinstance(r, EnhancementResult)

    def test_success_status_on_happy_path(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        r = enh.enhance(_make_request())
        assert r.success is True

    def test_backend_is_gfpgan(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        r = enh.enhance(_make_request())
        assert r.backend == EnhancerBackend.GFPGAN

    def test_output_image_is_ndarray(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        r = enh.enhance(_make_request())
        assert isinstance(r.output_image, np.ndarray)

    def test_num_faces_enhanced_matches_restored(self):
        enh = self._make_loaded()
        self._setup_restorer(enh, restored_faces=[_img(), _img()])
        r = enh.enhance(_make_request())
        assert r.num_faces_enhanced == 2

    def test_enhance_time_ms_positive(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        r = enh.enhance(_make_request())
        assert r.enhance_time_ms >= 0.0

    def test_not_loaded_returns_model_not_loaded(self):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        enh = GFPGANEnhancer(model_path="models/GFPGANv1.4.pth")
        r   = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.MODEL_NOT_LOADED

    def test_invalid_image_returns_invalid_input(self):
        enh = self._make_loaded()
        req = EnhancementRequest(image=np.zeros((64, 64), dtype=np.uint8))  # 2-D, not BGR
        r   = enh.enhance(req)
        assert r.status == EnhancementStatus.INVALID_INPUT

    def test_no_faces_returns_no_face_detected(self):
        enh = self._make_loaded()
        # restored_faces is empty list
        enh._restorer.enhance.return_value = ([], [], _img())
        r = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.NO_FACE_DETECTED

    def test_inference_exception_returns_inference_error(self):
        enh = self._make_loaded()
        enh._restorer.enhance.side_effect = RuntimeError("GFPGAN crashed")
        r = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.INFERENCE_ERROR

    def test_inference_exception_error_message_stored(self):
        enh = self._make_loaded()
        enh._restorer.enhance.side_effect = RuntimeError("specific error msg")
        r = enh.enhance(_make_request())
        assert "specific error msg" in r.error

    def test_inference_exception_returns_original_image(self):
        enh = self._make_loaded()
        enh._restorer.enhance.side_effect = RuntimeError("crash")
        img = _img()
        r   = enh.enhance(EnhancementRequest(image=img))
        assert r.output_image is img

    def test_total_calls_incremented(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        assert enh.total_calls == 0
        enh.enhance(_make_request())
        assert enh.total_calls == 1

    def test_total_calls_accumulates(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        for _ in range(3):
            enh.enhance(_make_request())
        assert enh.total_calls == 3

    def test_avg_inference_ms_positive_after_call(self):
        enh = self._make_loaded()
        self._setup_restorer(enh)
        enh.enhance(_make_request())
        assert enh.avg_inference_ms >= 0.0

    def test_upscale_factor_in_result(self):
        enh = self._make_loaded(upscale=2)
        self._setup_restorer(enh)
        r = enh.enhance(_make_request())
        assert r.upscale_factor == 2

    def test_none_output_image_falls_back(self):
        enh = self._make_loaded()
        # GFPGANer returns None as output_img (e.g. paste_back=False path)
        enh._restorer.enhance.return_value = (
            [_img()],  # cropped_faces
            [_img()],  # restored_faces
            None,      # output_img = None
        )
        r = enh.enhance(_make_request())
        # Should not crash; output_image should be an ndarray
        assert isinstance(r.output_image, np.ndarray)


# ============================================================
# 12. GFPGANEnhancer — release and repr
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestGFPGANEnhancerReleaseRepr:

    def _make_loaded(self):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        enh = GFPGANEnhancer(model_path="models/GFPGANv1.4.pth")
        enh._restorer  = MagicMock()
        enh._is_loaded = True
        return enh

    def test_release_clears_restorer(self):
        enh = self._make_loaded()
        enh.release()
        assert enh._restorer is None

    def test_release_sets_is_loaded_false(self):
        enh = self._make_loaded()
        enh.release()
        assert enh.is_loaded is False

    def test_repr_loaded(self):
        enh = self._make_loaded()
        assert "loaded" in repr(enh)

    def test_repr_not_loaded(self):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        enh = GFPGANEnhancer()
        assert "not loaded" in repr(enh)

    def test_repr_contains_upscale(self):
        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        enh = GFPGANEnhancer(upscale=4)
        assert "4" in repr(enh)

    def test_repr_contains_calls(self):
        enh = self._make_loaded()
        assert "calls=0" in repr(enh)


# ============================================================
# 13. CodeFormerEnhancer — construction
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestCodeFormerEnhancerConstruction:

    def _make(self, **kwargs):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        defaults = dict(model_path="models/codeformer.pth")
        defaults.update(kwargs)
        return CodeFormerEnhancer(**defaults)

    def test_default_model_path(self):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        enh = CodeFormerEnhancer()
        assert "codeformer" in enh.model_path.lower()

    def test_backend_is_codeformer(self):
        enh = self._make()
        assert enh.backend == EnhancerBackend.CODEFORMER

    def test_default_fidelity_weight(self):
        enh = self._make()
        assert enh.fidelity_weight == 0.5

    def test_custom_fidelity_weight(self):
        enh = self._make(fidelity_weight=0.8)
        assert enh.fidelity_weight == 0.8

    def test_default_upscale(self):
        enh = self._make()
        assert enh.upscale == 2

    def test_not_loaded_initially(self):
        enh = self._make()
        assert enh.is_loaded is False

    def test_codeformer_net_none_initially(self):
        enh = self._make()
        assert enh._codeformer_net is None

    def test_face_helper_none_initially(self):
        enh = self._make()
        assert enh._face_helper is None

    def test_bg_enhance_default_false(self):
        enh = self._make()
        assert enh.bg_enhance is False

    def test_custom_bg_enhance(self):
        enh = self._make(bg_enhance=True)
        assert enh.bg_enhance is True

    def test_device_stored(self):
        enh = self._make(device="cpu")
        assert enh.device == "cpu"

    def test_repr_contains_backend(self):
        enh = self._make()
        assert "codeformer" in repr(enh).lower()

    def test_repr_not_loaded(self):
        enh = self._make()
        assert "not loaded" in repr(enh)


# ============================================================
# 14. CodeFormerEnhancer — load_model error paths
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestCodeFormerEnhancerLoadModel:

    def _make(self, **kwargs):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        defaults = dict(model_path="models/codeformer.pth")
        defaults.update(kwargs)
        return CodeFormerEnhancer(**defaults)

    def test_raises_file_not_found_when_model_missing(self):
        enh = self._make(model_path="nonexistent/codeformer.pth")
        with pytest.raises(FileNotFoundError):
            enh.load_model()

    def test_is_loaded_false_after_failed_load(self):
        enh = self._make(model_path="nonexistent.pth")
        try:
            enh.load_model()
        except Exception:
            pass
        assert enh.is_loaded is False


# ============================================================
# 15. CodeFormerEnhancer — enhance() with mocked internals
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestCodeFormerEnhancerEnhance:

    def _make_loaded(self, **kwargs):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        enh = CodeFormerEnhancer(model_path="models/codeformer.pth", **kwargs)
        enh._is_loaded = True

        # Mock the CodeFormer net
        net_mock = MagicMock()
        restored_tensor = MagicMock()
        net_mock.return_value = [restored_tensor]
        enh._codeformer_net = net_mock

        # Mock the face helper
        helper_mock = MagicMock()
        cropped = _img(512, 512)
        helper_mock.cropped_faces = [cropped]
        helper_mock.paste_faces_to_input_image.return_value = _img(256, 256)
        enh._face_helper = helper_mock

        return enh

    def _setup_torch_mocks(self):
        """Return a context manager that patches torch for CodeFormer inference."""
        import sys
        torch_mock  = MagicMock()
        tensor_mock = MagicMock()
        tensor_mock.__getitem__ = MagicMock(return_value=MagicMock())
        torch_mock.no_grad.return_value.__enter__ = MagicMock(return_value=None)
        torch_mock.no_grad.return_value.__exit__  = MagicMock(return_value=False)
        torch_mock.device.return_value = "cpu"
        return patch.dict("sys.modules", {"torch": torch_mock})

    def test_not_loaded_returns_model_not_loaded(self):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        enh = CodeFormerEnhancer(model_path="models/codeformer.pth")
        r   = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.MODEL_NOT_LOADED

    def test_invalid_image_returns_invalid_input(self):
        enh = self._make_loaded()
        req = EnhancementRequest(image=np.zeros((64, 64), dtype=np.uint8))
        r   = enh.enhance(req)
        assert r.status == EnhancementStatus.INVALID_INPUT

    def test_no_cropped_faces_returns_no_face_detected(self):
        enh = self._make_loaded()
        enh._face_helper.cropped_faces = []
        with self._setup_torch_mocks():
            r = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.NO_FACE_DETECTED

    def test_inference_exception_returns_inference_error(self):
        enh = self._make_loaded()
        # Make the face_helper.align_warp_face raise
        enh._face_helper.align_warp_face.side_effect = RuntimeError("align failed")
        with self._setup_torch_mocks():
            r = enh.enhance(_make_request())
        assert r.status == EnhancementStatus.INFERENCE_ERROR

    def test_inference_exception_error_message_stored(self):
        enh = self._make_loaded()
        enh._face_helper.align_warp_face.side_effect = RuntimeError("specific msg")
        with self._setup_torch_mocks():
            r = enh.enhance(_make_request())
        assert r.error is not None
        assert "specific msg" in r.error

    def test_inference_exception_returns_original_image(self):
        enh = self._make_loaded()
        enh._face_helper.align_warp_face.side_effect = RuntimeError("crash")
        img = _img()
        with self._setup_torch_mocks():
            r = enh.enhance(EnhancementRequest(image=img))
        assert r.output_image is img

    def test_returns_enhancement_result_type(self):
        enh = self._make_loaded()
        enh._face_helper.cropped_faces = []
        with self._setup_torch_mocks():
            r = enh.enhance(_make_request())
        assert isinstance(r, EnhancementResult)

    def test_backend_is_codeformer(self):
        enh = self._make_loaded()
        enh._face_helper.cropped_faces = []
        with self._setup_torch_mocks():
            r = enh.enhance(_make_request())
        assert r.backend == EnhancerBackend.CODEFORMER


# ============================================================
# 16. CodeFormerEnhancer — release and repr
# ============================================================

@pytest.mark.unit
@pytest.mark.enhancer
class TestCodeFormerEnhancerReleaseRepr:

    def _make_loaded(self):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        enh = CodeFormerEnhancer(model_path="models/codeformer.pth")
        enh._codeformer_net = MagicMock()
        enh._face_helper    = MagicMock()
        enh._is_loaded      = True
        return enh

    def test_release_clears_net(self):
        enh = self._make_loaded()
        enh.release()
        assert enh._codeformer_net is None

    def test_release_clears_face_helper(self):
        enh = self._make_loaded()
        enh.release()
        assert enh._face_helper is None

    def test_release_sets_is_loaded_false(self):
        enh = self._make_loaded()
        enh.release()
        assert enh.is_loaded is False

    def test_repr_loaded(self):
        enh = self._make_loaded()
        assert "loaded" in repr(enh)

    def test_repr_not_loaded(self):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        enh = CodeFormerEnhancer()
        assert "not loaded" in repr(enh)

    def test_repr_contains_fidelity(self):
        from core.enhancer.codeformer_enhancer import CodeFormerEnhancer
        enh = CodeFormerEnhancer(fidelity_weight=0.7)
        assert "0.7" in repr(enh)
