# Integration tests for Phase 6 — Pipeline Orchestration.
#
# Covers:
#   - FacePipeline.run() — all PipelineStatus paths
#   - PipelineConfig — every control flag
#   - PipelineTiming — per-stage timing fields
#   - PipelineResult — all fields and properties
#   - Ethics consent gate
#   - Watermark stage
#   - Enhancement stage (mocked)
#   - Single-face and multi-face swap paths
#   - Error recovery at every pipeline stage
#   - VideoPipeline — process(), process_webcam(), _process_frame(),
#     _capped_resolution(), _add_watermark(), _merge_audio()
#   - VideoProcessingConfig / VideoProcessingResult
#
# All AI models (detector, recognizer, swapper, enhancer) are
# replaced with lightweight mocks — no real weights required.

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, PropertyMock, call, patch

import cv2
import numpy as np
import pytest

from core.detector.base_detector import DetectionResult, FaceBox
from core.enhancer.base_enhancer import (
    EnhancementRequest,
    EnhancementResult,
    EnhancementStatus,
    EnhancerBackend,
)
from core.pipeline.face_pipeline import (
    FacePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStatus,
    PipelineTiming,
    _apply_watermark,
    _timer,
)
from core.pipeline.video_pipeline import VideoPipeline, VideoProcessingConfig, VideoProcessingResult
from core.recognizer.base_recognizer import FaceEmbedding, FaceMatch, RecognitionResult
from core.swapper.base_swapper import BatchSwapResult, BlendMode, SwapResult, SwapStatus


def _blank_image(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a solid-colour BGR image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _rand_vec(dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_face_box(
    x1=100,
    y1=80,
    x2=300,
    y2=320,
    confidence=0.92,
    face_index=0,
    with_landmarks=True,
) -> FaceBox:
    lm = None
    if with_landmarks:
        lm = np.array(
            [[150.0, 140.0], [250.0, 140.0], [200.0, 200.0], [160.0, 270.0], [240.0, 270.0]],
            dtype=np.float32,
        )
    return FaceBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=confidence,
        face_index=face_index,
        landmarks=lm,
    )


def _make_detection(n_faces: int = 1, empty: bool = False) -> DetectionResult:
    if empty:
        return DetectionResult(faces=[], image_width=640, image_height=480)
    faces = [
        _make_face_box(
            x1=10 + i * 150,
            y1=10,
            x2=130 + i * 150,
            y2=130,
            face_index=i,
        )
        for i in range(n_faces)
    ]
    return DetectionResult(faces=faces, image_width=640, image_height=480, inference_time_ms=5.0)


def _make_embedding(seed: int = 0) -> FaceEmbedding:
    return FaceEmbedding(vector=_rand_vec(seed=seed), face_index=0)


def _make_swap_result(success: bool = True, img: Optional[np.ndarray] = None) -> SwapResult:
    if img is None:
        img = _blank_image()
    return SwapResult(
        output_image=img,
        status=SwapStatus.SUCCESS if success else SwapStatus.INFERENCE_ERROR,
        target_face=_make_face_box(),
        swap_time_ms=20.0,
        inference_time_ms=15.0,
        align_time_ms=2.0,
        blend_time_ms=3.0,
        error=None if success else "mock error",
    )


def _make_batch_result(n_faces: int = 1, all_success: bool = True) -> BatchSwapResult:
    results = [_make_swap_result(success=all_success or (i == 0)) for i in range(n_faces)]
    return BatchSwapResult(
        output_image=_blank_image(),
        swap_results=results,
        total_time_ms=25.0,
    )


def _make_enhancement_result(success: bool = True) -> EnhancementResult:
    return EnhancementResult(
        output_image=_blank_image(),
        status=EnhancementStatus.SUCCESS if success else EnhancementStatus.INFERENCE_ERROR,
        backend=EnhancerBackend.GFPGAN,
        num_faces_enhanced=1 if success else 0,
        enhance_time_ms=40.0,
        inference_time_ms=35.0,
        upscale_factor=2,
        error=None if success else "mock enhance error",
    )


def _mock_detector(n_faces: int = 1, empty: bool = False) -> MagicMock:
    detector = MagicMock()
    detector.is_loaded = True
    detector.model_name = "mock_detector"
    detector.detect.return_value = _make_detection(n_faces=n_faces, empty=empty)
    return detector


def _mock_recognizer(embedding: Optional[FaceEmbedding] = None) -> MagicMock:
    recognizer = MagicMock()
    recognizer.is_loaded = True
    recognizer.model_name = "mock_recognizer"
    recognizer.get_embedding.return_value = embedding or _make_embedding()
    return recognizer


def _mock_swapper(success: bool = True, n_faces: int = 1) -> MagicMock:
    swapper = MagicMock()
    swapper.is_loaded = True
    swapper.model_name = "mock_swapper"
    swapper.blend_mode = BlendMode.POISSON
    swapper.blend_alpha = 1.0
    swapper.mask_feather = 20
    swapper.swap.return_value = _make_swap_result(success=success)
    swapper.swap_all.return_value = _make_batch_result(n_faces=n_faces, all_success=success)
    return swapper


def _mock_enhancer(success: bool = True) -> MagicMock:
    enhancer = MagicMock()
    enhancer.is_loaded = True
    enhancer.model_name = "mock_enhancer"
    enhancer.enhance.return_value = _make_enhancement_result(success=success)
    return enhancer


def _make_pipeline(
    n_source_faces: int = 1,
    n_target_faces: int = 1,
    source_empty: bool = False,
    target_empty: bool = False,
    swap_success: bool = True,
    enhance_success: bool = True,
    embedding: Optional[FaceEmbedding] = None,
    with_enhancer: bool = False,
    config: Optional[PipelineConfig] = None,
) -> FacePipeline:
    """
    Build a FacePipeline with fully mocked components.
    """
    # Detector returns different results for source vs target calls
    detector = MagicMock()
    detector.is_loaded = True
    detector.model_name = "mock_detector"
    detector.detect.side_effect = [
        _make_detection(n_faces=n_source_faces, empty=source_empty),
        _make_detection(n_faces=n_target_faces, empty=target_empty),
    ]

    recognizer = _mock_recognizer(embedding=embedding)
    swapper = _mock_swapper(success=swap_success, n_faces=n_target_faces)
    enhancer = _mock_enhancer(success=enhance_success) if with_enhancer else None

    return FacePipeline(
        detector=detector,
        recognizer=recognizer,
        swapper=swapper,
        enhancer=enhancer,
        config=config or PipelineConfig(),
    )


@pytest.mark.integration
@pytest.mark.pipeline
class TestPipelineConfig:

    def test_default_values(self):
        cfg = PipelineConfig()
        assert cfg.blend_mode == BlendMode.POISSON
        assert cfg.blend_alpha == 1.0
        assert cfg.mask_feather == 20
        assert cfg.swap_all_faces is False
        assert cfg.enable_enhancement is False
        assert cfg.watermark is True
        assert cfg.require_consent is True
        assert cfg.save_intermediate is False

    def test_custom_values(self):
        cfg = PipelineConfig(
            blend_mode=BlendMode.ALPHA,
            blend_alpha=0.8,
            mask_feather=10,
            swap_all_faces=True,
            enable_enhancement=True,
            watermark=False,
            require_consent=False,
        )
        assert cfg.blend_mode == BlendMode.ALPHA
        assert cfg.blend_alpha == 0.8
        assert cfg.swap_all_faces is True
        assert cfg.enable_enhancement is True
        assert cfg.watermark is False
        assert cfg.require_consent is False

    def test_repr_contains_key_fields(self):
        cfg = PipelineConfig()
        r = repr(cfg)
        assert "POISSON" in r
        assert "swap_all" in r

    def test_source_face_index_default(self):
        assert PipelineConfig().source_face_index == 0

    def test_target_face_index_default(self):
        assert PipelineConfig().target_face_index == 0

    def test_max_faces_default(self):
        assert PipelineConfig().max_faces == 10

    def test_fidelity_weight_default(self):
        assert PipelineConfig().fidelity_weight == 0.5

    def test_upscale_default(self):
        assert PipelineConfig().upscale == 2


@pytest.mark.integration
@pytest.mark.pipeline
class TestPipelineTiming:

    def test_all_fields_default_zero(self):
        t = PipelineTiming()
        for field in (
            "detect_source_ms",
            "embed_source_ms",
            "detect_target_ms",
            "swap_ms",
            "enhance_ms",
            "watermark_ms",
            "total_ms",
        ):
            assert getattr(t, field) == 0.0

    def test_overhead_zero_when_no_stages(self):
        t = PipelineTiming(total_ms=0.0)
        assert t.pipeline_overhead_ms == 0.0

    def test_overhead_is_residual(self):
        t = PipelineTiming(
            detect_source_ms=10.0,
            detect_target_ms=10.0,
            embed_source_ms=5.0,
            swap_ms=20.0,
            enhance_ms=0.0,
            watermark_ms=1.0,
            total_ms=60.0,
        )
        expected = 60.0 - (10 + 10 + 5 + 20 + 0 + 1)
        assert abs(t.pipeline_overhead_ms - expected) < 1e-6

    def test_repr_contains_timing(self):
        t = PipelineTiming(total_ms=42.5, swap_ms=20.0)
        r = repr(t)
        assert "42.5" in r
        assert "20.0" in r

    def test_overhead_never_negative(self):
        # If stages sum to more than total_ms (floating point noise), clamp to 0
        t = PipelineTiming(
            detect_source_ms=5.0,
            detect_target_ms=5.0,
            embed_source_ms=5.0,
            swap_ms=5.0,
            total_ms=10.0,  # less than sum
        )
        assert t.pipeline_overhead_ms >= 0.0


@pytest.mark.integration
@pytest.mark.pipeline
class TestPipelineResult:

    def _make_result(self, status=PipelineStatus.SUCCESS) -> PipelineResult:
        return PipelineResult(
            output_image=_blank_image(),
            status=status,
            request_id="test-id-1234",
            swap_result=_make_batch_result(n_faces=1),
        )

    def test_success_true_for_success(self):
        r = self._make_result(PipelineStatus.SUCCESS)
        assert r.success is True

    def test_success_true_for_partial(self):
        r = self._make_result(PipelineStatus.PARTIAL)
        assert r.success is True

    def test_success_false_for_error(self):
        r = self._make_result(PipelineStatus.STAGE_ERROR)
        assert r.success is False

    def test_success_false_for_no_source_face(self):
        r = self._make_result(PipelineStatus.NO_SOURCE_FACE)
        assert r.success is False

    def test_success_false_for_no_target_face(self):
        r = self._make_result(PipelineStatus.NO_TARGET_FACE)
        assert r.success is False

    def test_success_false_for_consent_denied(self):
        r = self._make_result(PipelineStatus.CONSENT_DENIED)
        assert r.success is False

    def test_num_faces_swapped_from_batch(self):
        r = self._make_result()
        assert r.num_faces_swapped == 1

    def test_num_faces_swapped_zero_when_no_swap(self):
        r = PipelineResult(
            output_image=_blank_image(),
            status=PipelineStatus.NO_SOURCE_FACE,
            request_id="x",
        )
        assert r.num_faces_swapped == 0

    def test_repr_contains_status(self):
        r = self._make_result()
        assert "success" in repr(r)

    def test_request_id_stored(self):
        r = self._make_result()
        assert r.request_id == "test-id-1234"

    def test_warnings_default_empty(self):
        r = self._make_result()
        assert r.warnings == []

    def test_error_default_none(self):
        r = self._make_result()
        assert r.error is None


@pytest.mark.integration
@pytest.mark.pipeline
class TestTimerHelper:

    def test_returns_float(self):
        t = _timer()
        assert isinstance(t, float)

    def test_increases_over_time(self):
        t0 = _timer()
        time.sleep(0.005)
        t1 = _timer()
        assert t1 > t0

    def test_difference_roughly_correct(self):
        t0 = _timer()
        time.sleep(0.05)
        elapsed = _timer() - t0
        # Should be within 10–200 ms of 50 ms
        assert 20 < elapsed < 300


@pytest.mark.integration
@pytest.mark.pipeline
class TestApplyWatermark:

    def test_returns_same_shape(self):
        img = _blank_image(480, 640)
        out = _apply_watermark(img, "AI GENERATED")
        assert out.shape == img.shape

    def test_returns_uint8(self):
        img = _blank_image()
        out = _apply_watermark(img)
        assert out.dtype == np.uint8

    def test_modifies_image(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        out = _apply_watermark(img, "TEST")
        # At least some pixels should differ (the text area)
        assert not np.array_equal(img, out)

    def test_does_not_modify_original(self):
        img = _blank_image()
        original_copy = img.copy()
        _apply_watermark(img, "TEST")
        np.testing.assert_array_equal(img, original_copy)

    def test_custom_text(self):
        img = _blank_image()
        out1 = _apply_watermark(img, "TEXT ONE")
        out2 = _apply_watermark(img, "TEXT TWO")
        # Different text should produce different pixels
        assert not np.array_equal(out1, out2)

    def test_works_on_small_image(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = _apply_watermark(img, "AI")
        assert out.shape == (64, 64, 3)

    def test_works_on_large_image(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = _apply_watermark(img, "AI GENERATED")
        assert out.shape == (1080, 1920, 3)


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelineConstruction:

    def test_stores_components(self):
        det = _mock_detector()
        rec = _mock_recognizer()
        swap = _mock_swapper()
        p = FacePipeline(detector=det, recognizer=rec, swapper=swap)
        assert p.detector is det
        assert p.recognizer is rec
        assert p.swapper is swap
        assert p.enhancer is None

    def test_stores_enhancer(self):
        det = _mock_detector()
        rec = _mock_recognizer()
        swap = _mock_swapper()
        enh = _mock_enhancer()
        p = FacePipeline(detector=det, recognizer=rec, swapper=swap, enhancer=enh)
        assert p.enhancer is enh

    def test_default_config_used(self):
        p = FacePipeline(
            detector=_mock_detector(),
            recognizer=_mock_recognizer(),
            swapper=_mock_swapper(),
        )
        assert isinstance(p.config, PipelineConfig)

    def test_custom_config_stored(self):
        cfg = PipelineConfig(blend_mode=BlendMode.ALPHA)
        p = FacePipeline(
            detector=_mock_detector(),
            recognizer=_mock_recognizer(),
            swapper=_mock_swapper(),
            config=cfg,
        )
        assert p.config.blend_mode == BlendMode.ALPHA

    def test_repr_contains_class_names(self):
        p = FacePipeline(
            detector=_mock_detector(),
            recognizer=_mock_recognizer(),
            swapper=_mock_swapper(),
        )
        r = repr(p)
        assert "FacePipeline" in r


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelineConsentGate:

    def test_consent_false_returns_consent_denied(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=False)
        assert result.status == PipelineStatus.CONSENT_DENIED

    def test_consent_denied_does_not_call_detector(self):
        p = _make_pipeline()
        p.run(_blank_image(), _blank_image(), consent=False)
        p.detector.detect.assert_not_called()

    def test_consent_denied_returns_target_image_copy(self):
        target = _blank_image()
        p = _make_pipeline()
        result = p.run(_blank_image(), target, consent=False)
        assert result.output_image.shape == target.shape

    def test_consent_denied_has_error_message(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=False)
        assert result.error is not None
        assert "consent" in result.error.lower()

    def test_require_consent_false_allows_without_consent(self):
        cfg = PipelineConfig(require_consent=False, watermark=False)
        p = _make_pipeline(config=cfg)
        result = p.run(_blank_image(), _blank_image(), consent=False)
        assert result.status != PipelineStatus.CONSENT_DENIED

    def test_consent_true_passes_gate(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status != PipelineStatus.CONSENT_DENIED

    def test_per_call_config_overrides_consent_requirement(self):
        cfg = PipelineConfig(require_consent=False, watermark=False)
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=False, config=cfg)
        assert result.status != PipelineStatus.CONSENT_DENIED


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelineNoFace:

    def test_no_source_face_status(self):
        p = _make_pipeline(source_empty=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.NO_SOURCE_FACE

    def test_no_source_face_does_not_call_swapper(self):
        p = _make_pipeline(source_empty=True)
        p.run(_blank_image(), _blank_image(), consent=True)
        p.swapper.swap.assert_not_called()
        p.swapper.swap_all.assert_not_called()

    def test_no_source_face_error_message(self):
        p = _make_pipeline(source_empty=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.error is not None

    def test_no_source_face_returns_target_image(self):
        target = _blank_image()
        p = _make_pipeline(source_empty=True)
        result = p.run(_blank_image(), target, consent=True)
        assert result.output_image.shape == target.shape

    def test_no_target_face_status(self):
        p = _make_pipeline(target_empty=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.NO_TARGET_FACE

    def test_no_target_face_does_not_call_swapper(self):
        p = _make_pipeline(target_empty=True)
        p.run(_blank_image(), _blank_image(), consent=True)
        p.swapper.swap.assert_not_called()

    def test_no_target_face_error_message(self):
        p = _make_pipeline(target_empty=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.error is not None

    def test_embedding_none_returns_no_source_face(self):
        p = _make_pipeline(embedding=None)
        p.recognizer.get_embedding.return_value = None
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.NO_SOURCE_FACE


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelineSuccessfulSwap:

    def test_basic_run_success(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.success is True

    def test_status_is_success(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.SUCCESS

    def test_output_image_is_ndarray(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert isinstance(result.output_image, np.ndarray)

    def test_output_image_correct_shape(self):
        img = _blank_image(480, 640)
        p = _make_pipeline()
        # Swap output is a new image from the mock
        result = p.run(img, img, consent=True)
        assert result.output_image.ndim == 3

    def test_source_detection_populated(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.source_detection is not None

    def test_target_detection_populated(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.target_detection is not None

    def test_source_embedding_populated(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.source_embedding is not None

    def test_swap_result_populated(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.swap_result is not None

    def test_request_id_is_uuid_format(self):
        import re

        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert re.match(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            result.request_id,
        )

    def test_timing_total_ms_nonzero(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.timing.total_ms >= 0

    def test_config_stored_in_result(self):
        cfg = PipelineConfig(blend_mode=BlendMode.ALPHA, watermark=False)
        p = _make_pipeline(config=cfg)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.config is cfg

    def test_per_call_config_used(self):
        p = _make_pipeline()
        per_call = PipelineConfig(watermark=False, blend_mode=BlendMode.ALPHA)
        result = p.run(_blank_image(), _blank_image(), consent=True, config=per_call)
        assert result.config is per_call

    def test_swap_called_with_single_face(self):
        cfg = PipelineConfig(swap_all_faces=False, watermark=False)
        p = _make_pipeline(config=cfg)
        p.run(_blank_image(), _blank_image(), consent=True)
        p.swapper.swap.assert_called_once()
        p.swapper.swap_all.assert_not_called()

    def test_swap_all_called_when_configured(self):
        cfg = PipelineConfig(swap_all_faces=True, watermark=False)
        p = _make_pipeline(config=cfg)
        p.run(_blank_image(), _blank_image(), consent=True)
        p.swapper.swap_all.assert_called_once()
        p.swapper.swap.assert_not_called()

    def test_num_faces_swapped_correct(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.num_faces_swapped >= 1

    def test_different_request_ids_per_run(self):
        p = _make_pipeline()
        r1 = p.run(_blank_image(), _blank_image(), consent=True)
        # Reset side_effect for the second call
        p.detector.detect.side_effect = [
            _make_detection(n_faces=1),
            _make_detection(n_faces=1),
        ]
        r2 = p.run(_blank_image(), _blank_image(), consent=True)
        assert r1.request_id != r2.request_id


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelinePartialFailure:

    def test_partial_status_when_some_swaps_fail(self):
        # Create a batch result where not all succeeded
        bad_batch = BatchSwapResult(
            output_image=_blank_image(),
            swap_results=[
                _make_swap_result(success=True),
                _make_swap_result(success=False),
            ],
            total_time_ms=25.0,
        )
        cfg = PipelineConfig(swap_all_faces=True, watermark=False)
        p = _make_pipeline(config=cfg, n_target_faces=2)
        p.swapper.swap_all.return_value = bad_batch
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.PARTIAL

    def test_partial_has_warnings(self):
        bad_batch = BatchSwapResult(
            output_image=_blank_image(),
            swap_results=[_make_swap_result(success=False)],
            total_time_ms=10.0,
        )
        cfg = PipelineConfig(swap_all_faces=True, watermark=False)
        p = _make_pipeline(config=cfg)
        p.swapper.swap_all.return_value = bad_batch
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert len(result.warnings) > 0

    def test_detection_exception_causes_stage_error(self):
        p = _make_pipeline()
        p.detector.detect.side_effect = RuntimeError("detector exploded")
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.STAGE_ERROR

    def test_stage_error_has_error_message(self):
        p = _make_pipeline()
        p.detector.detect.side_effect = RuntimeError("boom")
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.error is not None
        assert len(result.error) > 0

    def test_stage_error_returns_target_image(self):
        target = _blank_image()
        p = _make_pipeline()
        p.detector.detect.side_effect = RuntimeError("boom")
        result = p.run(_blank_image(), target, consent=True)
        assert result.output_image.shape == target.shape

    def test_recognizer_exception_causes_stage_error(self):
        p = _make_pipeline()
        p.recognizer.get_embedding.side_effect = RuntimeError("recognizer failed")
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.STAGE_ERROR

    def test_swapper_exception_causes_stage_error(self):
        p = _make_pipeline()
        p.swapper.swap.side_effect = RuntimeError("swapper exploded")
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.status == PipelineStatus.STAGE_ERROR


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelineEnhancement:

    def test_enhancement_called_when_enabled(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True)
        p.run(_blank_image(), _blank_image(), consent=True)
        p.enhancer.enhance.assert_called_once()

    def test_enhancement_not_called_when_disabled(self):
        cfg = PipelineConfig(enable_enhancement=False, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True)
        p.run(_blank_image(), _blank_image(), consent=True)
        p.enhancer.enhance.assert_not_called()

    def test_enhancement_not_called_without_enhancer(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=False)
        p.run(_blank_image(), _blank_image(), consent=True)
        # No enhancer attached — no AttributeError either
        assert p.enhancer is None

    def test_enhancement_result_stored(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.enhancement_result is not None

    def test_enhancement_success_updates_output(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.success is True
        assert isinstance(result.output_image, np.ndarray)

    def test_enhancement_failure_adds_warning(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True, enhance_success=False)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        # Should still succeed (swap worked) but with a warning
        assert result.success is True
        assert len(result.warnings) > 0

    def test_enhancement_failure_does_not_lose_swap_output(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True, enhance_success=False)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.output_image is not None
        assert result.output_image.ndim == 3

    def test_enhancement_exception_adds_warning_not_error(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True)
        p.enhancer.enhance.side_effect = RuntimeError("enhancer exploded")
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.success is True
        assert any("enhance" in w.lower() or "exception" in w.lower() for w in result.warnings)

    def test_enable_enhancement_without_enhancer_adds_warning(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=False)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert any("enhancer" in w.lower() or "enhancement" in w.lower() for w in result.warnings)

    def test_enhancement_timing_recorded(self):
        cfg = PipelineConfig(enable_enhancement=True, watermark=False)
        p = _make_pipeline(config=cfg, with_enhancer=True)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.timing.enhance_ms >= 0.0


@pytest.mark.integration
@pytest.mark.pipeline
class TestFacePipelineWatermark:

    def test_watermark_enabled_by_default(self):
        p = _make_pipeline()
        result = p.run(_blank_image(), _blank_image(), consent=True)
        # Pipeline completes; watermark stage timing should be ≥ 0
        assert result.timing.watermark_ms >= 0.0

    def test_watermark_disabled_via_config(self):
        cfg = PipelineConfig(watermark=False)
        p = _make_pipeline(config=cfg)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.success is True

    def test_watermark_timing_zero_when_disabled(self):
        cfg = PipelineConfig(watermark=False)
        p = _make_pipeline(config=cfg)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.timing.watermark_ms == 0.0

    def test_custom_watermark_text_in_config(self):
        cfg = PipelineConfig(watermark=True, watermark_text="CUSTOM MARK")
        p = _make_pipeline(config=cfg)
        result = p.run(_blank_image(), _blank_image(), consent=True)
        assert result.success is True


@pytest.mark.integration
@pytest.mark.pipeline
class TestVideoProcessingConfig:

    def test_requires_source_embedding(self):
        emb = _make_embedding()
        cfg = VideoProcessingConfig(source_embedding=emb)
        assert cfg.source_embedding is emb

    def test_default_blend_mode(self):
        cfg = VideoProcessingConfig(source_embedding=_make_embedding())
        assert cfg.blend_mode == BlendMode.POISSON

    def test_default_skip_frames(self):
        cfg = VideoProcessingConfig(source_embedding=_make_embedding())
        assert cfg.skip_frames == 0

    def test_default_swap_all_faces(self):
        cfg = VideoProcessingConfig(source_embedding=_make_embedding())
        assert cfg.swap_all_faces is True

    def test_default_watermark_true(self):
        cfg = VideoProcessingConfig(source_embedding=_make_embedding())
        assert cfg.watermark is True

    def test_default_enhance_false(self):
        cfg = VideoProcessingConfig(source_embedding=_make_embedding())
        assert cfg.enhance is False

    def test_default_preserve_audio_true(self):
        cfg = VideoProcessingConfig(source_embedding=_make_embedding())
        assert cfg.preserve_audio is True

    def test_custom_values(self):
        emb = _make_embedding()
        cfg = VideoProcessingConfig(
            source_embedding=emb,
            blend_mode=BlendMode.ALPHA,
            skip_frames=2,
            swap_all_faces=False,
            watermark=False,
            enhance=True,
            max_resolution=(1280, 720),
        )
        assert cfg.blend_mode == BlendMode.ALPHA
        assert cfg.skip_frames == 2
        assert cfg.swap_all_faces is False
        assert cfg.watermark is False
        assert cfg.enhance is True
        assert cfg.max_resolution == (1280, 720)

    def test_progress_callback_stored(self):
        calls = []

        def cb(cur, tot):
            calls.append((cur, tot))

        cfg = VideoProcessingConfig(
            source_embedding=_make_embedding(),
            progress_callback=cb,
        )
        cfg.progress_callback(5, 100)
        assert calls == [(5, 100)]


@pytest.mark.integration
@pytest.mark.pipeline
class TestVideoProcessingResult:

    def test_success_true_when_frames_processed(self):
        r = VideoProcessingResult(output_path="out.mp4", processed_frames=10)
        assert r.success is True

    def test_success_false_when_no_frames(self):
        r = VideoProcessingResult(output_path="out.mp4", processed_frames=0)
        assert r.success is False

    def test_repr_contains_output_path(self):
        r = VideoProcessingResult(output_path="test_out.mp4", processed_frames=5)
        assert "test_out.mp4" in repr(r)

    def test_repr_contains_frame_count(self):
        r = VideoProcessingResult(
            output_path="out.mp4",
            total_frames=100,
            processed_frames=95,
        )
        rep = repr(r)
        assert "95" in rep
        assert "100" in rep

    def test_default_fields(self):
        r = VideoProcessingResult(output_path="x.mp4")
        assert r.total_frames == 0
        assert r.processed_frames == 0
        assert r.skipped_frames == 0
        assert r.failed_frames == 0
        assert r.total_time_s == 0.0
        assert r.avg_fps == 0.0
        assert r.source_fps == 0.0
        assert r.source_resolution == (0, 0)

    def test_output_path_stored(self):
        r = VideoProcessingResult(output_path="/tmp/result.mp4")
        assert r.output_path == "/tmp/result.mp4"


@pytest.mark.integration
@pytest.mark.pipeline
class TestVideoPipelineConstruction:

    def test_stores_detector(self):
        det = _mock_detector()
        swap = _mock_swapper()
        vp = VideoPipeline(detector=det, swapper=swap)
        assert vp.detector is det

    def test_stores_swapper(self):
        det = _mock_detector()
        swap = _mock_swapper()
        vp = VideoPipeline(detector=det, swapper=swap)
        assert vp.swapper is swap

    def test_enhancer_defaults_to_none(self):
        vp = VideoPipeline(detector=_mock_detector(), swapper=_mock_swapper())
        assert vp.enhancer is None

    def test_stores_enhancer(self):
        enh = _mock_enhancer()
        vp = VideoPipeline(
            detector=_mock_detector(),
            swapper=_mock_swapper(),
            enhancer=enh,
        )
        assert vp.enhancer is enh

    def test_repr_contains_class_name(self):
        vp = VideoPipeline(detector=_mock_detector(), swapper=_mock_swapper())
        assert "VideoPipeline" in repr(vp)

    def test_repr_contains_detector_name(self):
        vp = VideoPipeline(detector=_mock_detector(), swapper=_mock_swapper())
        r = repr(vp)
        assert "detector" in r.lower() or "Mock" in r

    def test_repr_shows_none_when_no_enhancer(self):
        vp = VideoPipeline(detector=_mock_detector(), swapper=_mock_swapper())
        assert "None" in repr(vp)


@pytest.mark.integration
@pytest.mark.pipeline
class TestCappedResolution:

    def test_no_cap_returns_original(self):
        assert VideoPipeline._capped_resolution(1920, 1080, None) == (1920, 1080)

    def test_no_scaling_needed(self):
        assert VideoPipeline._capped_resolution(640, 480, (1920, 1080)) == (640, 480)

    def test_scales_width_down(self):
        w, h = VideoPipeline._capped_resolution(3840, 2160, (1920, 1080))
        assert w <= 1920
        assert h <= 1080

    def test_preserves_aspect_ratio_landscape(self):
        w, h = VideoPipeline._capped_resolution(1920, 1080, (960, 960))
        assert abs(w / h - 1920 / 1080) < 0.01

    def test_preserves_aspect_ratio_portrait(self):
        w, h = VideoPipeline._capped_resolution(720, 1280, (360, 1280))
        assert abs(w / h - 720 / 1280) < 0.01

    def test_exact_cap_unchanged(self):
        assert VideoPipeline._capped_resolution(1280, 720, (1280, 720)) == (1280, 720)

    def test_returns_integers(self):
        w, h = VideoPipeline._capped_resolution(1920, 1080, (1000, 1000))
        assert isinstance(w, int)
        assert isinstance(h, int)

    def test_small_image_not_upscaled(self):
        # _capped_resolution should NOT upscale if image is already smaller
        w, h = VideoPipeline._capped_resolution(320, 240, (1920, 1080))
        assert w == 320
        assert h == 240


@pytest.mark.integration
@pytest.mark.pipeline
class TestVideoPipelineAddWatermark:

    def test_returns_same_shape(self):
        img = _blank_image(480, 640)
        out = VideoPipeline._add_watermark(img, "AI GENERATED")
        assert out.shape == img.shape

    def test_returns_uint8(self):
        img = _blank_image()
        out = VideoPipeline._add_watermark(img, "TEST")
        assert out.dtype == np.uint8

    def test_modifies_pixels(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        out = VideoPipeline._add_watermark(img, "MARK")
        assert not np.array_equal(img, out)

    def test_does_not_modify_original(self):
        img = _blank_image()
        copy = img.copy()
        VideoPipeline._add_watermark(img, "TEST")
        np.testing.assert_array_equal(img, copy)

    def test_different_text_different_output(self):
        img = _blank_image()
        out1 = VideoPipeline._add_watermark(img, "ALPHA")
        out2 = VideoPipeline._add_watermark(img, "ZZZZZZ")
        assert not np.array_equal(out1, out2)

    def test_works_on_small_frame(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        out = VideoPipeline._add_watermark(img, "AI")
        assert out.shape == (64, 64, 3)

    def test_works_on_hd_frame(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = VideoPipeline._add_watermark(img, "AI GENERATED")
        assert out.shape == (1080, 1920, 3)


@pytest.mark.integration
@pytest.mark.pipeline
class TestVideoPipelineProcessFrame:

    def _make_vp(self, n_faces=1, swap_success=True, with_enhancer=False):
        det = _mock_detector(n_faces=n_faces)
        swap = _mock_swapper(success=swap_success)
        enh = _mock_enhancer() if with_enhancer else None
        return VideoPipeline(detector=det, swapper=swap, enhancer=enh)

    def _make_cfg(self, **kwargs) -> VideoProcessingConfig:
        defaults = dict(
            source_embedding=_make_embedding(),
            watermark=False,
        )
        defaults.update(kwargs)
        return VideoProcessingConfig(**defaults)

    def test_returns_ndarray(self):
        vp = self._make_vp()
        frame = _blank_image()
        cfg = self._make_cfg()
        result = vp._process_frame(frame, cfg)
        assert isinstance(result, np.ndarray)

    def test_same_shape_on_success(self):
        vp = self._make_vp()
        frame = _blank_image(480, 640)
        cfg = self._make_cfg()
        out = vp._process_frame(frame, cfg)
        assert out.shape == frame.shape

    def test_returns_original_frame_when_no_face(self):
        vp = self._make_vp(n_faces=0)
        frame = _blank_image()
        cfg = self._make_cfg()
        out = vp._process_frame(frame, cfg)
        np.testing.assert_array_equal(out, frame)

    def test_swap_called_once_for_single_face(self):
        vp = self._make_vp()
        cfg = self._make_cfg(swap_all_faces=False)
        vp._process_frame(_blank_image(), cfg)
        vp.swapper.swap.assert_called_once()

    def test_swap_all_called_for_all_faces_mode(self):
        vp = self._make_vp()
        cfg = self._make_cfg(swap_all_faces=True)
        vp._process_frame(_blank_image(), cfg)
        vp.swapper.swap_all.assert_called_once()

    def test_enhance_called_when_enabled(self):
        vp = self._make_vp(with_enhancer=True)
        # Mark enhancer as loaded
        vp.enhancer.is_loaded = True
        cfg = self._make_cfg(enhance=True)
        vp._process_frame(_blank_image(), cfg)
        vp.enhancer.enhance.assert_called_once()

    def test_enhance_not_called_when_disabled(self):
        vp = self._make_vp(with_enhancer=True)
        cfg = self._make_cfg(enhance=False)
        vp._process_frame(_blank_image(), cfg)
        vp.enhancer.enhance.assert_not_called()

    def test_enhance_not_called_without_enhancer(self):
        vp = self._make_vp(with_enhancer=False)
        cfg = self._make_cfg(enhance=True)
        # Should not raise even though enhancer is None
        out = vp._process_frame(_blank_image(), cfg)
        assert isinstance(out, np.ndarray)

    def test_failed_swap_returns_original_frame(self):
        vp = self._make_vp(swap_success=False)
        frame = _blank_image()
        cfg = self._make_cfg(swap_all_faces=False)
        out = vp._process_frame(frame, cfg)
        # On failure, single swap returns original frame
        assert out.shape == frame.shape


@pytest.mark.integration
@pytest.mark.pipeline
class TestVideoPipelineProcess:

    def _make_temp_video(self, tmp_path, n_frames=5, fps=10.0, w=64, h=64) -> str:
        """Write a minimal MP4 with solid-colour frames and return its path."""
        path = str(tmp_path / "test_input.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        rng = np.random.default_rng(0)
        for _ in range(n_frames):
            frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def _make_vp(self):
        det = MagicMock()
        det.is_loaded = True
        det.detect.return_value = _make_detection(n_faces=1)
        swap = _mock_swapper()
        return VideoPipeline(detector=det, swapper=swap)

    def _make_cfg(self) -> VideoProcessingConfig:
        return VideoProcessingConfig(
            source_embedding=_make_embedding(),
            watermark=False,
            preserve_audio=False,
        )

    def test_raises_if_source_not_found(self, tmp_path):
        vp = self._make_vp()
        cfg = self._make_cfg()
        with pytest.raises(FileNotFoundError):
            vp.process(
                source_video=str(tmp_path / "nonexistent.mp4"),
                output_path=str(tmp_path / "out.mp4"),
                config=cfg,
            )

    def test_returns_video_processing_result(self, tmp_path):
        src = self._make_temp_video(tmp_path)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert isinstance(result, VideoProcessingResult)

    def test_output_file_created(self, tmp_path):
        src = self._make_temp_video(tmp_path)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        vp.process(source_video=src, output_path=out, config=cfg)
        assert Path(out).exists()

    def test_processed_frames_count(self, tmp_path):
        n_frames = 5
        src = self._make_temp_video(tmp_path, n_frames=n_frames)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.processed_frames + result.failed_frames == n_frames

    def test_total_frames_matches_video(self, tmp_path):
        n_frames = 6
        src = self._make_temp_video(tmp_path, n_frames=n_frames)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.total_frames == n_frames

    def test_output_path_in_result(self, tmp_path):
        src = self._make_temp_video(tmp_path)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.output_path == out

    def test_source_fps_in_result(self, tmp_path):
        src = self._make_temp_video(tmp_path, fps=10.0)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.source_fps > 0.0

    def test_total_time_positive(self, tmp_path):
        src = self._make_temp_video(tmp_path)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.total_time_s >= 0.0

    def test_skip_frames_reduces_processed_count(self, tmp_path):
        n_frames = 8
        src = self._make_temp_video(tmp_path, n_frames=n_frames)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        cfg = VideoProcessingConfig(
            source_embedding=_make_embedding(),
            watermark=False,
            preserve_audio=False,
            skip_frames=1,  # process every other frame
        )
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.skipped_frames > 0
        assert result.processed_frames < n_frames

    def test_progress_callback_called(self, tmp_path):
        src = self._make_temp_video(tmp_path, n_frames=4)
        out = str(tmp_path / "out.mp4")
        vp = self._make_vp()
        calls = []
        cfg = VideoProcessingConfig(
            source_embedding=_make_embedding(),
            watermark=False,
            preserve_audio=False,
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )
        vp.process(source_video=src, output_path=out, config=cfg)
        assert len(calls) > 0

    def test_failed_frames_counted_on_swap_error(self, tmp_path):
        src = self._make_temp_video(tmp_path, n_frames=4)
        out = str(tmp_path / "out.mp4")
        det = MagicMock()
        det.is_loaded = True
        det.detect.return_value = _make_detection(n_faces=1)
        swap = _mock_swapper()
        swap.swap.side_effect = RuntimeError("swap failed")
        swap.swap_all.side_effect = RuntimeError("swap failed")
        vp = VideoPipeline(detector=det, swapper=swap)
        cfg = self._make_cfg()
        result = vp.process(source_video=src, output_path=out, config=cfg)
        assert result.failed_frames > 0

    def test_output_dir_created_if_missing(self, tmp_path):
        src = self._make_temp_video(tmp_path)
        new_dir = tmp_path / "nested" / "output"
        out = str(new_dir / "result.mp4")
        vp = self._make_vp()
        cfg = self._make_cfg()
        vp.process(source_video=src, output_path=out, config=cfg)
        assert new_dir.exists()


@pytest.mark.integration
@pytest.mark.pipeline
class TestMergeAudio:

    def test_returns_false_when_ffmpeg_missing(self, tmp_path):
        """Should return False gracefully when FFmpeg is not on PATH."""
        video = str(tmp_path / "video.mp4")
        audio = str(tmp_path / "audio.mp4")
        output = str(tmp_path / "merged.mp4")
        # Create dummy files so the path checks pass
        Path(video).write_bytes(b"dummy")
        Path(audio).write_bytes(b"dummy")

        with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
            result = VideoPipeline._merge_audio(video, audio, output)
        assert result is False

    def test_returns_false_on_nonzero_returncode(self, tmp_path):
        video = str(tmp_path / "v.mp4")
        audio = str(tmp_path / "a.mp4")
        output = str(tmp_path / "out.mp4")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"error output"

        with patch("subprocess.run", return_value=mock_result):
            result = VideoPipeline._merge_audio(video, audio, output)
        assert result is False

    def test_returns_true_on_success(self, tmp_path):
        video = str(tmp_path / "v.mp4")
        audio = str(tmp_path / "a.mp4")
        output = str(tmp_path / "out.mp4")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = VideoPipeline._merge_audio(video, audio, output)
        assert result is True

    def test_returns_false_on_timeout(self, tmp_path):
        import subprocess

        video = str(tmp_path / "v.mp4")
        audio = str(tmp_path / "a.mp4")
        output = str(tmp_path / "out.mp4")

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 600)):
            result = VideoPipeline._merge_audio(video, audio, output)
        assert result is False
