# ============================================================
# AI Face Recognition & Face Swap
# core/pipeline/face_pipeline.py
# ============================================================
# Image-mode pipeline that orchestrates the full end-to-end
# face swap workflow in a single call:
#
#   Input Image
#       │
#       ▼
#   [1] YOLOv8 Detector      → DetectionResult
#       │
#       ▼
#   [2] InsightFace Recognizer → FaceEmbedding (source)
#       │
#       ▼
#   [3] InSwapper             → SwapResult / BatchSwapResult
#       │
#       ▼
#   [4] GFPGAN / CodeFormer   → EnhancementResult  (optional)
#       │
#       ▼
#   [5] Watermark             → output image        (optional)
#       │
#       ▼
#   PipelineResult
#
# The pipeline is designed to be:
#   - Stateless per call (all state lives in the component objects)
#   - Fault-tolerant (each stage returns partial results on failure)
#   - Observable (timing breakdown per stage)
#   - Ethics-aware (consent gate + watermark)
# ============================================================

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector.base_detector import BaseDetector, DetectionResult, FaceBox
from core.enhancer.base_enhancer import (
    BaseEnhancer,
    EnhancementRequest,
    EnhancementResult,
    EnhancementStatus,
)
from core.recognizer.base_recognizer import BaseRecognizer, FaceEmbedding
from core.swapper.base_swapper import (
    BaseSwapper,
    BatchSwapResult,
    BlendMode,
    SwapRequest,
    SwapResult,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Enumerations
# ============================================================

class PipelineStatus(Enum):
    """
    Overall status of a pipeline run.

    SUCCESS         — All requested stages completed successfully.
    PARTIAL         — Some stages succeeded but at least one failed
                      (e.g. enhancement failed but swap succeeded).
    NO_SOURCE_FACE  — No face was detected in the source image.
    NO_TARGET_FACE  — No face was detected in the target image.
    CONSENT_DENIED  — Ethics gate rejected the request.
    STAGE_ERROR     — A critical stage raised an unrecoverable error.
    """
    SUCCESS        = "success"
    PARTIAL        = "partial"
    NO_SOURCE_FACE = "no_source_face"
    NO_TARGET_FACE = "no_target_face"
    CONSENT_DENIED = "consent_denied"
    STAGE_ERROR    = "stage_error"


# ============================================================
# Configuration
# ============================================================

@dataclass
class PipelineConfig:
    """
    Single configuration object that controls all pipeline stages.

    Pass one of these to ``FacePipeline.run()`` to override the
    pipeline-level defaults for a specific call.

    Attributes:
        blend_mode:         Face compositing strategy.
        blend_alpha:        Global alpha weight [0.0, 1.0].
        mask_feather:       Blend mask edge blur radius (pixels).
        swap_all_faces:     If True, swap every face in the target.
                            If False, swap only *target_face_index*.
        max_faces:          Cap on faces to swap when swap_all=True.
        source_face_index:  Which face in the source image to use (0-based).
        target_face_index:  Which face to replace when swap_all=False.
        enable_enhancement: Run face enhancement after swap.
        fidelity_weight:    CodeFormer fidelity weight [0.0, 1.0].
        upscale:            Enhancement upscale factor.
        watermark:          Add "AI GENERATED" watermark to output.
        watermark_text:     Watermark text string.
        require_consent:    Reject calls where consent=False.
        save_intermediate:  Attach intermediate arrays to the result.
    """

    # Swap stage
    blend_mode:        BlendMode = BlendMode.POISSON
    blend_alpha:       float     = 1.0
    mask_feather:      int       = 20
    swap_all_faces:    bool      = False
    max_faces:         int       = 10
    source_face_index: int       = 0
    target_face_index: int       = 0

    # Enhancement stage
    enable_enhancement: bool  = False
    fidelity_weight:    float = 0.5
    upscale:            int   = 2

    # Ethics / output
    watermark:      bool = True
    watermark_text: str  = "AI GENERATED"
    require_consent: bool = True

    # Debug
    save_intermediate: bool = False

    def __repr__(self) -> str:
        return (
            f"PipelineConfig("
            f"blend={self.blend_mode.name}, "
            f"swap_all={self.swap_all_faces}, "
            f"enhance={self.enable_enhancement}, "
            f"watermark={self.watermark})"
        )


# ============================================================
# Stage timing
# ============================================================

@dataclass
class PipelineTiming:
    """
    Wall-clock time (ms) spent in each pipeline stage.

    All values are 0.0 if the corresponding stage was not executed.
    """

    detect_source_ms:  float = 0.0
    embed_source_ms:   float = 0.0
    detect_target_ms:  float = 0.0
    swap_ms:           float = 0.0
    enhance_ms:        float = 0.0
    watermark_ms:      float = 0.0
    total_ms:          float = 0.0

    @property
    def pipeline_overhead_ms(self) -> float:
        """Time spent in pipeline glue code (not in any AI model)."""
        stage_total = (
            self.detect_source_ms
            + self.embed_source_ms
            + self.detect_target_ms
            + self.swap_ms
            + self.enhance_ms
            + self.watermark_ms
        )
        return max(0.0, self.total_ms - stage_total)

    def __repr__(self) -> str:
        return (
            f"PipelineTiming("
            f"detect={self.detect_source_ms + self.detect_target_ms:.1f}ms, "
            f"embed={self.embed_source_ms:.1f}ms, "
            f"swap={self.swap_ms:.1f}ms, "
            f"enhance={self.enhance_ms:.1f}ms, "
            f"total={self.total_ms:.1f}ms)"
        )


# ============================================================
# Result
# ============================================================

@dataclass
class PipelineResult:
    """
    The complete output of a single ``FacePipeline.run()`` call.

    Attributes:
        output_image:       Final BGR output frame (watermarked if requested).
                            Equal to the target image if every stage failed.
        status:             Overall pipeline status.
        request_id:         Unique UUID string for tracing this run.
        source_detection:   Detection result from the source image.
        target_detection:   Detection result from the target image.
        source_embedding:   ArcFace embedding used as the swap source.
        swap_result:        Result of the swap stage.
        enhancement_result: Result of the enhancement stage (None if skipped).
        timing:             Per-stage timing breakdown.
        config:             The PipelineConfig used for this run.
        error:              High-level error message if status != SUCCESS.
        warnings:           Non-fatal warning messages accumulated during the run.
    """

    output_image:       np.ndarray
    status:             PipelineStatus
    request_id:         str
    source_detection:   Optional[DetectionResult]           = None
    target_detection:   Optional[DetectionResult]           = None
    source_embedding:   Optional[FaceEmbedding]             = None
    swap_result:        Optional[BatchSwapResult]           = None
    enhancement_result: Optional[EnhancementResult]         = None
    timing:             PipelineTiming                      = field(default_factory=PipelineTiming)
    config:             Optional[PipelineConfig]            = None
    error:              Optional[str]                       = None
    warnings:           List[str]                           = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if the pipeline completed without critical errors."""
        return self.status in (PipelineStatus.SUCCESS, PipelineStatus.PARTIAL)

    @property
    def num_faces_swapped(self) -> int:
        """Number of faces that were swapped successfully."""
        if self.swap_result is None:
            return 0
        return self.swap_result.num_swapped

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"status={self.status.value}, "
            f"faces_swapped={self.num_faces_swapped}, "
            f"timing={self.timing.total_ms:.1f}ms, "
            f"id={self.request_id[:8]}...)"
        )


# ============================================================
# FacePipeline
# ============================================================

class FacePipeline:
    """
    End-to-end image face swap pipeline.

    Orchestrates:
      - Face detection  (``BaseDetector``)
      - Face embedding  (``BaseRecognizer``)
      - Face swap       (``BaseSwapper``)
      - Face enhancement (``BaseEnhancer``, optional)
      - Watermarking    (built-in, optional)

    All component instances are injected at construction time so that
    the pipeline is fully testable with mocks.

    Example usage::

        detector   = YOLOFaceDetector("models/yolov8n-face.pt")
        recognizer = InsightFaceRecognizer()
        swapper    = InSwapper("models/inswapper_128.onnx")

        detector.load_model()
        recognizer.load_model()
        swapper.load_model()

        pipeline = FacePipeline(
            detector=detector,
            recognizer=recognizer,
            swapper=swapper,
        )

        result = pipeline.run(
            source_image=source_bgr,
            target_image=target_bgr,
            consent=True,
        )
        if result.success:
            cv2.imwrite("output.png", result.output_image)

    Args:
        detector:   Loaded face detector instance.
        recognizer: Loaded face recognizer instance.
        swapper:    Loaded face swapper instance.
        enhancer:   Optional loaded face enhancer instance.
        config:     Default ``PipelineConfig`` (can be overridden per call).
    """

    def __init__(
        self,
        detector:   BaseDetector,
        recognizer: BaseRecognizer,
        swapper:    BaseSwapper,
        enhancer:   Optional[BaseEnhancer] = None,
        config:     Optional[PipelineConfig] = None,
    ) -> None:
        self.detector   = detector
        self.recognizer = recognizer
        self.swapper    = swapper
        self.enhancer   = enhancer
        self.config     = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        *,
        consent: bool = False,
        config: Optional[PipelineConfig] = None,
    ) -> PipelineResult:
        """
        Run the full face swap pipeline on a single image pair.

        Args:
            source_image: BGR numpy array — the donor face image.
            target_image: BGR numpy array — the frame to modify.
            consent:      Explicit consent flag (must be True unless
                          ``config.require_consent=False``).
            config:       Per-call config override. Falls back to
                          the instance-level ``self.config``.

        Returns:
            ``PipelineResult`` — always non-None; check ``.success``.
        """
        cfg         = config or self.config
        request_id  = str(uuid.uuid4())
        t_total     = _timer()
        timing      = PipelineTiming()
        warnings:   List[str] = []

        logger.info(
            f"Pipeline run | id={request_id[:8]} | "
            f"source={source_image.shape} target={target_image.shape}"
        )

        # ── Ethics gate ──────────────────────────────────────────────
        if cfg.require_consent and not consent:
            logger.warning(f"Pipeline [{request_id[:8]}] rejected — consent=False")
            timing.total_ms = _timer() - t_total
            return PipelineResult(
                output_image=target_image.copy(),
                status=PipelineStatus.CONSENT_DENIED,
                request_id=request_id,
                timing=timing,
                config=cfg,
                error=(
                    "Consent was not provided. "
                    "Set consent=True to process this request."
                ),
            )

        # ── Stage 1: Detect faces in source image ────────────────────
        t0 = _timer()
        try:
            source_detection = self.detector.detect(source_image)
        except Exception as exc:
            timing.total_ms = _timer() - t_total
            return self._stage_error(
                target_image, request_id, timing, cfg,
                f"Source face detection failed: {exc}",
            )
        timing.detect_source_ms = _timer() - t0

        if source_detection.is_empty:
            timing.total_ms = _timer() - t_total
            return PipelineResult(
                output_image=target_image.copy(),
                status=PipelineStatus.NO_SOURCE_FACE,
                request_id=request_id,
                source_detection=source_detection,
                timing=timing,
                config=cfg,
                error="No face detected in the source image.",
            )

        # Pick the source face
        src_face_idx = min(cfg.source_face_index, len(source_detection.faces) - 1)
        source_face  = source_detection.faces[src_face_idx]

        # ── Stage 2: Extract source embedding ───────────────────────
        t0 = _timer()
        try:
            source_embedding = self.recognizer.get_embedding(
                source_image,
                bbox=(int(source_face.x1), int(source_face.y1),
                      int(source_face.x2), int(source_face.y2)),
            )
        except Exception as exc:
            timing.embed_source_ms = _timer() - t0
            timing.total_ms        = _timer() - t_total
            return self._stage_error(
                target_image, request_id, timing, cfg,
                f"Source embedding extraction failed: {exc}",
                source_detection=source_detection,
            )
        timing.embed_source_ms = _timer() - t0

        if source_embedding is None:
            timing.total_ms = _timer() - t_total
            return PipelineResult(
                output_image=target_image.copy(),
                status=PipelineStatus.NO_SOURCE_FACE,
                request_id=request_id,
                source_detection=source_detection,
                timing=timing,
                config=cfg,
                error="Failed to extract embedding from the source face.",
            )

        # ── Stage 3: Detect faces in target image ───────────────────
        t0 = _timer()
        try:
            target_detection = self.detector.detect(target_image)
        except Exception as exc:
            timing.detect_target_ms = _timer() - t0
            timing.total_ms         = _timer() - t_total
            return self._stage_error(
                target_image, request_id, timing, cfg,
                f"Target face detection failed: {exc}",
                source_detection=source_detection,
                source_embedding=source_embedding,
            )
        timing.detect_target_ms = _timer() - t0

        if target_detection.is_empty:
            timing.total_ms = _timer() - t_total
            return PipelineResult(
                output_image=target_image.copy(),
                status=PipelineStatus.NO_TARGET_FACE,
                request_id=request_id,
                source_detection=source_detection,
                target_detection=target_detection,
                source_embedding=source_embedding,
                timing=timing,
                config=cfg,
                error="No face detected in the target image.",
            )

        # ── Stage 4: Face swap ───────────────────────────────────────
        t0 = _timer()
        metadata = {"save_intermediate": cfg.save_intermediate}
        try:
            if cfg.swap_all_faces:
                swap_result = self.swapper.swap_all(
                    source_embedding=source_embedding,
                    target_image=target_image,
                    target_detection=target_detection,
                    blend_mode=cfg.blend_mode,
                    blend_alpha=cfg.blend_alpha,
                    mask_feather=cfg.mask_feather,
                    max_faces=cfg.max_faces,
                )
            else:
                # Swap single target face
                tgt_face_idx = min(cfg.target_face_index, len(target_detection.faces) - 1)
                tgt_face     = target_detection.faces[tgt_face_idx]

                single_req = SwapRequest(
                    source_embedding=source_embedding,
                    target_image=target_image,
                    target_face=tgt_face,
                    source_face_index=src_face_idx,
                    target_face_index=tgt_face_idx,
                    blend_mode=cfg.blend_mode,
                    blend_alpha=cfg.blend_alpha,
                    mask_feather=cfg.mask_feather,
                    metadata=metadata,
                )
                single_result = self.swapper.swap(single_req)

                # Wrap in a BatchSwapResult for a uniform result type
                swap_result = BatchSwapResult(
                    output_image=single_result.output_image,
                    swap_results=[single_result],
                    total_time_ms=single_result.swap_time_ms,
                )

        except Exception as exc:
            timing.swap_ms  = _timer() - t0
            timing.total_ms = _timer() - t_total
            return self._stage_error(
                target_image, request_id, timing, cfg,
                f"Face swap stage failed: {exc}",
                source_detection=source_detection,
                target_detection=target_detection,
                source_embedding=source_embedding,
            )
        timing.swap_ms = _timer() - t0

        swapped_image = swap_result.output_image

        # Collect swap warnings
        for sr in swap_result.swap_results:
            if not sr.success:
                warnings.append(
                    f"Face #{sr.target_face.face_index} swap failed: "
                    f"{sr.status.value} — {sr.error}"
                )

        # ── Stage 5: Face enhancement (optional) ────────────────────
        enhancement_result: Optional[EnhancementResult] = None

        if cfg.enable_enhancement and self.enhancer is not None:
            t0 = _timer()
            try:
                enh_req = EnhancementRequest(
                    image=swapped_image,
                    fidelity_weight=cfg.fidelity_weight,
                    upscale=cfg.upscale,
                    only_center_face=False,
                    paste_back=True,
                    full_frame=True,
                    metadata={"save_crops": cfg.save_intermediate},
                )
                enhancement_result = self.enhancer.enhance(enh_req)

                if enhancement_result.success:
                    swapped_image = enhancement_result.output_image
                else:
                    warnings.append(
                        f"Enhancement failed: {enhancement_result.error} — "
                        "using un-enhanced swap output."
                    )
            except Exception as exc:
                warnings.append(f"Enhancement stage raised exception: {exc}")
            finally:
                timing.enhance_ms = _timer() - t0

        elif cfg.enable_enhancement and self.enhancer is None:
            warnings.append(
                "enable_enhancement=True but no enhancer was provided to the pipeline."
            )

        # ── Stage 6: Watermark ───────────────────────────────────────
        t0 = _timer()
        if cfg.watermark:
            try:
                swapped_image = _apply_watermark(swapped_image, cfg.watermark_text)
            except Exception as exc:
                warnings.append(f"Watermark failed: {exc}")
        timing.watermark_ms = _timer() - t0

        # ── Finalise ─────────────────────────────────────────────────
        timing.total_ms = _timer() - t_total

        all_swaps_ok = swap_result.all_success
        status = PipelineStatus.SUCCESS if all_swaps_ok else PipelineStatus.PARTIAL

        logger.info(
            f"Pipeline [{request_id[:8]}] done | "
            f"status={status.value} | "
            f"swapped={swap_result.num_swapped}/{len(swap_result.swap_results)} | "
            f"total={timing.total_ms:.1f}ms"
        )
        if warnings:
            for w in warnings:
                logger.warning(f"Pipeline [{request_id[:8]}] warning: {w}")

        return PipelineResult(
            output_image=swapped_image,
            status=status,
            request_id=request_id,
            source_detection=source_detection,
            target_detection=target_detection,
            source_embedding=source_embedding,
            swap_result=swap_result,
            enhancement_result=enhancement_result,
            timing=timing,
            config=cfg,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stage_error(
        self,
        target_image:    np.ndarray,
        request_id:      str,
        timing:          PipelineTiming,
        cfg:             PipelineConfig,
        error:           str,
        *,
        source_detection: Optional[DetectionResult] = None,
        target_detection: Optional[DetectionResult] = None,
        source_embedding: Optional[FaceEmbedding]   = None,
    ) -> PipelineResult:
        """Return a STAGE_ERROR result with timing already finalised."""
        logger.error(f"Pipeline [{request_id[:8]}] stage error: {error}")
        return PipelineResult(
            output_image=target_image.copy(),
            status=PipelineStatus.STAGE_ERROR,
            request_id=request_id,
            source_detection=source_detection,
            target_detection=target_detection,
            source_embedding=source_embedding,
            timing=timing,
            config=cfg,
            error=error,
        )

    def __repr__(self) -> str:
        components = [
            f"detector={self.detector.__class__.__name__}",
            f"recognizer={self.recognizer.__class__.__name__}",
            f"swapper={self.swapper.__class__.__name__}",
        ]
        if self.enhancer is not None:
            components.append(f"enhancer={self.enhancer.__class__.__name__}")
        return f"FacePipeline({', '.join(components)})"


# ============================================================
# Module-level helpers
# ============================================================

def _timer() -> float:
    """Return current time in milliseconds (monotonic clock)."""
    return time.perf_counter() * 1000.0


def _apply_watermark(
    image: np.ndarray,
    text:  str = "AI GENERATED",
) -> np.ndarray:
    """
    Overlay a semi-transparent text watermark on *image*.

    Places the watermark in the bottom-right corner with a slight
    background shadow for legibility on any background colour.

    Args:
        image: BGR uint8 numpy array to watermark.
        text:  The text string to overlay.

    Returns:
        A copy of *image* with the watermark applied.
    """
    out    = image.copy()
    h, w   = out.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = max(0.4, min(w, h) / 800.0)
    thick  = max(1, int(scale * 1.5))

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    margin = int(min(w, h) * 0.02)

    x = w - tw - margin
    y = h - margin

    # Shadow
    cv2.putText(out, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    # Text (white with 70 % opacity via a blend)
    overlay = out.copy()
    cv2.putText(overlay, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    return out
