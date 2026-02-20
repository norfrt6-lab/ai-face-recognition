# POST /api/v1/swap — face swap endpoint.
#
# Accepts two uploaded images (source + target), runs the full
# detect → embed → swap → (optional enhance) → watermark
# pipeline and returns the swapped output image.
#
# Supported response modes:
#   - File download  (default) — returns image/png
#   - Base64 JSON   (return_base64=true) — returns SwapResponse JSON

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from api.metrics import SWAP_FACE_COUNT
from api.schemas.requests import BlendModeSchema, EnhancerBackendSchema
from api.schemas.requests import SwapRequest as SwapRequestSchema
from api.schemas.requests import swap_form_dep
from api.schemas.responses import (
    BoundingBox,
    ErrorResponse,
    SwappedFaceInfo,
    SwapResponse,
    SwapTimingBreakdown,
)
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Swap"])

_detector_breaker = CircuitBreaker("detector", failure_threshold=5, recovery_timeout=30.0)
_swapper_breaker = CircuitBreaker("swapper", failure_threshold=5, recovery_timeout=30.0)

# Maximum seconds for a single model inference call before timeout
_INFERENCE_TIMEOUT: float = 60.0


def _get_upload_limits() -> tuple[int, int, int]:
    """Return (max_bytes, max_dim, min_dim) from settings or defaults."""
    try:
        from config.settings import settings  # noqa: PLC0415

        return (
            settings.api.max_upload_bytes,
            settings.api.max_image_dimension,
            settings.api.min_image_dimension,
        )
    except Exception:
        return 10 * 1024 * 1024, 4096, 10


async def _decode_image(upload: UploadFile) -> np.ndarray:
    """Read an UploadFile and decode it to a BGR numpy array."""
    max_bytes, max_dim, min_dim = _get_upload_limits()

    # Pre-check Content-Length before reading full body into memory
    claimed_size = upload.size
    if claimed_size is not None and claimed_size > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({claimed_size} bytes). Maximum: {mb:.0f} MB.",
        )

    try:
        data = await upload.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read upload '{upload.filename}': {exc}",
        )

    if len(data) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(data)} bytes). Maximum: {mb:.0f} MB.",
        )

    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image '{upload.filename}': {exc}",
        )

    h, w = img.shape[:2]
    if h > max_dim or w > max_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large: {w}x{h}. Maximum dimension: {max_dim}px.",
        )
    if h < min_dim or w < min_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too small: {w}x{h}. Minimum dimension: {min_dim}px.",
        )

    return img


def _encode_image_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode a BGR numpy array to PNG/JPEG bytes."""
    ok, buf = cv2.imencode(fmt, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {fmt}")
    return buf.tobytes()


def _blend_mode_to_core(mode: BlendModeSchema):
    """Convert API BlendModeSchema to core BlendMode enum."""
    from core.swapper.base_swapper import BlendMode  # noqa: PLC0415

    mapping = {
        BlendModeSchema.alpha: BlendMode.ALPHA,
        BlendModeSchema.poisson: BlendMode.POISSON,
        BlendModeSchema.masked_alpha: BlendMode.MASKED_ALPHA,
    }
    return mapping.get(mode, BlendMode.POISSON)


def _save_output(image: np.ndarray, output_dir: Path, request_id: str) -> str:
    """
    Save the output image to disk and return a relative URL.

    Args:
        image:      BGR numpy array.
        output_dir: Directory to save to.
        request_id: Used as filename prefix.

    Returns:
        Relative URL string like '/api/v1/results/swap_<id>.png'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"swap_{request_id[:8]}.png"
    filepath = output_dir / filename
    cv2.imwrite(str(filepath), image)
    return f"/api/v1/results/{filename}"


@router.post(
    "/swap",
    summary="Face swap",
    description=(
        "Upload a **source** image (donor identity) and a **target** image "
        "(the scene to modify). The API detects faces in both images, extracts "
        "the source identity embedding, and injects it into the target face(s). "
        "\n\n"
        "Set **return_base64=true** to receive a JSON response with the swapped "
        "image encoded as a base64 PNG string instead of a raw image download."
        "\n\n"
        "⚠️ **Ethics**: `consent` must be `true`. You must have explicit consent "
        "from all individuals depicted in both images."
    ),
    responses={
        200: {
            "description": "Swapped image (PNG) or JSON with base64 payload.",
            "content": {
                "image/png": {"schema": {"type": "string", "format": "binary"}},
                "application/json": {"schema": SwapResponse.model_json_schema()},
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad request (validation / no face detected).",
        },
        422: {"description": "Missing or invalid form fields."},
        503: {"model": ErrorResponse, "description": "Pipeline components not ready."},
    },
)
async def swap_faces(
    request: Request,
    source_file: UploadFile = File(..., description="Source image — the donor identity."),
    target_file: UploadFile = File(..., description="Target image — the scene to modify."),
    params: SwapRequestSchema = Depends(swap_form_dep),
    return_base64: bool = Form(default=False),
) -> Response:
    """
    Perform a face swap between *source_file* and *target_file*.

    Form fields mirror ``SwapRequest`` schema fields plus:
      - ``return_base64`` — return JSON with base64 image instead of raw download.

    Returns:
        PNG image response (default) or ``SwapResponse`` JSON.
    """
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Unpack validated params
    blend_mode = params.blend_mode.value
    blend_alpha = params.blend_alpha
    mask_feather = params.mask_feather
    swap_all_faces = params.swap_all_faces
    max_faces = params.max_faces
    source_face_index = params.source_face_index
    target_face_index = params.target_face_index
    enhance = params.enhance
    enhancer_fidelity = params.enhancer_fidelity
    watermark = params.watermark

    logger.info(
        f"[{request_id[:8]}] POST /swap | "
        f"source={source_file.filename!r} target={target_file.filename!r} | "
        f"blend={blend_mode} enhance={enhance} swap_all={swap_all_faces}"
    )

    state = request.app.state
    for component in ("detector", "recognizer", "swapper"):
        obj = getattr(state, component, None)
        if obj is None or not getattr(obj, "is_loaded", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline component '{component}' is not ready.",
            )

    source_image = await _decode_image(source_file)
    target_image = await _decode_image(target_file)

    core_blend_mode = _blend_mode_to_core(params.blend_mode)

    loop = asyncio.get_running_loop()
    executor = getattr(state, "executor", None)

    try:
        _detector_breaker.check()
        source_detection = await asyncio.wait_for(
            loop.run_in_executor(executor, state.detector.detect, source_image),
            timeout=_INFERENCE_TIMEOUT,
        )
        _detector_breaker.record_success()
    except CircuitOpenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except asyncio.TimeoutError:
        _detector_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Source detection timed out after {_INFERENCE_TIMEOUT}s")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Face detection timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        _detector_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Source detection failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed on source image: {exc}",
        )

    if source_detection.is_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the source image.",
        )

    # Pick source face by index (clamp to valid range)
    src_idx = max(0, min(source_face_index, len(source_detection.faces) - 1))
    src_face = source_detection.faces[src_idx]

    try:
        src_bbox = (int(src_face.x1), int(src_face.y1), int(src_face.x2), int(src_face.y2))
        source_embedding = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(state.recognizer.get_embedding, source_image, bbox=src_bbox),
            ),
            timeout=_INFERENCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[{request_id[:8]}] Embedding extraction timed out after {_INFERENCE_TIMEOUT}s"
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Embedding extraction timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        logger.error(f"[{request_id[:8]}] Embedding extraction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract source face embedding: {exc}",
        )

    if source_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract a valid embedding from the source face.",
        )

    try:
        _detector_breaker.check()
        target_detection = await asyncio.wait_for(
            loop.run_in_executor(executor, state.detector.detect, target_image),
            timeout=_INFERENCE_TIMEOUT,
        )
        _detector_breaker.record_success()
    except CircuitOpenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except asyncio.TimeoutError:
        _detector_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Target detection timed out after {_INFERENCE_TIMEOUT}s")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Face detection timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        _detector_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Target detection failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed on target image: {exc}",
        )

    if target_detection.is_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the target image.",
        )

    try:
        _swapper_breaker.check()
        if swap_all_faces:
            batch_result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(
                        state.swapper.swap_all,
                        source_embedding=source_embedding,
                        target_image=target_image,
                        target_detection=target_detection,
                        blend_mode=core_blend_mode,
                        blend_alpha=blend_alpha,
                        mask_feather=mask_feather,
                        max_faces=max_faces,
                    ),
                ),
                timeout=_INFERENCE_TIMEOUT,
            )
            output_image = batch_result.output_image
            swap_results = batch_result.swap_results
        else:
            from core.swapper.base_swapper import SwapRequest as CoreSwapRequest  # noqa: PLC0415

            tgt_idx = max(0, min(target_face_index, len(target_detection.faces) - 1))
            tgt_face = target_detection.faces[tgt_idx]

            swap_req = CoreSwapRequest(
                source_embedding=source_embedding,
                target_image=target_image,
                target_face=tgt_face,
                blend_mode=core_blend_mode,
                blend_alpha=blend_alpha,
                mask_feather=mask_feather,
            )
            single_result = await asyncio.wait_for(
                loop.run_in_executor(executor, state.swapper.swap, swap_req),
                timeout=_INFERENCE_TIMEOUT,
            )
            output_image = single_result.output_image
            swap_results = [single_result]
        _swapper_breaker.record_success()

    except CircuitOpenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except asyncio.TimeoutError:
        _swapper_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Swap timed out after {_INFERENCE_TIMEOUT}s")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Face swap timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        _swapper_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Swap failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face swap failed: {exc}",
        )

    enhanced = False
    enhancer_obj = getattr(state, "enhancer", None)
    if enhance and enhancer_obj is not None and getattr(enhancer_obj, "is_loaded", False):
        try:
            from core.enhancer.base_enhancer import EnhancementRequest  # noqa: PLC0415

            enh_req = EnhancementRequest(
                image=output_image,
                fidelity_weight=enhancer_fidelity,
                full_frame=True,
                paste_back=True,
            )
            enh_result = await asyncio.wait_for(
                loop.run_in_executor(executor, enhancer_obj.enhance, enh_req),
                timeout=_INFERENCE_TIMEOUT,
            )
            if enh_result.success:
                output_image = enh_result.output_image
                enhanced = True
            else:
                logger.warning(f"[{request_id[:8]}] Enhancement failed: {enh_result.error}")
        except Exception as exc:
            logger.warning(f"[{request_id[:8]}] Enhancement error (skipped): {exc}")
    elif enhance and enhancer_obj is None:
        logger.warning(f"[{request_id[:8]}] enhance=True but no enhancer loaded — skipping.")

    watermarked = False
    if watermark:
        try:
            output_image = output_image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w = output_image.shape[:2]
            scale = max(0.4, min(w, h) / 800.0)
            thick = max(1, int(scale * 1.5))
            text = "AI GENERATED"
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            margin = int(min(w, h) * 0.02)
            x = w - tw - margin
            y = h - margin
            cv2.putText(
                output_image, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA
            )
            cv2.putText(
                output_image, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA
            )
            watermarked = True
        except Exception as exc:
            logger.warning(f"[{request_id[:8]}] Watermark failed: {exc}")

    output_url: Optional[str] = None
    try:
        output_dir = Path(getattr(state, "output_dir", "output"))
        output_url = _save_output(output_image, output_dir, request_id)
    except Exception as exc:
        logger.warning(f"[{request_id[:8]}] Could not save output: {exc}")

    total_ms = (time.perf_counter() - t_start) * 1000.0
    faces_info = []
    for sr in swap_results:
        fb = sr.target_face
        faces_info.append(
            SwappedFaceInfo(
                face_index=fb.face_index,
                bbox=BoundingBox(
                    x1=float(fb.x1),
                    y1=float(fb.y1),
                    x2=float(fb.x2),
                    y2=float(fb.y2),
                    confidence=float(fb.confidence),
                ),
                success=sr.success,
                status=sr.status.value,
                timing=SwapTimingBreakdown(
                    align_ms=sr.align_time_ms,
                    inference_ms=sr.inference_time_ms,
                    blend_ms=sr.blend_time_ms,
                    total_ms=sr.swap_time_ms,
                ),
                error=sr.error,
            )
        )

    num_swapped = sum(1 for sr in swap_results if sr.success)
    num_failed = len(swap_results) - num_swapped
    SWAP_FACE_COUNT.labels(status="success").inc(num_swapped)
    SWAP_FACE_COUNT.labels(status="failed").inc(num_failed)

    logger.info(
        f"[{request_id[:8]}] Swap complete | "
        f"swapped={num_swapped} failed={num_failed} "
        f"enhanced={enhanced} watermarked={watermarked} "
        f"total={total_ms:.1f}ms"
    )

    if return_base64:
        img_bytes = _encode_image_bytes(output_image, ".png")
        b64_str = base64.b64encode(img_bytes).decode("utf-8")

        return JSONResponse(
            content=SwapResponse(
                output_url=output_url,
                output_base64=b64_str,
                num_faces_swapped=num_swapped,
                num_faces_failed=num_failed,
                faces=faces_info,
                total_inference_ms=total_ms,
                blend_mode=blend_mode,
                enhanced=enhanced,
                watermarked=watermarked,
            ).model_dump(),
        )

    # Default: return raw PNG download
    img_bytes = _encode_image_bytes(output_image, ".png")
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={
            "X-Request-ID": request_id,
            "X-Faces-Swapped": str(num_swapped),
            "X-Processing-Ms": f"{total_ms:.1f}",
            "X-Enhanced": str(enhanced).lower(),
            "X-Watermarked": str(watermarked).lower(),
            "Content-Disposition": f'attachment; filename="swap_{request_id[:8]}.png"',
        },
    )
