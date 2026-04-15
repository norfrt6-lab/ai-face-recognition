"""POST /api/v1/swap/batch — batch face swap endpoint.

Accepts one source image and up to 10 target images.  Processes each target
independently (sequentially to avoid OOM) and returns a JSON list with per-image
results including the swapped image encoded as base64 PNG.

Maximum batch size: 10 images (returns HTTP 422 if exceeded).
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Batch Face Swap"])

_detector_breaker = CircuitBreaker("batch_detector", failure_threshold=5, recovery_timeout=30.0)
_swapper_breaker = CircuitBreaker("batch_swapper", failure_threshold=5, recovery_timeout=30.0)

_MAX_BATCH_SIZE: int = 10


async def _read_image(upload: UploadFile, label: str) -> np.ndarray:
    """Read an UploadFile and decode to a BGR numpy array.

    Args:
        upload: The uploaded file object.
        label:  Human-readable name for error messages.

    Returns:
        Decoded BGR numpy array.

    Raises:
        HTTPException 400: If the file cannot be read or decoded.
    """
    try:
        data = await upload.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read file '{label}': {exc}",
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image '{label}' — unsupported format.",
        )
    return img


def _encode_b64(image: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _swap_single(
    source_embedding: np.ndarray,
    target_image: np.ndarray,
    state,
) -> Dict[str, Any]:
    """Run detect → swap on a single target image.

    Args:
        source_embedding: Pre-extracted source face embedding.
        target_image:     BGR target image.
        state:            FastAPI app.state with detector and swapper.

    Returns:
        Dictionary with keys: success, image_b64, faces_swapped, error.
    """
    from core.swapper.base_swapper import BlendMode, SwapRequest  # noqa: PLC0415

    try:
        _detector_breaker.check()
        detection = state.detector.detect(target_image)
        _detector_breaker.record_success()
    except CircuitOpenError as exc:
        return {"success": False, "image_b64": None, "faces_swapped": 0, "error": str(exc)}
    except Exception as exc:
        _detector_breaker.record_failure()
        return {"success": False, "image_b64": None, "faces_swapped": 0, "error": f"Detection failed: {exc}"}

    if detection.is_empty:
        return {
            "success": False,
            "image_b64": None,
            "faces_swapped": 0,
            "error": "No face detected in target image.",
        }

    try:
        _swapper_breaker.check()
        tgt_face = detection.faces[0]
        swap_req = SwapRequest(
            source_embedding=source_embedding,
            target_image=target_image,
            target_face=tgt_face,
            blend_mode=BlendMode.POISSON,
        )
        result = state.swapper.swap(swap_req)
        _swapper_breaker.record_success()
    except CircuitOpenError as exc:
        return {"success": False, "image_b64": None, "faces_swapped": 0, "error": str(exc)}
    except Exception as exc:
        _swapper_breaker.record_failure()
        return {"success": False, "image_b64": None, "faces_swapped": 0, "error": f"Swap failed: {exc}"}

    if not result.success:
        return {
            "success": False,
            "image_b64": None,
            "faces_swapped": 0,
            "error": result.error or "Swap returned failure.",
        }

    try:
        b64 = _encode_b64(result.output_image)
    except Exception as exc:
        return {"success": False, "image_b64": None, "faces_swapped": 1, "error": f"Encode failed: {exc}"}

    return {"success": True, "image_b64": b64, "faces_swapped": 1, "error": None}


@router.post(
    "/swap/batch",
    summary="Batch face swap",
    description=(
        "Upload one **source** image and up to **10 target** images.  "
        "The source identity is swapped into the primary face of each target "
        "independently.  Processing is sequential to avoid OOM on constrained "
        "hardware.\n\n"
        "Returns a JSON array with one result object per target image.  "
        "Each object contains:\n"
        "- `index` — zero-based position in the submitted list\n"
        "- `success` — whether the swap succeeded\n"
        "- `image_b64` — base64-encoded PNG of the swapped image (null on failure)\n"
        "- `faces_swapped` — number of faces swapped (0 or 1)\n"
        "- `time_ms` — processing time for this target in milliseconds"
    ),
    responses={
        200: {"description": "JSON list of per-image swap results."},
        400: {"description": "Could not decode the source image."},
        422: {"description": "Batch size exceeds the maximum of 10."},
        503: {"description": "Pipeline components not ready."},
    },
)
async def batch_swap(
    request: Request,
    source_file: UploadFile = File(..., description="Source image (donor identity)."),
    target_files: List[UploadFile] = File(..., description="Up to 10 target images."),
) -> JSONResponse:
    """Swap the source identity into each target image in the batch."""
    t_total = time.perf_counter()

    # Enforce batch size limit
    if len(target_files) > _MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Batch size {len(target_files)} exceeds the maximum of "
                f"{_MAX_BATCH_SIZE} images per request."
            ),
        )

    state = request.app.state
    for component in ("detector", "recognizer", "swapper"):
        obj = getattr(state, component, None)
        if obj is None or not getattr(obj, "is_loaded", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline component '{component}' is not ready.",
            )

    # Decode source image
    source_image = await _read_image(source_file, "source")

    # Detect + embed source face
    try:
        _detector_breaker.check()
        src_detection = state.detector.detect(source_image)
        _detector_breaker.record_success()
    except CircuitOpenError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        _detector_breaker.record_failure()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Source face detection failed: {exc}",
        )

    if src_detection.is_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the source image.",
        )

    src_face = src_detection.faces[0]
    src_bbox = (int(src_face.x1), int(src_face.y1), int(src_face.x2), int(src_face.y2))
    try:
        from functools import partial  # noqa: PLC0415

        source_embedding = state.recognizer.get_embedding(source_image, bbox=src_bbox)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Source embedding extraction failed: {exc}",
        )

    if source_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract a valid embedding from the source face.",
        )

    # Process each target sequentially
    batch_results = []
    for idx, target_file in enumerate(target_files):
        t_item = time.perf_counter()
        target_image = await _read_image(target_file, f"target[{idx}]")
        item = _swap_single(source_embedding, target_image, state)
        elapsed_ms = (time.perf_counter() - t_item) * 1000.0
        batch_results.append(
            {
                "index": idx,
                "success": item["success"],
                "image_b64": item["image_b64"],
                "faces_swapped": item["faces_swapped"],
                "time_ms": round(elapsed_ms, 2),
            }
        )

    total_ms = (time.perf_counter() - t_total) * 1000.0
    succeeded = sum(1 for r in batch_results if r["success"])

    logger.info(
        f"[batch] targets={len(target_files)} "
        f"succeeded={succeeded} "
        f"failed={len(target_files) - succeeded} "
        f"total={total_ms:.1f}ms"
    )

    return JSONResponse(content=batch_results)
