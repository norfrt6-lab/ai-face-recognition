"""POST /api/v1/faces/attributes — face attribute estimation endpoint.

Accepts one image upload and returns geometric attributes for every detected
face: bounding box, confidence, area, aspect ratio, and centre point.

Only the face detector is required — no recognizer or swapper needed.  This
means the endpoint continues to work even when those heavier components are
not loaded.
"""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from api.schemas.attributes_schemas import (
    FaceAttributesItem,
    FaceAttributesResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Attributes"])


@router.post(
    "/faces/attributes",
    response_model=FaceAttributesResponse,
    summary="Face attribute estimation",
    description=(
        "Upload an image.  The API detects all faces using the face detector "
        "and returns per-face geometric attributes:\n\n"
        "- `face_index` — zero-based position in the detection result\n"
        "- `bbox` — `[x1, y1, x2, y2]` bounding box in absolute pixel coordinates\n"
        "- `confidence` — detection confidence score\n"
        "- `area` — bounding box area in pixels²\n"
        "- `aspect_ratio` — width-to-height ratio\n"
        "- `center` — `[cx, cy]` centre point of the bounding box\n\n"
        "Only the face **detector** is used; the recognizer and swapper do "
        "not need to be loaded for this endpoint to work."
    ),
    responses={
        200: {"description": "JSON list of face attribute objects."},
        400: {"description": "Could not read or decode the uploaded image."},
        503: {"description": "Face detector not ready."},
    },
)
async def face_attributes(
    request: Request,
    image_file: UploadFile = File(..., description="Input image to analyse."),
) -> JSONResponse:
    """Detect all faces in *image_file* and return their geometric attributes."""
    t_start = time.perf_counter()

    state = request.app.state
    detector = getattr(state, "detector", None)
    if detector is None or not getattr(detector, "is_loaded", True):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face detector is not ready.",
        )

    # Read and decode the uploaded image
    try:
        data = await image_file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded image: {exc}",
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the uploaded image — unsupported format.",
        )

    # Detect faces
    try:
        detection = detector.detect(img)
    except Exception as exc:
        logger.error(f"[attributes] Detection failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {exc}",
        )

    # Build attribute objects from FaceBox geometry
    faces: List[FaceAttributesItem] = []
    for face in detection.faces:
        cx, cy = face.center
        faces.append(
            FaceAttributesItem(
                face_index=face.face_index,
                bbox=[int(face.x1), int(face.y1), int(face.x2), int(face.y2)],
                confidence=round(float(face.confidence), 6),
                area=int(face.area),
                aspect_ratio=round(float(face.aspect_ratio), 4),
                center=[int(cx), int(cy)],
            )
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        f"[attributes] faces={detection.num_faces} "
        f"time={elapsed_ms:.1f}ms"
    )

    return JSONResponse(
        content=FaceAttributesResponse(
            num_faces=detection.num_faces,
            faces=faces,
            processing_time_ms=round(elapsed_ms, 2),
        ).model_dump()
    )
