"""POST /api/v1/anonymize — detect all faces and anonymize each one.

Supports three anonymization methods:
  - blur      : Apply Gaussian blur over each face region.
  - pixelate  : Downscale then upscale each face region (pixelation effect).
  - solid     : Fill each face region with a solid gray rectangle.

Returns the anonymized image as a PNG file download, with metadata in
response headers.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response

from api.schemas.anonymize_schemas import AnonymizeMethod
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Anonymization"])


# ---------------------------------------------------------------------------
# Anonymization helpers
# ---------------------------------------------------------------------------


def _blur_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Apply Gaussian blur in-place over the face region.

    Args:
        image: BGR numpy array (modified in-place).
        x1, y1, x2, y2: Bounding box coordinates (pixel space).
    """
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    # Kernel size must be odd and at least 1; scale with face size
    ksize = max(15, int(min(w, h) * 0.3))
    if ksize % 2 == 0:
        ksize += 1
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)


def _pixelate_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Apply pixelation in-place over the face region.

    Downscales the region to a small block size then upscales it back,
    creating a pixelated / mosaic effect.

    Args:
        image: BGR numpy array (modified in-place).
        x1, y1, x2, y2: Bounding box coordinates (pixel space).
    """
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    # Target a block size roughly 1/12 of the smaller dimension (min 4 px)
    block = max(4, min(w, h) // 12)
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated


def _solid_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Fill the face region in-place with a solid mid-gray rectangle.

    Args:
        image: BGR numpy array (modified in-place).
        x1, y1, x2, y2: Bounding box coordinates (pixel space).
    """
    image[y1:y2, x1:x2] = 128  # mid-gray in all channels


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/anonymize",
    summary="Anonymize all faces in an image",
    description=(
        "Upload an image.  The API detects **all** faces and anonymizes each "
        "face region using the selected *method*:\n\n"
        "- `blur` — Gaussian blur\n"
        "- `pixelate` — pixelation / mosaic effect\n"
        "- `solid` — solid gray fill\n\n"
        "Returns the anonymized image as a PNG file download."
    ),
    responses={
        200: {
            "description": "Anonymized PNG image.",
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
        },
        400: {"description": "Could not read or decode the uploaded image."},
        503: {"description": "Face detector not ready."},
    },
)
async def anonymize_faces(
    request: Request,
    image_file: UploadFile = File(..., description="Input image to anonymize."),
    method: AnonymizeMethod = Form(
        default=AnonymizeMethod.blur,
        description="Anonymization method: blur | pixelate | solid.",
    ),
) -> Response:
    """Detect all faces in *image_file* and anonymize each one using *method*."""
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

    # Detect all faces
    try:
        detection = detector.detect(img)
    except Exception as exc:
        logger.error(f"[anonymize] Detection failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {exc}",
        )

    output = img.copy()
    h_img, w_img = output.shape[:2]

    # Apply anonymization to each detected face
    _apply = {
        AnonymizeMethod.blur: _blur_region,
        AnonymizeMethod.pixelate: _pixelate_region,
        AnonymizeMethod.solid: _solid_region,
    }[method]

    for face in detection.faces:
        x1 = max(0, int(face.x1))
        y1 = max(0, int(face.y1))
        x2 = min(w_img, int(face.x2))
        y2 = min(h_img, int(face.y2))
        if x2 > x1 and y2 > y1:
            _apply(output, x1, y1, x2, y2)

    faces_anonymized = detection.num_faces
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        f"[anonymize] method={method.value} faces={faces_anonymized} "
        f"time={elapsed_ms:.1f}ms"
    )

    # Encode result as PNG
    ok, buf = cv2.imencode(".png", output)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode output image as PNG.",
        )

    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={
            "X-Faces-Anonymized": str(faces_anonymized),
            "X-Method": method.value,
            "X-Processing-Ms": f"{elapsed_ms:.1f}",
            "Content-Disposition": 'attachment; filename="anonymized.png"',
        },
    )
