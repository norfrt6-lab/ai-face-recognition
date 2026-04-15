"""POST /api/v1/similarity — compare two face images and return cosine similarity.

Accepts two uploaded images (face_a, face_b), detects the primary face in
each, extracts ArcFace embeddings via the loaded recognizer, computes cosine
similarity, and returns a JSON response indicating whether the two images
depict the same person.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from api.schemas.similarity_schemas import SimilarityResponse
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Face Similarity"])

_detector_breaker = CircuitBreaker("similarity_detector", failure_threshold=5, recovery_timeout=30.0)

# Cosine similarity threshold — faces with score >= this are considered the
# same person.  Matches the typical ArcFace operating point.
_DEFAULT_THRESHOLD: float = 0.6

_INFERENCE_TIMEOUT: float = 60.0


async def _read_image(upload: UploadFile, label: str) -> np.ndarray:
    """Read an UploadFile and decode it to a BGR numpy array.

    Args:
        upload: The uploaded file object.
        label:  Human-readable name for error messages ('face_a' / 'face_b').

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
            detail=f"Could not read upload '{label}': {exc}",
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image '{label}' — unsupported format.",
        )
    return img


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [0, 1] (clipped from [-1, 1]).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(a, b) / (norm_a * norm_b))
    # Clip to [0, 1] — ArcFace embeddings can produce slightly negative
    # cosine similarities for very different faces; we clamp for clarity.
    return float(np.clip(raw, 0.0, 1.0))


def _detect_and_embed(image: np.ndarray, state, label: str):
    """Detect the primary face in *image* and return its embedding.

    Args:
        image: BGR numpy array.
        state: FastAPI app.state with detector and recognizer.
        label: Name used for log messages.

    Returns:
        (detected: bool, embedding: np.ndarray | None)
    """
    try:
        _detector_breaker.check()
        detection = state.detector.detect(image)
        _detector_breaker.record_success()
    except CircuitOpenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        _detector_breaker.record_failure()
        logger.warning(f"[similarity] Detection failed for {label}: {exc}")
        return False, None

    if detection.is_empty:
        return False, None

    face = detection.faces[0]
    bbox = (int(face.x1), int(face.y1), int(face.x2), int(face.y2))
    try:
        embedding = state.recognizer.get_embedding(image, bbox=bbox)
    except Exception as exc:
        logger.warning(f"[similarity] Embedding extraction failed for {label}: {exc}")
        return True, None

    return True, embedding


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    summary="Face similarity comparison",
    description=(
        "Upload two images (**face_a** and **face_b**).  The API detects the "
        "primary face in each image, extracts a 512-dimensional ArcFace "
        "embedding, computes cosine similarity, and returns whether the two "
        "images depict the same person.\n\n"
        "Both images must contain at least one detectable face."
    ),
    responses={
        400: {"description": "No face detected in one or both images."},
        503: {"description": "Pipeline components (detector / recognizer) not ready."},
    },
)
async def face_similarity(
    request: Request,
    face_a: UploadFile = File(..., description="First face image."),
    face_b: UploadFile = File(..., description="Second face image."),
) -> JSONResponse:
    """Compare two face images and return their cosine similarity score."""
    t_start = time.perf_counter()

    state = request.app.state
    for component in ("detector", "recognizer"):
        obj = getattr(state, component, None)
        if obj is None or not getattr(obj, "is_loaded", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline component '{component}' is not ready.",
            )

    img_a = await _read_image(face_a, "face_a")
    img_b = await _read_image(face_b, "face_b")

    detected_a, emb_a = _detect_and_embed(img_a, state, "face_a")
    detected_b, emb_b = _detect_and_embed(img_b, state, "face_b")

    if not detected_a or emb_a is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected or embedding could not be extracted from face_a.",
        )
    if not detected_b or emb_b is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected or embedding could not be extracted from face_b.",
        )

    similarity = _cosine_similarity(emb_a, emb_b)
    is_same = similarity >= _DEFAULT_THRESHOLD
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    logger.info(
        f"[similarity] similarity={similarity:.4f} "
        f"same={is_same} "
        f"time={elapsed_ms:.1f}ms"
    )

    return JSONResponse(
        content=SimilarityResponse(
            similarity=round(similarity, 6),
            is_same_person=is_same,
            threshold=_DEFAULT_THRESHOLD,
            face_a_detected=detected_a,
            face_b_detected=detected_b,
            processing_time_ms=round(elapsed_ms, 2),
        ).model_dump()
    )
