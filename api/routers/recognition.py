from __future__ import annotations

import asyncio
import math
import time
import uuid
from functools import partial
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from api.schemas.requests import (
    RecognizeRequest as RecognizeRequestSchema,
    RegisterRequest as RegisterRequestSchema,
    recognize_form_dep,
    register_form_dep,
)
from api.schemas.responses import (
    BoundingBox,
    ErrorResponse,
    FaceAttributeResponse,
    FaceMatchResponse,
    LandmarkPoint,
    RecognizeResponse,
    RecognizedFace,
    RegisterResponse,
)
from api.metrics import RECOGNITION_COUNT
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Recognition"])

_detector_breaker = CircuitBreaker("detector", failure_threshold=5, recovery_timeout=30.0)
_recognizer_breaker = CircuitBreaker("recognizer", failure_threshold=5, recovery_timeout=30.0)

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


async def _decode_upload(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image file into a BGR numpy array."""
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
        raw = await upload.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot read upload '{upload.filename}': {exc}",
        )

    if len(raw) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(raw)} bytes). Maximum: {mb:.0f} MB.",
        )

    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot decode uploaded image '{upload.filename}': {exc}",
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


def _facebox_to_bbox(face) -> BoundingBox:
    """Convert a ``FaceBox`` to a ``BoundingBox`` response model."""
    return BoundingBox(
        x1=float(face.x1),
        y1=float(face.y1),
        x2=float(face.x2),
        y2=float(face.y2),
        confidence=float(face.confidence),
    )


def _facebox_to_landmarks(face) -> Optional[List[LandmarkPoint]]:
    """Convert FaceBox landmarks to a list of LandmarkPoint models."""
    if not face.has_landmarks or face.landmarks is None:
        return None
    return [
        LandmarkPoint(x=float(pt[0]), y=float(pt[1]))
        for pt in face.landmarks
    ]


def _check_components(state, *names: str) -> None:
    """
    Raise HTTP 503 if any required pipeline component is missing or unloaded.

    Args:
        state: FastAPI app.state object.
        names: Attribute names to check (e.g. 'detector', 'recognizer').
    """
    for name in names:
        obj = getattr(state, name, None)
        if obj is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline component '{name}' is not initialised.",
            )
        is_loaded = getattr(obj, "is_loaded", None)
        if is_loaded is not None and not is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline component '{name}' model is not loaded.",
            )


@router.post(
    "/recognize",
    response_model=RecognizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect and recognize faces",
    description=(
        "Upload an image to detect all faces and match each one against "
        "the registered face database. "
        "Returns identity matches, bounding boxes, landmarks, and "
        "optional demographic attributes per face.\n\n"
        "**Requires explicit consent** (`consent=true`)."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or request."},
        422: {"description": "Validation error."},
        503: {"model": ErrorResponse, "description": "Pipeline component not ready."},
    },
)
async def recognize(
    request:     Request,
    image:       UploadFile                = File(..., description="Image file (JPEG / PNG / WebP / BMP)."),
    params:      RecognizeRequestSchema    = Depends(recognize_form_dep),
) -> RecognizeResponse:
    """
    Detect all faces in *image* and attempt to match each one against
    the face database.

    Multi-part form fields
    ----------------------
    - **image**               JPEG / PNG / WebP / BMP file upload.
    - **top_k**               Max candidate matches per face (default 1).
    - **similarity_threshold** Override server threshold (optional).
    - **return_attributes**   Include age/gender predictions (default false).
    - **return_embeddings**   Include raw embedding vectors (default false).
    - **consent**             Must be ``true``.
    """
    t_start = time.perf_counter()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id[:8]}] POST /recognize — file={image.filename!r}")

    # Unpack validated params
    top_k = params.top_k
    similarity_threshold = params.similarity_threshold
    return_attributes = params.return_attributes

    _check_components(request.app.state, "detector", "recognizer")

    state      = request.app.state
    detector   = state.detector
    recognizer = state.recognizer
    face_db    = getattr(state, "face_database", None)

    img = await _decode_upload(image)
    h, w = img.shape[:2]

    loop = asyncio.get_running_loop()
    executor = getattr(state, "executor", None)

    try:
        _detector_breaker.check()
        detection = await asyncio.wait_for(
            loop.run_in_executor(executor, detector.detect, img),
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
        logger.error(f"[{request_id[:8]}] Detection timed out after {_INFERENCE_TIMEOUT}s")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Face detection timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        _detector_breaker.record_failure()
        logger.error(f"[{request_id[:8]}] Detection error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {exc}",
        )

    if detection.is_empty:
        return RecognizeResponse(
            num_faces_detected=0,
            num_faces_recognized=0,
            faces=[],
            inference_time_ms=(time.perf_counter() - t_start) * 1000,
            image_width=w,
            image_height=h,
        )

    if similarity_threshold is not None:
        thresh = similarity_threshold
    else:
        try:
            from config.settings import settings  # noqa: PLC0415
            thresh = settings.recognizer.similarity_threshold
        except Exception:
            thresh = 0.45

    recognized_faces: List[RecognizedFace] = []
    num_recognized = 0

    for face in detection.faces:
        # Extract embedding
        try:
            _recognizer_breaker.check()
            face_bbox = (int(face.x1), int(face.y1), int(face.x2), int(face.y2))
            embedding = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(recognizer.get_embedding, img, bbox=face_bbox),
                ),
                timeout=_INFERENCE_TIMEOUT,
            )
            _recognizer_breaker.record_success()
        except CircuitOpenError:
            embedding = None
        except asyncio.TimeoutError:
            _recognizer_breaker.record_failure()
            logger.warning(
                f"[{request_id[:8]}] Embedding timed out for face {face.face_index}"
            )
            embedding = None
        except Exception as exc:
            _recognizer_breaker.record_failure()
            logger.warning(
                f"[{request_id[:8]}] Embedding failed for face {face.face_index}: {exc}"
            )
            embedding = None

        # Attribute extraction
        attributes: Optional[FaceAttributeResponse] = None
        if return_attributes and embedding is not None:
            try:
                attr = await asyncio.wait_for(
                    loop.run_in_executor(
                        executor,
                        partial(recognizer.get_attributes, img, bbox=face_bbox),
                    ),
                    timeout=_INFERENCE_TIMEOUT,
                )
                if attr:
                    attributes = FaceAttributeResponse(
                        age=attr.age,
                        gender=attr.gender,
                        gender_score=attr.gender_score,
                    )
            except Exception:
                pass

        # Database search
        match_response: FaceMatchResponse
        if embedding is not None and face_db is not None:
            try:
                match = face_db.search(embedding, threshold=thresh)
                if match.is_known:
                    num_recognized += 1
                    # Look up FaceIdentity to get the UUID
                    identity_obj = face_db.get_identity(match.identity)
                    match_response = FaceMatchResponse(
                        identity_name=match.identity,
                        identity_id=identity_obj.identity_id if identity_obj else None,
                        similarity=float(match.similarity),
                        is_known=True,
                        threshold_used=thresh,
                    )
                else:
                    match_response = FaceMatchResponse(
                        identity_name=None,
                        identity_id=None,
                        similarity=float(match.similarity),
                        is_known=False,
                        threshold_used=thresh,
                    )
            except Exception as exc:
                logger.warning(
                    f"[{request_id[:8]}] DB search error for face {face.face_index}: {exc}"
                )
                match_response = FaceMatchResponse(
                    identity_name=None,
                    identity_id=None,
                    similarity=0.0,
                    is_known=False,
                    threshold_used=thresh,
                )
        else:
            # No database or embedding failed — return unknown
            match_response = FaceMatchResponse(
                identity_name=None,
                identity_id=None,
                similarity=0.0,
                is_known=False,
                threshold_used=thresh,
            )

        recognized_faces.append(
            RecognizedFace(
                face_index=face.face_index,
                bbox=_facebox_to_bbox(face),
                landmarks=_facebox_to_landmarks(face),
                attributes=attributes,
                match=match_response,
                embedding_norm=(
                    float(np.linalg.norm(embedding.vector))
                    if embedding is not None
                    else None
                ),
            )
        )

    RECOGNITION_COUNT.inc()
    inference_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[{request_id[:8]}] recognize done | "
        f"detected={len(detection.faces)} recognized={num_recognized} "
        f"time={inference_ms:.1f}ms"
    )

    return RecognizeResponse(
        num_faces_detected=len(detection.faces),
        num_faces_recognized=num_recognized,
        faces=recognized_faces,
        inference_time_ms=inference_ms,
        image_width=w,
        image_height=h,
    )


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new face identity",
    description=(
        "Upload an image and a name to register a new face identity "
        "in the face database, or add an embedding to an existing identity. "
        "\n\n**Requires explicit consent** (`consent=true`)."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or no face detected."},
        422: {"description": "Validation error."},
        503: {"model": ErrorResponse, "description": "Pipeline component not ready."},
    },
)
async def register(
    request:     Request,
    image:       UploadFile              = File(..., description="Face image (JPEG / PNG / WebP / BMP)."),
    params:      RegisterRequestSchema   = Depends(register_form_dep),
) -> RegisterResponse:
    """
    Register a face identity in the database.

    Multi-part form fields
    ----------------------
    - **image**        Face image upload.
    - **name**         Human-readable identity name (e.g. "Alice").
    - **identity_id**  Optional existing UUID to append to.
    - **overwrite**    Replace stored embeddings if True.
    - **consent**      Must be ``true``.
    """
    t_start    = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Unpack validated params
    name        = params.name
    identity_id = params.identity_id
    overwrite   = params.overwrite

    logger.info(
        f"[{request_id[:8]}] POST /register — "
        f"name={name!r} identity_id={identity_id!r}"
    )

    _check_components(request.app.state, "detector", "recognizer")

    state      = request.app.state
    detector   = state.detector
    recognizer = state.recognizer
    face_db    = getattr(state, "face_database", None)

    if face_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face database is not initialised.",
        )

    img = await _decode_upload(image)

    loop = asyncio.get_running_loop()
    executor = getattr(state, "executor", None)

    try:
        _detector_breaker.check()
        detection = await asyncio.wait_for(
            loop.run_in_executor(executor, detector.detect, img),
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
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Face detection timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        _detector_breaker.record_failure()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face detection failed: {exc}",
        )

    if detection.is_empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the uploaded image. Please upload a clear face photo.",
        )

    # Use the best (highest confidence) face for registration
    best_face = detection.best_face

    try:
        _recognizer_breaker.check()
        best_bbox = (int(best_face.x1), int(best_face.y1), int(best_face.x2), int(best_face.y2))
        embedding = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(recognizer.get_embedding, img, bbox=best_bbox),
            ),
            timeout=_INFERENCE_TIMEOUT,
        )
        _recognizer_breaker.record_success()
    except CircuitOpenError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except asyncio.TimeoutError:
        _recognizer_breaker.record_failure()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Embedding extraction timed out after {_INFERENCE_TIMEOUT:.0f}s.",
        )
    except Exception as exc:
        _recognizer_breaker.record_failure()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract face embedding: {exc}",
        )

    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Could not extract a face embedding from the detected face. "
                "Try a higher-quality or more front-facing photo."
            ),
        )

    try:
        if identity_id and not overwrite:
            # Append embedding to existing identity (looked up by name)
            existing = face_db.get_identity(name)
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Identity '{name}' not found in database.",
                )
            result = face_db.register(name=name, embedding=embedding)
            result_id = result.identity_id
            total_emb = result.num_embeddings
            added = 1
            msg = f"Identity '{name}' updated with 1 new embedding."
        elif identity_id and overwrite:
            # Replace all embeddings for an existing identity
            result = face_db.register(
                name=name, embedding=embedding, overwrite=True,
            )
            result_id = result.identity_id
            total_emb = result.num_embeddings
            added = 1
            msg = f"Identity '{name}' embeddings replaced (overwrite=true)."
        else:
            # Create new identity (or append if name already exists)
            result = face_db.register(name=name, embedding=embedding)
            result_id = result.identity_id
            total_emb = result.num_embeddings
            added = 1
            msg = f"New identity '{name}' registered successfully."
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[{request_id[:8]}] DB register error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register identity in database: {exc}",
        )

    inference_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"[{request_id[:8]}] register done | "
        f"id={result_id!r} name={name!r} "
        f"time={inference_ms:.1f}ms"
    )

    return RegisterResponse(
        identity_id=result_id,
        identity_name=name,
        embeddings_added=added,
        total_embeddings=total_emb,
        faces_detected=len(detection.faces),
        message=msg,
    )


@router.get(
    "/identities",
    summary="List registered identities",
    description="Return a paginated list of all identities in the face database.",
    responses={
        503: {"model": ErrorResponse, "description": "Face database not ready."},
    },
)
async def list_identities(
    request:     Request,
    page:        int            = 1,
    page_size:   int            = 50,
    name_filter: Optional[str]  = None,
) -> dict:
    # Validate pagination bounds
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    """
    Return a paginated list of registered face identities.

    Query parameters
    ----------------
    - **page**        Page number (1-based, default 1).
    - **page_size**   Items per page (default 50, max 200).
    - **name_filter** Optional case-insensitive substring filter on name.
    """
    state   = request.app.state
    face_db = getattr(state, "face_database", None)

    if face_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face database is not initialised.",
        )

    try:
        all_names = face_db.list_identities()  # returns List[str]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list identities: {exc}",
        )

    # Apply name filter (items are strings)
    if name_filter:
        nf = name_filter.lower()
        all_names = [n for n in all_names if nf in n.lower()]

    total = len(all_names)
    start = (page - 1) * page_size
    end   = start + page_size
    page_names = all_names[start:end]

    # Build rich response items from FaceIdentity objects
    items = []
    for n in page_names:
        identity = face_db.get_identity(n)
        if identity:
            items.append({
                "name":           identity.name,
                "identity_id":    identity.identity_id,
                "num_embeddings": identity.num_embeddings,
                "created_at":     identity.created_at,
                "updated_at":     identity.updated_at,
            })

    return {
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": math.ceil(total / page_size) if page_size > 0 else 0,
        "items":       items,
    }


@router.get(
    "/identities/{identity_id}",
    summary="Get a single registered identity",
    responses={
        404: {"model": ErrorResponse, "description": "Identity not found."},
        503: {"model": ErrorResponse, "description": "Face database not ready."},
    },
)
async def get_identity(request: Request, identity_id: str) -> dict:
    """Return metadata for a single registered identity by UUID."""
    face_db = getattr(request.app.state, "face_database", None)

    if face_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face database is not initialised.",
        )

    try:
        identity = face_db.get_identity(identity_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve identity: {exc}",
        )

    if identity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Identity '{identity_id}' not found.",
        )

    return {
        "identity_id":     identity.identity_id,
        "name":            identity.name,
        "num_embeddings":  identity.num_embeddings,
        "created_at":      identity.created_at,
        "updated_at":      identity.updated_at,
        "metadata":        identity.metadata,
    }


@router.delete(
    "/identities/{identity_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a registered identity",
    responses={
        404: {"model": ErrorResponse, "description": "Identity not found."},
        503: {"model": ErrorResponse, "description": "Face database not ready."},
    },
)
async def delete_identity(
    request:     Request,
    identity_id: str,
    confirm:     bool = Form(default=False),
) -> dict:
    """
    Remove an identity and all its embeddings from the face database.

    Form fields
    -----------
    - **confirm**  Must be ``true`` to prevent accidental deletion.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="confirm must be true to delete an identity.",
        )

    face_db = getattr(request.app.state, "face_database", None)
    if face_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face database is not initialised.",
        )

    try:
        removed = face_db.remove(identity_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete identity: {exc}",
        )

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Identity '{identity_id}' not found.",
        )

    logger.info(f"Identity deleted: {identity_id!r}")
    return {"deleted": True, "identity_id": identity_id}


@router.patch(
    "/identities/{identity_id}",
    status_code=status.HTTP_200_OK,
    summary="Rename a registered identity",
    responses={
        404: {"model": ErrorResponse, "description": "Identity not found."},
        503: {"model": ErrorResponse, "description": "Face database not ready."},
    },
)
async def rename_identity(
    request:     Request,
    identity_id: str,
    new_name:    str = Form(..., min_length=1, max_length=128),
) -> dict:
    """
    Rename an existing identity in the face database.

    Form fields
    -----------
    - **new_name**  New label for the identity.
    """
    if "/" in new_name or "\\" in new_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="new_name must not contain path separators.",
        )

    face_db = getattr(request.app.state, "face_database", None)
    if face_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face database is not initialised.",
        )

    try:
        renamed = face_db.rename(identity_id, new_name)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rename identity: {exc}",
        )

    if not renamed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Identity '{identity_id}' not found.",
        )

    logger.info(f"Identity {identity_id!r} renamed to {new_name!r}")
    return {"renamed": True, "identity_id": identity_id, "new_name": new_name}
