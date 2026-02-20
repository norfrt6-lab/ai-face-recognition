# Pydantic v2 response models for all FastAPI endpoints.
#
# These models define the exact JSON structure returned by:
#   GET  /api/v1/health
#   POST /api/v1/recognize
#   POST /api/v1/register
#   POST /api/v1/swap

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned face bounding box in pixel coordinates."""

    x1: float = Field(..., description="Left edge (pixels).")
    y1: float = Field(..., description="Top edge (pixels).")
    x2: float = Field(..., description="Right edge (pixels).")
    y2: float = Field(..., description="Bottom edge (pixels).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0, 1].")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    model_config = {"frozen": True}


class LandmarkPoint(BaseModel):
    """A single 2-D facial landmark point."""

    x: float = Field(..., description="X coordinate (pixels).")
    y: float = Field(..., description="Y coordinate (pixels).")

    model_config = {"frozen": True}


class FaceAttributeResponse(BaseModel):
    """Optional demographic attributes predicted for a face."""

    age: Optional[float] = Field(None, description="Estimated age in years.")
    gender: Optional[str] = Field(None, description="'M' or 'F'.")
    gender_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Gender prediction confidence [0, 1]."
    )


class DetectedFace(BaseModel):
    """A single face detection result."""

    face_index: int = Field(..., description="Zero-based index of this face in the image.")
    bbox: BoundingBox = Field(..., description="Bounding box in pixel coordinates.")
    landmarks: Optional[List[LandmarkPoint]] = Field(
        None,
        description="5-point facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth].",
    )
    attributes: Optional[FaceAttributeResponse] = Field(
        None, description="Predicted demographic attributes (if requested)."
    )


class ComponentStatus(str, Enum):
    """Status of an individual system component."""

    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status for a single pipeline component."""

    status: ComponentStatus = Field(..., description="Component health status.")
    loaded: bool = Field(..., description="Whether the model/component is loaded.")
    detail: Optional[str] = Field(None, description="Extra info or error message.")


class HealthResponse(BaseModel):
    """
    Response for GET /api/v1/health.

    Returns overall API status plus per-component health checks
    for every AI model in the pipeline.
    """

    status: ComponentStatus = Field(
        ..., description="Overall API health: 'ok' | 'degraded' | 'down'."
    )
    version: str = Field(..., description="Application version string.")
    environment: str = Field(..., description="Deployment environment (development / production).")
    uptime_seconds: float = Field(..., description="Seconds since the API process started.")
    components: Dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Per-component health map keyed by component name.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "environment": "development",
                "uptime_seconds": 42.3,
                "components": {
                    "detector": {"status": "ok", "loaded": True, "detail": None},
                    "recognizer": {"status": "ok", "loaded": True, "detail": None},
                    "swapper": {"status": "ok", "loaded": True, "detail": None},
                    "enhancer": {"status": "ok", "loaded": False, "detail": "disabled"},
                },
            }
        }
    }


class FaceMatchResponse(BaseModel):
    """A single face-to-identity match result."""

    identity_name: Optional[str] = Field(
        None, description="Name of the matched identity, or null if unknown."
    )
    identity_id: Optional[str] = Field(
        None, description="UUID of the matched identity, or null if unknown."
    )
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score [0, 1].")
    is_known: bool = Field(
        ..., description="True if similarity exceeded the recognition threshold."
    )
    threshold_used: float = Field(..., description="Cosine similarity threshold that was applied.")


class RecognizedFace(BaseModel):
    """Full recognition result for one detected face."""

    face_index: int = Field(..., description="Index of this face in the source image.")
    bbox: BoundingBox = Field(..., description="Face bounding box.")
    landmarks: Optional[List[LandmarkPoint]] = Field(None, description="5-point landmarks.")
    attributes: Optional[FaceAttributeResponse] = Field(None, description="Age / gender.")
    match: FaceMatchResponse = Field(..., description="Identity match result.")
    embedding_norm: Optional[float] = Field(
        None, description="L2 norm of the embedding (quality indicator)."
    )


class RecognizeResponse(BaseModel):
    """
    Response for POST /api/v1/recognize.

    Contains detection + recognition results for every face
    found in the uploaded image.
    """

    num_faces_detected: int = Field(..., description="Total faces found by the detector.")
    num_faces_recognized: int = Field(..., description="Faces matched to a known identity.")
    faces: List[RecognizedFace] = Field(
        default_factory=list, description="Per-face recognition results."
    )
    inference_time_ms: float = Field(
        ..., description="Total server-side inference time in milliseconds."
    )
    image_width: int = Field(..., description="Width of the uploaded image in pixels.")
    image_height: int = Field(..., description="Height of the uploaded image in pixels.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "num_faces_detected": 1,
                "num_faces_recognized": 1,
                "faces": [
                    {
                        "face_index": 0,
                        "bbox": {"x1": 100, "y1": 80, "x2": 300, "y2": 320, "confidence": 0.97},
                        "landmarks": None,
                        "attributes": {"age": 28.5, "gender": "F", "gender_score": 0.92},
                        "match": {
                            "identity_name": "Alice",
                            "identity_id": "uuid-1234",
                            "similarity": 0.87,
                            "is_known": True,
                            "threshold_used": 0.45,
                        },
                        "embedding_norm": 1.0,
                    }
                ],
                "inference_time_ms": 34.2,
                "image_width": 640,
                "image_height": 480,
            }
        }
    }


class RegisterResponse(BaseModel):
    """
    Response for POST /api/v1/register.

    Confirms that a new identity (or additional embeddings for an
    existing identity) was saved to the face database.
    """

    identity_id: str = Field(..., description="UUID of the registered (or updated) identity.")
    identity_name: str = Field(..., description="Name label stored for this identity.")
    embeddings_added: int = Field(
        ..., description="Number of face embeddings that were added in this request."
    )
    total_embeddings: int = Field(
        ..., description="Total embeddings stored for this identity after the update."
    )
    faces_detected: int = Field(..., description="Number of faces detected in the uploaded image.")
    message: str = Field(..., description="Human-readable status message.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "identity_id": "a1b2c3d4-1234-5678-abcd-ef0123456789",
                "identity_name": "Alice",
                "embeddings_added": 1,
                "total_embeddings": 3,
                "faces_detected": 1,
                "message": "Identity 'Alice' updated with 1 new embedding(s).",
            }
        }
    }


class SwapTimingBreakdown(BaseModel):
    """Detailed timing breakdown for a single face swap operation."""

    align_ms: float = Field(..., description="Face alignment time (ms).")
    inference_ms: float = Field(..., description="ONNX Runtime inference time (ms).")
    blend_ms: float = Field(..., description="Paste-back / blending time (ms).")
    total_ms: float = Field(..., description="Total wall-clock time for this swap (ms).")


class SwappedFaceInfo(BaseModel):
    """Per-face swap result summary."""

    face_index: int = Field(..., description="Index of the target face that was swapped.")
    bbox: BoundingBox = Field(..., description="Bounding box of the swapped face.")
    success: bool = Field(..., description="Whether this face was swapped successfully.")
    status: str = Field(..., description="Swap status string (e.g. 'success', 'align_error').")
    timing: Optional[SwapTimingBreakdown] = Field(
        None, description="Timing breakdown for this face swap."
    )
    error: Optional[str] = Field(None, description="Error description if success=False, else null.")


class SwapResponse(BaseModel):
    """
    Response for POST /api/v1/swap.

    Contains the URL / base64 of the swapped output image,
    per-face swap results, and timing metadata.
    """

    output_url: Optional[str] = Field(
        None,
        description=(
            "URL to download the swapped output image. "
            "Present when the server saves results to disk."
        ),
    )
    output_base64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded PNG of the swapped output image. "
            "Present when return_base64=true was requested."
        ),
    )
    num_faces_swapped: int = Field(..., description="Number of faces successfully swapped.")
    num_faces_failed: int = Field(..., description="Number of faces that failed to swap.")
    faces: List[SwappedFaceInfo] = Field(
        default_factory=list, description="Per-face swap result details."
    )
    total_inference_ms: float = Field(
        ..., description="Total server-side processing time in milliseconds."
    )
    blend_mode: str = Field(
        ..., description="Blend mode used: 'poisson' | 'alpha' | 'masked_alpha'."
    )
    enhanced: bool = Field(
        ..., description="True if GFPGAN / CodeFormer post-processing was applied."
    )
    watermarked: bool = Field(
        ..., description="True if an AI-generated watermark was embedded in the output."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "output_url": "/api/v1/results/swap_abc123.png",
                "output_base64": None,
                "num_faces_swapped": 1,
                "num_faces_failed": 0,
                "faces": [
                    {
                        "face_index": 0,
                        "bbox": {"x1": 100, "y1": 80, "x2": 300, "y2": 320, "confidence": 0.96},
                        "success": True,
                        "status": "success",
                        "timing": {
                            "align_ms": 2.1,
                            "inference_ms": 18.4,
                            "blend_ms": 3.7,
                            "total_ms": 24.2,
                        },
                        "error": None,
                    }
                ],
                "total_inference_ms": 24.2,
                "blend_mode": "poisson",
                "enhanced": False,
                "watermarked": True,
            }
        }
    }


class ErrorDetail(BaseModel):
    """A single structured error detail."""

    field: Optional[str] = Field(None, description="Field name the error relates to (if any).")
    message: str = Field(..., description="Human-readable error description.")
    code: Optional[str] = Field(None, description="Machine-readable error code.")


class ErrorResponse(BaseModel):
    """
    Standardised error envelope returned for all 4xx / 5xx responses.

    All API errors use this shape so clients can handle them uniformly.
    """

    error: str = Field(..., description="Short error category (e.g. 'validation_error').")
    message: str = Field(..., description="Human-readable description of the error.")
    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Optional list of per-field or per-item error details.",
    )
    request_id: Optional[str] = Field(
        None, description="Unique request ID for tracing (from X-Request-ID header)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "no_face_detected",
                "message": "No face was detected in the uploaded image.",
                "details": [],
                "request_id": "req_abc123",
            }
        }
    }


__all__ = [
    # Shared
    "BoundingBox",
    "LandmarkPoint",
    "FaceAttributeResponse",
    "DetectedFace",
    # Health
    "ComponentStatus",
    "ComponentHealth",
    "HealthResponse",
    # Recognition
    "FaceMatchResponse",
    "RecognizedFace",
    "RecognizeResponse",
    # Register
    "RegisterResponse",
    # Swap
    "SwapTimingBreakdown",
    "SwappedFaceInfo",
    "SwapResponse",
    # Errors
    "ErrorDetail",
    "ErrorResponse",
]
