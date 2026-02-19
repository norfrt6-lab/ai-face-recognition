# ============================================================
# api/schemas/__init__.py
# API Schema Package — re-exports all request/response models
# ============================================================

from api.schemas.requests import (
    BaseAPIRequest,
    BlendModeSchema,
    EnhancerBackendSchema,
    RecognizeRequest,
    RegisterRequest,
    SwapRequest,
    DeleteIdentityRequest,
    RenameIdentityRequest,
    ListIdentitiesRequest,
)

from api.schemas.responses import (
    BoundingBox,
    LandmarkPoint,
    FaceAttributeResponse,
    DetectedFace,
    ComponentStatus,
    ComponentHealth,
    HealthResponse,
    FaceMatchResponse,
    RecognizedFace,
    RecognizeResponse,
    RegisterResponse,
    SwapTimingBreakdown,
    SwappedFaceInfo,
    SwapResponse,
    ErrorDetail,
    ErrorResponse,
)

__all__ = [
    # ── Enums ──────────────────────────────────────────────
    "BlendModeSchema",
    "EnhancerBackendSchema",
    "ComponentStatus",
    # ── Request models ─────────────────────────────────────
    "BaseAPIRequest",
    "RecognizeRequest",
    "RegisterRequest",
    "SwapRequest",
    "DeleteIdentityRequest",
    "RenameIdentityRequest",
    "ListIdentitiesRequest",
    # ── Shared response primitives ─────────────────────────
    "BoundingBox",
    "LandmarkPoint",
    "FaceAttributeResponse",
    "DetectedFace",
    # ── Health ─────────────────────────────────────────────
    "ComponentHealth",
    "HealthResponse",
    # ── Recognition ────────────────────────────────────────
    "FaceMatchResponse",
    "RecognizedFace",
    "RecognizeResponse",
    # ── Registration ───────────────────────────────────────
    "RegisterResponse",
    # ── Swap ───────────────────────────────────────────────
    "SwapTimingBreakdown",
    "SwappedFaceInfo",
    "SwapResponse",
    # ── Errors ─────────────────────────────────────────────
    "ErrorDetail",
    "ErrorResponse",
]
