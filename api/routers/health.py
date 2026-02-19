# ============================================================
# AI Face Recognition & Face Swap
# api/routers/health.py
# ============================================================
# GET /api/v1/health — liveness + readiness check endpoint.
#
# Returns overall API status plus per-component health for
# every AI model in the pipeline (detector, recognizer,
# swapper, enhancer).
#
# Used by:
#   - Docker HEALTHCHECK (curl -f http://localhost:8000/api/v1/health)
#   - Kubernetes readiness / liveness probes
#   - docker-compose depends_on healthcheck
#   - Streamlit UI status indicator
# ============================================================

from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter, Request

from api.schemas.responses import (
    ComponentHealth,
    ComponentStatus,
    HealthResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Module-level start time for uptime calculation
_START_TIME: float = time.perf_counter()


# ============================================================
# Endpoint
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the overall API health status and per-component "
        "readiness for every AI model in the pipeline. "
        "An HTTP 200 with status='ok' or status='degraded' means "
        "the API is running. HTTP 503 means the API is down."
    ),
    responses={
        200: {"description": "API is healthy or degraded but running."},
        503: {"description": "API is down or not ready."},
    },
)
async def health_check(request: Request) -> HealthResponse:
    """
    Liveness + readiness probe.

    Inspects the ``app.state`` object for loaded pipeline components
    and reports their individual status.  The overall status is:

    - ``ok``       — all required components are loaded.
    - ``degraded`` — at least one optional component is not loaded.
    - ``down``     — a required component is missing.
    """
    uptime = (time.perf_counter() - _START_TIME)

    # ── Collect per-component health ────────────────────────────────
    components: Dict[str, ComponentHealth] = {}
    overall = ComponentStatus.OK

    state = request.app.state

    # Detector
    components["detector"] = _check_component(
        state, "detector", required=True
    )
    if components["detector"].status == ComponentStatus.DOWN:
        overall = ComponentStatus.DOWN

    # Recognizer
    components["recognizer"] = _check_component(
        state, "recognizer", required=True
    )
    if components["recognizer"].status == ComponentStatus.DOWN:
        overall = ComponentStatus.DOWN

    # Swapper
    components["swapper"] = _check_component(
        state, "swapper", required=True
    )
    if components["swapper"].status == ComponentStatus.DOWN:
        overall = ComponentStatus.DOWN

    # Enhancer (optional — degraded but not down if missing)
    components["enhancer"] = _check_component(
        state, "enhancer", required=False
    )
    if (
        overall == ComponentStatus.OK
        and components["enhancer"].status != ComponentStatus.OK
    ):
        overall = ComponentStatus.DEGRADED

    # Face database (optional)
    components["face_database"] = _check_component(
        state, "face_database", required=False
    )
    if (
        overall == ComponentStatus.OK
        and components["face_database"].status != ComponentStatus.OK
    ):
        overall = ComponentStatus.DEGRADED

    # ── App metadata ─────────────────────────────────────────────────
    try:
        from config.settings import settings  # noqa: PLC0415
        version     = settings.app_version
        environment = settings.environment
    except Exception:
        version     = "unknown"
        environment = "unknown"

    logger.debug(f"Health check: overall={overall.value} uptime={uptime:.1f}s")

    return HealthResponse(
        status=overall,
        version=version,
        environment=environment,
        uptime_seconds=uptime,
        components=components,
    )


# ============================================================
# Helpers
# ============================================================

def _check_component(
    state,
    name: str,
    required: bool,
) -> ComponentHealth:
    """
    Inspect ``state.<name>`` and return a ``ComponentHealth`` object.

    A component is considered "loaded" if:
      - The attribute exists on state AND
      - The object has an ``is_loaded`` property that returns True,
        OR the object is not None (for non-model components like the
        face database).

    Args:
        state:    FastAPI ``app.state`` object.
        name:     Attribute name to inspect.
        required: If True, an absent/unloaded component causes DOWN
                  status; if False it causes DEGRADED.

    Returns:
        ``ComponentHealth`` with status, loaded flag, and detail.
    """
    obj = getattr(state, name, None)

    if obj is None:
        status = ComponentStatus.DOWN if required else ComponentStatus.DEGRADED
        return ComponentHealth(
            status=status,
            loaded=False,
            detail=f"Component '{name}' not initialised.",
        )

    # Check is_loaded attribute if present
    is_loaded = getattr(obj, "is_loaded", None)
    if is_loaded is not None:
        loaded = bool(is_loaded)
        if not loaded:
            status = ComponentStatus.DOWN if required else ComponentStatus.DEGRADED
            return ComponentHealth(
                status=status,
                loaded=False,
                detail=f"Component '{name}' exists but model is not loaded.",
            )

    return ComponentHealth(
        status=ComponentStatus.OK,
        loaded=True,
        detail=None,
    )
