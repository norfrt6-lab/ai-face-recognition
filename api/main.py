# ============================================================
# AI Face Recognition & Face Swap
# api/main.py
# ============================================================
# FastAPI application entry point.
#
# Responsibilities:
#   - Create and configure the FastAPI app instance
#   - Lifespan handler: load all AI models on startup,
#     release them on shutdown
#   - Register all routers under /api/v1
#   - CORS middleware
#   - Request ID middleware (X-Request-ID header)
#   - Global exception handlers (validation, HTTP, generic)
#   - Static file serving for output images
#
# Run with:
#   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers import health, recognition, swap
from api.schemas.responses import ErrorDetail, ErrorResponse
from utils.logger import get_logger, setup_from_settings

# Configure logger from settings before any other logging
setup_from_settings()

logger = get_logger(__name__)


# ============================================================
# Lifespan — model loading / teardown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    On startup:
      - Load settings from config
      - Initialise and load all AI pipeline components
      - Attach them to app.state for use in route handlers

    On shutdown:
      - Release all loaded models and free GPU memory
    """
    logger.info("=" * 60)
    logger.info("AI Face Recognition & Face Swap API — starting up")
    logger.info("=" * 60)

    # ── Load settings ────────────────────────────────────────────────
    try:
        from config.settings import settings  # noqa: PLC0415
        logger.info(
            f"Settings loaded | env={settings.environment} "
            f"v={settings.app_version}"
        )
    except Exception as exc:
        logger.warning(f"Could not load settings: {exc} — using defaults.")
        settings = None

    # ── Attach output dir to state ───────────────────────────────────
    output_dir = (
        Path(settings.storage.output_dir)
        if settings
        else Path("output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    app.state.output_dir = output_dir

    # ── Initialise detector ─────────────────────────────────────────
    app.state.detector = None
    try:
        from core.detector.yolo_detector import YOLOFaceDetector  # noqa: PLC0415

        model_path = (
            settings.detector.model_path
            if settings
            else "models/yolov8n-face.pt"
        )
        device = (
            settings.detector.device
            if settings
            else "cpu"
        )
        detector = YOLOFaceDetector(
            model_path=model_path,
            device=device,
        )
        detector.load_model()
        app.state.detector = detector
        logger.success(f"Detector loaded: {detector.model_name!r}")
    except Exception as exc:
        logger.error(f"Failed to load detector: {exc}")

    # ── Initialise recognizer ────────────────────────────────────────
    app.state.recognizer = None
    try:
        from core.recognizer.insightface_recognizer import InsightFaceRecognizer  # noqa: PLC0415

        model_pack = (
            settings.recognizer.model_pack
            if settings
            else "buffalo_l"
        )
        model_root = (
            settings.recognizer.model_root
            if settings
            else "models"
        )
        recognizer = InsightFaceRecognizer(
            model_pack=model_pack,
            model_root=model_root,
        )
        recognizer.load_model()
        app.state.recognizer = recognizer
        logger.success(f"Recognizer loaded: {recognizer.model_name!r}")
    except Exception as exc:
        logger.error(f"Failed to load recognizer: {exc}")

    # ── Initialise swapper ───────────────────────────────────────────
    app.state.swapper = None
    try:
        from core.swapper.inswapper import InSwapper  # noqa: PLC0415

        swap_model_path = (
            settings.swapper.model_path
            if settings
            else "models/inswapper_128.onnx"
        )
        providers = (
            settings.swapper.providers
            if settings
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        swapper = InSwapper(
            model_path=swap_model_path,
            providers=providers,
        )
        swapper.load_model()
        app.state.swapper = swapper
        logger.success(f"Swapper loaded: {swapper.model_name!r}")
    except Exception as exc:
        logger.error(f"Failed to load swapper: {exc}")

    # ── Initialise enhancer (optional) ───────────────────────────────
    app.state.enhancer = None
    try:
        if settings and settings.enhancer.backend != "none":
            backend = settings.enhancer.backend

            if backend == "gfpgan":
                from core.enhancer.gfpgan_enhancer import GFPGANEnhancer  # noqa: PLC0415

                enhancer = GFPGANEnhancer(
                    model_path=settings.enhancer.gfpgan_model_path,
                    upscale=settings.enhancer.upscale,
                    only_center_face=settings.enhancer.only_center_face,
                )
                enhancer.load_model()
                app.state.enhancer = enhancer
                logger.success(f"Enhancer loaded: GFPGAN ({enhancer.model_name!r})")

            elif backend == "codeformer":
                from core.enhancer.codeformer_enhancer import CodeFormerEnhancer  # noqa: PLC0415

                enhancer = CodeFormerEnhancer(
                    model_path=settings.enhancer.codeformer_model_path,
                    fidelity_weight=settings.enhancer.fidelity_weight,
                    upscale=settings.enhancer.upscale,
                )
                enhancer.load_model()
                app.state.enhancer = enhancer
                logger.success(f"Enhancer loaded: CodeFormer ({enhancer.model_name!r})")
        else:
            logger.info("Enhancer disabled (backend=none or no settings).")
    except Exception as exc:
        logger.warning(f"Failed to load enhancer (non-critical): {exc}")

    # ── Initialise face database ─────────────────────────────────────
    app.state.face_database = None
    app.state.face_database_path = None
    try:
        from core.recognizer.face_database import FaceDatabase  # noqa: PLC0415

        db_path = (
            settings.recognizer.database_path
            if settings and hasattr(settings.recognizer, "database_path")
            else "cache/face_db.pkl"
        )
        db_file = Path(db_path)
        if db_file.exists():
            face_db = FaceDatabase.load(db_path)
            logger.success(
                f"Face database loaded: {db_file} "
                f"({face_db.count} identities)"
            )
        else:
            face_db = FaceDatabase()
            logger.info(f"Face database: starting fresh (will save to {db_path})")
        app.state.face_database = face_db
        app.state.face_database_path = str(db_path)
    except Exception as exc:
        logger.warning(f"Failed to initialise face database: {exc}")

    logger.info("Startup complete — all components initialised.")
    logger.info("=" * 60)

    # ── Yield (application runs here) ───────────────────────────────
    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    logger.info("Shutting down — releasing pipeline components...")

    for attr in ("detector", "recognizer", "swapper", "enhancer"):
        obj = getattr(app.state, attr, None)
        if obj is not None:
            try:
                obj.release()
                logger.info(f"Released: {attr}")
            except Exception as exc:
                logger.warning(f"Error releasing {attr}: {exc}")

    # Save face database
    face_db = getattr(app.state, "face_database", None)
    db_path = getattr(app.state, "face_database_path", None)
    if face_db is not None and db_path is not None:
        try:
            face_db.save(db_path)
            logger.info(f"Face database saved to {db_path}.")
        except Exception as exc:
            logger.warning(f"Failed to save face database: {exc}")

    logger.info("Shutdown complete.")


# ============================================================
# App factory
# ============================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Separated into a factory function so tests can create isolated
    app instances without triggering lifespan model loading.

    Returns:
        Configured ``FastAPI`` instance.
    """
    # ── Load settings for app metadata ──────────────────────────────
    try:
        from config.settings import settings  # noqa: PLC0415
        _version     = settings.app_version
        _title       = settings.app_name
        _environment = settings.environment
        _debug       = _environment == "development"
        _api_prefix   = settings.api.api_prefix
    except Exception:
        _version      = "1.0.0"
        _title        = "AI Face Recognition & Swap"
        _debug        = False
        _api_prefix   = "/api/v1"

    app = FastAPI(
        title=_title,
        version=_version,
        description=(
            "# AI Face Recognition & Face Swap API\n\n"
            "REST API for:\n"
            "- **Face Detection** — YOLOv8 real-time bounding-box detection\n"
            "- **Face Recognition** — InsightFace ArcFace 512-dim embeddings\n"
            "- **Face Swap** — inswapper_128.onnx identity injection\n"
            "- **Face Enhancement** — GFPGAN v1.4 / CodeFormer restoration\n\n"
            "All operations require explicit `consent=true`. "
            "Do not use this technology to create non-consensual deepfakes."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        debug=_debug,
    )

    # ── Middleware (CORS + Rate Limiter + Request ID) ─────────────────
    from api.middleware.cors import configure_middleware  # noqa: PLC0415
    configure_middleware(app)

    # ── Routers ──────────────────────────────────────────────────────
    app.include_router(health.router,      prefix=_api_prefix)
    app.include_router(recognition.router, prefix=_api_prefix)
    app.include_router(swap.router,        prefix=_api_prefix)

    # ── Static files (output images) ────────────────────────────────
    try:
        from config.settings import settings as _settings  # noqa: PLC0415
        output_dir = Path(_settings.storage.output_dir)
    except Exception:
        output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        f"{_api_prefix}/results",
        StaticFiles(directory=str(output_dir)),
        name="results",
    )

    # ── Exception handlers ───────────────────────────────────────────
    _register_exception_handlers(app)

    logger.info(f"FastAPI app created | version={_version} prefix={_api_prefix}")
    return app


# ============================================================
# Exception handlers
# ============================================================

def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on *app*."""

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return a structured 422 for Pydantic / FastAPI validation errors."""
        details = []
        for error in exc.errors():
            loc   = " → ".join(str(l) for l in error.get("loc", []))
            msg   = error.get("msg", "Validation error")
            code  = error.get("type", "validation_error")
            details.append(ErrorDetail(field=loc or None, message=msg, code=code))

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                message="One or more request fields failed validation.",
                details=details,
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        """Return a structured JSON body for all HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=_status_to_error_code(exc.status_code),
                message=str(exc.detail),
                details=[],
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Catch-all handler — prevents stack traces leaking to clients."""
        logger.exception(f"Unhandled exception on {request.url}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred. Please try again later.",
                details=[],
                request_id=getattr(request.state, "request_id", None),
            ).model_dump(),
        )


def _status_to_error_code(status_code: int) -> str:
    """Map an HTTP status code to a short machine-readable error string."""
    mapping = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        408: "request_timeout",
        413: "payload_too_large",
        422: "unprocessable_entity",
        429: "too_many_requests",
        500: "internal_server_error",
        502: "bad_gateway",
        503: "service_unavailable",
        504: "gateway_timeout",
    }
    return mapping.get(status_code, f"http_{status_code}")


# ============================================================
# App instance (module-level for Uvicorn)
# ============================================================

app = create_app()


# ============================================================
# Root redirect
# ============================================================

@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect clients to the API docs."""
    return JSONResponse(
        content={
            "message": "AI Face Recognition & Swap API",
            "docs":    "/docs",
            "redoc":   "/redoc",
            "health":  "/api/v1/health",
        }
    )
