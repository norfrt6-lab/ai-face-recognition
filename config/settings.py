from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: two levels up from this file (config/settings.py)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent


class DetectorSettings(BaseSettings):
    """YOLOv8 face detector settings."""

    model_config = SettingsConfigDict(env_prefix="DETECTOR_", extra="ignore")

    # Model path or Ultralytics hub name
    model_path: str = Field(
        default="models/yolov8n-face.pt",
        description="Path to YOLOv8 face model weights (.pt file).",
    )
    # Detection confidence threshold
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Minimum confidence score to accept a detected face.",
    )
    # IoU threshold for NMS
    iou_threshold: float = Field(
        default=0.45,
        ge=0.01,
        le=1.0,
        description="Intersection-over-Union threshold for Non-Maximum Suppression.",
    )
    # Max number of faces to detect per frame
    max_faces: int = Field(
        default=20,
        ge=1,
        description="Maximum number of faces to return per image/frame.",
    )
    # Input image size for YOLO inference
    input_size: int = Field(
        default=640,
        description="Input resolution (square) for YOLOv8 inference.",
    )
    # Device: 'cpu', 'cuda', 'cuda:0', 'mps'
    device: str = Field(
        default="cpu",
        description="Inference device: 'cpu', 'cuda', 'cuda:0', or 'mps'.",
    )


class RecognizerSettings(BaseSettings):
    """InsightFace face recognizer / analyser settings."""

    model_config = SettingsConfigDict(env_prefix="RECOGNIZER_", extra="ignore")

    # InsightFace model pack name
    model_pack: str = Field(
        default="buffalo_l",
        description="InsightFace model pack name (e.g. buffalo_l, buffalo_s).",
    )
    # Directory where InsightFace stores downloaded models
    model_root: str = Field(
        default="models",
        description="Root directory for InsightFace model downloads.",
    )
    # Cosine similarity threshold for face identity match
    similarity_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for accepting a face identity match.",
    )
    # Embedding vector dimension (ArcFace = 512)
    embedding_dim: int = Field(
        default=512,
        description="Dimensionality of the face embedding vector.",
    )
    # Detection size passed to InsightFace analyser
    det_size: tuple[int, int] = Field(
        default=(640, 640),
        description="Detection input resolution for InsightFace analyser.",
    )
    # Execution providers for InsightFace ONNX models
    providers: List[str] = Field(
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="ONNX Runtime execution providers in priority order.",
    )


class SwapperSettings(BaseSettings):
    """Face swap engine settings (inswapper_128)."""

    model_config = SettingsConfigDict(env_prefix="SWAPPER_", extra="ignore")

    # Path to inswapper ONNX model
    model_path: str = Field(
        default="models/inswapper_128.onnx",
        description="Path to inswapper_128.onnx face swap model.",
    )
    # Execution providers for ONNX Runtime
    providers: List[str] = Field(
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        description="ONNX Runtime execution providers in priority order.",
    )
    # Enable Poisson blending for seamless face merge
    enable_blending: bool = Field(
        default=True,
        description="Apply Poisson blending to smooth face boundary after swap.",
    )
    # Blend mask feather radius in pixels
    blend_feather: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Pixel radius for feathering the face blend mask.",
    )


class EnhancerSettings(BaseSettings):
    """Face enhancement / restoration settings (GFPGAN / CodeFormer)."""

    model_config = SettingsConfigDict(env_prefix="ENHANCER_", extra="ignore")

    # Which enhancer backend to use
    backend: Literal["gfpgan", "codeformer", "none"] = Field(
        default="gfpgan",
        description="Face enhancement backend: 'gfpgan', 'codeformer', or 'none'.",
    )
    # GFPGAN model path
    gfpgan_model_path: str = Field(
        default="models/GFPGANv1.4.pth",
        description="Path to GFPGAN v1.4 model weights.",
    )
    # CodeFormer model path
    codeformer_model_path: str = Field(
        default="models/codeformer.pth",
        description="Path to CodeFormer model weights.",
    )
    # Enhancement strength / fidelity weight (0.0 = max enhancement, 1.0 = original)
    fidelity_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Balance between enhancement and identity fidelity. "
            "0.0 = maximum enhancement, 1.0 = preserve original details."
        ),
    )
    # Upscale factor applied during enhancement
    upscale: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Super-resolution upscale factor during face enhancement.",
    )
    # Only enhance faces, not background
    only_center_face: bool = Field(
        default=False,
        description="When True, enhance only the most central detected face.",
    )


class APISettings(BaseSettings):
    """FastAPI server settings."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field(default="0.0.0.0", description="API server bind host.")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port.")
    workers: int = Field(default=1, ge=1, description="Number of Uvicorn worker processes.")
    reload: bool = Field(default=False, description="Enable hot-reload (development only).")
    debug: bool = Field(default=False, description="Enable debug mode.")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://127.0.0.1:8501"],
        description="Allowed CORS origins (include Streamlit UI origin).",
    )

    # File upload limits
    max_upload_size_mb: int = Field(
        default=50,
        ge=1,
        description="Maximum allowed upload file size in megabytes.",
    )
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp", "image/bmp"],
        description="Accepted MIME types for image uploads.",
    )
    allowed_video_types: List[str] = Field(
        default=["video/mp4", "video/avi", "video/quicktime", "video/x-matroska"],
        description="Accepted MIME types for video uploads.",
    )

    # API versioning prefix
    api_prefix: str = Field(default="/api/v1", description="URL prefix for all API routes.")

    # Rate limiting (requests per minute per IP)
    rate_limit_per_minute: int = Field(
        default=60,
        description="Maximum API requests per minute per client IP.",
    )

    # API key authentication (comma-separated keys; empty = auth disabled)
    api_keys: List[str] = Field(
        default_factory=list,
        description="Accepted API keys for X-API-Key header. Empty list disables auth.",
    )

    # Upload constraints
    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum upload file size in bytes (default 10 MB).",
    )
    max_image_dimension: int = Field(
        default=4096,
        description="Maximum width or height for uploaded images in pixels.",
    )
    min_image_dimension: int = Field(
        default=10,
        description="Minimum width or height for uploaded images in pixels.",
    )


class UISettings(BaseSettings):
    """Streamlit UI settings."""

    model_config = SettingsConfigDict(env_prefix="UI_", extra="ignore")

    page_title: str = Field(default="AI Face Recognition & Swap", description="Browser tab title.")
    page_icon: str = Field(default="🤖", description="Browser tab favicon emoji.")
    layout: Literal["centered", "wide"] = Field(
        default="wide", description="Streamlit page layout."
    )
    # Backend API base URL for UI → API calls
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the FastAPI backend (used by Streamlit).",
    )
    # Max preview image width in pixels
    preview_width: int = Field(
        default=800, description="Maximum display width for preview images in the UI."
    )


class StorageSettings(BaseSettings):
    """File storage settings."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")

    # Upload directory
    upload_dir: Path = Field(
        default=ROOT_DIR / "uploads",
        description="Directory for temporarily storing uploaded files.",
    )
    # Output directory for processed results
    output_dir: Path = Field(
        default=ROOT_DIR / "output",
        description="Directory for saving processed output files.",
    )
    # Cache directory for intermediate results
    cache_dir: Path = Field(
        default=ROOT_DIR / "cache",
        description="Directory for caching intermediate pipeline results.",
    )
    # Auto-cleanup uploads older than N hours (0 = disabled)
    cleanup_after_hours: int = Field(
        default=24,
        ge=0,
        description="Automatically delete files older than this many hours. 0 = disabled.",
    )

    @field_validator("upload_dir", "output_dir", "cache_dir", mode="before")
    @classmethod
    def create_dirs(cls, v: str | Path) -> Path:
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"Cannot create directory {path}: {exc}") from exc
        return path


class LoggingSettings(BaseSettings):
    """Logging settings (Loguru-based)."""

    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level to emit.",
    )
    # Log file path (None = stdout only)
    file_path: Optional[Path] = Field(
        default=ROOT_DIR / "logs" / "app.log",
        description="Path to log file. Set to null/empty to disable file logging.",
    )
    # Log rotation size
    rotation: str = Field(
        default="10 MB",
        description="Loguru rotation threshold (e.g. '10 MB', '1 day').",
    )
    # Log retention
    retention: str = Field(
        default="7 days",
        description="How long to retain rotated log files.",
    )
    # Structured JSON logs (useful for production / log aggregators)
    json_logs: bool = Field(
        default=False,
        description="Emit logs as JSON objects (for log aggregation pipelines).",
    )


class EthicsSettings(BaseSettings):
    """Ethics & safety guardrails."""

    model_config = SettingsConfigDict(env_prefix="ETHICS_", extra="ignore")

    # Require explicit consent flag on all swap API calls
    require_consent: bool = Field(
        default=True,
        description="Require a consent=true flag on all face swap requests.",
    )
    # Watermark all AI-generated output images
    watermark_output: bool = Field(
        default=True,
        description="Embed an invisible + visible watermark on swapped output.",
    )
    watermark_text: str = Field(
        default="AI GENERATED",
        description="Text to overlay on watermarked output images.",
    )
    # NSFW detection gate (not yet implemented — requires additional classifier model)
    enable_nsfw_filter: bool = Field(
        default=False,
        description="Run NSFW classifier on input before processing. Not yet implemented.",
    )


class Settings(BaseSettings):
    """
    Master settings object.

    Priority (highest → lowest):
      1. Environment variables  (e.g. DETECTOR_CONFIDENCE_THRESHOLD=0.6)
      2. .env file              (loaded from project root)
      3. Default values below
    """

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App meta
    app_name: str = Field(default="AI Face Recognition & Swap", description="Application name.")
    app_version: str = Field(default="1.0.0", description="Application version string.")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment.",
    )

    # Sub-settings (nested)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    recognizer: RecognizerSettings = Field(default_factory=RecognizerSettings)
    swapper: SwapperSettings = Field(default_factory=SwapperSettings)
    enhancer: EnhancerSettings = Field(default_factory=EnhancerSettings)
    api: APISettings = Field(default_factory=APISettings)
    ui: UISettings = Field(default_factory=UISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ethics: EthicsSettings = Field(default_factory=EthicsSettings)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def models_dir(self) -> Path:
        return ROOT_DIR / "models"


settings = Settings()
