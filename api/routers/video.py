"""POST /api/v1/swap/video — video face swap endpoint.

Accepts a source image and a target video file as multipart uploads plus
processing parameters as form fields.  Saves uploads to a temporary directory,
runs the VideoPipeline, and streams the output video back as a file download.

Enforces a maximum video upload size (default 50 MB, configurable via
settings or the VIDEO_MAX_UPLOAD_MB environment variable).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Video Face Swap"])

# Default maximum video file size (bytes).  Can be overridden via settings or
# the VIDEO_MAX_UPLOAD_MB environment variable.
_DEFAULT_MAX_VIDEO_MB: int = 50
_BYTES_PER_MB: int = 1024 * 1024


def _get_max_video_bytes() -> int:
    """Return the maximum allowed video upload size in bytes."""
    try:
        mb = int(os.environ.get("VIDEO_MAX_UPLOAD_MB", _DEFAULT_MAX_VIDEO_MB))
    except (ValueError, TypeError):
        mb = _DEFAULT_MAX_VIDEO_MB
    return mb * _BYTES_PER_MB


@router.post(
    "/swap/video",
    summary="Video face swap",
    description=(
        "Upload a **source** image (donor identity) and a **target** video "
        "file.  The API extracts the source face embedding and applies it to "
        "every frame of the video.\n\n"
        "Processing parameters (all optional form fields):\n"
        "- `skip_frames` — process every Nth frame (1 = every frame, default 1)\n"
        "- `enhance` — apply face enhancement on each processed frame\n"
        "- `preserve_audio` — keep the original audio track (default true)\n"
        "- `swap_all` — swap all detected faces per frame (default false)\n\n"
        f"Maximum video upload size: {_DEFAULT_MAX_VIDEO_MB} MB."
    ),
    response_class=FileResponse,
    responses={
        200: {
            "description": "Processed video file download.",
            "content": {"video/mp4": {}},
        },
        400: {"description": "Invalid source image or video file."},
        413: {"description": "Video file exceeds the size limit."},
        503: {"description": "Pipeline components not ready."},
    },
)
async def video_swap(
    request: Request,
    source_file: UploadFile = File(..., description="Source image — the donor identity."),
    target_video: UploadFile = File(..., description="Target video file to process."),
    skip_frames: int = Form(default=1, ge=1, description="Process every Nth frame."),
    enhance: bool = Form(default=False, description="Apply face enhancement."),
    preserve_audio: bool = Form(default=True, description="Preserve original audio track."),
    swap_all: bool = Form(default=False, description="Swap all faces per frame."),
) -> FileResponse:
    """Swap the source identity into every frame of the target video."""
    t_start = time.perf_counter()
    max_bytes = _get_max_video_bytes()

    state = request.app.state
    for component in ("detector", "recognizer", "swapper"):
        obj = getattr(state, component, None)
        if obj is None or not getattr(obj, "is_loaded", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Pipeline component '{component}' is not ready.",
            )

    tmp_dir = tempfile.mkdtemp(prefix="face_swap_video_")
    source_path = Path(tmp_dir) / "source.jpg"
    video_path = Path(tmp_dir) / "target.mp4"
    output_path = Path(tmp_dir) / "output.mp4"

    try:
        # --- Read and validate source image ---
        try:
            src_data = await source_file.read()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read source image: {exc}",
            )
        src_arr = np.frombuffer(src_data, dtype=np.uint8)
        src_img = cv2.imdecode(src_arr, cv2.IMREAD_COLOR)
        if src_img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not decode source image — unsupported format.",
            )

        # --- Read and validate target video ---
        # Pre-check Content-Length if available
        claimed_size = target_video.size
        if claimed_size is not None and claimed_size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Video file too large ({claimed_size} bytes). "
                    f"Maximum: {max_bytes // _BYTES_PER_MB} MB."
                ),
            )
        try:
            video_data = await target_video.read()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not read target video: {exc}",
            )
        if len(video_data) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"Video file too large ({len(video_data)} bytes). "
                    f"Maximum: {max_bytes // _BYTES_PER_MB} MB."
                ),
            )

        # Write uploads to temp files
        cv2.imwrite(str(source_path), src_img)
        video_path.write_bytes(video_data)

        # --- Detect + embed source face ---
        src_detection = state.detector.detect(src_img)
        if src_detection.is_empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the source image.",
            )
        src_face = src_detection.faces[0]
        src_bbox = (int(src_face.x1), int(src_face.y1), int(src_face.x2), int(src_face.y2))
        source_embedding = state.recognizer.get_embedding(src_img, bbox=src_bbox)
        if source_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract a valid embedding from the source face.",
            )

        # --- Build VideoProcessingConfig and run pipeline ---
        from core.pipeline.video_pipeline import (  # noqa: PLC0415
            VideoPipeline,
            VideoProcessingConfig,
        )
        from core.recognizer.base_recognizer import FaceEmbedding  # noqa: PLC0415
        from core.swapper.base_swapper import BlendMode  # noqa: PLC0415

        face_embedding = FaceEmbedding(embedding=source_embedding)
        config = VideoProcessingConfig(
            source_embedding=face_embedding,
            skip_frames=max(0, skip_frames - 1),  # API uses 1-based; config uses 0-based
            enhance=enhance,
            preserve_audio=preserve_audio,
            swap_all_faces=swap_all,
            blend_mode=BlendMode.POISSON,
        )

        enhancer = getattr(state, "enhancer", None) if enhance else None

        pipeline = VideoPipeline(
            detector=state.detector,
            recognizer=state.recognizer,
            swapper=state.swapper,
            enhancer=enhancer,
        )

        logger.info(
            f"[video] Processing: {target_video.filename!r} | "
            f"skip_frames={skip_frames} enhance={enhance} "
            f"preserve_audio={preserve_audio} swap_all={swap_all}"
        )

        try:
            result = pipeline.process(
                source_video=str(video_path),
                output_path=str(output_path),
                config=config,
            )
        except Exception as exc:
            logger.error(f"[video] Pipeline failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Video processing failed: {exc}",
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Video pipeline completed but output file is empty.",
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            f"[video] Done | frames={result.processed_frames}/{result.total_frames} "
            f"time={elapsed_ms:.0f}ms"
        )

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename="swapped_video.mp4",
            headers={
                "X-Frames-Processed": str(result.processed_frames),
                "X-Total-Frames": str(result.total_frames),
                "X-Processing-Ms": f"{elapsed_ms:.0f}",
            },
        )

    except HTTPException:
        # Clean up temp dir on early error
        _cleanup_dir(tmp_dir)
        raise
    except Exception as exc:
        _cleanup_dir(tmp_dir)
        logger.exception(f"[video] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during video processing: {exc}",
        )
    # Note: on success the temp dir is NOT deleted here because FileResponse
    # streams the file lazily.  In production, a background task or periodic
    # cleanup job should purge the temp directory.  For correctness the
    # directory is cleaned up on any exception path above.


def _cleanup_dir(path: str) -> None:
    """Best-effort removal of a temporary directory tree."""
    import shutil  # noqa: PLC0415

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
