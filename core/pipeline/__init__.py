try:
    from core.pipeline.face_pipeline import (
        FacePipeline,
        PipelineConfig,
        PipelineResult,
        PipelineStatus,
        PipelineTiming,
    )
except ImportError:  # pragma: no cover
    FacePipeline = None  # type: ignore[assignment,misc]
    PipelineConfig = None  # type: ignore[assignment,misc]
    PipelineResult = None  # type: ignore[assignment,misc]
    PipelineStatus = None  # type: ignore[assignment,misc]
    PipelineTiming = None  # type: ignore[assignment,misc]

try:
    from core.pipeline.video_pipeline import (
        VideoPipeline,
        VideoProcessingConfig,
        VideoProcessingResult,
    )
except ImportError:  # pragma: no cover
    VideoPipeline = None  # type: ignore[assignment,misc]
    VideoProcessingConfig = None  # type: ignore[assignment,misc]
    VideoProcessingResult = None  # type: ignore[assignment,misc]

__all__ = [
    # Image pipeline
    "FacePipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStatus",
    "PipelineTiming",
    # Video pipeline
    "VideoPipeline",
    "VideoProcessingConfig",
    "VideoProcessingResult",
]
