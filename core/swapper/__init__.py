try:
    from core.swapper.base_swapper import (
        BaseSwapper,
        BlendMode,
        SwapStatus,
        SwapRequest,
        SwapResult,
        BatchSwapResult,
        get_reference_points,
        estimate_norm,
        norm_crop,
        estimate_landmarks_from_bbox,
        paste_back,
        paste_back_poisson,
    )
except ImportError:  # pragma: no cover
    BaseSwapper = None  # type: ignore[assignment,misc]
    BlendMode = None  # type: ignore[assignment,misc]
    SwapStatus = None  # type: ignore[assignment,misc]
    SwapRequest = None  # type: ignore[assignment,misc]
    SwapResult = None  # type: ignore[assignment,misc]
    BatchSwapResult = None  # type: ignore[assignment,misc]
    get_reference_points = None  # type: ignore[assignment]
    estimate_norm = None  # type: ignore[assignment]
    norm_crop = None  # type: ignore[assignment]
    estimate_landmarks_from_bbox = None  # type: ignore[assignment]
    paste_back = None  # type: ignore[assignment]
    paste_back_poisson = None  # type: ignore[assignment]

try:
    from core.swapper.inswapper import InSwapper
except ImportError:  # pragma: no cover
    InSwapper = None  # type: ignore[assignment,misc]

__all__ = [
    # Abstract base + data types
    "BaseSwapper",
    "BlendMode",
    "SwapStatus",
    "SwapRequest",
    "SwapResult",
    "BatchSwapResult",
    # Alignment utilities
    "get_reference_points",
    "estimate_norm",
    "norm_crop",
    "estimate_landmarks_from_bbox",
    "paste_back",
    "paste_back_poisson",
    # Concrete implementation
    "InSwapper",
]
