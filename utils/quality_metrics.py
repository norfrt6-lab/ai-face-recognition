"""Image quality comparison utilities for evaluating face-swap output.

Provides standard image-quality metrics — MSE, PSNR, and SSIM — along with
a convenience ``evaluate_swap_quality()`` function that returns a structured
``QualityReport`` dataclass with a human-readable interpretation.

All functions operate on BGR or RGB numpy uint8 arrays of the same shape.

Usage::

    from utils.quality_metrics import evaluate_swap_quality

    report = evaluate_swap_quality(original_frame, swapped_frame)
    print(f"PSNR={report.psnr:.2f} dB  SSIM={report.ssim:.4f}  ({report.interpretation})")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level metric functions
# ---------------------------------------------------------------------------


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute the Mean Squared Error between two images.

    Args:
        original:  Reference image (uint8 or float32 numpy array).
        processed: Processed image with the same shape as *original*.

    Returns:
        MSE value (>= 0.0).  A value of 0.0 means the images are identical.

    Raises:
        ValueError: If the shapes do not match.
    """
    _check_shapes(original, processed)
    a = original.astype(np.float64)
    b = processed.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels.

    PSNR = 10 * log10(MAX_I^2 / MSE)

    For uint8 images MAX_I = 255.  Returns ``float('inf')`` when the images
    are identical (MSE = 0).

    Args:
        original:  Reference image (uint8 numpy array).
        processed: Processed image with the same shape as *original*.

    Returns:
        PSNR in dB.  Higher is better; values above 40 dB are considered
        excellent for face-swap quality.

    Raises:
        ValueError: If the shapes do not match.
    """
    mse = compute_mse(original, processed)
    if mse == 0.0:
        return float("inf")
    max_pixel = 255.0
    return float(10.0 * np.log10((max_pixel ** 2) / mse))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images.

    Attempts to use ``scipy.signal.fftconvolve`` for the sliding-window
    mean/variance calculation (fast path).  Falls back to a simplified
    global-statistics version if scipy is not available.

    SSIM values range from -1 to 1; a value of 1 means the images are
    structurally identical.  For practical purposes values above 0.95 are
    considered excellent.

    Args:
        original:  Reference image (uint8 or float32 numpy array, H×W or H×W×C).
        processed: Processed image with the same shape as *original*.

    Returns:
        Mean SSIM across all channels.

    Raises:
        ValueError: If the shapes do not match.
    """
    _check_shapes(original, processed)

    # Convert to float64 in [0, 255] space
    a = original.astype(np.float64)
    b = processed.astype(np.float64)

    # If multi-channel, compute per-channel and average
    if a.ndim == 3:
        scores = [_ssim_single_channel(a[:, :, c], b[:, :, c]) for c in range(a.shape[2])]
        return float(np.mean(scores))

    return _ssim_single_channel(a, b)


# ---------------------------------------------------------------------------
# Dataclass + evaluation
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Quality metrics for a pair of (original, processed) images.

    Attributes:
        psnr:           Peak Signal-to-Noise Ratio in dB.
        ssim:           Structural Similarity Index (-1 to 1, higher is better).
        mse:            Mean Squared Error (lower is better).
        interpretation: Human-readable quality label: 'excellent', 'good',
                        'fair', or 'poor'.
    """

    psnr: float
    ssim: float
    mse: float
    interpretation: str

    def __str__(self) -> str:
        return (
            f"QualityReport(PSNR={self.psnr:.2f} dB, "
            f"SSIM={self.ssim:.4f}, "
            f"MSE={self.mse:.2f}, "
            f"quality={self.interpretation})"
        )


def evaluate_swap_quality(
    original: np.ndarray,
    swapped: np.ndarray,
) -> QualityReport:
    """Compute all three quality metrics and return a QualityReport.

    Interpretation thresholds (PSNR-based, industry-standard for lossy
    compression / image processing):

    ============  ========  =======
    Label         PSNR (dB) SSIM
    ============  ========  =======
    excellent     >= 40     >= 0.95
    good          >= 30     >= 0.80
    fair          >= 20     >= 0.60
    poor          < 20      < 0.60
    ============  ========  =======

    The PSNR value takes precedence for the label.

    Args:
        original: Original / reference BGR image.
        swapped:  Swapped / processed BGR image of the same shape.

    Returns:
        QualityReport with psnr, ssim, mse, and interpretation fields.
    """
    mse = compute_mse(original, swapped)
    psnr = compute_psnr(original, swapped)
    ssim = compute_ssim(original, swapped)

    if psnr == float("inf") or psnr >= 40.0:
        interpretation = "excellent"
    elif psnr >= 30.0:
        interpretation = "good"
    elif psnr >= 20.0:
        interpretation = "fair"
    else:
        interpretation = "poor"

    logger.debug(
        f"[quality_metrics] MSE={mse:.2f} PSNR={psnr:.2f}dB "
        f"SSIM={ssim:.4f} quality={interpretation}"
    )

    return QualityReport(psnr=psnr, ssim=ssim, mse=mse, interpretation=interpretation)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    """Raise ValueError if *a* and *b* do not have the same shape."""
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: original={a.shape} vs processed={b.shape}. "
            "Both images must have the same dimensions and channels."
        )


def _ssim_single_channel(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM for a single 2-D (H, W) float64 channel.

    Uses a Gaussian-weighted sliding window via scipy.signal.fftconvolve
    when available, otherwise falls back to a simplified global-statistics
    approximation.

    SSIM formula:
        SSIM(x,y) = (2μ_x μ_y + C1)(2σ_xy + C2) /
                    ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

    Constants C1 = (0.01 * 255)^2, C2 = (0.03 * 255)^2 (Wang et al. 2004).

    Args:
        a: 2-D float64 array (single channel, values in [0, 255]).
        b: 2-D float64 array with the same shape as *a*.

    Returns:
        SSIM score as a float.
    """
    C1 = (0.01 * 255) ** 2  # 6.5025
    C2 = (0.03 * 255) ** 2  # 58.5225

    try:
        return _ssim_scipy(a, b, C1, C2)
    except ImportError:
        return _ssim_global(a, b, C1, C2)


def _ssim_scipy(a: np.ndarray, b: np.ndarray, C1: float, C2: float) -> float:
    """SSIM using a Gaussian sliding window (fast path — requires scipy).

    Raises:
        ImportError: If scipy is not installed.
    """
    from scipy.signal import fftconvolve  # noqa: PLC0415

    # Build a 11×11 Gaussian kernel (sigma=1.5) — matches Wang et al. reference
    kernel_size = 11
    sigma = 1.5
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float64)
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss_1d /= gauss_1d.sum()
    kernel = np.outer(gauss_1d, gauss_1d)

    def _convolve(arr: np.ndarray) -> np.ndarray:
        return fftconvolve(arr, kernel, mode="valid")

    mu_a = _convolve(a)
    mu_b = _convolve(b)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a2 = _convolve(a * a) - mu_a2
    sigma_b2 = _convolve(b * b) - mu_b2
    sigma_ab = _convolve(a * b) - mu_ab

    numerator = (2.0 * mu_ab + C1) * (2.0 * sigma_ab + C2)
    denominator = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)

    ssim_map = numerator / np.where(denominator == 0, 1e-10, denominator)
    return float(ssim_map.mean())


def _ssim_global(a: np.ndarray, b: np.ndarray, C1: float, C2: float) -> float:
    """Simplified SSIM using global image statistics (no scipy required).

    This is a fall-back that is less accurate than the sliding-window version
    but requires no external dependencies.

    Args:
        a: 2-D float64 channel.
        b: 2-D float64 channel.
        C1, C2: Stability constants.

    Returns:
        SSIM approximation.
    """
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    sigma_a2 = float(np.var(a))
    sigma_b2 = float(np.var(b))
    sigma_ab = float(np.mean((a - mu_a) * (b - mu_b)))

    numerator = (2.0 * mu_a * mu_b + C1) * (2.0 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)

    if denominator == 0:
        return 1.0
    return float(numerator / denominator)
