"""Unit tests for utils.quality_metrics.

Uses synthetic numpy arrays — no model loading or real images required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils.quality_metrics import (
    QualityReport,
    compute_mse,
    compute_psnr,
    compute_ssim,
    evaluate_swap_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: int, shape: tuple = (64, 64, 3)) -> np.ndarray:
    """Return an image filled with a constant uint8 value."""
    return np.full(shape, value, dtype=np.uint8)


def _random_image(shape: tuple = (64, 64, 3), seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# MSE tests
# ---------------------------------------------------------------------------


class TestComputeMse:
    def test_identical_images_have_zero_mse(self):
        img = _random_image()
        assert compute_mse(img, img.copy()) == pytest.approx(0.0)

    def test_maximally_different_images(self):
        """Black vs white images → MSE = 255²."""
        black = _solid(0)
        white = _solid(255)
        expected = 255.0 ** 2
        assert compute_mse(black, white) == pytest.approx(expected)

    def test_mse_is_symmetric(self):
        a = _random_image(seed=1)
        b = _random_image(seed=2)
        assert compute_mse(a, b) == pytest.approx(compute_mse(b, a))

    def test_shape_mismatch_raises(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.zeros((20, 20, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_mse(a, b)

    def test_known_mse_value(self):
        """Verify MSE against a manually computed value."""
        a = np.array([[[0, 0, 0]], [[255, 255, 255]]], dtype=np.uint8)
        b = np.array([[[255, 255, 255]], [[0, 0, 0]]], dtype=np.uint8)
        # Each of 6 pixels has (255)^2 difference → mean = 255^2
        assert compute_mse(a, b) == pytest.approx(255.0 ** 2)


# ---------------------------------------------------------------------------
# PSNR tests
# ---------------------------------------------------------------------------


class TestComputePsnr:
    def test_identical_images_return_inf(self):
        img = _random_image()
        assert compute_psnr(img, img.copy()) == float("inf")

    def test_psnr_decreases_with_more_noise(self):
        """Higher MSE → lower PSNR."""
        rng = np.random.default_rng(0)
        original = _random_image()
        noisy_light = np.clip(original.astype(int) + rng.integers(0, 5, original.shape), 0, 255).astype(np.uint8)
        noisy_heavy = np.clip(original.astype(int) + rng.integers(0, 80, original.shape), 0, 255).astype(np.uint8)
        assert compute_psnr(original, noisy_light) > compute_psnr(original, noisy_heavy)

    def test_psnr_formula(self):
        """Manually verify the PSNR formula for a known MSE."""
        a = _solid(0)
        b = _solid(10)
        mse = 100.0  # 10^2
        expected = 10.0 * math.log10((255.0 ** 2) / mse)
        assert compute_psnr(a, b) == pytest.approx(expected, rel=1e-5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_psnr(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((16, 16, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# SSIM tests
# ---------------------------------------------------------------------------


class TestComputeSsim:
    def test_identical_images_have_ssim_near_one(self):
        img = _random_image()
        ssim = compute_ssim(img, img.copy())
        assert ssim == pytest.approx(1.0, abs=1e-4)

    def test_ssim_between_zero_and_one_for_similar_images(self):
        rng = np.random.default_rng(7)
        original = _random_image()
        noisy = np.clip(original.astype(int) + rng.integers(-10, 10, original.shape), 0, 255).astype(np.uint8)
        ssim = compute_ssim(original, noisy)
        assert 0.0 <= ssim <= 1.0

    def test_dissimilar_images_have_lower_ssim(self):
        a = _random_image(seed=1)
        b = _random_image(seed=99)  # different random seed → very different content
        ssim_same = compute_ssim(a, a.copy())
        ssim_diff = compute_ssim(a, b)
        assert ssim_same > ssim_diff

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_ssim(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((16, 8, 3), dtype=np.uint8))

    def test_grayscale_2d_input(self):
        """SSIM should work on 2-D (H, W) arrays as well."""
        img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        ssim = compute_ssim(img, img.copy())
        assert ssim == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# QualityReport / evaluate_swap_quality tests
# ---------------------------------------------------------------------------


class TestEvaluateSwapQuality:
    def test_returns_quality_report(self):
        img = _random_image()
        report = evaluate_swap_quality(img, img.copy())
        assert isinstance(report, QualityReport)

    def test_identical_images_give_excellent(self):
        img = _random_image()
        report = evaluate_swap_quality(img, img.copy())
        assert report.interpretation == "excellent"
        assert report.mse == pytest.approx(0.0)
        assert report.psnr == float("inf")

    def test_slightly_noisy_image_is_good_or_excellent(self):
        rng = np.random.default_rng(3)
        original = _random_image()
        noisy = np.clip(original.astype(int) + rng.integers(-3, 3, original.shape), 0, 255).astype(np.uint8)
        report = evaluate_swap_quality(original, noisy)
        assert report.interpretation in {"excellent", "good"}

    def test_heavily_distorted_image_is_poor(self):
        black = _solid(0)
        white = _solid(255)
        report = evaluate_swap_quality(black, white)
        assert report.interpretation == "poor"
        assert report.mse == pytest.approx(255.0 ** 2)

    def test_interpretation_values_are_valid(self):
        valid_labels = {"excellent", "good", "fair", "poor"}
        for seed in range(5):
            rng = np.random.default_rng(seed)
            a = _random_image(seed=seed)
            b = _random_image(seed=seed + 100)
            report = evaluate_swap_quality(a, b)
            assert report.interpretation in valid_labels

    def test_report_str_contains_metrics(self):
        img = _random_image()
        report = evaluate_swap_quality(img, img.copy())
        s = str(report)
        assert "PSNR" in s
        assert "SSIM" in s
        assert "MSE" in s
