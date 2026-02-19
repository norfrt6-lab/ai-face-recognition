# ============================================================
# AI Face Recognition & Face Swap
# core/enhancer/gfpgan_enhancer.py
# ============================================================
# GFPGAN v1.4 face restoration enhancer.
#
# GFPGAN (Generative Facial Prior GAN) restores degraded faces
# by leveraging rich facial priors from a pretrained GAN.
# It excels at:
#   - Removing compression artifacts and blur from swapped faces
#   - Restoring fine facial details (eyes, teeth, skin texture)
#   - Optional background upsampling via Real-ESRGAN
#
# Model:      GFPGANv1.4.pth
# Input:      BGR face crop or full frame (any resolution)
# Output:     Enhanced BGR image (upscaled by self.upscale factor)
#
# Pipeline per enhance() call:
#   1. Validate input image
#   2. Pad to minimum size if needed
#   3. Run GFPGAN restorer (detects + enhances faces internally)
#   4. Crop back to original aspect ratio if padded
#   5. Return EnhancementResult with output + timing metadata
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.detector.base_detector import FaceBox
from core.enhancer.base_enhancer import (
    BaseEnhancer,
    EnhancementRequest,
    EnhancementResult,
    EnhancementStatus,
    EnhancerBackend,
    find_center_face,
    pad_image_for_enhancement,
    unpad_image,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class GFPGANEnhancer(BaseEnhancer):
    """
    GFPGAN v1.4 face enhancement engine.

    Wraps the ``gfpgan`` Python package's ``GFPGANer`` restorer and
    translates its output into the standardised ``EnhancementResult``
    type used across this pipeline.

    The restorer internally runs a face detector (RetinaFace), extracts
    each face crop, applies the GFPGAN network, and pastes the enhanced
    faces back into the original image.

    Usage::

        from core.enhancer.gfpgan_enhancer import GFPGANEnhancer
        from core.enhancer.base_enhancer import EnhancementRequest

        with GFPGANEnhancer("models/GFPGANv1.4.pth") as enh:
            result = enh.enhance(
                EnhancementRequest(image=bgr_frame, upscale=2)
            )
            if result.success:
                cv2.imwrite("enhanced.png", result.output_image)

    Attributes:
        model_path:        Path to GFPGANv1.4.pth weights file.
        arch:              Model architecture string ('clean' or 'RestoreFormer').
        channel_multiplier: Channel width multiplier (default 2).
        bg_upsampler:      Optional background upsampler ('realesrgan' or None).
        upscale:           Output upscale factor (1, 2, or 4).
        only_center_face:  Enhance only the most central face.
        paste_back:        Paste enhanced faces back into the full frame.
        device:            Inference device ('auto' | 'cpu' | 'cuda').
    """

    def __init__(
        self,
        model_path: str = "models/GFPGANv1.4.pth",
        arch: str = "clean",
        channel_multiplier: int = 2,
        bg_upsampler: Optional[str] = None,
        upscale: int = 2,
        only_center_face: bool = False,
        paste_back: bool = True,
        device: str = "auto",
    ) -> None:
        super().__init__(
            model_path=model_path,
            backend=EnhancerBackend.GFPGAN,
            upscale=upscale,
            only_center_face=only_center_face,
            paste_back=paste_back,
            device=device,
        )
        self.arch               = arch
        self.channel_multiplier = channel_multiplier
        self.bg_upsampler_name  = bg_upsampler  # string name; actual object built in load_model

        # Populated by load_model()
        self._restorer      = None   # GFPGANer instance
        self._bg_upsampler  = None   # Optional RealESRGANer

    # ------------------------------------------------------------------
    # BaseEnhancer interface — load_model
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load GFPGANv1.4.pth and (optionally) the Real-ESRGAN background
        upsampler.

        Raises:
            FileNotFoundError: If ``self.model_path`` does not exist.
            ImportError:       If the ``gfpgan`` package is not installed.
            RuntimeError:      On any other loading failure.
        """
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GFPGAN model not found: {self.model_path}\n"
                "Download it with: python utils/download_models.py --minimum"
            )

        try:
            from gfpgan import GFPGANer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "gfpgan package is not installed. Install it with:\n"
                "  pip install gfpgan>=1.3.8"
            ) from exc

        # Resolve device
        device_str = self._resolve_device()
        logger.info(
            f"Loading GFPGAN | arch={self.arch} | upscale={self.upscale}x "
            f"| device={device_str} | bg_upsampler={self.bg_upsampler_name}"
        )

        # Optionally build the Real-ESRGAN background upsampler
        bg_upsampler_obj = None
        if self.bg_upsampler_name == "realesrgan":
            bg_upsampler_obj = self._build_realesrgan_upsampler(device_str)

        try:
            self._restorer = GFPGANer(
                model_path=str(path),
                upscale=self.upscale,
                arch=self.arch,
                channel_multiplier=self.channel_multiplier,
                bg_upsampler=bg_upsampler_obj,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise GFPGANer from {self.model_path}: {exc}"
            ) from exc

        self._bg_upsampler = bg_upsampler_obj
        self._is_loaded    = True
        logger.success(
            f"GFPGANEnhancer loaded | "
            f"arch={self.arch} | upscale={self.upscale}x | "
            f"bg_upsampler={'yes' if bg_upsampler_obj else 'no'}"
        )

    # ------------------------------------------------------------------
    # BaseEnhancer interface — enhance
    # ------------------------------------------------------------------

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        """
        Enhance face(s) in ``request.image`` using GFPGAN v1.4.

        Steps:
          1. Validate the input image.
          2. Pad to a minimum workable size if needed.
          3. Run the GFPGAN restorer (internally detects + enhances).
          4. Crop back if padding was added.
          5. Apply only-center-face filtering if requested.
          6. Return ``EnhancementResult``.

        Args:
            request: ``EnhancementRequest`` with image + parameters.

        Returns:
            ``EnhancementResult`` — always non-None; check ``.success``.
        """
        t0 = self._timer()

        # Guard: model must be loaded
        if not self._is_loaded or self._restorer is None:
            return self._make_failed_result(
                EnhancementStatus.MODEL_NOT_LOADED,
                request.image,
                "Model not loaded. Call load_model() first.",
                t0,
            )

        # Validate input
        try:
            self._validate_image(request.image)
        except ValueError as exc:
            return self._make_failed_result(
                EnhancementStatus.INVALID_INPUT,
                request.image,
                str(exc),
                t0,
            )

        # Determine effective parameters for this call
        only_center = (
            request.only_center_face
            if request.only_center_face is not None
            else self.only_center_face
        )
        paste_back = (
            request.paste_back
            if request.paste_back is not None
            else self.paste_back
        )

        # Pad to minimum size (GFPGAN needs at least 128px sides)
        padded, padding = pad_image_for_enhancement(request.image, min_size=128)
        was_padded = any(p > 0 for p in padding)

        # Run GFPGAN
        t_inf = self._timer()
        try:
            cropped_faces, restored_faces, output_img = self._restorer.enhance(
                padded,
                has_aligned=False,
                only_center_face=only_center,
                paste_back=paste_back,
            )
        except Exception as exc:
            logger.error(f"GFPGAN inference error: {exc}")
            return self._make_failed_result(
                EnhancementStatus.INFERENCE_ERROR,
                request.image,
                f"GFPGAN inference error: {exc}",
                t0,
            )
        inference_time = self._timer() - t_inf

        # If no faces were detected / enhanced
        num_enhanced = len(restored_faces) if restored_faces else 0
        if num_enhanced == 0:
            logger.warning("GFPGAN: no faces detected or restored.")
            return self._make_failed_result(
                EnhancementStatus.NO_FACE_DETECTED,
                request.image,
                "GFPGAN detected no faces in the input image.",
                t0,
            )

        # output_img may be None if paste_back=False; fall back to padded input
        if output_img is None:
            output_img = padded

        # Remove padding if we added it
        if was_padded and output_img is not None:
            # The output may have been upscaled — scale padding accordingly
            scale = output_img.shape[0] / padded.shape[0]
            scaled_padding = tuple(int(p * scale) for p in padding)
            output_img = unpad_image(output_img, scaled_padding)

        # Update stats
        with self._stats_lock:
            self._total_calls     += 1
            self._total_inference += inference_time
        total_time = self._timer() - t0

        logger.debug(
            f"GFPGAN enhance | faces={num_enhanced} | "
            f"inf={inference_time:.1f}ms total={total_time:.1f}ms"
        )

        return EnhancementResult(
            output_image=output_img,
            status=EnhancementStatus.SUCCESS,
            backend=EnhancerBackend.GFPGAN,
            num_faces_enhanced=num_enhanced,
            enhance_time_ms=total_time,
            inference_time_ms=inference_time,
            upscale_factor=self.upscale,
            face_crops=restored_faces if request.metadata.get("save_crops") else None,
        )

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free the GFPGAN restorer and background upsampler."""
        self._restorer     = None
        self._bg_upsampler = None
        super().release()
        logger.info("GFPGANEnhancer released.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_realesrgan_upsampler(self, device_str: str):
        """
        Build a Real-ESRGAN background upsampler for GFPGAN.

        Real-ESRGAN enhances the non-face background regions when
        ``bg_upsampler='realesrgan'`` is requested.

        Args:
            device_str: Resolved device string ('cuda' or 'cpu').

        Returns:
            RealESRGANer instance, or None if the package is missing.
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: PLC0415
            from realesrgan import RealESRGANer              # noqa: PLC0415

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=2,
            )
            upsampler = RealESRGANer(
                scale=2,
                model_path="https://github.com/xinntao/Real-ESRGAN/"
                           "releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=device_str.startswith("cuda"),
            )
            logger.info("Real-ESRGAN background upsampler loaded.")
            return upsampler
        except ImportError:
            logger.warning(
                "realesrgan package not installed — background upsampling disabled. "
                "Install with: pip install realesrgan"
            )
            return None
        except Exception as exc:
            logger.warning(f"Failed to load Real-ESRGAN upsampler: {exc}")
            return None

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"GFPGANEnhancer("
            f"arch={self.arch!r}, "
            f"upscale={self.upscale}x, "
            f"bg_upsampler={self.bg_upsampler_name!r}, "
            f"device={self.device!r}, "
            f"status={status}, "
            f"calls={self._total_calls})"
        )
