# ============================================================
# AI Face Recognition & Face Swap
# core/enhancer/codeformer_enhancer.py
# ============================================================
# CodeFormer face enhancement implementation.
#
# CodeFormer is a robust blind face restoration algorithm based
# on a learned discrete codebook prior.  Unlike GFPGAN, it
# exposes a "fidelity weight" (w) that lets you control the
# trade-off between restoration quality and identity fidelity:
#
#   w = 0.0  →  maximum restoration quality (may alter identity)
#   w = 1.0  →  maximum identity preservation (less restoration)
#
# Model:   codeformer.pth  (~370 MB)
# Input:   512 × 512 BGR face crop (aligned)
# Output:  512 × 512 BGR restored face crop
#
# Pipeline per enhance() call:
#   1. Validate input image
#   2. Detect / locate face regions (via facexlib)
#   3. Align and warp each face to 512 × 512
#   4. Run CodeFormer inference with fidelity_weight
#   5. Paste enhanced crops back into the original frame
#   6. Apply upscaling if requested
# ============================================================

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

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


# ============================================================
# CodeFormerEnhancer
# ============================================================

class CodeFormerEnhancer(BaseEnhancer):
    """
    Face enhancement engine backed by CodeFormer.

    CodeFormer uses a VQ-VAE codebook prior for blind face restoration,
    giving high-quality results with controllable fidelity via
    ``fidelity_weight``.

    Usage::

        with CodeFormerEnhancer("models/codeformer.pth") as enh:
            result = enh.enhance(
                EnhancementRequest(
                    image=frame,
                    fidelity_weight=0.7,
                    upscale=2,
                    full_frame=True,
                )
            )
            if result.success:
                cv2.imwrite("enhanced.jpg", result.output_image)

    Attributes:
        model_path:        Path to ``codeformer.pth``.
        fidelity_weight:   Default fidelity weight [0.0, 1.0].
        upscale:           Default output upscale factor (1, 2, or 4).
        only_center_face:  Enhance only the most central face by default.
        paste_back:        Paste enhanced crops back into source frame.
        device:            Inference device ('auto' | 'cpu' | 'cuda').
        bg_enhance:        Also enhance the background using Real-ESRGAN.
    """

    #: Native resolution expected by the CodeFormer model
    _MODEL_RESOLUTION: int = 512

    def __init__(
        self,
        model_path: str = "models/codeformer.pth",
        fidelity_weight: float = 0.5,
        upscale: int = 2,
        only_center_face: bool = False,
        paste_back: bool = True,
        device: str = "auto",
        bg_enhance: bool = False,
        bg_upsampler: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            backend=EnhancerBackend.CODEFORMER,
            upscale=upscale,
            only_center_face=only_center_face,
            paste_back=paste_back,
            device=device,
        )
        self.fidelity_weight = float(fidelity_weight)
        self.bg_enhance      = bool(bg_enhance)
        self.bg_upsampler    = bg_upsampler

        # Internal CodeFormer net (set by load_model)
        self._codeformer_net = None
        self._face_helper    = None
        self._bg_upsampler_obj = None

    # ------------------------------------------------------------------
    # BaseEnhancer interface — load_model
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the CodeFormer model weights and initialise facexlib helper.

        Raises:
            FileNotFoundError: If ``self.model_path`` does not exist.
            ImportError:       If required packages are not installed.
            RuntimeError:      On any other loading failure.
        """
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"CodeFormer model not found: {self.model_path}\n"
                "Download it with: python utils/download_models.py --minimum"
            )

        # ── Validate required imports ───────────────────────────────
        self._check_imports()

        resolved_device = self._resolve_device()
        logger.info(
            f"Loading CodeFormer | path={self.model_path!r} | "
            f"device={resolved_device} | fidelity={self.fidelity_weight}"
        )

        try:
            import torch  # noqa: PLC0415
            from basicsr.utils.registry import ARCH_REGISTRY  # noqa: PLC0415
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper  # noqa: PLC0415

            # ── Build model architecture ────────────────────────────
            # CodeFormer is registered in BasicSR's architecture registry
            net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            net = net.to(resolved_device)

            # ── Load checkpoint ─────────────────────────────────────
            checkpoint = torch.load(
                str(path),
                map_location=torch.device(resolved_device),
            )
            # Checkpoint may be nested under 'params_ema' or 'params'
            state_dict = (
                checkpoint.get("params_ema")
                or checkpoint.get("params")
                or checkpoint
            )
            net.load_state_dict(state_dict, strict=False)
            net.eval()

            self._codeformer_net = net

            # ── Initialise face restoration helper ─────────────────
            self._face_helper = FaceRestoreHelper(
                upscale_factor=self.upscale,
                face_size=self._MODEL_RESOLUTION,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="png",
                use_parse=True,
                device=torch.device(resolved_device),
            )

            # ── Optional background upsampler ───────────────────────
            if self.bg_enhance:
                self._bg_upsampler_obj = self._load_bg_upsampler(resolved_device)

            self._is_loaded = True
            logger.success(
                f"CodeFormer loaded | device={resolved_device} | "
                f"fidelity_default={self.fidelity_weight}"
            )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CodeFormer from {self.model_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # BaseEnhancer interface — enhance
    # ------------------------------------------------------------------

    def enhance(self, request: EnhancementRequest) -> EnhancementResult:
        """
        Enhance face(s) in ``request.image`` using CodeFormer.

        Args:
            request: ``EnhancementRequest`` with image + parameters.

        Returns:
            ``EnhancementResult`` — always non-None; check ``.success``.
        """
        t0 = self._timer()

        # Guard: model must be loaded
        if not self._is_loaded or self._codeformer_net is None:
            return self._make_failed_result(
                EnhancementStatus.MODEL_NOT_LOADED,
                request.image,
                "Model not loaded. Call load_model() first.",
                t0,
            )

        # Guard: validate input
        try:
            self._validate_image(request.image)
        except ValueError as exc:
            return self._make_failed_result(
                EnhancementStatus.INVALID_INPUT,
                request.image,
                str(exc),
                t0,
            )

        fidelity = request.fidelity_weight
        upscale  = request.upscale if request.upscale else self.upscale

        try:
            import torch  # noqa: PLC0415

            resolved_device = self._resolve_device()
            device = torch.device(resolved_device)

            # ── Prepare face restoration helper ─────────────────────
            helper = self._face_helper
            helper.clean_all()
            helper.read_image(request.image)
            helper.get_face_landmarks_5(
                only_center_face=request.only_center_face,
                resize=640,
                eye_dist_threshold=5,
            )
            helper.align_warp_face()

            if len(helper.cropped_faces) == 0:
                return self._make_failed_result(
                    EnhancementStatus.NO_FACE_DETECTED,
                    request.image,
                    "No faces detected by facexlib FaceRestoreHelper.",
                    t0,
                )

            # ── Inference loop ───────────────────────────────────────
            t_inf  = self._timer()
            face_crops: List[np.ndarray] = []

            for idx, cropped_face in enumerate(helper.cropped_faces):
                # BGR → RGB tensor in [-1, 1]
                face_t = self._preprocess_face(cropped_face, device)

                with torch.no_grad():
                    output = self._codeformer_net(
                        face_t,
                        w=fidelity,
                        adain=True,
                    )[0]

                restored = self._postprocess_face(output)
                face_crops.append(restored)
                helper.add_restored_face(restored)

            inference_time = self._timer() - t_inf

            # ── Background upsampling (optional) ────────────────────
            if self.bg_enhance and self._bg_upsampler_obj is not None:
                helper.get_inverse_affine(None)
                output_image = helper.paste_faces_to_input_image(
                    upsample_img=self._bg_upsampler_obj.enhance(
                        request.image, outscale=upscale
                    )[0],
                )
            elif request.paste_back:
                helper.get_inverse_affine(None)
                output_image = helper.paste_faces_to_input_image()
                # Resize to target upscale if needed
                if upscale > 1:
                    h, w = output_image.shape[:2]
                    output_image = cv2.resize(
                        output_image,
                        (w * upscale, h * upscale),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
            else:
                # Return the enhanced face crop(s) directly
                output_image = face_crops[0] if face_crops else request.image

        except Exception as exc:
            logger.error(f"CodeFormer inference error: {exc}")
            return self._make_failed_result(
                EnhancementStatus.INFERENCE_ERROR,
                request.image,
                f"CodeFormer inference error: {exc}",
                t0,
            )

        # ── Statistics ───────────────────────────────────────────────
        with self._stats_lock:
            self._total_calls     += 1
            self._total_inference += inference_time
        num_enhanced = len(face_crops)
        total_time   = self._timer() - t0

        logger.debug(
            f"CodeFormer | faces={num_enhanced} | "
            f"fidelity={fidelity:.2f} | "
            f"inf={inference_time:.1f}ms | "
            f"total={total_time:.1f}ms"
        )

        return EnhancementResult(
            output_image=output_image,
            status=EnhancementStatus.SUCCESS,
            backend=EnhancerBackend.CODEFORMER,
            num_faces_enhanced=num_enhanced,
            enhance_time_ms=total_time,
            inference_time_ms=inference_time,
            upscale_factor=upscale,
            face_crops=face_crops if request.metadata.get("save_crops") else None,
        )

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free the model and face helper from GPU/CPU memory."""
        self._codeformer_net   = None
        self._face_helper      = None
        self._bg_upsampler_obj = None
        super().release()
        logger.info("CodeFormerEnhancer released.")

    # ------------------------------------------------------------------
    # Internal preprocessing / postprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_face(face_bgr: np.ndarray, device) -> "torch.Tensor":
        """
        Convert a BGR uint8 face crop to a (1, 3, 512, 512) float32
        tensor in [-1, 1] on *device*.

        Args:
            face_bgr: (H, W, 3) BGR uint8 face crop.
            device:   torch.device to move the tensor to.

        Returns:
            (1, 3, H, W) float32 tensor in [-1.0, 1.0].
        """
        import torch  # noqa: PLC0415

        # BGR → RGB
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # HWC → CHW, [0, 255] → [-1, 1]
        tensor = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = tensor.transpose(2, 0, 1)            # CHW
        tensor = torch.from_numpy(tensor).unsqueeze(0)  # NCHW
        return tensor.to(device)

    @staticmethod
    def _postprocess_face(output_tensor) -> np.ndarray:
        """
        Convert a model output tensor back to a BGR uint8 face crop.

        Args:
            output_tensor: (1, 3, H, W) or (3, H, W) float32 in [-1, 1].

        Returns:
            (H, W, 3) BGR uint8 image.
        """
        import torch  # noqa: PLC0415

        out = output_tensor.squeeze(0).float().detach().cpu()
        # [-1, 1] → [0, 255]
        out = (out * 0.5 + 0.5).clamp(0, 1)
        # CHW → HWC
        out = out.permute(1, 2, 0).numpy()
        out = (out * 255.0).round().astype(np.uint8)
        # RGB → BGR
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out

    # ------------------------------------------------------------------
    # Internal: background upsampler
    # ------------------------------------------------------------------

    def _load_bg_upsampler(self, device: str):
        """
        Load a Real-ESRGAN background upsampler.

        Returns the upsampler object or None if loading fails.
        """
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: PLC0415
            from realesrgan import RealESRGANer  # noqa: PLC0415

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path="models/RealESRGAN_x2plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=device.startswith("cuda"),
            )
            logger.info("CodeFormer background upsampler (Real-ESRGAN) loaded.")
            return bg_upsampler
        except Exception as exc:
            logger.warning(
                f"Could not load Real-ESRGAN background upsampler: {exc}. "
                "Background enhancement will be skipped."
            )
            return None

    # ------------------------------------------------------------------
    # Internal: import guard
    # ------------------------------------------------------------------

    @staticmethod
    def _check_imports() -> None:
        """Raise ImportError with a helpful message if deps are missing."""
        missing = []
        for pkg in ("torch", "basicsr", "facexlib"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                f"CodeFormerEnhancer requires: {', '.join(missing)}.\n"
                "Install with: pip install torch basicsr facexlib"
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return (
            f"CodeFormerEnhancer("
            f"model={self.model_name!r}, "
            f"fidelity={self.fidelity_weight:.2f}, "
            f"upscale={self.upscale}x, "
            f"device={self.device!r}, "
            f"bg_enhance={self.bg_enhance}, "
            f"status={status}, "
            f"calls={self._total_calls})"
        )
