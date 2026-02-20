# Concrete implementation of BaseSwapper using the
# inswapper_128.onnx model (InsightFace / roop lineage).
#
# Model:       inswapper_128.onnx
# Input size:  128 × 128 aligned face crop
# Inference:   ONNX Runtime (CUDA or CPU)
#
# Pipeline per swap call:
#   1. Align source face → 128×128 canonical crop
#   2. Preprocess crop   → (1, 3, 128, 128) float32 tensor
#   3. Encode source embedding → latent code via model emap
#   4. ONNX inference    → swapped 128×128 face patch
#   5. Post-process      → BGR uint8
#   6. Paste-back        → Poisson or alpha blend into target frame

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Module-level onnxruntime import — allows test patches via
# `patch("core.swapper.inswapper.ort", ...)`.
# Importing here (not inside load_model) also means the ImportError
# is raised at import time when ort is truly absent, giving a clear
# message rather than a confusing attribute error later.
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

from core.detector.base_detector import FaceBox
from core.recognizer.base_recognizer import FaceEmbedding
from core.swapper.base_swapper import (
    BaseSwapper,
    BlendMode,
    SwapRequest,
    SwapResult,
    SwapStatus,
    _make_crop_mask,
    estimate_landmarks_from_bbox,
    norm_crop,
    paste_back,
    paste_back_poisson,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# Mean and std used by inswapper (same as ArcFace / most InsightFace models)
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Native model resolution
_MODEL_INPUT_SIZE = 128


class InSwapper(BaseSwapper):
    """
    Face swap engine backed by ``inswapper_128.onnx``.

    The model takes:
      - A 128 × 128 aligned face crop (the *target* face to replace).
      - A latent identity vector derived from the *source* embedding via
        the model's internal ``emap`` (identity mapping matrix).

    It outputs a 128 × 128 BGR image with the target face replaced by
    the source identity while preserving the target pose, expression,
    and lighting.

    Usage::

        from core.swapper.inswapper import InSwapper
        from core.swapper.base_swapper import SwapRequest, BlendMode

        with InSwapper("models/inswapper_128.onnx") as swapper:
            result = swapper.swap(
                SwapRequest(
                    source_embedding=embedding,
                    target_image=frame,
                    target_face=face_box,
                    blend_mode=BlendMode.POISSON,
                )
            )
            if result.success:
                cv2.imwrite("output.jpg", result.output_image)

    Attributes:
        model_path:    Path to the ``.onnx`` file.
        providers:     ONNX Runtime execution providers.
        blend_mode:    Default compositing strategy.
        blend_alpha:   Default alpha weight (1.0 = fully swapped).
        mask_feather:  Default Gaussian feather radius for mask edges.
        input_size:    Model native input resolution (128).
    """

    def __init__(
        self,
        model_path: str = "models/inswapper_128.onnx",
        providers: Optional[List[str]] = None,
        blend_mode: BlendMode = BlendMode.POISSON,
        blend_alpha: float = 1.0,
        mask_feather: int = 20,
        input_size: int = _MODEL_INPUT_SIZE,
    ) -> None:
        super().__init__(
            model_path=model_path,
            providers=providers,
            blend_mode=blend_mode,
            blend_alpha=blend_alpha,
            mask_feather=mask_feather,
            input_size=input_size,
        )

        # ONNX session (set by load_model)
        self._session = None

        # Model metadata (populated during load)
        self._input_name:  Optional[str] = None   # aligned-face tensor name
        self._latent_name: Optional[str] = None   # identity latent tensor name
        self._output_name: Optional[str] = None   # swapped-face tensor name
        self._emap:        Optional[np.ndarray] = None  # (512, 512) identity matrix

        # Inference statistics (cumulative, guarded by _stats_lock)
        self._stats_lock               = threading.Lock()
        self._total_calls:     int   = 0
        self._total_inference: float = 0.0  # ms

    # ------------------------------------------------------------------
    # BaseSwapper interface — load_model
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load ``inswapper_128.onnx`` and extract metadata.

        Raises:
            FileNotFoundError: If ``self.model_path`` does not exist.
            ImportError:       If ``onnxruntime`` is not installed.
            RuntimeError:      On any other loading failure.
        """
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"inswapper model not found: {self.model_path}\n"
                "Download it with: python utils/download_models.py --minimum"
            )

        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. Install it with:\n"
                "  GPU: pip install onnxruntime-gpu>=1.18.0\n"
                "  CPU: pip install onnxruntime>=1.18.0"
            )

        # Resolve providers to those available in this ORT installation
        resolved_providers = self._resolve_providers(self.providers)
        logger.info(
            f"Loading inswapper_128.onnx | providers={resolved_providers}"
        )

        try:
            sess_opts = ort.SessionOptions()  # type: ignore[union-attr]
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore[union-attr]
            )
            sess_opts.log_severity_level = 3  # suppress verbose ORT logs

            self._session = ort.InferenceSession(  # type: ignore[union-attr]
                str(path),
                providers=resolved_providers,
                sess_options=sess_opts,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create ONNX Runtime session for {self.model_path}: {exc}"
            ) from exc

        inputs  = self._session.get_inputs()
        outputs = self._session.get_outputs()

        if len(inputs) < 2:
            raise RuntimeError(
                f"Expected ≥2 inputs in inswapper model, found {len(inputs)}. "
                "The model file may be corrupt or incompatible."
            )

        # Convention used by the public inswapper_128.onnx:
        #   inputs[0] → target face crop  (1, 3, 128, 128)
        #   inputs[1] → latent code        (1, 512)
        #   outputs[0]→ swapped face crop  (1, 3, 128, 128)
        self._input_name  = inputs[0].name
        self._latent_name = inputs[1].name
        self._output_name = outputs[0].name

        # The emap is stored as an initialiser / weight in the ONNX graph.
        # It maps a 512-dim ArcFace embedding → 512-dim latent code that
        # the model can condition on.
        self._emap = self._extract_emap()

        self._is_loaded = True
        logger.success(
            f"inswapper_128 loaded | "
            f"input={self._input_name!r} latent={self._latent_name!r} "
            f"output={self._output_name!r} | "
            f"emap={'found' if self._emap is not None else 'not found (fallback)'}"
        )

    # ------------------------------------------------------------------
    # BaseSwapper interface — swap
    # ------------------------------------------------------------------

    def swap(self, request: SwapRequest) -> SwapResult:
        """
        Replace the face in ``request.target_face`` with the source identity.

        Steps:
          1. Validate inputs.
          2. Extract / estimate 5-point landmarks for alignment.
          3. Affine-warp target face crop → 128 × 128 canonical space.
          4. Pre-process crop tensor.
          5. Build latent identity vector from source embedding.
          6. Run ONNX inference.
          7. Post-process output tensor → BGR crop.
          8. Paste swapped crop back into target frame.

        Args:
            request: ``SwapRequest`` with source embedding + target face.

        Returns:
            ``SwapResult`` — always non-None; check ``.success`` or ``.status``.
        """
        t0 = self._timer()

        # Guard: model must be loaded
        if not self._is_loaded or self._session is None:
            return self._make_failed_result(
                SwapStatus.MODEL_NOT_LOADED,
                request.target_image,
                request.target_face,
                "Model not loaded. Call load_model() first.",
                t0,
            )

        # Guard: validate target image
        try:
            self._validate_image(request.target_image)
        except ValueError as exc:
            return self._make_failed_result(
                SwapStatus.NO_TARGET_FACE,
                request.target_image,
                request.target_face,
                str(exc),
                t0,
            )

        t_align = self._timer()
        landmarks = self._get_landmarks(request.target_face)

        aligned_crop, affine_M = norm_crop(
            request.target_image,
            landmarks,
            output_size=self.input_size,
        )

        if aligned_crop is None or affine_M is None:
            return self._make_failed_result(
                SwapStatus.ALIGN_ERROR,
                request.target_image,
                request.target_face,
                "Affine alignment failed — norm_crop returned None.",
                t0,
            )

        align_time = self._timer() - t_align

        crop_tensor = self._preprocess(aligned_crop)    # (1, 3, 128, 128)

        latent = self._build_latent(request.source_embedding)  # (1, 512)

        t_inf = self._timer()
        try:
            outputs = self._session.run(
                [self._output_name],
                {
                    self._input_name:  crop_tensor,
                    self._latent_name: latent,
                },
            )
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error(f"inswapper inference error: {exc}")
            return self._make_failed_result(
                SwapStatus.INFERENCE_ERROR,
                request.target_image,
                request.target_face,
                f"ONNX inference error: {exc}",
                t0,
            )
        inference_time = self._timer() - t_inf

        swapped_crop = self._postprocess(outputs[0])    # (128, 128, 3) BGR uint8

        t_blend = self._timer()

        blend_mode = request.blend_mode if request.blend_mode is not None else self.blend_mode
        feather    = request.mask_feather if request.mask_feather is not None else self.mask_feather
        alpha      = request.blend_alpha  if request.blend_alpha  is not None else self.blend_alpha

        try:
            output_frame = self._paste_back(
                original=request.target_image,
                swapped_crop=swapped_crop,
                affine_M=affine_M,
                blend_mode=blend_mode,
                alpha=alpha,
                feather=feather,
            )
        except (RuntimeError, ValueError, cv2.error) as exc:
            logger.error(f"paste-back error: {exc}")
            return self._make_failed_result(
                SwapStatus.BLEND_ERROR,
                request.target_image,
                request.target_face,
                f"Paste-back / blend error: {exc}",
                t0,
            )

        blend_time = self._timer() - t_blend

        with self._stats_lock:
            self._total_calls     += 1
            self._total_inference += inference_time
        total_time = self._timer() - t0

        logger.debug(
            f"swap | face_idx={request.target_face.face_index} | "
            f"align={align_time:.1f}ms "
            f"inf={inference_time:.1f}ms "
            f"blend={blend_time:.1f}ms "
            f"total={total_time:.1f}ms"
        )

        return SwapResult(
            output_image=output_frame,
            status=SwapStatus.SUCCESS,
            target_face=request.target_face,
            swap_time_ms=total_time,
            inference_time_ms=inference_time,
            align_time_ms=align_time,
            blend_time_ms=blend_time,
            intermediate={
                "aligned_crop":  aligned_crop,
                "swapped_crop":  swapped_crop,
                "affine_matrix": affine_M,
            } if request.metadata.get("save_intermediate") else None,
        )

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Free the ONNX session and clear model metadata."""
        self._session     = None
        self._emap        = None
        self._input_name  = None
        self._latent_name = None
        self._output_name = None
        super().release()
        logger.info("InSwapper released.")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def avg_inference_ms(self) -> float:
        """Average ONNX inference time per call in milliseconds."""
        with self._stats_lock:
            if self._total_calls == 0:
                return 0.0
            return self._total_inference / self._total_calls

    @property
    def total_calls(self) -> int:
        """Total number of swap() calls since model was loaded."""
        with self._stats_lock:
            return self._total_calls

    def reset_stats(self) -> None:
        """Reset cumulative inference statistics."""
        with self._stats_lock:
            self._total_calls     = 0
            self._total_inference = 0.0

    # ------------------------------------------------------------------
    # Internal preprocessing / postprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """
        Convert a BGR uint8 crop to a normalised (1, 3, 128, 128)
        float32 tensor suitable for inswapper input.

        Normalisation:  pixel = (pixel / 255.0 - 0.5) / 0.5
                        → range [-1.0, 1.0]

        Args:
            crop: (H, W, 3) BGR uint8 image.

        Returns:
            (1, 3, H, W) float32 numpy array.
        """
        # BGR → RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # HWC → CHW, [0, 255] → [-1.0, 1.0]
        tensor = rgb.astype(np.float32) / 255.0           # [0, 1]
        tensor = (tensor - _MEAN) / _STD                  # [-1, 1]
        tensor = tensor.transpose(2, 0, 1)                # CHW
        tensor = np.expand_dims(tensor, axis=0)           # NCHW
        return tensor

    @staticmethod
    def _postprocess(output: np.ndarray) -> np.ndarray:
        """
        Convert the model output tensor back to a BGR uint8 image.

        Args:
            output: (1, 3, H, W) float32 tensor in [-1.0, 1.0].

        Returns:
            (H, W, 3) BGR uint8 image.
        """
        # Remove batch dim, CHW → HWC
        img = output[0].transpose(1, 2, 0)                # HWC, [-1, 1]

        # [-1, 1] → [0, 255]
        img = (img * _STD + _MEAN) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        # RGB → BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    # ------------------------------------------------------------------
    # Internal: latent identity construction
    # ------------------------------------------------------------------

    def _build_latent(self, embedding: FaceEmbedding) -> np.ndarray:
        """
        Map a 512-dim ArcFace embedding to a (1, 512) latent code via
        the model's internal ``emap`` matrix.

        The emap is a learned linear projection that adapts the external
        ArcFace embedding space to the model's internal identity space.
        If the emap was not extracted from the model (rare), the raw
        normalised embedding is used directly as a fallback.

        Args:
            embedding: ``FaceEmbedding`` with a normalised ``.vector``.

        Returns:
            (1, 512) float32 latent code tensor.
        """
        vec = embedding.vector.astype(np.float32).flatten()  # (512,)

        # L2-normalise the embedding
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        if self._emap is not None:
            # Project through emap: (512,) @ (512, 512) → (512,)
            latent = vec @ self._emap
            # Normalise the projected latent
            lat_norm = np.linalg.norm(latent)
            if lat_norm > 1e-8:
                latent = latent / lat_norm
        else:
            latent = vec

        return latent.reshape(1, -1).astype(np.float32)  # (1, 512)

    def _extract_emap(self) -> Optional[np.ndarray]:
        """
        Extract the ``emap`` weight matrix from the ONNX model graph.

        The emap is stored as a graph initialiser with shape (512, 512).
        Iterates over all initialisers to find one matching that shape.

        Returns:
            (512, 512) float32 numpy array, or None if not found.
        """
        if self._session is None:
            return None

        try:
            import onnx  # noqa: PLC0415

            model_proto = onnx.load(str(self.model_path))
            for initializer in model_proto.graph.initializer:
                shape = list(initializer.dims)
                if shape == [512, 512]:
                    arr = np.array(initializer.float_data, dtype=np.float32)
                    if arr.size == 512 * 512:
                        return arr.reshape(512, 512)
                    # raw_data path
                    if initializer.raw_data:
                        arr = np.frombuffer(
                            initializer.raw_data, dtype=np.float32
                        ).copy()
                        if arr.size == 512 * 512:
                            return arr.reshape(512, 512)
        except ImportError:
            logger.warning(
                "onnx package not installed — cannot extract emap. "
                "Raw embedding will be used as latent (slightly lower quality). "
                "Install with: pip install onnx"
            )
        except Exception as exc:
            logger.warning(f"emap extraction failed ({exc}); using raw embedding fallback.")

        return None

    # ------------------------------------------------------------------
    # Internal: paste-back dispatch
    # ------------------------------------------------------------------

    def _paste_back(
        self,
        original: np.ndarray,
        swapped_crop: np.ndarray,
        affine_M: np.ndarray,
        blend_mode: BlendMode,
        alpha: float,
        feather: int,
    ) -> np.ndarray:
        """
        Dispatch to the correct paste-back strategy based on *blend_mode*.

        Args:
            original:     BGR target frame.
            swapped_crop: BGR swapped face crop (128 × 128).
            affine_M:     (2, 3) forward affine matrix.
            blend_mode:   Which compositing approach to use.
            alpha:        Global alpha weight applied after blending.
            feather:      Mask edge softening radius in pixels.

        Returns:
            BGR frame with the swapped face pasted back.
        """
        if blend_mode == BlendMode.POISSON:
            swapped = paste_back_poisson(
                original=original,
                swapped_crop=swapped_crop,
                affine_matrix=affine_M,
                mask=_make_crop_mask(self.input_size, feather=0),
            )
        elif blend_mode in (BlendMode.ALPHA, BlendMode.MASKED_ALPHA):
            mask = _make_crop_mask(self.input_size, feather=feather)
            swapped = paste_back(
                original=original,
                swapped_crop=swapped_crop,
                affine_matrix=affine_M,
                blend_mask=mask,
                feather=feather,
            )
        else:
            # Unknown mode — default to alpha
            mask = _make_crop_mask(self.input_size, feather=feather)
            swapped = paste_back(
                original=original,
                swapped_crop=swapped_crop,
                affine_matrix=affine_M,
                blend_mask=mask,
                feather=feather,
            )

        # Apply global alpha blend against the original frame
        if alpha < 1.0:
            swapped = cv2.addWeighted(
                swapped,  alpha,
                original, 1.0 - alpha,
                0,
            )

        return swapped

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        providers = getattr(self, "providers", [])
        return (
            f"InSwapper("
            f"model={self.model_name!r}, "
            f"blend={self.blend_mode.name}, "
            f"input_size={self.input_size}, "
            f"providers={providers}, "
            f"status={status}, "
            f"calls={self._total_calls})"
        )
