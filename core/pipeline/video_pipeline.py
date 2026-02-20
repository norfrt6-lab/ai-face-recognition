# Frame-by-frame video processing pipeline.
#
# Orchestrates the full detect → recognise → swap → enhance
# pipeline over every frame of a video file or webcam stream.
#
# Key features:
#   - tqdm progress bar with live FPS display
#   - skip_frames  — process every Nth frame (performance tuning)
#   - Error recovery — bad frames are logged and skipped
#   - FFmpeg audio merge — preserve original audio track
#   - Thread-safe frame queue for smooth I/O
#   - Progress callbacks for UI integration (Streamlit)
#   - Source/target mode: one source identity → all target faces

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from core.detector.base_detector import BaseDetector, DetectionResult
from core.recognizer.base_recognizer import BaseRecognizer, FaceEmbedding
from core.swapper.base_swapper import BaseSwapper, BlendMode, SwapRequest
from core.swapper.inswapper import InSwapper
from core.enhancer.base_enhancer import BaseEnhancer, EnhancementRequest
from core.tracker.iou_tracker import IoUTracker
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoProcessingConfig:
    """
    Configuration for a single video pipeline run.

    Attributes:
        source_embedding:    ArcFace embedding of the donor identity whose
                             face will be injected into every target frame.
        blend_mode:          Compositing mode for paste-back.
        blend_alpha:         Global blend strength [0.0, 1.0].
        mask_feather:        Gaussian feather radius for blend mask (px).
        swap_all_faces:      Replace every detected face per frame.
                             When False only the largest face is swapped.
        max_faces:           Cap on simultaneous face swaps per frame.
        skip_frames:         Process only every (skip_frames+1)-th frame.
                             0 = process every frame.
                             1 = process every other frame, etc.
        enhance:             Run face enhancement after each swap.
        preserve_audio:      Merge original audio into the output file
                             using FFmpeg.
        output_fps:          Override output frame rate (None = source FPS).
        output_format:       Video container format ('mp4', 'avi', 'mkv').
        output_codec:        FFmpeg codec string for VideoWriter (e.g. 'mp4v').
        max_resolution:      (width, height) cap — frames are resized down
                             if either dimension exceeds these values.
        watermark:           Overlay an "AI GENERATED" text watermark on
                             every output frame.
        watermark_text:      Text for the watermark overlay.
        save_intermediate:   Save every raw + swapped frame pair to disk
                             for debugging.
        progress_callback:   Optional callable(current_frame, total_frames)
                             called after each processed frame — used by
                             Streamlit to update a progress bar.
    """

    source_embedding:  FaceEmbedding

    blend_mode:        BlendMode  = BlendMode.POISSON
    blend_alpha:       float      = 1.0
    mask_feather:      int        = 20
    swap_all_faces:    bool       = True
    max_faces:         int        = 10
    skip_frames:       int        = 0
    enhance:           bool       = False
    preserve_audio:    bool       = True
    output_fps:        Optional[float]  = None
    output_format:     str        = "mp4"
    output_codec:      str        = "mp4v"
    max_resolution:    Optional[tuple]  = None   # (w, h)
    watermark:         bool       = True
    watermark_text:    str        = "AI GENERATED"
    enable_tracking:   bool       = True
    tracking_iou:      float      = 0.3
    tracking_max_age:  int        = 5
    save_intermediate: bool       = False
    progress_callback: Optional[Callable[[int, int], None]] = field(
        default=None, repr=False
    )


@dataclass
class VideoProcessingResult:
    """
    Summary result from a complete video pipeline run.

    Attributes:
        output_path:       Path to the processed output video file.
        total_frames:      Total frames in the source video.
        processed_frames:  Frames that went through swap/enhance.
        skipped_frames:    Frames skipped due to skip_frames setting.
        failed_frames:     Frames where swap/detect raised an error.
        total_time_s:      Wall-clock time for the entire run (seconds).
        avg_fps:           Average processed frames per second.
        source_fps:        Original video frame rate.
        source_resolution: (width, height) of the source video.
    """

    output_path:        str
    total_frames:       int   = 0
    processed_frames:   int   = 0
    skipped_frames:     int   = 0
    failed_frames:      int   = 0
    total_time_s:       float = 0.0
    avg_fps:            float = 0.0
    source_fps:         float = 0.0
    source_resolution:  tuple = field(default_factory=lambda: (0, 0))

    @property
    def success(self) -> bool:
        return self.processed_frames > 0

    def __repr__(self) -> str:
        return (
            f"VideoProcessingResult("
            f"output={self.output_path!r}, "
            f"frames={self.processed_frames}/{self.total_frames}, "
            f"failed={self.failed_frames}, "
            f"avg_fps={self.avg_fps:.1f}, "
            f"time={self.total_time_s:.1f}s)"
        )


class VideoPipeline:
    """
    Frame-by-frame face swap video pipeline.

    Reads a source video, applies face detection → swap → (optional)
    enhancement on each frame, and writes the result to a new video
    file, optionally merging the original audio track back in via
    FFmpeg.

    Usage::

        pipeline = VideoPipeline(
            detector=YOLOFaceDetector("models/yolov8n-face.pt"),
            swapper=InSwapper("models/inswapper_128.onnx"),
        )

        cfg = VideoProcessingConfig(
            source_embedding=embedding,
            blend_mode=BlendMode.POISSON,
            skip_frames=0,
            enhance=False,
            preserve_audio=True,
        )

        result = pipeline.process(
            source_video="input.mp4",
            output_path="output.mp4",
            config=cfg,
        )
        print(result)

    Args:
        detector:  Loaded ``BaseDetector`` instance.
        swapper:   Loaded ``BaseSwapper`` instance.
        enhancer:  Optional loaded ``BaseEnhancer`` instance.
    """

    def __init__(
        self,
        detector:  BaseDetector,
        swapper:   BaseSwapper,
        enhancer:  Optional[BaseEnhancer] = None,
    ) -> None:
        self.detector  = detector
        self.swapper   = swapper
        self.enhancer  = enhancer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        source_video: str,
        output_path:  str,
        config:       VideoProcessingConfig,
    ) -> VideoProcessingResult:
        """
        Process *source_video* end-to-end and write the result to
        *output_path*.

        Args:
            source_video: Path to the input video file.
            output_path:  Path to write the processed output video.
            config:       ``VideoProcessingConfig`` controlling every
                          aspect of the processing run.

        Returns:
            ``VideoProcessingResult`` with timing and frame statistics.

        Raises:
            FileNotFoundError: If *source_video* does not exist.
            RuntimeError:      If the VideoCapture or VideoWriter
                               cannot be opened.
        """
        source_path = Path(source_video)
        if not source_path.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source_video}")

        source_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_res    = (src_w, src_h)

        output_fps    = config.output_fps or source_fps
        out_w, out_h  = self._capped_resolution(src_w, src_h, config.max_resolution)

        logger.info(
            f"VideoPipeline: {source_video!r} → {output_path!r} | "
            f"{total_frames} frames @ {source_fps:.2f} fps | "
            f"resolution {src_w}x{src_h} → {out_w}x{out_h} | "
            f"skip_frames={config.skip_frames}"
        )

        fourcc = cv2.VideoWriter_fourcc(*config.output_codec)
        # Write to a temp file first so we can merge audio after
        temp_video_path = str(out_path.with_suffix(f".tmp_noaudio{out_path.suffix}"))
        writer = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            output_fps,
            (out_w, out_h),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open VideoWriter for {output_path} "
                f"(codec={config.output_codec!r}, size={out_w}x{out_h})"
            )

        # Initialise face tracker for temporal consistency (local to this run)
        tracker: Optional[IoUTracker] = None
        if config.enable_tracking:
            tracker = IoUTracker(
                iou_threshold=config.tracking_iou,
                max_age=config.tracking_max_age,
            )

        t_start          = time.perf_counter()
        processed_frames = 0
        skipped_frames   = 0
        failed_frames    = 0
        frame_idx        = 0

        with tqdm(
            total=total_frames,
            desc="Processing",
            unit="frame",
            dynamic_ncols=True,
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize to capped resolution if needed
                if (src_w, src_h) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                # Skip frames if configured
                if config.skip_frames > 0 and (frame_idx % (config.skip_frames + 1)) != 0:
                    writer.write(frame)
                    skipped_frames += 1
                    frame_idx      += 1
                    pbar.update(1)
                    continue

                try:
                    output_frame = self._process_frame(frame, config, tracker)
                    processed_frames += 1
                except Exception as exc:
                    logger.warning(
                        f"Frame {frame_idx} failed: {exc} — writing original."
                    )
                    output_frame = frame
                    failed_frames += 1

                # Watermark
                if config.watermark:
                    output_frame = self._add_watermark(
                        output_frame, config.watermark_text
                    )

                writer.write(output_frame)
                frame_idx += 1
                pbar.update(1)

                # FPS display
                elapsed = time.perf_counter() - t_start
                fps_now = processed_frames / elapsed if elapsed > 0 else 0.0
                pbar.set_postfix(fps=f"{fps_now:.1f}", refresh=False)

                # Progress callback (Streamlit / UI)
                if config.progress_callback:
                    try:
                        config.progress_callback(frame_idx, total_frames)
                    except Exception:
                        pass  # never let callback crash the pipeline

        cap.release()
        writer.release()

        total_time = time.perf_counter() - t_start
        avg_fps    = processed_frames / total_time if total_time > 0 else 0.0

        logger.info(
            f"Video processing done | "
            f"processed={processed_frames} skipped={skipped_frames} "
            f"failed={failed_frames} | "
            f"time={total_time:.1f}s avg_fps={avg_fps:.1f}"
        )

        final_output = str(out_path)
        if config.preserve_audio:
            merged = self._merge_audio(
                video_path=temp_video_path,
                audio_source=str(source_path),
                output_path=final_output,
            )
            if not merged:
                # Audio merge failed — just rename the silent video
                import shutil
                shutil.move(temp_video_path, final_output)
        else:
            import shutil
            shutil.move(temp_video_path, final_output)

        # Clean up temp file if still present
        temp_path_obj = Path(temp_video_path)
        if temp_path_obj.exists() and temp_path_obj != out_path:
            try:
                temp_path_obj.unlink()
            except OSError:
                pass

        return VideoProcessingResult(
            output_path=final_output,
            total_frames=frame_idx,
            processed_frames=processed_frames,
            skipped_frames=skipped_frames,
            failed_frames=failed_frames,
            total_time_s=total_time,
            avg_fps=avg_fps,
            source_fps=source_fps,
            source_resolution=source_res,
        )

    def process_webcam(
        self,
        config:       VideoProcessingConfig,
        output_path:  Optional[str] = None,
        camera_index: int = 0,
        max_frames:   Optional[int] = None,
        display:      bool = True,
    ) -> VideoProcessingResult:
        """
        Process live webcam frames until the user presses 'q'.

        Args:
            config:       ``VideoProcessingConfig``.
            output_path:  Optional file path to record the output.
            camera_index: OpenCV camera index (0 = default webcam).
            max_frames:   Stop after this many frames (None = unlimited).
            display:      Show a live preview window via cv2.imshow.

        Returns:
            ``VideoProcessingResult`` summary.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {camera_index}.")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = self._capped_resolution(src_w, src_h, config.max_resolution)

        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*config.output_codec)
            writer = cv2.VideoWriter(
                str(out_path), fourcc, src_fps, (out_w, out_h)
            )

        tracker: Optional[IoUTracker] = None
        if config.enable_tracking:
            tracker = IoUTracker(
                iou_threshold=config.tracking_iou,
                max_age=config.tracking_max_age,
            )

        t_start          = time.perf_counter()
        processed_frames = 0
        failed_frames    = 0
        frame_idx        = 0

        logger.info(
            f"Webcam pipeline started | camera={camera_index} | "
            f"resolution={src_w}x{src_h}"
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if (src_w, src_h) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h))

                try:
                    output_frame = self._process_frame(frame, config, tracker)
                    processed_frames += 1
                except Exception as exc:
                    logger.debug(f"Webcam frame {frame_idx} error: {exc}")
                    output_frame = frame
                    failed_frames += 1

                if config.watermark:
                    output_frame = self._add_watermark(
                        output_frame, config.watermark_text
                    )

                if writer:
                    writer.write(output_frame)

                if display:
                    cv2.imshow("AI Face Swap — Press Q to quit", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        total_time = time.perf_counter() - t_start
        avg_fps    = processed_frames / total_time if total_time > 0 else 0.0

        logger.info(
            f"Webcam pipeline ended | frames={processed_frames} "
            f"failed={failed_frames} | avg_fps={avg_fps:.1f}"
        )

        return VideoProcessingResult(
            output_path=output_path or "",
            total_frames=frame_idx,
            processed_frames=processed_frames,
            failed_frames=failed_frames,
            total_time_s=total_time,
            avg_fps=avg_fps,
            source_fps=src_fps,
            source_resolution=(src_w, src_h),
        )

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame:   np.ndarray,
        config:  VideoProcessingConfig,
        tracker: Optional[IoUTracker] = None,
    ) -> np.ndarray:
        """
        Run detect → swap → (optional enhance) on a single BGR frame.

        Args:
            frame:   Input BGR frame.
            config:  Pipeline configuration.
            tracker: Optional IoU tracker for temporal consistency.

        Returns:
            Processed BGR frame (same shape as input).
        """
        detection: DetectionResult = self.detector.detect(frame)

        if detection.is_empty:
            if tracker is not None:
                tracker.update([])  # age existing tracks
            return frame   # no faces → return unmodified

        # Assign persistent track IDs across frames
        if tracker is not None:
            tracked = tracker.update(detection.faces)
            detection = DetectionResult(
                faces=tracked,
                image_width=detection.image_width,
                image_height=detection.image_height,
                inference_time_ms=detection.inference_time_ms,
                frame_index=detection.frame_index,
                metadata=detection.metadata,
            )

        if config.swap_all_faces:
            batch = self.swapper.swap_all(
                source_embedding=config.source_embedding,
                target_image=frame,
                target_detection=detection,
                blend_mode=config.blend_mode,
                blend_alpha=config.blend_alpha,
                mask_feather=config.mask_feather,
                max_faces=config.max_faces,
            )
            output_frame = batch.output_image
        else:
            best_face = detection.best_face
            if best_face is None:
                return frame
            req = SwapRequest(
                source_embedding=config.source_embedding,
                target_image=frame,
                target_face=best_face,
                blend_mode=config.blend_mode,
                blend_alpha=config.blend_alpha,
                mask_feather=config.mask_feather,
            )
            result = self.swapper.swap(req)
            output_frame = result.output_image if result.success else frame

        if config.enhance and self.enhancer is not None and self.enhancer.is_loaded:
            enh_req = EnhancementRequest(
                image=output_frame,
                full_frame=True,
                only_center_face=False,
                paste_back=True,
            )
            enh_result = self.enhancer.enhance(enh_req)
            if enh_result.success:
                # Resize back to original frame dimensions if upscaling changed size
                h0, w0 = frame.shape[:2]
                h1, w1 = enh_result.output_image.shape[:2]
                if (h1, w1) != (h0, w0):
                    output_frame = cv2.resize(
                        enh_result.output_image, (w0, h0),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    output_frame = enh_result.output_image

        return output_frame

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _capped_resolution(
        w: int,
        h: int,
        max_res: Optional[tuple],
    ) -> tuple:
        """
        Return (new_w, new_h) scaled down to fit within *max_res*.
        Preserves aspect ratio. If *max_res* is None, returns (w, h).

        Args:
            w:       Source width.
            h:       Source height.
            max_res: (max_width, max_height) tuple or None.

        Returns:
            (capped_width, capped_height) as a tuple of ints.
        """
        if max_res is None:
            return (w, h)

        max_w, max_h = max_res
        scale = min(max_w / w, max_h / h, 1.0)
        # Round down to even numbers — many codecs (H.264) require even dims
        new_w = int(w * scale) & ~1
        new_h = int(h * scale) & ~1
        return (max(2, new_w), max(2, new_h))

    @staticmethod
    def _add_watermark(
        frame: np.ndarray,
        text:  str,
    ) -> np.ndarray:
        """
        Overlay a semi-transparent text watermark on *frame*.

        The text is drawn in the bottom-left corner with a dark
        background for readability.

        Args:
            frame: BGR input frame.
            text:  Watermark text string.

        Returns:
            Frame with watermark applied (same shape).
        """
        frame = frame.copy()
        h, w  = frame.shape[:2]

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, w / 1280 * 0.6)
        thickness  = max(1, int(w / 640))

        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        margin = int(h * 0.02)
        x = margin
        y = h - margin

        # Draw semi-transparent background rectangle
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 4, y - th - baseline - 4),
            (x + tw + 4, y + baseline + 4),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw text
        cv2.putText(
            frame, text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    @staticmethod
    def _merge_audio(
        video_path:   str,
        audio_source: str,
        output_path:  str,
    ) -> bool:
        """
        Merge the audio track from *audio_source* into *video_path*
        using FFmpeg.

        Args:
            video_path:   Path to the silent processed video.
            audio_source: Path to the original video (audio donor).
            output_path:  Path for the final video with audio.

        Returns:
            True if merge succeeded, False otherwise.
        """
        import subprocess  # noqa: PLC0415

        cmd = [
            "ffmpeg",
            "-y",                    # overwrite output
            "-i",  video_path,       # processed (silent) video
            "-i",  audio_source,     # original (audio donor)
            "-map", "0:v:0",         # take video from first input
            "-map", "1:a?",          # take audio from second (optional)
            "-c:v", "copy",          # no re-encode video
            "-c:a", "aac",           # re-encode audio to AAC
            "-shortest",             # match shorter stream length
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,         # 10-minute hard timeout
            )
            if result.returncode == 0:
                logger.info(f"Audio merged → {output_path}")
                return True
            else:
                err = result.stderr.decode("utf-8", errors="replace")[-500:]
                logger.warning(f"FFmpeg audio merge failed: {err}")
                return False
        except FileNotFoundError:
            logger.warning(
                "FFmpeg not found — audio will not be preserved. "
                "Install FFmpeg and ensure it is on PATH."
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg audio merge timed out.")
            return False
        except Exception as exc:
            logger.warning(f"Audio merge error: {exc}")
            return False

    def __repr__(self) -> str:
        return (
            f"VideoPipeline("
            f"detector={self.detector.__class__.__name__}, "
            f"swapper={self.swapper.__class__.__name__}, "
            f"enhancer={self.enhancer.__class__.__name__ if self.enhancer else None})"
        )
