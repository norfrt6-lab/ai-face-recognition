from __future__ import annotations

import os
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


class VideoMeta:
    """Metadata container for a video file."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
        duration_sec: float,
        codec: str,
        has_audio: bool,
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration_sec = duration_sec
        self.codec = codec
        self.has_audio = has_audio

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return (width, height)."""
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"VideoMeta(path={self.path.name!r}, "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, "
            f"frames={self.frame_count}, "
            f"duration={self.duration_sec:.2f}s, "
            f"codec={self.codec!r}, "
            f"audio={self.has_audio})"
        )


def get_video_meta(video_path: str | Path) -> VideoMeta:
    """
    Read video metadata without decoding frames.

    Args:
        video_path: Path to the input video file.

    Returns:
        VideoMeta instance populated with video properties.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
    finally:
        cap.release()

    # Detect audio via ffprobe (graceful fallback if ffprobe unavailable)
    has_audio = _probe_audio(path)

    meta = VideoMeta(
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        codec=codec,
        has_audio=has_audio,
    )
    logger.debug(f"Video meta: {meta}")
    return meta


def _probe_audio(path: Path) -> bool:
    """Return True if the video file contains at least one audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "audio"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # ffprobe not installed or timed out — assume no audio
        return False


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    show_progress: bool = True,
) -> List[Path]:
    """
    Extract frames from a video file and save them as PNG images.

    Args:
        video_path:     Path to the input video.
        output_dir:     Directory where extracted frame images will be saved.
        skip_frames:    Number of frames to skip between each extracted frame
                        (0 = extract every frame, 1 = every other frame, …).
        max_frames:     Maximum number of frames to extract (None = all).
        resize:         Optional (width, height) to resize each frame.
        show_progress:  Show a tqdm progress bar.

    Returns:
        Sorted list of Paths to the extracted frame images.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    meta = get_video_meta(video_path)
    total = meta.frame_count
    saved_paths: List[Path] = []
    frame_idx = 0
    saved_idx = 0

    step = skip_frames + 1  # e.g. skip_frames=1 → take every 2nd frame

    pbar = tqdm(
        total=min(total, max_frames) if max_frames else total,
        desc="Extracting frames",
        unit="frame",
        disable=not show_progress,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if max_frames and saved_idx >= max_frames:
                    break

                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LANCZOS4)

                frame_filename = output_dir / f"frame_{saved_idx:06d}.png"
                cv2.imwrite(str(frame_filename), frame)
                saved_paths.append(frame_filename)
                saved_idx += 1
                pbar.update(1)

            frame_idx += 1
    finally:
        cap.release()
        pbar.close()

    logger.info(f"Extracted {len(saved_paths)} frames → {output_dir}")
    return sorted(saved_paths)


def iter_frames(
    video_path: str | Path,
    *,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
    as_rgb: bool = False,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Lazily yield (frame_index, frame_array) from a video without saving to disk.

    Useful for streaming pipelines where frames are processed on-the-fly.

    Args:
        video_path:   Path to the input video file.
        skip_frames:  Frames to skip between yields.
        max_frames:   Stop after yielding this many frames.
        resize:       Optional (width, height) to resize each frame.
        as_rgb:       If True, convert BGR → RGB before yielding.

    Yields:
        Tuple of (frame_index, numpy array HxWxC).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    step = skip_frames + 1
    frame_idx = 0
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                if max_frames and yielded >= max_frames:
                    break

                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LANCZOS4)

                if as_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                yield frame_idx, frame
                yielded += 1

            frame_idx += 1
    finally:
        cap.release()


def iter_webcam(
    device_id: int = 0,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    as_rgb: bool = False,
) -> Iterator[np.ndarray]:
    """
    Lazily yield frames from a webcam / capture device.

    Args:
        device_id:  OpenCV capture device index (0 = default webcam).
        width:      Requested capture width in pixels.
        height:     Requested capture height in pixels.
        fps:        Requested capture FPS.
        as_rgb:     Convert BGR → RGB before yielding.

    Yields:
        numpy arrays of shape (H, W, 3).

    Note:
        The caller is responsible for stopping iteration (break / KeyboardInterrupt).
    """
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at device_id={device_id}")

    logger.info(
        f"Webcam opened: device={device_id}, "
        f"res={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"fps={cap.get(cv2.CAP_PROP_FPS)}"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Webcam read failed — stopping.")
                break
            if as_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
    finally:
        cap.release()
        logger.info("Webcam released.")


class VideoWriter:
    """
    Context-manager wrapper around cv2.VideoWriter.

    Usage::

        with VideoWriter("output.mp4", fps=25.0, size=(1280, 720)) as vw:
            for frame in frames:
                vw.write(frame)
    """

    # Mapping of file extensions → fourcc codes
    _FOURCC_MAP = {
        ".mp4": "mp4v",
        ".avi": "XVID",
        ".mov": "mp4v",
        ".mkv": "X264",
        ".wmv": "WMV2",
    }

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        size: Tuple[int, int],
        *,
        fourcc: Optional[str] = None,
        is_color: bool = True,
    ) -> None:
        """
        Args:
            output_path: Destination file path (extension determines codec).
            fps:         Frames per second for the output video.
            size:        (width, height) of each frame.
            fourcc:      Override codec FourCC string (e.g. 'mp4v', 'H264').
            is_color:    True for colour frames, False for greyscale.
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.size = size  # (width, height)
        self.is_color = is_color
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

        # Resolve fourcc
        ext = self.output_path.suffix.lower()
        _fourcc_str = fourcc or self._FOURCC_MAP.get(ext, "mp4v")
        self._fourcc = cv2.VideoWriter_fourcc(*_fourcc_str)

    def open(self) -> "VideoWriter":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            self._fourcc,
            self.fps,
            self.size,
            self.is_color,
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"cv2.VideoWriter failed to open: {self.output_path} "
                f"(fourcc={self._fourcc}, fps={self.fps}, size={self.size})"
            )
        logger.debug(
            f"VideoWriter opened: {self.output_path.name} "
            f"| {self.size[0]}x{self.size[1]} @ {self.fps:.2f} fps"
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single frame. Automatically resizes if the frame does not
        match the writer's configured resolution.

        Args:
            frame: BGR numpy array of shape (H, W, 3) or (H, W) for greyscale.
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter is not open. Call open() or use as context manager.")

        h, w = frame.shape[:2]
        if (w, h) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)

        self._writer.write(frame)
        self._frames_written += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                f"VideoWriter closed: {self.output_path.name} "
                f"({self._frames_written} frames written)"
            )

    # Context manager support
    def __enter__(self) -> "VideoWriter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @property
    def frames_written(self) -> int:
        return self._frames_written


def frames_to_video(
    frame_paths: List[str | Path],
    output_path: str | Path,
    fps: float,
    *,
    size: Optional[Tuple[int, int]] = None,
    show_progress: bool = True,
) -> Path:
    """
    Assemble a list of frame image files into a video file.

    Args:
        frame_paths:    Ordered list of paths to frame images (PNG/JPG).
        output_path:    Destination video file path.
        fps:            Output video frame rate.
        size:           (width, height) override. If None, inferred from first frame.
        show_progress:  Show tqdm progress bar.

    Returns:
        Path to the written video file.
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty — no frames to write.")

    output_path = Path(output_path)

    # Infer size from first frame if not provided
    if size is None:
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        h, w = first.shape[:2]
        size = (w, h)

    with VideoWriter(output_path, fps=fps, size=size) as vw:
        for fpath in tqdm(
            frame_paths, desc="Writing video", unit="frame", disable=not show_progress
        ):
            frame = cv2.imread(str(fpath))
            if frame is None:
                logger.warning(f"Skipping unreadable frame: {fpath}")
                continue
            vw.write(frame)

    logger.info(f"Video assembled → {output_path}")
    return output_path


def extract_audio(video_path: str | Path, audio_out: str | Path) -> Optional[Path]:
    """
    Extract the audio track from a video file using ffmpeg.

    Args:
        video_path: Source video file.
        audio_out:  Destination audio file (e.g. 'audio.aac' or 'audio.mp3').

    Returns:
        Path to the extracted audio file, or None if extraction failed /
        no audio stream was found.
    """
    video_path = Path(video_path)
    audio_out = Path(audio_out)
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    if not _probe_audio(video_path):
        logger.debug(f"No audio stream found in {video_path.name}")
        return None

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # overwrite output
                "-i",
                str(video_path),
                "-vn",  # no video
                "-acodec",
                "copy",
                str(audio_out),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        logger.info(f"Audio extracted → {audio_out}")
        return audio_out
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Audio extraction failed: {exc}")
        return None


def merge_audio_video(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Merge a (muted) processed video with the original audio track using ffmpeg.

    Args:
        video_path:   Path to the video file (no audio / muted).
        audio_path:   Path to the audio file to embed.
        output_path:  Destination for the merged output.

    Returns:
        Path to the output file.

    Raises:
        RuntimeError: If ffmpeg is not installed or merging fails.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        logger.info(f"Audio+Video merged → {output_path}")
        return output_path
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it with: "
            "winget install ffmpeg  (Windows) / brew install ffmpeg (macOS) / "
            "apt install ffmpeg (Ubuntu)"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg merge failed: {exc.stderr.decode()}")


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    frame_processor,  # Callable[[np.ndarray], np.ndarray]
    *,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    preserve_audio: bool = True,
    show_progress: bool = True,
) -> Path:
    """
    High-level helper: read video → apply frame_processor → write output.

    Optionally preserves the original audio track (requires ffmpeg).

    Args:
        input_path:       Source video file.
        output_path:      Destination video file.
        frame_processor:  Callable that takes a BGR frame (np.ndarray)
                          and returns a processed BGR frame.
        skip_frames:      Skip N frames between each processed frame.
        max_frames:       Stop after processing this many frames.
        preserve_audio:   Attempt to copy the original audio to the output.
        show_progress:    Show a tqdm progress bar.

    Returns:
        Path to the output video file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    meta = get_video_meta(input_path)

    effective_fps = meta.fps / (skip_frames + 1) if skip_frames > 0 else meta.fps

    with VideoWriter(output_path, fps=effective_fps, size=meta.resolution) as vw:
        total = max_frames or meta.frame_count
        pbar = tqdm(
            total=total,
            desc="Processing video",
            unit="frame",
            disable=not show_progress,
        )
        try:
            for _, frame in iter_frames(
                input_path,
                skip_frames=skip_frames,
                max_frames=max_frames,
            ):
                processed = frame_processor(frame)
                vw.write(processed)
                pbar.update(1)
        finally:
            pbar.close()

    # Preserve original audio
    if preserve_audio and meta.has_audio:
        logger.info("Merging original audio track into output...")
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
            tmp_audio = Path(tmp.name)

        try:
            extracted = extract_audio(input_path, tmp_audio)
            if extracted:
                # Write to a temp output then replace
                tmp_output = output_path.with_suffix(".merged.mp4")
                merge_audio_video(output_path, extracted, tmp_output)
                output_path.unlink(missing_ok=True)
                tmp_output.rename(output_path)
        finally:
            tmp_audio.unlink(missing_ok=True)

    logger.success(f"Video processing complete → {output_path}")
    return output_path


def get_frame_at(
    video_path: str | Path,
    frame_index: int,
    *,
    as_rgb: bool = False,
) -> np.ndarray:
    """
    Read a single frame at a specific index from a video file.

    Args:
        video_path:   Path to the video.
        frame_index:  Zero-based frame index.
        as_rgb:       Convert BGR → RGB.

    Returns:
        numpy array of shape (H, W, 3).

    Raises:
        IndexError:  If frame_index is beyond the video's frame count.
        RuntimeError: If the frame cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if frame_index < 0:
        cap.release()
        raise IndexError(f"frame_index must be non-negative, got {frame_index}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index >= total:
        cap.release()
        raise IndexError(f"frame_index {frame_index} is out of range (total={total})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")

    if as_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def resize_video(
    input_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
    *,
    show_progress: bool = True,
) -> Path:
    """
    Resize all frames of a video to (width, height).

    Args:
        input_path:     Source video.
        output_path:    Destination video.
        width:          Target frame width.
        height:         Target frame height.
        show_progress:  tqdm progress.

    Returns:
        Path to the resized video.
    """
    return process_video(
        input_path,
        output_path,
        frame_processor=lambda f: cv2.resize(f, (width, height), interpolation=cv2.INTER_LANCZOS4),
        show_progress=show_progress,
    )
