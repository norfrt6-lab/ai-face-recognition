from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

# ── Type aliases ────────────────────────────────────────────
# All internal frames are np.ndarray in BGR uint8 (OpenCV convention)
Frame = np.ndarray  # shape (H, W, 3) or (H, W, 4)  dtype=uint8
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Point = Tuple[int, int]  # (x, y)


def load_image(
    source: Union[str, Path, bytes, np.ndarray],
    color_mode: str = "BGR",
) -> Frame:
    """
    Load an image from a file path, raw bytes, or pass-through ndarray.

    Args:
        source:     File path, bytes, or already-loaded ndarray.
        color_mode: Desired output color space: 'BGR' | 'RGB' | 'GRAY'.

    Returns:
        Image as np.ndarray in the requested color space.

    Raises:
        ValueError: If the source cannot be decoded or is an unsupported type.
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
        # If already BGR, convert only if another mode is requested
        if color_mode == "RGB":
            img = bgr_to_rgb(img)
        elif color_mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"OpenCV could not decode image: {path}")
    elif isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("OpenCV could not decode image from bytes.")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Normalise channel count to 3 (BGR)
    img = normalise_channels(img)

    if color_mode == "RGB":
        img = bgr_to_rgb(img)
    elif color_mode == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def save_image(
    image: Frame,
    dest: Union[str, Path],
    quality: int = 95,
) -> Path:
    """
    Save an image ndarray (BGR) to disk.

    Args:
        image:   BGR ndarray.
        dest:    Output file path (.jpg / .png / .webp …).
        quality: JPEG / WebP quality (1-100), ignored for PNG.

    Returns:
        Resolved output path.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ext = dest.suffix.lower()
    params: List[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif ext == ".png":
        # PNG compression 0-9, map quality 0-100 → 0-9 inverted
        png_compression = max(0, min(9, (100 - quality) // 10))
        params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    ok = cv2.imwrite(str(dest), image, params)
    if not ok:
        raise IOError(f"Failed to save image to: {dest}")
    logger.debug(f"Image saved → {dest}")
    return dest


def image_to_bytes(image: Frame, ext: str = ".jpg", quality: int = 95) -> bytes:
    """Encode an image ndarray to raw bytes (e.g. for API responses)."""
    params: List[int] = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"cv2.imencode failed for ext={ext}")
    return buf.tobytes()


def image_to_base64(image: Frame, ext: str = ".jpg", quality: int = 95) -> str:
    """Convert image ndarray to a Base64-encoded data URL string."""
    raw = image_to_bytes(image, ext=ext, quality=quality)
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def base64_to_image(data_url_or_b64: str) -> Frame:
    """Decode a Base64 string or data URL back to a BGR ndarray."""
    if data_url_or_b64.startswith("data:"):
        data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
    raw = base64.b64decode(data_url_or_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode Base64 image data.")
    return img


def bgr_to_rgb(image: Frame) -> Frame:
    """Convert BGR ndarray to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: Frame) -> Frame:
    """Convert RGB ndarray to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: Frame) -> Image.Image:
    """Convert BGR ndarray to PIL Image (RGB)."""
    return Image.fromarray(bgr_to_rgb(image))


def pil_to_bgr(pil_image: Image.Image) -> Frame:
    """Convert PIL Image to BGR ndarray."""
    arr = np.array(pil_image.convert("RGB"))
    return rgb_to_bgr(arr)


def normalise_channels(image: np.ndarray) -> Frame:
    """
    Ensure the image has exactly 3 channels (BGR).
    Converts BGRA → BGR, GRAY → BGR as needed.
    """
    if image.ndim == 2:
        # Grayscale → BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        c = image.shape[2]
        if c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if c == 3:
            return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_image(
    image: Frame,
    width: Optional[int] = None,
    height: Optional[int] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> Frame:
    """
    Resize an image to a target width and/or height.
    If only one dimension is provided, aspect ratio is preserved.
    """
    h, w = image.shape[:2]

    if width is None and height is None:
        return image
    if h == 0 or w == 0:
        return image
    if width is None:
        scale = height / h
        width = max(1, int(round(w * scale)))
    elif height is None:
        scale = width / w
        height = max(1, int(round(h * scale)))

    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_to_max(image: Frame, max_side: int) -> Frame:
    """
    Resize image so that the longer side equals *max_side*,
    preserving aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    if w >= h:
        return resize_image(image, width=max_side)
    return resize_image(image, height=max_side)


def letterbox(
    image: Frame,
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Frame, float, Tuple[int, int]]:
    """
    Resize + pad image to *target_size* (width, height) using letterboxing.

    Returns:
        (padded_image, scale_factor, (pad_left, pad_top))
    """
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2

    out = np.full((th, tw, 3), color, dtype=np.uint8)
    out[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized
    return out, scale, (pad_left, pad_top)


def crop_image(
    image: Frame,
    bbox: BBox,
    padding: float = 0.0,
) -> Frame:
    """
    Crop the region defined by *bbox* (x1, y1, x2, y2) from an image.

    Args:
        image:   Source image (BGR).
        bbox:    Bounding box (x1, y1, x2, y2) in pixel coordinates.
        padding: Fractional padding to add around the bbox (e.g. 0.1 = 10%).

    Returns:
        Cropped BGR ndarray.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    if padding > 0.0:
        bw, bh = x2 - x1, y2 - y1
        px = int(bw * padding)
        py = int(bh * padding)
        x1, y1 = x1 - px, y1 - py
        x2, y2 = x2 + px, y2 + py

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2].copy()


def pad_to_square(
    image: Frame,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Frame:
    """
    Pad an image to a square by adding borders to the shorter axis.
    """
    h, w = image.shape[:2]
    diff = abs(h - w)
    half = diff // 2
    rest = diff - half

    if w < h:
        return cv2.copyMakeBorder(image, 0, 0, half, rest, cv2.BORDER_CONSTANT, value=color)
    if h < w:
        return cv2.copyMakeBorder(image, half, rest, 0, 0, cv2.BORDER_CONSTANT, value=color)
    return image


def expand_bbox(
    bbox: BBox,
    image_shape: Tuple[int, int],
    scale: float = 1.3,
) -> BBox:
    """
    Expand a bounding box by *scale* around its center, clamped to image bounds.

    Args:
        bbox:        Original (x1, y1, x2, y2).
        image_shape: (height, width) of the source image.
        scale:       Expansion factor (1.3 = 30% larger on each side).

    Returns:
        Expanded (x1, y1, x2, y2).
    """
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = max(0, int(cx - bw / 2))
    ny1 = max(0, int(cy - bh / 2))
    nx2 = min(w, int(cx + bw / 2))
    ny2 = min(h, int(cy + bh / 2))
    return nx1, ny1, nx2, ny2


def align_face(
    image: Frame,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> Tuple[Frame, np.ndarray]:
    """Align a face to a canonical crop using 5-point landmarks.

    Delegates to the shared norm_crop implementation in base_swapper.
    """
    from core.swapper.base_swapper import norm_crop

    crop, M = norm_crop(image, landmarks, output_size=output_size)
    if crop is None or M is None:
        raise ValueError("Could not estimate affine transform from landmarks.")
    return crop, M


def paste_face_back(
    target: Frame,
    swapped_face: Frame,
    bbox: BBox,
    mask: Optional[np.ndarray] = None,
    blend_feather: int = 20,
) -> Frame:
    """
    Paste a swapped (or enhanced) face back into the target frame at *bbox*.

    Args:
        target:       Full-resolution target BGR frame.
        swapped_face: Swapped/enhanced face BGR crop (will be resized to bbox).
        bbox:         (x1, y1, x2, y2) region in the target to paste into.
        mask:         Optional alpha mask (H, W) uint8. Auto-generated if None.
        blend_feather: Gaussian blur radius for mask edge feathering.

    Returns:
        New BGR frame with the face pasted back.
    """
    x1, y1, x2, y2 = bbox
    region_w, region_h = x2 - x1, y2 - y1

    # Resize swapped face to match bbox size
    face_resized = cv2.resize(swapped_face, (region_w, region_h), interpolation=cv2.INTER_LINEAR)

    if mask is None:
        mask = _create_face_mask(region_h, region_w, blend_feather)

    # Normalise mask to float [0, 1]
    mask_f = mask.astype(np.float32) / 255.0
    if mask_f.ndim == 2:
        mask_f = mask_f[:, :, np.newaxis]

    result = target.copy()
    roi = result[y1:y2, x1:x2].astype(np.float32)
    face_f = face_resized.astype(np.float32)

    blended = face_f * mask_f + roi * (1.0 - mask_f)
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result


def _create_face_mask(height: int, width: int, feather: int = 20) -> np.ndarray:
    from utils.mask_utils import ellipse_mask

    return ellipse_mask(height, width, feather=feather)


def alpha_blend(
    src: Frame,
    dst: Frame,
    alpha: float = 1.0,
) -> Frame:
    """
    Simple per-pixel alpha blend:  out = src * alpha + dst * (1 - alpha).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(src, alpha, dst, 1.0 - alpha, 0).astype(np.uint8)


def poisson_blend(
    src: Frame,
    dst: Frame,
    center: Optional[Tuple[int, int]] = None,
    mask: Optional[np.ndarray] = None,
) -> Frame:
    """
    Poisson blending (seamlessClone) for photorealistic face compositing.

    Args:
        src:    Source patch (the face to paste) — same size as dst.
        dst:    Destination (full frame).
        center: (x, y) center point in *dst* to paste into. Defaults to image center.
        mask:   Region mask (H, W) uint8. Defaults to white rectangle.

    Returns:
        Blended BGR frame.
    """
    if center is None:
        center = (dst.shape[1] // 2, dst.shape[0] // 2)

    if mask is None:
        h, w = src.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)

    try:
        blended = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    except cv2.error as exc:
        logger.warning(f"seamlessClone failed ({exc}), falling back to alpha blend.")
        blended = alpha_blend(src, dst, alpha=0.9)

    return blended


def masked_blend(
    src: Frame,
    dst: Frame,
    mask: np.ndarray,
) -> Frame:
    """
    Blend src into dst using a per-pixel float or uint8 mask.

    Args:
        src:  Source image (H, W, 3) BGR.
        dst:  Destination image (H, W, 3) BGR.
        mask: Blend mask (H, W) uint8 or float32 in range [0, 255] or [0, 1].

    Returns:
        Blended BGR image.
    """
    if mask.ndim != 2:
        raise ValueError(f"masked_blend: mask must be 2-D (H, W), got shape {mask.shape}")
    if mask.shape[:2] != src.shape[:2] or mask.shape[:2] != dst.shape[:2]:
        raise ValueError(
            f"masked_blend: shape mismatch — src={src.shape}, dst={dst.shape}, mask={mask.shape}"
        )

    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    # Normalise to [0, 1] if the values look like uint8
    if mask.max() > 1.0:
        mask = mask / 255.0

    mask = np.clip(mask[:, :, np.newaxis], 0.0, 1.0)
    blended = src.astype(np.float32) * mask + dst.astype(np.float32) * (1.0 - mask)
    return blended.astype(np.uint8)


def add_watermark(
    image: Frame,
    text: str = "AI GENERATED",
    position: str = "bottom_right",
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.65,
) -> Frame:
    """
    Overlay a text watermark on an image.

    Args:
        image:      BGR source image.
        text:       Watermark text.
        position:   Anchor: 'bottom_right' | 'bottom_left' | 'top_right' | 'top_left' | 'center'.
        font_scale: OpenCV font scale.
        color:      Text BGR colour.
        alpha:      Transparency of watermark layer (0 = invisible, 1 = opaque).

    Returns:
        Watermarked BGR image.
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = int(min(h, w) * 0.02) + 5

    if position == "bottom_right":
        org = (w - tw - margin, h - margin)
    elif position == "bottom_left":
        org = (margin, h - margin)
    elif position == "top_right":
        org = (w - tw - margin, th + margin)
    elif position == "top_left":
        org = (margin, th + margin)
    else:  # center
        org = ((w - tw) // 2, (h + th) // 2)

    # Shadow for readability
    shadow_color = (0, 0, 0)
    cv2.putText(
        overlay,
        text,
        (org[0] + 1, org[1] + 1),
        font,
        font_scale,
        shadow_color,
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(overlay, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def draw_bboxes(
    image: Frame,
    bboxes: List[BBox],
    labels: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> Frame:
    """
    Draw bounding boxes (and optional labels + confidence scores) on an image.

    Args:
        image:       BGR source image (will NOT be modified in-place).
        bboxes:      List of (x1, y1, x2, y2) boxes.
        labels:      Optional label strings aligned to *bboxes*.
        confidences: Optional float scores aligned to *bboxes*.
        color:       Box BGR colour.
        thickness:   Line thickness in pixels.
        font_scale:  Label font scale.

    Returns:
        A new BGR image with boxes drawn.
    """
    out = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        parts: List[str] = []
        if labels and i < len(labels):
            parts.append(labels[i])
        if confidences and i < len(confidences):
            parts.append(f"{confidences[i]:.2f}")

        if parts:
            label_text = " | ".join(parts)
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            bg_y1 = max(y1 - th - 8, 0)
            cv2.rectangle(out, (x1, bg_y1), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out,
                label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return out


def draw_landmarks(
    image: Frame,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = (0, 165, 255),
    radius: int = 3,
) -> Frame:
    """
    Draw facial landmark points on an image.

    Args:
        image:     BGR source image.
        landmarks: (N, 2) array of (x, y) landmark coordinates.
        color:     Point BGR colour.
        radius:    Circle radius in pixels.

    Returns:
        Image with landmarks drawn.
    """
    out = image.copy()
    for x, y in landmarks.astype(int):
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return out


def draw_faces(
    image: Frame,
    bboxes: List[BBox],
    landmarks_list: Optional[List[np.ndarray]] = None,
    identities: Optional[List[str]] = None,
    confidences: Optional[List[float]] = None,
    box_color: Tuple[int, int, int] = (0, 230, 0),
) -> Frame:
    """
    Convenience wrapper: draw boxes, landmarks, and identity labels together.
    """
    out = draw_bboxes(
        image,
        bboxes,
        labels=identities,
        confidences=confidences,
        color=box_color,
    )
    if landmarks_list:
        for lm in landmarks_list:
            if lm is not None:
                out = draw_landmarks(out, lm)
    return out


def side_by_side(
    left: Frame,
    right: Frame,
    label_left: str = "Original",
    label_right: str = "Result",
    separator_width: int = 4,
    separator_color: Tuple[int, int, int] = (200, 200, 200),
) -> Frame:
    """
    Concatenate two images side-by-side for comparison display.
    Both images are resized to the same height as the taller one.
    """
    h = max(left.shape[0], right.shape[0])
    left_r = resize_image(left, height=h) if left.shape[0] != h else left
    right_r = resize_image(right, height=h) if right.shape[0] != h else right

    # Add labels
    def _label(img: Frame, text: str) -> Frame:
        labeled = img.copy()
        cv2.putText(
            labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            labeled, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA
        )
        return labeled

    left_r = _label(left_r, label_left)
    right_r = _label(right_r, label_right)

    sep = np.full((h, separator_width, 3), separator_color, dtype=np.uint8)
    return np.concatenate([left_r, sep, right_r], axis=1)


def is_valid_image(data: bytes) -> bool:
    """Return True if *data* can be decoded as a valid image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img is not None


def compute_brightness(image: Frame) -> float:
    """
    Compute mean perceptual brightness of a BGR image (0.0 – 255.0).
    Uses the V channel of HSV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def compute_blur_score(image: Frame) -> float:
    """
    Estimate image sharpness using the variance of the Laplacian.
    Higher = sharper.  A value < 100 is typically considered blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurry(image: Frame, threshold: float = 100.0) -> bool:
    """Return True if the image is considered blurry."""
    return compute_blur_score(image) < threshold


def get_image_info(image: Frame) -> dict:
    """
    Return a dict with basic image metadata.

    Returns:
        {
            "height": int,
            "width": int,
            "channels": int,
            "dtype": str,
            "brightness": float,
            "blur_score": float,
            "is_blurry": bool,
        }
    """
    h, w = image.shape[:2]
    c = image.shape[2] if image.ndim == 3 else 1
    brightness = compute_brightness(image) if c >= 3 else 0.0
    blur_score = compute_blur_score(image)
    return {
        "height": h,
        "width": w,
        "channels": c,
        "dtype": str(image.dtype),
        "brightness": round(brightness, 2),
        "blur_score": round(blur_score, 2),
        "is_blurry": is_blurry(image),
    }
