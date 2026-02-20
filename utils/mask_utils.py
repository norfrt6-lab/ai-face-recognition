from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# ── Type aliases ─────────────────────────────────────────────
Frame     = np.ndarray          # (H, W, 3) BGR uint8
Mask      = np.ndarray          # (H, W)    uint8  0-255
BBox      = Tuple[int, int, int, int]   # (x1, y1, x2, y2)
Landmarks = np.ndarray          # (N, 2) float32  (x, y) pairs


def ellipse_mask(
    height: int,
    width: int,
    *,
    center: Optional[Tuple[int, int]] = None,
    axes: Optional[Tuple[int, int]] = None,
    angle: float = 0.0,
    feather: int = 20,
) -> Mask:
    """
    Create a soft elliptical mask centred in a (height × width) canvas.

    Args:
        height:  Canvas height in pixels.
        width:   Canvas width in pixels.
        center:  (cx, cy) override.  Defaults to image centre.
        axes:    (semi_x, semi_y) radii.  Defaults to ~45 % of canvas dims.
        angle:   Rotation angle of the ellipse in degrees.
        feather: Gaussian blur sigma for edge softening.

    Returns:
        uint8 mask — white (255) inside the ellipse, black (0) outside,
        with softened edges when feather > 0.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    cx = center[0] if center else width  // 2
    cy = center[1] if center else height // 2
    ax = axes[0]   if axes   else int(width  * 0.45)
    ay = axes[1]   if axes   else int(height * 0.48)

    cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)

    if feather > 0:
        ksize = _odd_kernel(feather)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


def rectangle_mask(
    height: int,
    width: int,
    *,
    bbox: Optional[BBox] = None,
    padding: int = 0,
    feather: int = 0,
) -> Mask:
    """
    Create a rectangular mask, optionally padded and feathered.

    Args:
        height:  Canvas height.
        width:   Canvas width.
        bbox:    (x1, y1, x2, y2) region to fill white.
                 Defaults to the full canvas.
        padding: Pixel padding to subtract from each side (inset).
        feather: Gaussian blur sigma for edge softening.

    Returns:
        uint8 mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if bbox is None:
        x1, y1, x2, y2 = 0, 0, width, height
    else:
        x1, y1, x2, y2 = bbox

    x1 = max(0, x1 + padding)
    y1 = max(0, y1 + padding)
    x2 = min(width,  x2 - padding)
    y2 = min(height, y2 - padding)

    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if feather > 0:
        ksize = _odd_kernel(feather)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


def full_mask(height: int, width: int) -> Mask:
    """Return an all-white (255) mask of the given dimensions."""
    return np.full((height, width), 255, dtype=np.uint8)


def empty_mask(height: int, width: int) -> Mask:
    """Return an all-black (0) mask of the given dimensions."""
    return np.zeros((height, width), dtype=np.uint8)


def convex_hull_mask(
    height: int,
    width: int,
    landmarks: Landmarks,
    *,
    feather: int = 20,
    expand_px: int = 0,
) -> Mask:
    """
    Create a convex hull mask from 2-D landmark points.

    Useful for tightly masking just the face polygon formed by the
    outer boundary of detected landmarks (e.g. dlib 68-point or
    MediaPipe 468-point sets).

    Args:
        height:     Canvas height.
        width:      Canvas width.
        landmarks:  (N, 2) array of (x, y) points (float or int).
        feather:    Edge softening blur radius.
        expand_px:  Dilate the hull by this many pixels before feathering.

    Returns:
        uint8 mask (H × W).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = landmarks.astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    if expand_px > 0:
        mask = dilate_mask(mask, expand_px)

    if feather > 0:
        ksize = _odd_kernel(feather)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


def face_oval_mask(
    height: int,
    width: int,
    landmarks_5pt: Landmarks,
    *,
    scale: float = 1.5,
    feather: int = 25,
) -> Mask:
    """
    Create an elliptical face oval mask from 5-point landmarks
    (left_eye, right_eye, nose, left_mouth, right_mouth).

    The ellipse is fitted to the bounding box of the 5 points and
    optionally scaled outward to cover the full face including forehead
    and chin.

    Args:
        height:          Canvas height.
        width:           Canvas width.
        landmarks_5pt:   (5, 2) landmark array.
        scale:           Scale factor for the ellipse axes (>1 = larger).
        feather:         Edge softening blur radius.

    Returns:
        uint8 mask (H × W).
    """
    pts = landmarks_5pt.astype(np.float32)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())

    # Estimate axis radii from extent of landmarks
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()

    # Heuristic: face is taller than the inter-eye / mouth span
    ax = int((x_range / 2) * scale * 1.3)
    ay = int((y_range / 2) * scale * 2.0)

    return ellipse_mask(
        height,
        width,
        center=(int(cx), int(cy)),
        axes=(ax, ay),
        feather=feather,
    )


def landmarks_region_mask(
    height: int,
    width: int,
    landmarks: Landmarks,
    region_indices: List[int],
    *,
    feather: int = 10,
) -> Mask:
    """
    Create a filled polygon mask for a specific facial region
    (e.g. left eye, lips, nose) identified by a subset of landmark indices.

    Args:
        height:          Canvas height.
        width:           Canvas width.
        landmarks:       Full (N, 2) landmark array.
        region_indices:  Indices into *landmarks* that form the region contour.
        feather:         Edge softening blur radius.

    Returns:
        uint8 mask (H × W).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts  = landmarks[region_indices].astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if feather > 0:
        ksize = _odd_kernel(feather)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


def face_bbox_mask(
    frame_height: int,
    frame_width: int,
    bbox: BBox,
    *,
    padding_top: float = 0.20,
    padding_sides: float = 0.08,
    padding_bottom: float = 0.05,
    feather: int = 25,
    shape: str = "ellipse",
) -> Mask:
    """
    Create a face mask from a detection bounding box with configurable
    padding and shape.

    Args:
        frame_height:    Full frame height.
        frame_width:     Full frame width.
        bbox:            Face bounding box (x1, y1, x2, y2).
        padding_top:     Fractional padding above bbox (for forehead).
        padding_sides:   Fractional horizontal padding.
        padding_bottom:  Fractional padding below bbox (for chin).
        feather:         Edge softening blur radius in pixels.
        shape:           'ellipse' | 'rectangle' | 'rounded_rectangle'.

    Returns:
        uint8 mask with the same (H × W) as the full frame.
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    # Apply padding
    px1 = max(0, x1 - int(bw * padding_sides))
    px2 = min(frame_width,  x2 + int(bw * padding_sides))
    py1 = max(0, y1 - int(bh * padding_top))
    py2 = min(frame_height, y2 + int(bh * padding_bottom))

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    rw   = px2 - px1
    rh   = py2 - py1

    if shape == "ellipse":
        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2
        cv2.ellipse(mask, (cx, cy), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
    elif shape == "rounded_rectangle":
        radius = min(rw, rh) // 6
        _draw_rounded_rect(mask, (px1, py1, px2, py2), radius)
    else:  # rectangle
        cv2.rectangle(mask, (px1, py1), (px2, py2), 255, -1)

    if feather > 0:
        ksize = _odd_kernel(feather)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


def multi_face_mask(
    frame_height: int,
    frame_width: int,
    bboxes: List[BBox],
    *,
    feather: int = 25,
    shape: str = "ellipse",
) -> Mask:
    """
    Create a combined mask covering all detected faces.

    Returns the union (max) of individual per-face masks.
    """
    combined = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for bbox in bboxes:
        face_m = face_bbox_mask(
            frame_height, frame_width, bbox,
            feather=feather, shape=shape
        )
        combined = np.maximum(combined, face_m)
    return combined


def dilate_mask(mask: Mask, radius: int) -> Mask:
    """
    Dilate (expand) a mask by *radius* pixels using an elliptical kernel.

    Args:
        mask:   Input uint8 mask.
        radius: Structuring element radius in pixels.

    Returns:
        Dilated uint8 mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_kernel(radius), _odd_kernel(radius))
    )
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: Mask, radius: int) -> Mask:
    """
    Erode (shrink) a mask by *radius* pixels.

    Args:
        mask:   Input uint8 mask.
        radius: Structuring element radius in pixels.

    Returns:
        Eroded uint8 mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_kernel(radius), _odd_kernel(radius))
    )
    return cv2.erode(mask, kernel, iterations=1)


def feather_mask(mask: Mask, radius: int) -> Mask:
    """
    Soften mask edges with Gaussian blur.

    Args:
        mask:   Input uint8 mask.
        radius: Blur radius / sigma.

    Returns:
        Feathered uint8 mask.
    """
    if radius <= 0:
        return mask
    ksize = _odd_kernel(radius)
    return cv2.GaussianBlur(mask, (ksize, ksize), radius)


def invert_mask(mask: Mask) -> Mask:
    """Return the bitwise inverse of a mask."""
    return cv2.bitwise_not(mask)


def combine_masks(
    mask_a: Mask,
    mask_b: Mask,
    mode: str = "union",
) -> Mask:
    """
    Combine two masks using a logical operation.

    Args:
        mask_a: First uint8 mask.
        mask_b: Second uint8 mask.
        mode:   'union' (OR) | 'intersection' (AND) | 'subtract' (A - B) | 'xor'.

    Returns:
        Combined uint8 mask.
    """
    if mode == "union":
        return cv2.bitwise_or(mask_a, mask_b)
    if mode == "intersection":
        return cv2.bitwise_and(mask_a, mask_b)
    if mode == "subtract":
        return cv2.subtract(mask_a, mask_b)
    if mode == "xor":
        return cv2.bitwise_xor(mask_a, mask_b)
    raise ValueError(f"Unknown mask combine mode: {mode!r}. "
                     f"Choose 'union' | 'intersection' | 'subtract' | 'xor'.")


def threshold_mask(
    mask: Mask,
    threshold: int = 128,
    *,
    feather: int = 0,
) -> Mask:
    """
    Binarise a soft mask at *threshold*, optionally re-feathering afterwards.

    Args:
        mask:       Input uint8 mask (may be soft / gradient).
        threshold:  Binarisation threshold (0-255).
        feather:    Re-feather after thresholding.

    Returns:
        Hard (or re-softened) binary mask.
    """
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    if feather > 0:
        binary = feather_mask(binary, feather)
    return binary


def crop_mask_to_bbox(mask: Mask, bbox: BBox) -> Mask:
    """
    Zero out all mask pixels *outside* the given bounding box.

    Args:
        mask: Input (H, W) uint8 mask.
        bbox: (x1, y1, x2, y2) region to keep.

    Returns:
        Masked-out uint8 mask.
    """
    result = np.zeros_like(mask)
    x1, y1, x2, y2 = bbox
    result[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return result


def apply_mask_blend(
    src: Frame,
    dst: Frame,
    mask: Mask,
) -> Frame:
    """
    Blend *src* into *dst* using *mask* as per-pixel alpha.

    Formula::

        out = src * (mask / 255) + dst * (1 - mask / 255)

    Args:
        src:  Source image (H, W, 3) BGR — the "new" content.
        dst:  Destination image (H, W, 3) BGR — the "background".
        mask: Blend alpha mask (H, W) uint8.

    Returns:
        Blended BGR image.
    """
    _assert_same_hw(src, dst, mask)
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]   # broadcast over channels
    blended = src.astype(np.float32) * alpha + dst.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def poisson_mask_blend(
    src: Frame,
    dst: Frame,
    mask: Mask,
    *,
    center: Optional[Tuple[int, int]] = None,
    clone_mode: int = cv2.NORMAL_CLONE,
) -> Frame:
    """
    Seamless (Poisson) clone with an explicit mask.

    The mask must be a soft or hard foreground region in *src* that will
    be cloned into *dst* at *center*.

    Args:
        src:        Source patch — same size as dst.
        dst:        Destination image — same size as src.
        mask:       Region-of-interest mask (H, W) uint8.
        center:     (cx, cy) in *dst* to paste at.  Defaults to dst centre.
        clone_mode: cv2.NORMAL_CLONE | cv2.MIXED_CLONE | cv2.MONOCHROME_TRANSFER.

    Returns:
        Blended BGR image (same size as dst).
    """
    _assert_same_hw(src, dst, mask)

    if center is None:
        h, w = dst.shape[:2]
        center = (w // 2, h // 2)

    # seamlessClone requires a hard mask
    hard_mask = threshold_mask(mask, threshold=1)

    try:
        result = cv2.seamlessClone(src, dst, hard_mask, center, clone_mode)
    except cv2.error as exc:
        logger.warning(f"Poisson blend failed ({exc}), falling back to alpha blend.")
        result = apply_mask_blend(src, dst, mask)

    return result


def region_copy_blend(
    src: Frame,
    dst: Frame,
    mask: Mask,
    bbox: BBox,
) -> Frame:
    """
    Copy-paste the region defined by *bbox* from *src* into *dst*
    using *mask* for smooth blending.

    Unlike full-image blending, this operates only inside *bbox* for
    efficiency.

    Args:
        src:   Source frame (same size as dst).
        dst:   Destination frame.
        mask:  Full-frame blend mask (H, W) uint8.
        bbox:  (x1, y1, x2, y2) region to operate within.

    Returns:
        New frame with the region blended in.
    """
    x1, y1, x2, y2 = bbox
    result = dst.copy()

    src_roi  = src[y1:y2,  x1:x2]
    dst_roi  = dst[y1:y2,  x1:x2]
    mask_roi = mask[y1:y2, x1:x2]

    blended_roi = apply_mask_blend(src_roi, dst_roi, mask_roi)
    result[y1:y2, x1:x2] = blended_roi
    return result


def edge_aware_mask(
    image: Frame,
    base_mask: Mask,
    *,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
    post_feather: int = 5,
) -> Mask:
    """
    Refine a coarse mask using a bilateral-filtered edge map so that
    the mask boundary respects natural image edges (skin vs hair, etc.).

    Args:
        image:        BGR frame — used only for edge extraction.
        base_mask:    Initial coarse uint8 mask.
        sigma_color:  Bilateral filter color sigma.
        sigma_space:  Bilateral filter spatial sigma.
        post_feather: Light feather applied after edge refinement.

    Returns:
        Refined uint8 mask.
    """
    # Bilateral filter preserves edges while smoothing flat regions
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Use edges as inhibitor: erode mask near strong edges
    edge_dilated = dilate_mask(edges, radius=3)
    inhibitor    = invert_mask(edge_dilated)

    refined = cv2.bitwise_and(base_mask, inhibitor)
    # Combine with original to avoid over-erosion
    refined = cv2.addWeighted(base_mask, 0.7, refined, 0.3, 0)

    if post_feather > 0:
        refined = feather_mask(refined, post_feather)

    return refined.astype(np.uint8)


def skin_color_mask(
    image: Frame,
    *,
    feather: int = 15,
    dilate: int = 10,
) -> Mask:
    """
    Generate a rough skin-tone mask using HSV colour thresholding.

    Works best for front-facing faces under reasonable lighting.
    Should be used as a *guide* mask, not a precise segmentation.

    Args:
        image:   BGR input frame.
        feather: Edge softening radius.
        dilate:  Dilation to fill small gaps in skin regions.

    Returns:
        uint8 skin mask (H × W).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Heuristic skin colour range in HSV
    lower = np.array([0,  20,  70], dtype=np.uint8)
    upper = np.array([20, 180, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Additional range for slightly darker / redder skin
    lower2 = np.array([170, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)
    mask2  = cv2.inRange(hsv, lower2, upper2)

    combined = cv2.bitwise_or(mask, mask2)

    if dilate > 0:
        combined = dilate_mask(combined, dilate)
    if feather > 0:
        combined = feather_mask(combined, feather)

    return combined


def mask_coverage(mask: Mask) -> float:
    """
    Return the fraction of non-zero pixels in the mask [0.0, 1.0].

    Useful for sanity-checking that a generated mask is non-trivial.
    """
    total = mask.size
    nonzero = int(np.count_nonzero(mask))
    return nonzero / total if total > 0 else 0.0


def mask_bounding_box(mask: Mask) -> Optional[BBox]:
    """
    Return the tight bounding box (x1, y1, x2, y2) around non-zero pixels,
    or None if the mask is empty.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def is_empty_mask(mask: Mask, threshold: float = 0.001) -> bool:
    """
    Return True if the mask has fewer than *threshold* fraction of
    non-zero pixels (i.e. effectively empty).
    """
    return mask_coverage(mask) < threshold


def resize_mask(
    mask: Mask,
    target_size: Tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> Mask:
    """
    Resize a mask to *target_size* (width, height).

    Bilinear interpolation is used by default to preserve soft edges.
    """
    resized = cv2.resize(mask, target_size, interpolation=interpolation)
    return resized.astype(np.uint8)


def mask_to_3channel(mask: Mask) -> Frame:
    """
    Convert a single-channel (H, W) uint8 mask to a 3-channel (H, W, 3)
    image by stacking it across all channels.

    Useful for multiplying masks against BGR image arrays.
    """
    return cv2.merge([mask, mask, mask])


def mask_to_float(mask: Mask) -> np.ndarray:
    """Normalise a uint8 mask to float32 in [0.0, 1.0]."""
    return mask.astype(np.float32) / 255.0


def float_to_mask(arr: np.ndarray) -> Mask:
    """Convert a float32 array in [0.0, 1.0] back to uint8 [0, 255]."""
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def visualize_mask(
    image: Frame,
    mask: Mask,
    *,
    color: Tuple[int, int, int] = (0, 200, 255),
    alpha: float = 0.45,
    show_contour: bool = True,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
) -> Frame:
    """
    Overlay a mask on an image for debugging and visualisation.

    Args:
        image:            BGR source frame.
        mask:             uint8 mask to visualise.
        color:            BGR tint colour for the masked region.
        alpha:            Opacity of the tint overlay.
        show_contour:     Draw the mask contour boundary.
        contour_color:    BGR colour for the contour line.
        contour_thickness: Contour line thickness in pixels.

    Returns:
        Visualised BGR frame (copy of original, not modified in-place).
    """
    vis   = image.copy()
    tint  = np.zeros_like(image, dtype=np.uint8)
    tint[:] = color

    alpha_f = mask.astype(np.float32) / 255.0
    alpha_f = alpha_f[:, :, np.newaxis] * alpha

    vis = (
        vis.astype(np.float32) * (1.0 - alpha_f) +
        tint.astype(np.float32) * alpha_f
    ).astype(np.uint8)

    if show_contour:
        hard = threshold_mask(mask, threshold=127)
        contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, contour_color, contour_thickness)

    return vis


def visualize_all_masks(
    image: Frame,
    masks: List[Mask],
    *,
    alpha: float = 0.35,
) -> Frame:
    """
    Overlay multiple masks on an image, each with a distinct colour.

    Args:
        image:  BGR source frame.
        masks:  List of uint8 (H, W) masks.
        alpha:  Per-mask overlay opacity.

    Returns:
        Visualised BGR frame.
    """
    # Distinct BGR colours for up to 8 masks
    palette = [
        (0, 200, 255),   # yellow
        (0, 165, 255),   # orange
        (0, 255, 0),     # green
        (255, 0, 0),     # blue
        (255, 0, 255),   # magenta
        (0, 255, 255),   # cyan
        (128, 0, 255),   # purple
        (255, 128, 0),   # light blue
    ]
    vis = image.copy()
    for i, mask in enumerate(masks):
        color = palette[i % len(palette)]
        vis = visualize_mask(vis, mask, color=color, alpha=alpha, show_contour=True)
    return vis


def _odd_kernel(radius: int) -> int:
    """
    Convert a radius/sigma value to an odd kernel size suitable for
    cv2 blur functions (minimum 1).

    For a sigma ``s``, a common rule-of-thumb kernel size is ``2*s+1``.
    We then ensure the result is odd and at least 1.
    """
    ksize = max(1, int(radius) * 2 + 1)
    return ksize if ksize % 2 == 1 else ksize + 1


def _draw_rounded_rect(
    mask: Mask,
    bbox: BBox,
    radius: int,
) -> None:
    """
    Draw a filled rounded rectangle into *mask* (in-place).

    Args:
        mask:   Target uint8 mask canvas.
        bbox:   (x1, y1, x2, y2) bounding box.
        radius: Corner radius in pixels.
    """
    x1, y1, x2, y2 = bbox
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Four corner circles
    cv2.circle(mask, (x1 + r, y1 + r), r, 255, -1)
    cv2.circle(mask, (x2 - r, y1 + r), r, 255, -1)
    cv2.circle(mask, (x1 + r, y2 - r), r, 255, -1)
    cv2.circle(mask, (x2 - r, y2 - r), r, 255, -1)

    # Three rectangles to fill the body
    cv2.rectangle(mask, (x1 + r, y1),     (x2 - r, y2),     255, -1)
    cv2.rectangle(mask, (x1,     y1 + r), (x1 + r, y2 - r), 255, -1)
    cv2.rectangle(mask, (x2 - r, y1 + r), (x2,     y2 - r), 255, -1)
