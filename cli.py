"""Command-line interface for the AI Face Recognition & Swap pipeline.

Usage examples::

    python cli.py swap --source src.jpg --target tgt.jpg --output out.jpg
    python cli.py swap --source src.jpg --target tgt.jpg --output out.jpg \\
        --enhance --no-watermark --blend poisson --swap-all --consent

    python cli.py video --source src.jpg --target video.mp4 --output out.mp4 \\
        --enhance --skip-frames 1

    python cli.py info --image path.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so relative imports work when the
# script is executed directly (python cli.py …).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger, setup_from_settings  # noqa: E402

setup_from_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image(path: str):
    """Load an image from *path* and return a BGR numpy array.

    Raises SystemExit(1) if the file does not exist or cannot be decoded.
    """
    import cv2

    p = Path(path)
    if not p.exists():
        logger.error(f"Image file not found: {path}")
        sys.exit(1)
    img = cv2.imread(str(p))
    if img is None:
        logger.error(f"Could not decode image: {path}")
        sys.exit(1)
    return img


def _load_detector():
    """Load and return a YOLOFaceDetector using settings or defaults."""
    try:
        from config.settings import settings  # noqa: PLC0415

        model_path = settings.detector.model_path
        device = settings.detector.device
    except Exception:
        model_path = str(PROJECT_ROOT / "models" / "yolov8n-face.pt")
        device = "cpu"

    from core.detector.yolo_detector import YOLOFaceDetector  # noqa: PLC0415

    detector = YOLOFaceDetector(model_path=model_path, device=device)
    detector.load_model()
    return detector


def _load_recognizer():
    """Load and return an InsightFaceRecognizer using settings or defaults."""
    try:
        from config.settings import settings  # noqa: PLC0415

        model_pack = settings.recognizer.model_pack
        model_root = settings.recognizer.model_root
    except Exception:
        model_pack = "buffalo_l"
        model_root = str(PROJECT_ROOT / "models")

    from core.recognizer.insightface_recognizer import InsightFaceRecognizer  # noqa: PLC0415

    recognizer = InsightFaceRecognizer(model_pack=model_pack, model_root=model_root)
    recognizer.load_model()
    return recognizer


def _load_swapper():
    """Load and return an InSwapper using settings or defaults."""
    try:
        from config.settings import settings  # noqa: PLC0415

        model_path = settings.swapper.model_path
        providers = settings.swapper.providers
    except Exception:
        model_path = str(PROJECT_ROOT / "models" / "inswapper_128.onnx")
        providers = ["CPUExecutionProvider"]

    from core.swapper.inswapper import InSwapper  # noqa: PLC0415

    swapper = InSwapper(model_path=model_path, providers=providers)
    swapper.load_model()
    return swapper


def _load_enhancer():
    """Load and return an enhancer (GFPGAN or CodeFormer) or None."""
    try:
        from config.settings import settings  # noqa: PLC0415

        backend = settings.enhancer.backend
        if backend == "none":
            return None
        if backend == "gfpgan":
            from core.enhancer.gfpgan_enhancer import GFPGANEnhancer  # noqa: PLC0415

            enh = GFPGANEnhancer(
                model_path=settings.enhancer.gfpgan_model_path,
                upscale=settings.enhancer.upscale,
                only_center_face=settings.enhancer.only_center_face,
            )
            enh.load_model()
            return enh
        if backend == "codeformer":
            from core.enhancer.codeformer_enhancer import CodeFormerEnhancer  # noqa: PLC0415

            enh = CodeFormerEnhancer(
                model_path=settings.enhancer.codeformer_model_path,
                fidelity_weight=settings.enhancer.fidelity_weight,
                upscale=settings.enhancer.upscale,
            )
            enh.load_model()
            return enh
    except Exception as exc:
        logger.warning(f"Could not load enhancer: {exc}")
    return None


# ---------------------------------------------------------------------------
# Subcommand: swap
# ---------------------------------------------------------------------------


def cmd_swap(args: argparse.Namespace) -> int:
    """Run a single image face swap."""
    import cv2

    from core.swapper.base_swapper import BlendMode, SwapRequest  # noqa: PLC0415

    if not args.consent:
        logger.error(
            "Consent flag is required.  Add --consent to confirm you have "
            "explicit consent from all depicted individuals."
        )
        return 1

    logger.info("Loading source image …")
    source_image = _load_image(args.source)
    logger.info("Loading target image …")
    target_image = _load_image(args.target)

    logger.info("Loading detector …")
    try:
        detector = _load_detector()
    except Exception as exc:
        logger.error(f"Detector load failed: {exc}")
        return 1

    logger.info("Loading recognizer …")
    try:
        recognizer = _load_recognizer()
    except Exception as exc:
        logger.error(f"Recognizer load failed: {exc}")
        return 1

    logger.info("Loading swapper …")
    try:
        swapper = _load_swapper()
    except Exception as exc:
        logger.error(f"Swapper load failed: {exc}")
        return 1

    # Detect source face
    src_detection = detector.detect(source_image)
    if src_detection.is_empty:
        logger.error("No face detected in source image.")
        return 1
    src_face = src_detection.faces[0]

    # Extract source embedding
    src_bbox = (int(src_face.x1), int(src_face.y1), int(src_face.x2), int(src_face.y2))
    source_embedding = recognizer.get_embedding(source_image, bbox=src_bbox)
    if source_embedding is None:
        logger.error("Could not extract embedding from source face.")
        return 1

    # Detect target faces
    tgt_detection = detector.detect(target_image)
    if tgt_detection.is_empty:
        logger.error("No face detected in target image.")
        return 1

    # Resolve blend mode
    blend_map = {
        "poisson": BlendMode.POISSON,
        "alpha": BlendMode.ALPHA,
        "seamless": BlendMode.POISSON,  # alias
    }
    blend_mode = blend_map.get(args.blend, BlendMode.POISSON)

    output_image = target_image.copy()

    if args.swap_all:
        batch_result = swapper.swap_all(
            source_embedding=source_embedding,
            target_image=output_image,
            target_detection=tgt_detection,
            blend_mode=blend_mode,
        )
        output_image = batch_result.output_image
        n_swapped = sum(1 for r in batch_result.swap_results if r.success)
    else:
        tgt_face = tgt_detection.faces[0]
        swap_req = SwapRequest(
            source_embedding=source_embedding,
            target_image=output_image,
            target_face=tgt_face,
            blend_mode=blend_mode,
        )
        result = swapper.swap(swap_req)
        output_image = result.output_image
        n_swapped = 1 if result.success else 0

    logger.info(f"Faces swapped: {n_swapped}")

    # Optional enhancement
    if args.enhance:
        enhancer = _load_enhancer()
        if enhancer is not None:
            from core.enhancer.base_enhancer import EnhancementRequest  # noqa: PLC0415

            enh_req = EnhancementRequest(image=output_image, full_frame=True, paste_back=True)
            enh_result = enhancer.enhance(enh_req)
            if enh_result.success:
                output_image = enh_result.output_image
                logger.info("Enhancement applied.")
            else:
                logger.warning(f"Enhancement failed: {enh_result.error}")
        else:
            logger.warning("--enhance requested but no enhancer available.")

    # Optional watermark
    if not args.no_watermark:
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = output_image.shape[:2]
        scale = max(0.4, min(w, h) / 800.0)
        thick = max(1, int(scale * 1.5))
        text = "AI GENERATED"
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        margin = int(min(w, h) * 0.02)
        x = w - tw - margin
        y = h - margin
        cv2.putText(output_image, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1)
        cv2.putText(output_image, text, (x, y), font, scale, (255, 255, 255), thick)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), output_image)
    if not ok:
        logger.error(f"Failed to write output image to {out_path}")
        return 1
    logger.success(f"Output written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: video
# ---------------------------------------------------------------------------


def cmd_video(args: argparse.Namespace) -> int:
    """Run face swap over a video file."""
    from core.pipeline.video_pipeline import VideoPipeline, VideoProcessingConfig  # noqa: PLC0415

    logger.info("Loading source image …")
    source_image = _load_image(args.source)

    target_path = Path(args.target)
    if not target_path.exists():
        logger.error(f"Target video not found: {args.target}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pipeline components …")
    try:
        detector = _load_detector()
        recognizer = _load_recognizer()
        swapper = _load_swapper()
    except Exception as exc:
        logger.error(f"Failed to load pipeline components: {exc}")
        return 1

    enhancer = _load_enhancer() if args.enhance else None

    try:
        from config.settings import settings  # noqa: PLC0415

        providers = settings.swapper.providers
    except Exception:
        providers = ["CPUExecutionProvider"]

    config = VideoProcessingConfig(
        skip_frames=args.skip_frames,
        enhance=args.enhance,
        preserve_audio=True,
        swap_all_faces=False,
    )

    pipeline = VideoPipeline(
        detector=detector,
        recognizer=recognizer,
        swapper=swapper,
        enhancer=enhancer,
        config=config,
    )

    logger.info(f"Processing video: {target_path} → {output_path}")
    try:
        result = pipeline.process_video(
            source_image=source_image,
            input_path=str(target_path),
            output_path=str(output_path),
        )
        logger.success(
            f"Video processing complete | frames={result.frames_processed} "
            f"swapped={result.frames_swapped} "
            f"time={result.total_time_s:.1f}s"
        )
    except Exception as exc:
        logger.error(f"Video processing failed: {exc}")
        return 1

    return 0


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """Detect faces in an image and print bounding boxes + count."""
    logger.info(f"Loading image: {args.image}")
    image = _load_image(args.image)

    logger.info("Loading detector …")
    try:
        detector = _load_detector()
    except Exception as exc:
        logger.error(f"Detector load failed: {exc}")
        return 1

    result = detector.detect(image)
    h, w = image.shape[:2]

    print(f"Image:  {args.image}")
    print(f"Size:   {w}x{h} px")
    print(f"Faces:  {result.num_faces}")
    for face in result.faces:
        print(
            f"  [{face.face_index}] "
            f"bbox=({face.x1},{face.y1},{face.x2},{face.y2}) "
            f"size={face.width}x{face.height} "
            f"conf={face.confidence:.3f}"
        )

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "AI Face Recognition & Swap — command-line interface.\n\n"
            "Subcommands:\n"
            "  swap   Swap faces between two images.\n"
            "  video  Swap faces across a video file.\n"
            "  info   Detect and report faces in an image."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------
    # swap subcommand
    # ------------------------------------------------------------------
    swap_parser = subparsers.add_parser(
        "swap",
        help="Swap a face from a source image into a target image.",
        description=(
            "Detect the primary face in SOURCE, extract its ArcFace embedding, "
            "then inject that identity into TARGET (one or all faces)."
        ),
    )
    swap_parser.add_argument(
        "--source",
        required=True,
        metavar="SRC",
        help="Path to the source image (donor identity).",
    )
    swap_parser.add_argument(
        "--target",
        required=True,
        metavar="TGT",
        help="Path to the target image (scene to modify).",
    )
    swap_parser.add_argument(
        "--output",
        required=True,
        metavar="OUT",
        help="Output file path (PNG/JPG).",
    )
    swap_parser.add_argument(
        "--enhance",
        action="store_true",
        default=False,
        help="Apply face enhancement (GFPGAN / CodeFormer) after swap.",
    )
    swap_parser.add_argument(
        "--no-watermark",
        action="store_true",
        default=False,
        help="Disable the 'AI GENERATED' watermark on the output image.",
    )
    swap_parser.add_argument(
        "--blend",
        choices=["poisson", "alpha", "seamless"],
        default="poisson",
        help="Blending mode for compositing the swapped face (default: poisson).",
    )
    swap_parser.add_argument(
        "--swap-all",
        action="store_true",
        default=False,
        help="Swap ALL detected faces in the target (default: swap only the primary face).",
    )
    swap_parser.add_argument(
        "--consent",
        action="store_true",
        default=False,
        help=(
            "Confirm explicit consent from all depicted individuals. "
            "Required to run the swap."
        ),
    )

    # ------------------------------------------------------------------
    # video subcommand
    # ------------------------------------------------------------------
    video_parser = subparsers.add_parser(
        "video",
        help="Swap faces across every frame of a video file.",
        description=(
            "Extract the source identity from SOURCE_IMAGE and apply it to "
            "every face detected across TARGET_VIDEO, writing the result to OUTPUT."
        ),
    )
    video_parser.add_argument(
        "--source",
        required=True,
        metavar="SRC",
        help="Path to the source image (donor identity).",
    )
    video_parser.add_argument(
        "--target",
        required=True,
        metavar="VIDEO",
        help="Path to the input video file.",
    )
    video_parser.add_argument(
        "--output",
        required=True,
        metavar="OUT",
        help="Path to the output video file.",
    )
    video_parser.add_argument(
        "--enhance",
        action="store_true",
        default=False,
        help="Apply face enhancement on each processed frame.",
    )
    video_parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Process every Nth frame; copy intermediate frames unmodified "
            "(default: 1 — process every frame)."
        ),
    )

    # ------------------------------------------------------------------
    # info subcommand
    # ------------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="Detect faces in an image and print bounding box information.",
        description=(
            "Run the face detector on IMAGE and print the number of faces "
            "found together with each bounding box and confidence score."
        ),
    )
    info_parser.add_argument(
        "--image",
        required=True,
        metavar="IMG",
        help="Path to the image to inspect.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "swap": cmd_swap,
        "video": cmd_video,
        "info": cmd_info,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        exit_code = handler(args)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        exit_code = 130
    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
