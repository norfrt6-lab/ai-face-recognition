# ============================================================
# AI Face Recognition & Face Swap - Model Downloader
# ============================================================
# Auto-downloads all required model weights:
#   - YOLOv8n-face          (.pt)
#   - InsightFace buffalo_l  (via insightface SDK)
#   - inswapper_128.onnx    (Hugging Face)
#   - GFPGANv1.4.pth        (GitHub release)
#   - CodeFormer.pth        (GitHub release)
# ============================================================

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests
from loguru import logger
from tqdm import tqdm

# ── Project root (two levels up from utils/download_models.py)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = ROOT_DIR / "models"


# ============================================================
# Model Registry
# ============================================================

@dataclass
class ModelSpec:
    """Specification for a single downloadable model file."""

    name: str
    filename: str                          # Local filename inside MODELS_DIR
    url: str                               # Primary download URL
    mirrors: List[str] = field(default_factory=list)   # Fallback URLs
    sha256: Optional[str] = None           # Expected SHA-256 hex digest (optional verify)
    subdirectory: Optional[str] = None     # Sub-folder inside MODELS_DIR
    unzip: bool = False                    # Unzip after download?
    post_download: Optional[str] = None   # Name of a special post-download action key

    @property
    def local_path(self) -> Path:
        base = MODELS_DIR / self.subdirectory if self.subdirectory else MODELS_DIR
        return base / self.filename

    @property
    def dest_dir(self) -> Path:
        return MODELS_DIR / self.subdirectory if self.subdirectory else MODELS_DIR


# Official / mirrored download URLs
_MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ── YOLOv8n-face ──────────────────────────────────────────────────────
    # Trained on WIDERFace by akanametov — lightweight & fast
    "yolov8n-face": ModelSpec(
        name="YOLOv8n-face",
        filename="yolov8n-face.pt",
        url="https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
        mirrors=[
            "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
        ],
        sha256=None,  # Update with known hash after first download
    ),
    # ── YOLOv8s-face (higher accuracy alternative) ────────────────────────
    "yolov8s-face": ModelSpec(
        name="YOLOv8s-face",
        filename="yolov8s-face.pt",
        url="https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8s-face.pt",
        mirrors=[],
        sha256=None,
    ),
    # ── InsightFace buffalo_l ─────────────────────────────────────────────
    # Downloaded via the insightface Python SDK (handled separately below)
    "buffalo_l": ModelSpec(
        name="InsightFace buffalo_l",
        filename="buffalo_l.zip",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        mirrors=[
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l.zip",
        ],
        sha256=None,
        subdirectory=None,     # will be extracted to models/buffalo_l/
        unzip=True,
        post_download="buffalo_l_extract",
    ),
    # ── inswapper_128.onnx ───────────────────────────────────────────────
    "inswapper_128": ModelSpec(
        name="inswapper_128 (face swap ONNX)",
        filename="inswapper_128.onnx",
        url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
        mirrors=[
            "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        ],
        sha256=None,
    ),
    # ── GFPGANv1.4.pth ───────────────────────────────────────────────────
    "gfpgan_v1.4": ModelSpec(
        name="GFPGANv1.4 (face restoration)",
        filename="GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        mirrors=[
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/GFPGANv1.4.pth",
        ],
        sha256=None,
    ),
    # ── CodeFormer.pth ────────────────────────────────────────────────────
    "codeformer": ModelSpec(
        name="CodeFormer (face restoration)",
        filename="codeformer.pth",
        url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        mirrors=[
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/codeformer.pth",
        ],
        sha256=None,
    ),
    # ── detection_Resnet50_Final.pth  (RetinaFace — used by GFPGAN) ───────
    "retinaface_resnet50": ModelSpec(
        name="RetinaFace ResNet50 (face parsing)",
        filename="detection_Resnet50_Final.pth",
        url="https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        mirrors=[],
        sha256=None,
        subdirectory="facexlib",
    ),
    # ── parsing_parsenet.pth  (face parsing — used by GFPGAN) ─────────────
    "parsenet": ModelSpec(
        name="ParseNet (face parsing)",
        filename="parsing_parsenet.pth",
        url="https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        mirrors=[],
        sha256=None,
        subdirectory="facexlib",
    ),
}


# ============================================================
# Core Download Logic
# ============================================================

def _download_file(
    url: str,
    dest_path: Path,
    *,
    chunk_size: int = 1024 * 1024,   # 1 MB chunks
    timeout: int = 30,
    show_progress: bool = True,
) -> bool:
    """
    Download a single file from *url* to *dest_path*.

    Args:
        url:            HTTP(S) URL of the file to download.
        dest_path:      Local file path to write to.
        chunk_size:     Number of bytes per download chunk.
        timeout:        Connection timeout in seconds.
        show_progress:  Show a tqdm download bar.

    Returns:
        True on success, False on failure.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (AI-Face-Recognition model downloader)"}
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()

        total_bytes = int(response.headers.get("content-length", 0))
        filename = dest_path.name

        pbar = tqdm(
            total=total_bytes if total_bytes > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"  ↓ {filename}",
            disable=not show_progress,
            dynamic_ncols=True,
        )

        with open(tmp_path, "wb") as f, pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Rename .part → final name only on success
        tmp_path.rename(dest_path)
        return True

    except requests.exceptions.RequestException as exc:
        logger.error(f"Download failed [{url}]: {exc}")
        tmp_path.unlink(missing_ok=True)
        return False
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user.")
        tmp_path.unlink(missing_ok=True)
        raise


def _verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """
    Verify a file's SHA-256 checksum.

    Args:
        file_path:     Path to the file to verify.
        expected_hash: Expected lowercase hex SHA-256 digest.

    Returns:
        True if the digest matches, False otherwise.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        actual = sha256.hexdigest()
        if actual.lower() != expected_hash.lower():
            logger.error(
                f"SHA-256 mismatch for {file_path.name}!\n"
                f"  expected: {expected_hash}\n"
                f"  got:      {actual}"
            )
            return False
        logger.debug(f"SHA-256 verified OK: {file_path.name}")
        return True
    except IOError as exc:
        logger.error(f"Cannot read file for checksum: {exc}")
        return False


def _download_with_fallback(
    spec: ModelSpec,
    *,
    show_progress: bool = True,
) -> bool:
    """
    Try downloading from the primary URL, then from mirrors on failure.

    Args:
        spec:           ModelSpec describing the file to download.
        show_progress:  Show tqdm progress bar.

    Returns:
        True if the file was successfully downloaded.
    """
    urls = [spec.url] + spec.mirrors
    dest = spec.local_path

    for attempt, url in enumerate(urls, start=1):
        logger.info(f"  Attempt {attempt}/{len(urls)}: {url}")
        success = _download_file(url, dest, show_progress=show_progress)
        if success:
            # SHA-256 verification
            if spec.sha256:
                if not _verify_sha256(dest, spec.sha256):
                    logger.warning("Checksum failed — trying next mirror.")
                    dest.unlink(missing_ok=True)
                    continue
            else:
                logger.warning(
                    f"  [WARN] Skipping checksum verification for {spec.name} — sha256 is None. "
                    "File integrity cannot be guaranteed."
                )
            return True
        logger.warning(f"  ✗ Failed from {url}")

    logger.error(f"All download sources exhausted for: {spec.name}")
    return False


# ============================================================
# Post-Download Actions
# ============================================================

def _post_buffalo_l_extract(spec: ModelSpec) -> bool:
    """
    Extract buffalo_l.zip into models/buffalo_l/ directory.

    The ZIP contains a top-level 'buffalo_l/' folder; we extract
    it directly into MODELS_DIR so the result is models/buffalo_l/*.
    """
    zip_path = spec.local_path
    extract_dir = MODELS_DIR / "buffalo_l"

    if not zip_path.exists():
        logger.error(f"ZIP file not found: {zip_path}")
        return False

    logger.info(f"  Extracting {zip_path.name} → {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # List members to understand structure
            members = zf.namelist()
            logger.debug(f"  ZIP members ({len(members)}): {members[:5]} ...")

            # Check if zip has a root folder
            has_root = all(m.startswith("buffalo_l/") for m in members if m.strip())

            if has_root:
                # Extract in-place — zip already has buffalo_l/ prefix
                zf.extractall(MODELS_DIR)
            else:
                # No root folder — extract into buffalo_l/ sub-dir
                extract_dir.mkdir(parents=True, exist_ok=True)
                zf.extractall(extract_dir)

        # Remove the zip after successful extraction
        zip_path.unlink(missing_ok=True)
        logger.success(f"  buffalo_l extracted → {extract_dir}")
        return True

    except zipfile.BadZipFile as exc:
        logger.error(f"Bad ZIP file: {exc}")
        zip_path.unlink(missing_ok=True)
        return False


_POST_DOWNLOAD_ACTIONS: Dict[str, Callable[[ModelSpec], bool]] = {
    "buffalo_l_extract": _post_buffalo_l_extract,
}


# ============================================================
# InsightFace SDK-based Download (alternative to manual)
# ============================================================

def _download_buffalo_l_via_sdk() -> bool:
    """
    Use the insightface Python SDK to download & cache the buffalo_l model pack.
    This is the preferred method if insightface is installed.

    Returns:
        True on success, False on failure.
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis

        logger.info("  Using InsightFace SDK to download buffalo_l model pack...")
        # Instantiate triggers automatic model download to ~/.insightface/models/
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.success("  buffalo_l downloaded and ready via InsightFace SDK.")
        return True

    except ImportError:
        logger.warning("  insightface not installed — falling back to manual download.")
        return False
    except Exception as exc:
        logger.warning(f"  InsightFace SDK download failed ({exc}) — trying manual.")
        return False


# ============================================================
# Public API
# ============================================================

def is_model_present(key: str) -> bool:
    """
    Check whether a model's file(s) already exist on disk.

    For buffalo_l, checks the extracted directory contents.

    Args:
        key: Registry key (e.g. 'yolov8n-face', 'buffalo_l', 'inswapper_128').

    Returns:
        True if the model appears to be present.
    """
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {key!r}")

    spec = _MODEL_REGISTRY[key]

    if key == "buffalo_l":
        # Check the extracted directory instead of the zip
        buffalo_dir = MODELS_DIR / "buffalo_l"
        return buffalo_dir.exists() and any(buffalo_dir.iterdir())

    return spec.local_path.exists() and spec.local_path.stat().st_size > 0


def download_model(
    key: str,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> bool:
    """
    Download a single model by its registry key.

    Args:
        key:            Registry key (see _MODEL_REGISTRY).
        force:          Re-download even if the file already exists.
        show_progress:  Show tqdm download bar.

    Returns:
        True if the model is available after this call (already present or
        just downloaded), False if download failed.

    Raises:
        KeyError: If *key* is not in the model registry.
    """
    if key not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key: {key!r}. "
            f"Available keys: {list(_MODEL_REGISTRY.keys())}"
        )

    spec = _MODEL_REGISTRY[key]

    # ── buffalo_l: prefer InsightFace SDK then manual ZIP ─────────────────
    if key == "buffalo_l":
        if not force and is_model_present("buffalo_l"):
            logger.info(f"  [SKIP] {spec.name} already present.")
            return True
        logger.info(f"Downloading: {spec.name}")
        if _download_buffalo_l_via_sdk():
            return True
        # Fall through to manual ZIP download

    elif not force and is_model_present(key):
        logger.info(f"  [SKIP] {spec.name} already present.")
        return True
    else:
        logger.info(f"Downloading: {spec.name}")

    # Ensure destination directory exists
    spec.dest_dir.mkdir(parents=True, exist_ok=True)

    # Download
    ok = _download_with_fallback(spec, show_progress=show_progress)
    if not ok:
        return False

    # Post-download action (e.g. unzip)
    if spec.post_download and spec.post_download in _POST_DOWNLOAD_ACTIONS:
        action = _POST_DOWNLOAD_ACTIONS[spec.post_download]
        ok = action(spec)
        if not ok:
            logger.error(f"Post-download action failed for: {spec.name}")
            return False

    logger.success(f"  ✓ {spec.name} ready.")
    return True


def download_all_models(
    *,
    force: bool = False,
    skip: Optional[List[str]] = None,
    show_progress: bool = True,
) -> Dict[str, bool]:
    """
    Download all models defined in the registry.

    Args:
        force:          Re-download models that already exist.
        skip:           List of model keys to skip (e.g. ['codeformer']).
        show_progress:  Show tqdm progress bars.

    Returns:
        Dict mapping model key → True (success) / False (failure).
    """
    skip = skip or []
    results: Dict[str, bool] = {}

    logger.info("=" * 60)
    logger.info("AI Face Recognition — Model Downloader")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for key, spec in _MODEL_REGISTRY.items():
        if key in skip:
            logger.info(f"  [SKIP] {spec.name} (skipped by caller)")
            results[key] = True
            continue

        results[key] = download_model(key, force=force, show_progress=show_progress)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    ok_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - ok_count
    logger.info(f"Download summary: {ok_count} succeeded, {fail_count} failed")

    for key, success in results.items():
        status = "✓" if success else "✗"
        spec = _MODEL_REGISTRY[key]
        logger.info(f"  [{status}] {spec.name}")

    if fail_count > 0:
        failed_keys = [k for k, v in results.items() if not v]
        logger.warning(
            f"Some models failed to download: {failed_keys}. "
            "Check your internet connection or download them manually."
        )

    logger.info("=" * 60)
    return results


def download_minimum_models(
    *,
    force: bool = False,
    show_progress: bool = True,
) -> Dict[str, bool]:
    """
    Download only the minimum set of models required for basic operation:
      - yolov8n-face   (face detection)
      - buffalo_l      (face analysis + recognition)
      - inswapper_128  (face swap)

    GFPGAN / CodeFormer enhancement is optional and skipped here.

    Returns:
        Dict mapping model key → success bool.
    """
    minimum_keys = ["yolov8n-face", "buffalo_l", "inswapper_128"]
    results = {}
    for key in minimum_keys:
        results[key] = download_model(key, force=force, show_progress=show_progress)
    return results


def check_all_models() -> Dict[str, bool]:
    """
    Check which models are already present on disk without downloading.

    Returns:
        Dict mapping model key → True (present) / False (missing).
    """
    status: Dict[str, bool] = {}
    for key in _MODEL_REGISTRY:
        present = is_model_present(key)
        status[key] = present

    logger.info("Model status check:")
    for key, present in status.items():
        icon = "✓" if present else "✗"
        spec = _MODEL_REGISTRY[key]
        logger.info(f"  [{icon}] {spec.name}  ({spec.local_path})")

    return status


def get_model_path(key: str) -> Path:
    """
    Return the expected local path for a model by its registry key.

    Args:
        key: Registry key (e.g. 'yolov8n-face', 'inswapper_128').

    Returns:
        Resolved Path.  Does NOT check whether the file exists.

    Raises:
        KeyError: If *key* is not recognised.
    """
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {key!r}")

    if key == "buffalo_l":
        return MODELS_DIR / "buffalo_l"

    return _MODEL_REGISTRY[key].local_path


# ============================================================
# CLI Entry Point
# ============================================================

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog="download_models",
        description="Download AI model weights for the Face Recognition & Swap pipeline.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models (including optional enhancers).",
    )
    parser.add_argument(
        "--minimum",
        action="store_true",
        default=True,
        help="Download only the minimum required models (default).",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="KEY",
        help=f"Download a single model by key. Choices: {list(_MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model already exists.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which models are already present (no download).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="KEY",
        default=[],
        help="Model keys to skip when using --all.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check existing files against known SHA-256 hashes (no download).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point. Returns exit code (0 = success, 1 = failure)."""
    args = _parse_args()

    if args.check:
        status = check_all_models()
        missing = [k for k, v in status.items() if not v]
        return 0 if not missing else 1

    if args.verify_only:
        logger.info("Verifying existing model files against known SHA-256 hashes...")
        any_failed = False
        for key, spec in _MODEL_REGISTRY.items():
            if not is_model_present(key):
                logger.warning(f"  [SKIP] {spec.name} — not downloaded")
                continue
            if spec.sha256 is None:
                logger.warning(f"  [WARN] {spec.name} — no sha256 hash configured, cannot verify")
                any_failed = True
            else:
                path = spec.local_path
                if _verify_sha256(path, spec.sha256):
                    logger.info(f"  [OK]   {spec.name}")
                else:
                    logger.error(f"  [FAIL] {spec.name}")
                    any_failed = True
        return 1 if any_failed else 0

    if args.dry_run:
        logger.info("Dry run — showing what would be downloaded:")
        for key, spec in _MODEL_REGISTRY.items():
            present = is_model_present(key)
            action = "SKIP (already present)" if present else "DOWNLOAD"
            logger.info(f"  [{action}] {spec.name} → {spec.local_path}")
            if not present:
                logger.info(f"           URL: {spec.url}")
        return 0

    if args.model:
        success = download_model(args.model, force=args.force)
        return 0 if success else 1

    if args.all:
        results = download_all_models(force=args.force, skip=args.skip)
    else:
        # Default: minimum
        results = download_minimum_models(force=args.force)

    failed = [k for k, v in results.items() if not v]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
