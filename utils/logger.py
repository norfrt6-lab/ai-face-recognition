from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger

# Re-export the loguru logger so the rest of the app imports from here
logger = _logger

# Track whether setup has been called
_configured = False


def setup_logger(
    level: str = "INFO",
    file_path: Optional[Path | str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip",
    json_logs: bool = False,
    colorize: bool = True,
) -> None:
    """
    Configure the global Loguru logger.

    Call this ONCE at application startup (e.g. in main.py / app.py).
    Subsequent calls are ignored unless force=True equivalent logic is added.

    Args:
        level:       Minimum log level (TRACE | DEBUG | INFO | SUCCESS |
                     WARNING | ERROR | CRITICAL).
        file_path:   Optional path to a log file.  None = stdout only.
        rotation:    Loguru rotation rule  (e.g. "10 MB", "1 day", "00:00").
        retention:   How long to keep rotated files (e.g. "7 days", "4 weeks").
        compression: Archive format for rotated files ("zip", "gz", "tar").
        json_logs:   Emit structured JSON instead of human-readable text.
                     Useful when forwarding logs to Elastic / Loki / Datadog.
        colorize:    Add ANSI colour codes to the stdout sink.
    """
    global _configured

    # Remove the default Loguru sink (stderr with its default format)
    _logger.remove()

    # ------------------------------------------------------------------ #
    # Format strings
    # ------------------------------------------------------------------ #
    _TEXT_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <9}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    _PLAIN_FORMAT = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <9} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # ------------------------------------------------------------------ #
    # Stdout sink
    # ------------------------------------------------------------------ #
    if json_logs:
        _logger.add(
            sys.stdout,
            level=level,
            serialize=True,          # Loguru built-in JSON serialisation
            backtrace=True,
            diagnose=False,          # Avoid leaking variable values in prod
            enqueue=True,            # Thread-safe async logging
        )
    else:
        _logger.add(
            sys.stdout,
            level=level,
            format=_TEXT_FORMAT,
            colorize=colorize,
            backtrace=True,
            diagnose=level in ("DEBUG", "TRACE"),
            enqueue=True,
        )

    # ------------------------------------------------------------------ #
    # File sink (optional)
    # ------------------------------------------------------------------ #
    if file_path:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        _logger.add(
            str(log_path),
            level=level,
            format=_PLAIN_FORMAT if not json_logs else "{message}",
            serialize=json_logs,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=False,
            enqueue=True,
            encoding="utf-8",
        )

    _configured = True
    _logger.debug(
        "Logger initialised | level={} | file={} | json={}",
        level,
        file_path or "stdout only",
        json_logs,
    )


def get_logger(name: str):
    """
    Return a child logger bound to the given module name.

    Usage::

        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Hello from {}", __name__)

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A Loguru logger with the ``name`` context bound.
    """
    return _logger.bind(name=name)


# ------------------------------------------------------------------ #
# Convenience: setup with defaults from config/settings.py
# ------------------------------------------------------------------ #

def setup_from_settings() -> None:
    """
    Initialise the logger using values from ``config.settings``.

    This avoids a circular-import by doing the settings import lazily
    inside the function body.
    """
    try:
        from config.settings import settings  # noqa: PLC0415

        log_cfg = settings.logging
        setup_logger(
            level=log_cfg.level,
            file_path=log_cfg.file_path,
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            json_logs=log_cfg.json_logs,
            colorize=not settings.is_production,
        )
    except Exception as exc:  # pragma: no cover
        # Fall back to a safe default so the app can still start
        setup_logger(level="INFO")
        _logger.warning("Could not load settings for logger setup: {}", exc)


# ------------------------------------------------------------------ #
# Auto-setup with safe defaults if the module is imported before
# setup_logger() / setup_from_settings() is explicitly called.
# ------------------------------------------------------------------ #

if not _configured:
    setup_logger(level="DEBUG", colorize=True)
