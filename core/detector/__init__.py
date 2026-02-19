# ============================================================
# AI Face Recognition & Face Swap
# core/detector/__init__.py
# ============================================================

from core.detector.base_detector import (
    BaseDetector,
    DetectionResult,
    FaceBox,
    face_box_from_xyxy,
)
from core.detector.yolo_detector import YOLOFaceDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "FaceBox",
    "face_box_from_xyxy",
    "YOLOFaceDetector",
]
