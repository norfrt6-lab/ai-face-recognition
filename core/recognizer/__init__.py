# ============================================================
# AI Face Recognition & Face Swap - Core Recognizer Module
# ============================================================

from core.recognizer.base_recognizer import BaseRecognizer, FaceMatch
from core.recognizer.insightface_recognizer import InsightFaceRecognizer
from core.recognizer.face_database import FaceDatabase

__all__ = [
    "BaseRecognizer",
    "FaceMatch",
    "InsightFaceRecognizer",
    "FaceDatabase",
]
