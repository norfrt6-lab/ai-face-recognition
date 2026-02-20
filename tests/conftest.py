"""Shared pytest fixtures for all test modules."""

from __future__ import annotations

import numpy as np
import pytest

from core.detector.base_detector import FaceBox
from core.recognizer.base_recognizer import FaceEmbedding


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_image() -> np.ndarray:
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def random_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_box() -> FaceBox:
    return FaceBox(
        x1=100, y1=80, x2=300, y2=320,
        confidence=0.92,
        face_index=0,
    )


@pytest.fixture
def sample_face_box_with_landmarks() -> FaceBox:
    lm = np.array(
        [[150.0, 140.0], [250.0, 140.0], [200.0, 200.0],
         [160.0, 270.0], [240.0, 270.0]],
        dtype=np.float32,
    )
    return FaceBox(
        x1=100, y1=80, x2=300, y2=320,
        confidence=0.92, face_index=0, landmarks=lm,
    )


@pytest.fixture
def random_embedding() -> FaceEmbedding:
    rng = np.random.default_rng(7)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return FaceEmbedding(vector=v, face_index=0)
