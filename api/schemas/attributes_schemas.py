"""Pydantic response models for the face attributes endpoint."""

from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field


class FaceAttributesItem(BaseModel):
    """Geometric attributes for a single detected face."""

    face_index: int = Field(
        ...,
        description="Zero-based index of this face in the detection result.",
    )
    bbox: List[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in absolute pixel coordinates.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score.",
    )
    area: int = Field(
        ...,
        ge=0,
        description="Bounding box area in pixels² (width × height).",
    )
    aspect_ratio: float = Field(
        ...,
        ge=0.0,
        description="Width-to-height ratio of the bounding box.",
    )
    center: List[int] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Centre point [cx, cy] of the bounding box.",
    )


class FaceAttributesResponse(BaseModel):
    """Response body for POST /api/v1/faces/attributes."""

    num_faces: int = Field(
        ...,
        ge=0,
        description="Total number of faces detected.",
    )
    faces: List[FaceAttributesItem] = Field(
        ...,
        description="List of per-face geometry attributes.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds.",
    )
