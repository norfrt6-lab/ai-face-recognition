"""Pydantic response model for the face similarity endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SimilarityResponse(BaseModel):
    """Response body for POST /api/v1/similarity."""

    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the two face embeddings (0–1).",
    )
    is_same_person: bool = Field(
        ...,
        description="True when similarity >= threshold.",
    )
    threshold: float = Field(
        ...,
        description="Similarity threshold used to determine is_same_person.",
    )
    face_a_detected: bool = Field(
        ...,
        description="Whether a face was successfully detected in image A.",
    )
    face_b_detected: bool = Field(
        ...,
        description="Whether a face was successfully detected in image B.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds.",
    )
