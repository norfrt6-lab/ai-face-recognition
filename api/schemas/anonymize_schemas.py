"""Pydantic request/response models for the anonymize endpoint."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AnonymizeMethod(str, Enum):
    """Supported face anonymization methods."""

    blur = "blur"
    pixelate = "pixelate"
    solid = "solid"


class AnonymizeResponse(BaseModel):
    """Response metadata returned alongside the anonymized image download."""

    faces_anonymized: int = Field(
        ...,
        description="Number of face regions that were anonymized.",
    )
    method: AnonymizeMethod = Field(
        ...,
        description="The anonymization method that was applied.",
    )
    processing_time_ms: float = Field(
        ...,
        description="Total server-side processing time in milliseconds.",
    )
