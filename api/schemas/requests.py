# ============================================================
# AI Face Recognition & Face Swap
# api/schemas/requests.py
# ============================================================
# Pydantic v2 request models for all FastAPI endpoints.
#
# Schemas:
#   RecognizeRequest   — POST /api/v1/recognize
#   RegisterRequest    — POST /api/v1/register
#   SwapRequest        — POST /api/v1/swap
#   HealthRequest      — (no body, included for completeness)
# ============================================================

from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# Enumerations
# ============================================================

class BlendModeSchema(str, Enum):
    """Compositing strategy for face paste-back."""
    alpha        = "alpha"
    poisson      = "poisson"
    masked_alpha = "masked_alpha"


class EnhancerBackendSchema(str, Enum):
    """Which face enhancement model to apply after swap."""
    gfpgan      = "gfpgan"
    codeformer  = "codeformer"
    none        = "none"


# ============================================================
# Shared base
# ============================================================

class BaseAPIRequest(BaseModel):
    """Common fields shared across all requests."""

    model_config = {"str_strip_whitespace": True, "extra": "forbid"}


# ============================================================
# Recognition request
# ============================================================

class RecognizeRequest(BaseAPIRequest):
    """
    POST /api/v1/recognize

    Ask the API to detect all faces in the uploaded image and
    attempt to match each one against the registered face database.

    The image is supplied as a form-data file upload (not JSON),
    so this schema covers only the *auxiliary* JSON fields that
    can be sent alongside the file.

    Fields:
        top_k:               Return at most this many candidate matches
                             per face (default 1 = best match only).
        similarity_threshold: Override the server-side default cosine
                             similarity threshold for this request.
        return_attributes:   Include age / gender predictions in the
                             response (slightly slower).
        return_embeddings:   Include raw 512-dim embedding vectors in
                             the response (useful for debugging).
        consent:             Must be True — explicit consent required
                             by the ethics policy.
    """

    top_k: Annotated[int, Field(ge=1, le=20)] = Field(
        default=1,
        description="Number of top identity candidates to return per detected face.",
    )
    similarity_threshold: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default=None,
        description=(
            "Cosine similarity threshold override for this request. "
            "Falls back to server default when omitted."
        ),
    )
    return_attributes: bool = Field(
        default=False,
        description="Include predicted age and gender in the response.",
    )
    return_embeddings: bool = Field(
        default=False,
        description="Include raw 512-dim ArcFace embedding vectors in the response.",
    )
    consent: bool = Field(
        default=False,
        description="Explicit consent flag required by the ethics policy.",
    )

    @field_validator("consent")
    @classmethod
    def consent_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "consent must be true. "
                "You must have explicit consent from all individuals in the image."
            )
        return v


# ============================================================
# Registration request
# ============================================================

class RegisterRequest(BaseAPIRequest):
    """
    POST /api/v1/register

    Register a new face identity (or add an embedding to an
    existing one) in the face database.

    The face image is supplied as a form-data file upload.

    Fields:
        name:            Human-readable identity label (e.g. "Alice").
        identity_id:     Optional existing identity UUID to append to.
                         If omitted a new identity is created.
        overwrite:       If True and *identity_id* already exists,
                         replace all stored embeddings instead of appending.
        metadata:        Optional free-form JSON object attached to the
                         identity record (e.g. {"role": "employee"}).
        consent:         Explicit consent flag.
    """

    name: Annotated[str, Field(min_length=1, max_length=128)] = Field(
        description="Human-readable label for the identity (e.g. 'Alice').",
    )
    identity_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing identity UUID to append an embedding to. "
            "Leave blank to create a new identity."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "When True and identity_id is provided, replace stored embeddings "
            "instead of appending."
        ),
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional free-form metadata dict attached to the identity record.",
    )
    consent: bool = Field(
        default=False,
        description="Explicit consent flag required by the ethics policy.",
    )

    @field_validator("name")
    @classmethod
    def name_no_slashes(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("name must not contain path separators.")
        return v

    @field_validator("consent")
    @classmethod
    def consent_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "consent must be true. "
                "You must have explicit consent from the person being registered."
            )
        return v


# ============================================================
# Swap request
# ============================================================

class SwapRequest(BaseAPIRequest):
    """
    POST /api/v1/swap

    Perform a face swap: replace the face(s) in the *target* image
    with the identity from the *source* image.

    Both images are supplied as form-data file uploads.
    This schema carries the auxiliary control fields.

    Fields:
        blend_mode:         Compositing strategy (poisson recommended).
        blend_alpha:        Global blend strength [0.0, 1.0].
                            1.0 = fully swapped, 0.0 = original unchanged.
        mask_feather:       Mask edge blur radius in pixels.
        enhance:            Run face enhancement (GFPGAN/CodeFormer) after swap.
        enhancer_backend:   Which enhancer to use when enhance=True.
        enhancer_fidelity:  Fidelity weight for CodeFormer [0.0, 1.0].
        swap_all_faces:     Swap every detected face in the target image.
                            When False, only the largest/most-confident face
                            is swapped.
        max_faces:          Cap on how many faces to swap when swap_all=True.
        source_face_index:  Index of the face to use from the *source* image
                            when multiple faces are present (0-based).
        target_face_index:  Index of the face to replace in the *target* image
                            when swap_all_faces=False.
        watermark:          Embed an "AI GENERATED" watermark on the output.
        consent:            Explicit consent flag.
    """

    blend_mode: BlendModeSchema = Field(
        default=BlendModeSchema.poisson,
        description="Compositing strategy used to paste the swapped face back.",
    )
    blend_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0,
        description="Global alpha strength: 1.0 = fully swapped, 0.0 = original.",
    )
    mask_feather: Annotated[int, Field(ge=0, le=100)] = Field(
        default=20,
        description="Gaussian blur radius (pixels) applied to the blend mask edges.",
    )
    enhance: bool = Field(
        default=False,
        description="Apply face enhancement (GFPGAN / CodeFormer) after swap.",
    )
    enhancer_backend: EnhancerBackendSchema = Field(
        default=EnhancerBackendSchema.gfpgan,
        description="Enhancement backend to use when enhance=True.",
    )
    enhancer_fidelity: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description=(
            "CodeFormer fidelity weight. "
            "0.0 = maximum quality enhancement, 1.0 = maximum identity preservation."
        ),
    )
    swap_all_faces: bool = Field(
        default=False,
        description="When True, replace every detected face in the target image.",
    )
    max_faces: Annotated[int, Field(ge=1, le=50)] = Field(
        default=10,
        description="Maximum number of faces to swap when swap_all_faces=True.",
    )
    source_face_index: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Which face in the source image to use as the donor identity (0-based).",
    )
    target_face_index: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description=(
            "Which face in the target image to replace when swap_all_faces=False (0-based)."
        ),
    )
    watermark: bool = Field(
        default=True,
        description='Embed an "AI GENERATED" watermark on all swapped outputs.',
    )
    consent: bool = Field(
        default=False,
        description="Explicit consent flag required by the ethics policy.",
    )

    @field_validator("consent")
    @classmethod
    def consent_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "consent must be true. "
                "You must have explicit consent from all individuals involved."
            )
        return v

    @model_validator(mode="after")
    def validate_face_indices(self) -> "SwapRequest":
        if self.swap_all_faces and self.target_face_index != 0:
            # When swapping all faces, target_face_index is ignored — warn via value reset
            pass  # Not an error, just informational
        return self


# ============================================================
# Database management requests
# ============================================================

class DeleteIdentityRequest(BaseAPIRequest):
    """
    DELETE /api/v1/identities/{identity_id}

    Remove a registered identity from the face database.

    Fields:
        confirm:   Must be True to prevent accidental deletion.
    """

    confirm: bool = Field(
        default=False,
        description="Set to True to confirm deletion of the identity record.",
    )

    @field_validator("confirm")
    @classmethod
    def confirm_must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError("confirm must be true to delete an identity.")
        return v


class RenameIdentityRequest(BaseAPIRequest):
    """
    PATCH /api/v1/identities/{identity_id}

    Rename an existing identity in the face database.

    Fields:
        new_name:   New human-readable label for the identity.
    """

    new_name: Annotated[str, Field(min_length=1, max_length=128)] = Field(
        description="New label to assign to this identity.",
    )

    @field_validator("new_name")
    @classmethod
    def name_no_slashes(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("new_name must not contain path separators.")
        return v


class ListIdentitiesRequest(BaseAPIRequest):
    """
    GET /api/v1/identities  (query params)

    Optional filters for listing registered identities.

    Fields:
        page:       1-based page number.
        page_size:  Number of identities per page.
        name_filter: Optional substring filter on identity name.
    """

    page: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Page number (1-based).",
    )
    page_size: Annotated[int, Field(ge=1, le=200)] = Field(
        default=50,
        description="Number of identities to return per page.",
    )
    name_filter: Optional[str] = Field(
        default=None,
        description="Optional case-insensitive substring filter on identity name.",
    )
