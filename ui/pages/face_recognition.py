# Streamlit page for face recognition and identity registration.
#
# Features:
#   - Upload an image and detect + recognize all faces
#   - View identity matches with confidence scores
#   - Register new identities from uploaded photos
#   - Browse and manage the face database
#   - Visualize bounding boxes and landmarks on detected faces

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


import os as _os
_DEFAULT_API_URL = _os.getenv("UI_API_BASE_URL", "http://localhost:8000")
_SECTION_RECOGNIZE = "🔍 Recognize Faces"
_SECTION_REGISTER  = "➕ Register Identity"
_SECTION_DATABASE  = "🗄️ Face Database"

_CONFIDENCE_COLORS = {
    "high":   "#00C851",   # green  ≥ 0.75
    "medium": "#FF8800",   # orange 0.50 – 0.74
    "low":    "#FF4444",   # red    < 0.50
    "unknown": "#AAAAAA",  # grey   not recognized
}


def _api_url() -> str:
    return st.session_state.get("api_base_url", _DEFAULT_API_URL)


def _post_recognize(
    image_bytes: bytes,
    filename: str,
    return_attributes: bool = True,
    similarity_threshold: Optional[float] = None,
) -> Optional[Dict]:
    """Call POST /api/v1/recognize and return the parsed JSON (or None on error)."""
    url = f"{_api_url()}/api/v1/recognize"

    data: Dict[str, Any] = {
        "consent":           "true",
        "return_attributes": str(return_attributes).lower(),
        "top_k":             "5",
    }
    if similarity_threshold is not None:
        data["similarity_threshold"] = str(similarity_threshold)

    files = {"image": (filename, image_bytes, "image/jpeg")}

    try:
        resp = requests.post(url, files=files, data=data, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('message', resp.text)}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to API at `{_api_url()}`. "
            "Make sure the API server is running."
        )
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _post_register(
    image_bytes: bytes,
    filename: str,
    name: str,
    identity_id: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[Dict]:
    """Call POST /api/v1/register and return the parsed JSON (or None on error)."""
    url = f"{_api_url()}/api/v1/register"

    data: Dict[str, Any] = {
        "name":      name,
        "consent":   "true",
        "overwrite": str(overwrite).lower(),
    }
    if identity_id:
        data["identity_id"] = identity_id

    files = {"image": (filename, image_bytes, "image/jpeg")}

    try:
        resp = requests.post(url, files=files, data=data, timeout=60)
        if resp.status_code in (200, 201):
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('message', resp.text)}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at `{_api_url()}`.")
        return None
    except Exception as exc:
        st.error(f"Registration failed: {exc}")
        return None


def _get_identities(
    page: int = 1,
    page_size: int = 50,
    name_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Call GET /api/v1/identities and return parsed JSON."""
    url    = f"{_api_url()}/api/v1/identities"
    params = {"page": page, "page_size": page_size}
    if name_filter:
        params["name_filter"] = name_filter

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at `{_api_url()}`.")
        return None
    except Exception as exc:
        st.error(f"Failed to list identities: {exc}")
        return None


def _delete_identity(identity_id: str) -> bool:
    """Call DELETE /api/v1/identities/{id} and return True on success."""
    url  = f"{_api_url()}/api/v1/identities/{identity_id}"
    data = {"confirm": "true"}
    try:
        resp = requests.delete(url, data=data, timeout=15)
        return resp.status_code == 200
    except Exception as exc:
        st.error(f"Delete failed: {exc}")
        return False


def _rename_identity(identity_id: str, new_name: str) -> bool:
    """Call PATCH /api/v1/identities/{id} and return True on success."""
    url  = f"{_api_url()}/api/v1/identities/{identity_id}"
    data = {"new_name": new_name}
    try:
        resp = requests.patch(url, data=data, timeout=15)
        return resp.status_code == 200
    except Exception as exc:
        st.error(f"Rename failed: {exc}")
        return False


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _annotate_image(
    image: Image.Image,
    faces: List[Dict],
) -> Image.Image:
    """
    Draw bounding boxes and labels on *image* for each face in *faces*.

    Args:
        image: PIL Image (RGB).
        faces: List of recognized face dicts from the API response.

    Returns:
        Annotated PIL Image (RGB copy).
    """
    annotated = image.copy().convert("RGB")
    draw      = ImageDraw.Draw(annotated, "RGBA")

    # Try to load a reasonable font; fall back to default if unavailable
    try:
        font_label = ImageFont.truetype("arial.ttf", size=14)
        font_small = ImageFont.truetype("arial.ttf", size=11)
    except (IOError, OSError):
        font_label = ImageFont.load_default()
        font_small = font_label

    for face in faces:
        bbox  = face.get("bbox", {})
        match = face.get("match", {})
        attrs = face.get("attributes") or {}

        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        is_known    = match.get("is_known", False)
        identity    = match.get("identity_name") or "Unknown"
        similarity  = match.get("similarity", 0.0)
        conf_detect = bbox.get("confidence", 0.0)

        # Choose box colour
        if not is_known:
            box_hex = _CONFIDENCE_COLORS["unknown"]
        elif similarity >= 0.75:
            box_hex = _CONFIDENCE_COLORS["high"]
        elif similarity >= 0.50:
            box_hex = _CONFIDENCE_COLORS["medium"]
        else:
            box_hex = _CONFIDENCE_COLORS["low"]

        # Parse hex colour to RGB
        r = int(box_hex[1:3], 16)
        g = int(box_hex[3:5], 16)
        b = int(box_hex[5:7], 16)

        # Semi-transparent fill
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(r, g, b, 255),
            fill=(r, g, b, 40),
            width=2,
        )

        # Label background
        label = f"{identity}"
        if is_known:
            label += f"  {similarity:.0%}"

        bbox_text = draw.textbbox((x1, y1 - 20), label, font=font_label)
        pad = 3
        draw.rectangle(
            [bbox_text[0] - pad, bbox_text[1] - pad,
             bbox_text[2] + pad, bbox_text[3] + pad],
            fill=(r, g, b, 200),
        )
        draw.text((x1, y1 - 20), label, fill=(255, 255, 255, 255), font=font_label)

        # Age / gender sub-label
        if attrs:
            age    = attrs.get("age")
            gender = attrs.get("gender")
            parts  = []
            if age is not None:
                parts.append(f"age {age:.0f}")
            if gender:
                parts.append(gender)
            if parts:
                sub_label = "  ".join(parts)
                draw.text(
                    (x1 + 3, y2 + 3),
                    sub_label,
                    fill=(r, g, b, 220),
                    font=font_small,
                )

        # Draw landmarks
        landmarks = face.get("landmarks") or []
        for lm in landmarks:
            lx = int(lm.get("x", 0))
            ly = int(lm.get("y", 0))
            draw.ellipse(
                [lx - 2, ly - 2, lx + 2, ly + 2],
                fill=(255, 215, 0, 200),
            )

    return annotated


def _render_recognize_section() -> None:
    """Render the 'Recognize Faces' tab content."""

    st.subheader("Upload an image to detect and recognize faces")
    st.caption(
        "The image will be sent to the API server, which will detect all faces, "
        "match them against the registered identity database, and return results."
    )

    col_upload, col_settings = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="recog_upload",
        )
        
        # Consent checkbox
        consent = st.checkbox(
            "✅ I confirm I have explicit consent from all individuals in this image.",
            key="recog_consent",
            value=True,
        )

    with col_settings:
        st.markdown("**Settings**")
        show_attributes = st.checkbox("Show age / gender", value=True, key="recog_attrs")
        threshold_override = st.slider(
            "Similarity threshold",
            min_value=0.20,
            max_value=0.95,
            value=0.45,
            step=0.05,
            key="recog_threshold",
            help="Faces with similarity above this value are considered 'known'.",
        )
        annotate_image = st.checkbox("Annotate image", value=True, key="recog_annotate")

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return
    
    if not consent:
        st.warning("⚠️ Please check the consent box to proceed with face recognition.")
        return

    image_bytes = uploaded.read()
    pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.divider()

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button(
            "🔍 Recognize",
            type="primary",
            use_container_width=True,
            key="recog_run_btn",
            disabled=not uploaded or not consent,
        )

    if not run_btn and "recog_last_result" not in st.session_state:
        img_col, _ = st.columns([2, 1])
        with img_col:
            st.image(pil_image, caption="Uploaded image", use_container_width=True)
        return

    # Cache result so re-renders don't re-call the API
    if run_btn:
        with st.spinner("Running face recognition …"):
            t0     = time.perf_counter()
            result = _post_recognize(
                image_bytes=image_bytes,
                filename=uploaded.name,
                return_attributes=show_attributes,
                similarity_threshold=threshold_override,
            )
            elapsed = (time.perf_counter() - t0) * 1000
        if result is not None:
            st.session_state["recog_last_result"] = result
            st.session_state["recog_last_image"]  = pil_image
            st.session_state["recog_elapsed_ms"]  = elapsed
        else:
            return

    result    = st.session_state.get("recog_last_result", {})
    pil_image = st.session_state.get("recog_last_image", pil_image)
    elapsed   = st.session_state.get("recog_elapsed_ms", 0.0)

    faces     = result.get("faces", [])
    n_det     = result.get("num_faces_detected", 0)
    n_rec     = result.get("num_faces_recognized", 0)
    inf_ms    = result.get("inference_time_ms", elapsed)
    img_w     = result.get("image_width", pil_image.width)
    img_h     = result.get("image_height", pil_image.height)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faces detected",    n_det)
    m2.metric("Faces recognized",  n_rec)
    m3.metric("Inference time",    f"{inf_ms:.0f} ms")
    m4.metric("Image size",        f"{img_w}×{img_h}")

    st.divider()

    img_col, res_col = st.columns([3, 2])

    with img_col:
        if annotate_image and faces:
            display_img = _annotate_image(pil_image, faces)
        else:
            display_img = pil_image
        st.image(display_img, caption="Detection results", use_container_width=True)

    with res_col:
        if not faces:
            st.warning("No faces detected in this image.")
        else:
            st.markdown(f"**{n_det} face(s) found:**")
            for face in faces:
                idx    = face.get("face_index", 0)
                match  = face.get("match", {})
                attrs  = face.get("attributes") or {}
                bbox   = face.get("bbox", {})

                is_known   = match.get("is_known", False)
                identity   = match.get("identity_name") or "Unknown"
                similarity = match.get("similarity", 0.0)
                det_conf   = bbox.get("confidence", 0.0)

                # Colour badge
                if not is_known:
                    badge = "🔴"
                elif similarity >= 0.75:
                    badge = "🟢"
                elif similarity >= 0.50:
                    badge = "🟡"
                else:
                    badge = "🟠"

                with st.expander(
                    f"{badge} Face #{idx} — {identity}",
                    expanded=(idx == 0),
                ):
                    st.markdown(f"**Identity:** `{identity}`")
                    if is_known:
                        st.markdown(f"**Match confidence:** `{similarity:.1%}`")
                        st.progress(min(similarity, 1.0))
                    else:
                        st.markdown("**Status:** Not in database")
                        if similarity > 0:
                            st.markdown(f"**Closest match score:** `{similarity:.1%}`")

                    st.markdown(f"**Detection confidence:** `{det_conf:.1%}`")

                    if attrs:
                        st.markdown("**Attributes:**")
                        age    = attrs.get("age")
                        gender = attrs.get("gender")
                        if age is not None:
                            st.markdown(f"- Age: `{age:.0f}` years")
                        if gender:
                            st.markdown(f"- Gender: `{gender}`")

                    st.markdown(
                        f"**Bbox:** x1={bbox.get('x1',0):.0f} "
                        f"y1={bbox.get('y1',0):.0f} "
                        f"x2={bbox.get('x2',0):.0f} "
                        f"y2={bbox.get('y2',0):.0f}"
                    )


def _render_register_section() -> None:
    """Render the 'Register Identity' tab content."""

    st.subheader("Register a new face identity")
    st.caption(
        "Upload a clear, front-facing photo and provide a name. "
        "The API will detect the best face, extract its embedding, "
        "and store it in the face database."
    )

    with st.form("register_form", clear_on_submit=False):
        col_img, col_form = st.columns([1, 1])

        with col_img:
            uploaded = st.file_uploader(
                "Face image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                key="reg_upload",
            )
            if uploaded:
                pil_preview = Image.open(io.BytesIO(uploaded.read()))
                uploaded.seek(0)
                st.image(pil_preview, caption="Preview", use_container_width=True)

        with col_form:
            name = st.text_input(
                "Identity name *",
                placeholder="e.g. Alice Smith",
                key="reg_name",
                help="A unique human-readable label for this person.",
            )
            st.markdown("---")
            st.markdown("**Advanced**")
            identity_id = st.text_input(
                "Existing identity UUID (optional)",
                placeholder="Leave blank to create new",
                key="reg_identity_id",
                help=(
                    "Provide an existing UUID to add more embeddings to "
                    "an existing identity (multi-shot registration)."
                ),
            )
            overwrite = st.checkbox(
                "Overwrite existing embeddings",
                value=False,
                key="reg_overwrite",
                help="Replace all stored embeddings instead of appending.",
            )
            st.markdown("---")
            consent_check = st.checkbox(
                "✅ I have explicit consent from the person being registered",
                key="reg_consent",
            )

        submitted = st.form_submit_button(
            "➕ Register Identity",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    if not uploaded:
        st.error("Please upload a face image.")
        return
    if not name.strip():
        st.error("Please enter a name for this identity.")
        return
    if not consent_check:
        st.error(
            "You must confirm that you have explicit consent from the person "
            "being registered."
        )
        return

    image_bytes = uploaded.read()

    with st.spinner(f"Registering '{name}' …"):
        result = _post_register(
            image_bytes=image_bytes,
            filename=uploaded.name,
            name=name.strip(),
            identity_id=identity_id.strip() or None,
            overwrite=overwrite,
        )

    if result:
        st.success(f"✅ {result.get('message', 'Identity registered successfully.')}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Identity name",     result.get("identity_name", name))
        col_b.metric("Embeddings added",  result.get("embeddings_added", 1))
        col_c.metric("Total embeddings",  result.get("total_embeddings", 1))

        st.code(
            f"Identity ID: {result.get('identity_id', 'N/A')}",
            language="text",
        )

        # Invalidate the database cache so the DB tab refreshes
        if "db_identities_cache" in st.session_state:
            del st.session_state["db_identities_cache"]


def _render_database_section() -> None:
    """Render the 'Face Database' tab content."""

    st.subheader("Registered face identities")
    st.caption("Browse, search, rename, or delete identities stored in the face database.")

    search_col, btn_col = st.columns([3, 1])
    with search_col:
        name_filter = st.text_input(
            "Search by name",
            placeholder="Type to filter …",
            key="db_search",
            label_visibility="collapsed",
        )
    with btn_col:
        refresh_btn = st.button(
            "🔄 Refresh",
            use_container_width=True,
            key="db_refresh",
        )

    # Fetch identities (cache in session state until refresh)
    if refresh_btn or "db_identities_cache" not in st.session_state:
        with st.spinner("Loading database …"):
            data = _get_identities(page_size=200, name_filter=name_filter or None)
        if data is not None:
            st.session_state["db_identities_cache"] = data
    else:
        data = st.session_state.get("db_identities_cache")

    if data is None:
        st.error("Failed to load the face database.")
        return

    total = data.get("total", 0)
    items = data.get("items", [])

    # Apply client-side filter too (in case we loaded cached data)
    if name_filter:
        nf    = name_filter.lower()
        items = [i for i in items if nf in i.get("name", "").lower()]

    st.markdown(f"**{total} identit{'y' if total == 1 else 'ies'} registered**")

    if not items:
        st.info(
            "No identities found. "
            "Go to the **Register Identity** tab to add the first one."
        )
        return

    st.divider()

    for identity in items:
        uid       = identity.get("identity_id", "")
        name      = identity.get("name", "Unknown")
        num_emb   = identity.get("num_embeddings", "?")
        created   = identity.get("created_at")

        created_str = ""
        if created:
            try:
                import datetime  # noqa: PLC0415
                created_str = datetime.datetime.fromtimestamp(
                    float(created)
                ).strftime("%Y-%m-%d %H:%M")
            except Exception:
                created_str = str(created)

        with st.expander(f"👤  {name}  —  {num_emb} embedding(s)", expanded=False):
            col_info, col_actions = st.columns([2, 1])

            with col_info:
                st.markdown(f"**UUID:** `{uid}`")
                if created_str:
                    st.markdown(f"**Registered:** {created_str}")
                st.markdown(f"**Embeddings stored:** {num_emb}")

            with col_actions:
                st.markdown("**Rename**")
                new_name_key = f"rename_{uid}"
                new_name = st.text_input(
                    "New name",
                    value=name,
                    key=new_name_key,
                    label_visibility="collapsed",
                )
                rename_btn = st.button(
                    "💾 Save name",
                    key=f"rename_btn_{uid}",
                    use_container_width=True,
                )
                if rename_btn:
                    if new_name.strip() and new_name.strip() != name:
                        if _rename_identity(uid, new_name.strip()):
                            st.success(f"Renamed to '{new_name.strip()}'")
                            if "db_identities_cache" in st.session_state:
                                del st.session_state["db_identities_cache"]
                            st.rerun()
                    else:
                        st.info("Name unchanged.")

                st.markdown("---")

                confirm_key = f"del_confirm_{uid}"
                confirmed   = st.checkbox(
                    "Confirm delete",
                    key=confirm_key,
                )
                delete_btn = st.button(
                    "🗑️ Delete",
                    key=f"del_btn_{uid}",
                    use_container_width=True,
                    type="secondary",
                )
                if delete_btn:
                    if confirmed:
                        if _delete_identity(uid):
                            st.success(f"Identity '{name}' deleted.")
                            if "db_identities_cache" in st.session_state:
                                del st.session_state["db_identities_cache"]
                            st.rerun()
                        else:
                            st.error("Delete failed.")
                    else:
                        st.warning("Check 'Confirm delete' first.")


def render(api_base_url: str = _DEFAULT_API_URL) -> None:
    """
    Render the full Face Recognition page.

    This function is called by ``ui/app.py`` and receives the
    API base URL from the global sidebar settings.

    Args:
        api_base_url: Base URL of the FastAPI backend.
    """
    st.session_state["api_base_url"] = api_base_url

    st.title("🔍 Face Recognition")
    st.markdown(
        "Detect, identify, and manage faces using the ArcFace recognition pipeline."
    )

    tab_recognize, tab_register, tab_database = st.tabs([
        _SECTION_RECOGNIZE,
        _SECTION_REGISTER,
        _SECTION_DATABASE,
    ])

    with tab_recognize:
        _render_recognize_section()

    with tab_register:
        _render_register_section()

    with tab_database:
        _render_database_section()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Recognition — AI Face Swap",
        page_icon="🔍",
        layout="wide",
    )
    render()
