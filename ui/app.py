# Streamlit web UI entry point.
#
# Pages:
#   🔄 Face Swap       — upload source + target, run swap
#   🔍 Recognition     — detect + identify faces in an image
#   📋 Identities      — browse / manage the face database
#   ℹ️  About           — project info, ethics notice
#
# Run with:
#   streamlit run ui/app.py --server.port 8501

from __future__ import annotations

import io
import os
import time
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# Configure logger from settings at startup
try:
    from utils.logger import setup_from_settings
    setup_from_settings()
except Exception:
    pass


st.set_page_config(
    page_title="AI Face Recognition & Swap",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "**AI Face Recognition & Face Swap**\n\n"
            "Powered by YOLOv8 · InsightFace · inswapper_128 · GFPGAN\n\n"
            "⚠️ Use responsibly and with explicit consent."
        ),
    },
)


DEFAULT_API_URL = os.getenv("UI_API_BASE_URL", "http://localhost:8000")
API_TIMEOUT     = 60   # seconds per request


def _init_state() -> None:
    """Initialise session-state keys on first run."""
    defaults = {
        "api_url":        DEFAULT_API_URL,
        "api_healthy":    None,   # None = not checked yet
        "api_version":    "—",
        "api_components": {},
        "last_health_check": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def _check_health(force: bool = False) -> dict:
    """
    Call GET /api/v1/health and cache result for 10 seconds.

    Returns the parsed JSON dict, or an empty dict on failure.
    """
    now = time.time()
    if not force and (now - st.session_state.last_health_check) < 10:
        return {}   # use cached status

    try:
        resp = requests.get(
            f"{_api_url()}/api/v1/health",
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.api_healthy    = data.get("status", "unknown")
            st.session_state.api_version    = data.get("version", "—")
            st.session_state.api_components = data.get("components", {})
            st.session_state.last_health_check = now
            return data
        else:
            st.session_state.api_healthy = "down"
            return {}
    except Exception:
        st.session_state.api_healthy = "down"
        return {}


def _post_form(
    endpoint:  str,
    fields:    dict,
    files:     Optional[dict] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """
    POST multipart/form-data to *endpoint* and return (json, error).

    Args:
        endpoint: Path after the API base URL (e.g. '/api/v1/swap').
        fields:   Dict of form field name → value.
        files:    Dict of field name → (filename, bytes, mime_type).

    Returns:
        Tuple of (response_json_or_none, error_message_or_none).
    """
    url = f"{_api_url()}{endpoint}"
    try:
        data  = {k: str(v) if not isinstance(v, bool) else str(v).lower()
                 for k, v in fields.items()}
        fdata = {}
        if files:
            for fname, (filename, content, mime) in files.items():
                fdata[fname] = (filename, content, mime)

        resp = requests.post(url, data=data, files=fdata, timeout=API_TIMEOUT)

        if resp.status_code in (200, 201):
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct:
                return resp.json(), None
            else:
                # Raw image bytes — wrap in a dict
                return {"_raw_bytes": resp.content, "_content_type": ct}, None
        else:
            try:
                detail = resp.json().get("message") or resp.json().get("detail") or resp.text
            except Exception:
                detail = resp.text
            return None, f"HTTP {resp.status_code}: {detail}"
    except requests.ConnectionError:
        return None, (
            f"Cannot connect to API at {_api_url()}. "
            "Make sure the backend is running."
        )
    except requests.Timeout:
        return None, f"Request timed out after {API_TIMEOUT}s."
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _img_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def _render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.title("🤖 AI Face Swap")
        st.caption("Recognition & Swap Pipeline")
        st.divider()

        with st.expander("⚙️ API Settings", expanded=False):
            new_url = st.text_input(
                "Backend URL",
                value=st.session_state.api_url,
                help="Base URL of the FastAPI backend.",
            )
            if new_url != st.session_state.api_url:
                st.session_state.api_url       = new_url
                st.session_state.api_healthy   = None
                st.session_state.last_health_check = 0.0

            if st.button("🔄 Refresh Health", use_container_width=True):
                _check_health(force=True)

            _check_health()
        health_status = st.session_state.api_healthy

        if health_status == "ok":
            st.success(
                f"✅ API online · v{st.session_state.api_version}",
                icon=None,
            )
        elif health_status == "degraded":
            st.warning(
                f"⚠️ API degraded · v{st.session_state.api_version}",
            )
        else:
            st.error("❌ API offline — start the backend.")

        components = st.session_state.api_components
        if components:
            st.caption("Components")
            for name, info in components.items():
                loaded = info.get("loaded", False)
                icon   = "🟢" if loaded else "🔴"
                st.caption(f"{icon} {name}")

        st.divider()

        page = st.radio(
            "Navigate",
            options=[
                "🔄 Face Swap",
                "🔍 Face Recognition",
                "📋 Identity Database",
                "ℹ️ About",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.caption("⚠️ Use with explicit consent only.")

    return page


def _page_face_swap() -> None:
    st.header("🔄 Face Swap")
    st.caption(
        "Upload a **source** image (whose face to copy) and a "
        "**target** image (where to put it). The API will detect, "
        "align, and swap the face."
    )

    consent = st.checkbox(
        "✅ I confirm I have **explicit consent** from all individuals "
        "in both images for this face swap.",
        value=False,
        key="swap_consent",
    )
    if not consent:
        st.warning(
            "You must confirm consent before running a face swap. "
            "Do not use this tool on images of people who have not agreed."
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Image")
        st.caption("The donor — whose face identity will be used.")
        source_file = st.file_uploader(
            "Upload source image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="swap_source",
            label_visibility="collapsed",
        )
        if source_file:
            st.image(source_file, use_column_width=True)

    with col2:
        st.subheader("Target Image")
        st.caption("The scene — whose face(s) will be replaced.")
        target_file = st.file_uploader(
            "Upload target image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="swap_target",
            label_visibility="collapsed",
        )
        if target_file:
            st.image(target_file, use_column_width=True)

    with st.expander("⚙️ Swap Options", expanded=False):
        opt_col1, opt_col2, opt_col3 = st.columns(3)

        with opt_col1:
            blend_mode   = st.selectbox(
                "Blend Mode",
                options=["poisson", "alpha", "masked_alpha"],
                index=0,
                help="poisson = seamless clone (best quality), alpha = fast",
            )
            blend_alpha  = st.slider(
                "Blend Alpha",
                min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="1.0 = fully swapped, 0.0 = original unchanged",
            )
            mask_feather = st.slider(
                "Mask Feather (px)",
                min_value=0, max_value=60, value=20, step=2,
            )

        with opt_col2:
            swap_all     = st.checkbox(
                "Swap All Faces",
                value=False,
                help="Replace every detected face, not just the primary one.",
            )
            max_faces    = st.number_input(
                "Max Faces",
                min_value=1, max_value=20, value=5, step=1,
            )
            watermark    = st.checkbox("Add Watermark", value=True)

        with opt_col3:
            enhance      = st.checkbox(
                "Enhance After Swap",
                value=False,
                help="Run GFPGAN / CodeFormer to remove artifacts (slower).",
            )
            enh_backend  = st.selectbox(
                "Enhancer",
                options=["gfpgan", "codeformer", "none"],
                index=0,
                disabled=not enhance,
            )
            enh_fidelity = st.slider(
                "Fidelity Weight (CodeFormer)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=not enhance,
                help="0.0 = max enhancement, 1.0 = max identity preservation.",
            )
        with opt_col3:
            return_base64 = st.checkbox(
                "Return as Base64 JSON",
                value=False,
                help="Return JSON with base64 image instead of raw download.",
            )

    st.divider()
    run_disabled = not consent or not source_file or not target_file
    run_clicked  = st.button(
        "🚀 Run Face Swap",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
    )

    if run_disabled and (source_file or target_file):
        if not consent:
            st.caption("⬆️ Check the consent box above to enable.")
        elif not source_file:
            st.caption("⬆️ Upload a source image.")
        elif not target_file:
            st.caption("⬆️ Upload a target image.")

    if run_clicked and consent and source_file and target_file:
        with st.spinner("Swapping faces… this may take a moment."):
            source_bytes = source_file.read()
            target_bytes = target_file.read()

            fields = {
                "blend_mode":       blend_mode,
                "blend_alpha":      blend_alpha,
                "mask_feather":     mask_feather,
                "swap_all_faces":   swap_all,
                "max_faces":        max_faces,
                "enhance":          enhance,
                "enhancer_backend": enh_backend,
                "enhancer_fidelity": enh_fidelity,
                "watermark":        watermark,
                "return_base64":    return_base64,
                "consent":          True,
            }
            files = {
                "source_file": (source_file.name, source_bytes, "image/jpeg"),
                "target_file": (target_file.name, target_bytes, "image/jpeg"),
            }

            result, error = _post_form("/api/v1/swap", fields, files)

        if error:
            st.error(f"❌ Swap failed: {error}")
        elif result:
            st.success("✅ Swap completed!")
            st.divider()

            # Check if we got raw image bytes back
            if "_raw_bytes" in result:
                raw = result["_raw_bytes"]
                st.subheader("Result")
                st.image(raw, use_column_width=True)
                st.download_button(
                    label="⬇️ Download Output",
                    data=raw,
                    file_name="swapped_output.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                # JSON response with base64
                import base64
                b64 = result.get("output_base64")
                if b64:
                    img_bytes = base64.b64decode(b64)
                    st.subheader("Result")
                    st.image(img_bytes, use_column_width=True)
                    st.download_button(
                        label="⬇️ Download Output",
                        data=img_bytes,
                        file_name="swapped_output.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                # Metadata
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                meta_col1.metric("Faces Swapped",  result.get("num_faces_swapped", "—"))
                meta_col2.metric("Faces Failed",   result.get("num_faces_failed",  "—"))
                meta_col3.metric("Enhanced",       "Yes" if result.get("enhanced") else "No")
                meta_col4.metric("Time (ms)",      f"{result.get('total_inference_ms', 0):.0f}")

                # Per-face details
                faces = result.get("faces", [])
                if faces:
                    with st.expander("Per-face details", expanded=False):
                        for f in faces:
                            st.json(f)


def _page_recognition() -> None:
    st.header("🔍 Face Recognition")
    st.caption(
        "Upload an image to detect all faces and match each one "
        "against the registered identity database."
    )

    consent = st.checkbox(
        "✅ I confirm I have **explicit consent** from all individuals in this image.",
        value=False,
        key="recog_consent",
    )

    st.divider()

    upload = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="recog_image",
    )
    if upload:
        st.image(upload, use_column_width=False, width=500)

    with st.expander("⚙️ Options", expanded=False):
        opt1, opt2 = st.columns(2)
        with opt1:
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.1, max_value=1.0, value=0.45, step=0.05,
                help="Minimum cosine similarity to declare a match.",
            )
            return_attrs = st.checkbox("Return Age/Gender", value=True)
        with opt2:
            top_k = st.number_input("Top-K Candidates", min_value=1, max_value=10, value=1)

    st.divider()
    run_disabled = not consent or not upload
    if st.button(
        "🔍 Recognize",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
    ):
        with st.spinner("Detecting and recognizing faces…"):
            img_bytes = upload.read()
            fields = {
                "top_k":               top_k,
                "similarity_threshold": threshold,
                "return_attributes":   return_attrs,
                "return_embeddings":   False,
                "consent":             True,
            }
            files = {"image": (upload.name, img_bytes, "image/jpeg")}
            result, error = _post_form("/api/v1/recognize", fields, files)

        if error:
            st.error(f"❌ Recognition failed: {error}")
        elif result:
            num_det   = result.get("num_faces_detected", 0)
            num_recog = result.get("num_faces_recognized", 0)

            st.success(
                f"✅ Detected **{num_det}** face(s) — "
                f"recognized **{num_recog}** known identit{'y' if num_recog == 1 else 'ies'}."
            )

            meta1, meta2, meta3 = st.columns(3)
            meta1.metric("Faces Detected",    num_det)
            meta2.metric("Faces Recognized",  num_recog)
            meta3.metric("Inference (ms)",    f"{result.get('inference_time_ms', 0):.0f}")

            faces = result.get("faces", [])
            if not faces:
                st.info("No faces returned in the response.")
                return

            for i, face in enumerate(faces):
                with st.expander(
                    f"Face #{face.get('face_index', i)} — "
                    f"{'✅ ' + face['match']['identity_name'] if face['match']['is_known'] else '❓ Unknown'}",
                    expanded=True,
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        bbox = face.get("bbox", {})
                        st.write("**Bounding Box**")
                        st.json(bbox)
                        attrs = face.get("attributes")
                        if attrs:
                            st.write("**Attributes**")
                            st.write(
                                f"Age: `{attrs.get('age', '—')}` | "
                                f"Gender: `{attrs.get('gender', '—')}` "
                                f"({attrs.get('gender_score', 0):.0%})"
                            )
                    with c2:
                        match = face.get("match", {})
                        if match.get("is_known"):
                            st.success(
                                f"**{match['identity_name']}**\n\n"
                                f"Similarity: `{match['similarity']:.3f}` "
                                f"(threshold: `{match['threshold_used']}`)"
                            )
                        else:
                            st.warning(
                                f"**Unknown**\n\n"
                                f"Best similarity: `{match.get('similarity', 0):.3f}` "
                                f"< threshold `{match.get('threshold_used', 0.45)}`"
                            )


def _page_identities() -> None:
    st.header("📋 Identity Database")
    st.caption("Browse, register, and manage face identities.")

    tab_browse, tab_register = st.tabs(["Browse Identities", "Register New Identity"])

    with tab_browse:
        st.subheader("Registered Identities")

        col_search, col_refresh = st.columns([4, 1])
        with col_search:
            name_filter = st.text_input("Filter by name", placeholder="e.g. Alice")
        with col_refresh:
            st.write("")
            refresh = st.button("🔄 Refresh", use_container_width=True)

        if refresh or "identities_data" not in st.session_state:
            with st.spinner("Loading identities…"):
                url = f"{_api_url()}/api/v1/identities"
                params: dict = {}
                if name_filter:
                    params["name_filter"] = name_filter
                try:
                    resp = requests.get(url, params=params, timeout=10)
                    if resp.status_code == 200:
                        st.session_state.identities_data = resp.json()
                    else:
                        st.error(f"Failed to load identities: HTTP {resp.status_code}")
                        st.session_state.identities_data = None
                except Exception as exc:
                    st.error(f"Cannot reach API: {exc}")
                    st.session_state.identities_data = None

        data = st.session_state.get("identities_data")
        if data is not None:
            total = data.get("total", 0)
            items = data.get("items", [])
            st.caption(f"**{total}** identit{'y' if total == 1 else 'ies'} registered.")

            if not items:
                st.info(
                    "No identities registered yet. "
                    "Use the **Register** tab to add people to the database."
                )
            else:
                for item in items:
                    with st.container():
                        c1, c2, c3 = st.columns([3, 2, 2])
                        c1.write(f"**{item.get('name', '—')}**")
                        c2.caption(f"Embeddings: {item.get('num_embeddings', '—')}")
                        c3.caption(f"ID: `{str(item.get('identity_id', ''))[:8]}…`")
                    st.divider()

    with tab_register:
        st.subheader("Register a New Face Identity")
        st.caption(
            "Upload a clear, front-facing photo and enter the person's name. "
            "The API will extract a 512-dim ArcFace embedding and save it."
        )

        reg_consent = st.checkbox(
            "✅ I confirm I have **explicit consent** from this person to register their face.",
            value=False,
            key="reg_consent",
        )

        reg_name  = st.text_input("Identity Name", placeholder="e.g. Alice")
        reg_image = st.file_uploader(
            "Face photo",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="reg_image",
        )
        if reg_image:
            st.image(reg_image, width=300)

        reg_disabled = not reg_consent or not reg_name.strip() or not reg_image
        if st.button(
            "📝 Register Identity",
            disabled=reg_disabled,
            use_container_width=True,
            type="primary",
        ):
            with st.spinner(f"Registering '{reg_name}'…"):
                img_bytes = reg_image.read()
                fields = {
                    "name":    reg_name.strip(),
                    "consent": True,
                }
                files = {"image": (reg_image.name, img_bytes, "image/jpeg")}
                result, error = _post_form("/api/v1/register", fields, files)

            if error:
                st.error(f"❌ Registration failed: {error}")
            elif result:
                st.success(
                    f"✅ **{result.get('identity_name')}** registered!\n\n"
                    f"Identity ID: `{result.get('identity_id', '')}`\n\n"
                    f"{result.get('message', '')}"
                )
                # Clear cached identities so the Browse tab refreshes
                if "identities_data" in st.session_state:
                    del st.session_state["identities_data"]


def _page_about() -> None:
    st.header("ℹ️ About")

    st.markdown(
        """
        ## AI Face Recognition & Face Swap

        This application is a research and educational demonstration of
        state-of-the-art face AI techniques.

        ### Pipeline

        | Stage | Technology | Purpose |
        |-------|-----------|---------|
        | Face Detection | **YOLOv8** (Ultralytics) | Real-time bounding-box detection |
        | Face Analysis | **InsightFace** buffalo_l | ArcFace 512-dim embeddings + landmarks |
        | Face Swap | **inswapper_128.onnx** | Identity injection between faces |
        | Face Enhancement | **GFPGAN v1.4** / CodeFormer | Post-swap artifact removal |
        | Backend API | **FastAPI** + Uvicorn | REST API |
        | Frontend UI | **Streamlit** | This interface |

        ### Architecture

        ```
        Source Image ──► YOLOv8 Detector ──► InsightFace ──► ArcFace Embedding
                                                                      │
        Target Image ──► YOLOv8 Detector ──► FaceBox ────► InSwapper ◄┘
                                                               │
                                                         GFPGAN/CodeFormer
                                                               │
                                                        Output Image + Watermark
        ```

        ---

        ### ⚠️ Ethics & Safety

        > **This tool must only be used with the explicit, informed consent
        > of all individuals depicted in the source and target images.**

        Safeguards built into this system:
        - **Consent gate**: every API call requires `consent=true`
        - **Watermark**: all AI-generated outputs are stamped "AI GENERATED"
        - **Request logging**: all swap operations are logged (metadata only)
        - **Local processing**: no data is sent to third parties

        **Do NOT use this technology to:**
        - Create non-consensual deepfakes
        - Impersonate individuals without consent
        - Generate misleading or harmful content

        The authors accept no responsibility for misuse of this software.

        ---

        ### Licenses

        | Model | License |
        |-------|---------|
        | YOLOv8 | AGPL-3.0 |
        | InsightFace | MIT |
        | inswapper_128.onnx | Non-commercial research only |
        | GFPGAN | Apache 2.0 |
        | CodeFormer | S-Lab License (non-commercial) |
        """
    )


def main() -> None:
    _init_state()

    page = _render_sidebar()

    if page == "🔄 Face Swap":
        _page_face_swap()
    elif page == "🔍 Face Recognition":
        _page_recognition()
    elif page == "📋 Identity Database":
        _page_identities()
    elif page == "ℹ️ About":
        _page_about()


if __name__ == "__main__":
    main()
