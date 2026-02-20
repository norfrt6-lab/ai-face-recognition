# Streamlit page for the face swap feature.
#
# Layout:
#   ┌─────────────────┬─────────────────┬─────────────────┐
#   │  Source Image   │  Target Image   │  Output Image   │
#   │  (donor face)   │  (scene)        │  (swapped)      │
#   └─────────────────┴─────────────────┴─────────────────┘
#   ┌─────────────────────────────────────────────────────┐
#   │  Swap Settings (sidebar)                            │
#   └─────────────────────────────────────────────────────┘

from __future__ import annotations

import base64
import io
import time
from typing import Optional

import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Face Swap — AI Face Recognition",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


import os as _os
API_BASE_URL_DEFAULT = _os.getenv("UI_API_BASE_URL", "http://localhost:8000")
SWAP_ENDPOINT        = "/api/v1/swap"
HEALTH_ENDPOINT      = "/api/v1/health"
MAX_IMAGE_SIZE_MB    = 50
SUPPORTED_FORMATS    = ["jpg", "jpeg", "png", "webp", "bmp"]


def _init_session_state() -> None:
    """Initialise all session state keys with defaults."""
    defaults = {
        "swap_result_image": None,
        "swap_result_info":  None,
        "api_url":           API_BASE_URL_DEFAULT,
        "api_healthy":       None,
        "last_health_check": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


def _get_api_url() -> str:
    return st.session_state.get("api_url", API_BASE_URL_DEFAULT).rstrip("/")


def _check_api_health(force: bool = False) -> bool:
    """
    Ping the health endpoint.  Caches the result for 30 seconds.

    Returns True if the API is healthy or degraded (i.e. reachable).
    """
    now = time.time()
    if not force and (now - st.session_state.last_health_check) < 30:
        return st.session_state.api_healthy or False

    try:
        resp = requests.get(
            f"{_get_api_url()}{HEALTH_ENDPOINT}",
            timeout=5,
        )
        healthy = resp.status_code == 200
        data    = resp.json() if healthy else {}
        st.session_state.api_healthy    = healthy
        st.session_state.last_health_check = now
        st.session_state.api_health_data   = data
        return healthy
    except Exception:
        st.session_state.api_healthy    = False
        st.session_state.last_health_check = now
        return False


def _call_swap_api(
    source_bytes:      bytes,
    target_bytes:      bytes,
    source_filename:   str,
    target_filename:   str,
    blend_mode:        str,
    blend_alpha:       float,
    mask_feather:      int,
    swap_all_faces:    bool,
    max_faces:         int,
    source_face_index: int,
    target_face_index: int,
    enhance:           bool,
    enhancer_backend:  str,
    enhancer_fidelity: float,
    watermark:         bool,
    consent:           bool = False,
) -> dict:
    """
    Call POST /api/v1/swap with return_base64=true.

    Returns a dict with keys:
        success (bool), image (PIL.Image or None), info (dict), error (str)
    """
    url = f"{_get_api_url()}{SWAP_ENDPOINT}"

    files = {
        "source_file": (source_filename, source_bytes, "image/jpeg"),
        "target_file": (target_filename, target_bytes, "image/jpeg"),
    }
    data = {
        "blend_mode":         blend_mode,
        "blend_alpha":        str(blend_alpha),
        "mask_feather":       str(mask_feather),
        "swap_all_faces":     str(swap_all_faces).lower(),
        "max_faces":          str(max_faces),
        "source_face_index":  str(source_face_index),
        "target_face_index":  str(target_face_index),
        "enhance":            str(enhance).lower(),
        "enhancer_backend":   enhancer_backend,
        "enhancer_fidelity":  str(enhancer_fidelity),
        "watermark":          str(watermark).lower(),
        "return_base64":      "true",
        "consent":            str(consent).lower(),
    }

    try:
        resp = requests.post(url, files=files, data=data, timeout=120)
        resp.raise_for_status()
        payload = resp.json()

        # Decode base64 image
        b64 = payload.get("output_base64")
        if b64:
            img_bytes = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            image = None

        return {
            "success": True,
            "image":   image,
            "info":    payload,
            "error":   None,
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "image":   None,
            "info":    {},
            "error":   (
                f"Cannot connect to API at {_get_api_url()}. "
                "Make sure the API server is running."
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "image":   None,
            "info":    {},
            "error":   "The API request timed out (>120s). Try a smaller image.",
        }
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("message", str(exc))
        except Exception:
            detail = str(exc)
        return {
            "success": False,
            "image":   None,
            "info":    {},
            "error":   f"API error {exc.response.status_code}: {detail}",
        }
    except Exception as exc:
        return {
            "success": False,
            "image":   None,
            "info":    {},
            "error":   f"Unexpected error: {exc}",
        }


def _render_sidebar() -> dict:
    """
    Render the sidebar and return the current settings dict.
    """
    with st.sidebar:
        st.title("⚙️ Swap Settings")
        st.divider()

        st.subheader("🔗 API Connection")
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_url,
            help="URL of the FastAPI backend (e.g. http://localhost:8000).",
        )
        st.session_state.api_url = api_url

        col_ping, col_status = st.columns([1, 2])
        with col_ping:
            if st.button("Ping", use_container_width=True):
                _check_api_health(force=True)

        with col_status:
            healthy = st.session_state.get("api_healthy")
            if healthy is True:
                st.success("● Online")
            elif healthy is False:
                st.error("● Offline")
            else:
                st.info("● Unknown")

        # Show component status if health data available
        health_data = st.session_state.get("api_health_data", {})
        components  = health_data.get("components", {})
        if components:
            with st.expander("Component Status", expanded=False):
                for name, comp in components.items():
                    icon = "✅" if comp.get("status") == "ok" else "⚠️"
                    loaded = "loaded" if comp.get("loaded") else "not loaded"
                    st.text(f"{icon} {name}: {loaded}")

        st.divider()

        st.subheader("🎨 Blending")
        blend_mode = st.selectbox(
            "Blend Mode",
            options=["poisson", "alpha", "masked_alpha"],
            index=0,
            help=(
                "**Poisson** — seamless boundary (recommended).\n"
                "**Alpha** — fast ellipse blend.\n"
                "**Masked Alpha** — skin-aware alpha blend."
            ),
        )
        blend_alpha = st.slider(
            "Blend Strength",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="1.0 = fully swapped, 0.0 = original unchanged.",
        )
        mask_feather = st.slider(
            "Mask Feather (px)",
            min_value=0,
            max_value=60,
            value=20,
            step=2,
            help="Gaussian blur radius applied to blend mask edges.",
        )

        st.divider()

        st.subheader("👤 Face Selection")
        swap_all_faces = st.toggle(
            "Swap All Faces in Target",
            value=False,
            help="When ON, every detected face in the target is replaced.",
        )
        col_src, col_tgt = st.columns(2)
        with col_src:
            source_face_index = st.number_input(
                "Source Face #",
                min_value=0,
                max_value=49,
                value=0,
                step=1,
                help="Index of the donor face (0 = first detected).",
            )
        with col_tgt:
            target_face_index = st.number_input(
                "Target Face #",
                min_value=0,
                max_value=49,
                value=0,
                step=1,
                disabled=swap_all_faces,
                help="Index of the face to replace (0 = first detected).",
            )
        if swap_all_faces:
            max_faces = st.slider(
                "Max Faces to Swap",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
            )
        else:
            max_faces = 10

        st.divider()

        st.subheader("✨ Enhancement")
        enhance = st.toggle(
            "Enable Face Enhancement",
            value=False,
            help="Apply GFPGAN / CodeFormer after swap to restore face quality.",
        )
        if enhance:
            enhancer_backend = st.selectbox(
                "Backend",
                options=["gfpgan", "codeformer"],
                index=0,
            )
            enhancer_fidelity = st.slider(
                "Fidelity Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help=(
                    "CodeFormer only: 0.0 = max quality, "
                    "1.0 = max identity preservation."
                ),
            )
        else:
            enhancer_backend  = "gfpgan"
            enhancer_fidelity = 0.5

        st.divider()

        st.subheader("📤 Output")
        watermark = st.toggle(
            "Watermark Output",
            value=True,
            help='Embed "AI GENERATED" text on the output image.',
        )

        st.divider()

        st.subheader("⚠️ Ethics")
        st.warning(
            "You must confirm that you have "
            "**explicit consent** from all individuals depicted.\n\n"
            "Do NOT create non-consensual deepfakes."
        )
        consent = st.checkbox(
            "I have explicit consent from all individuals depicted",
            value=False,
            key="swap_consent",
        )

    return {
        "blend_mode":        blend_mode,
        "blend_alpha":       blend_alpha,
        "mask_feather":      mask_feather,
        "swap_all_faces":    swap_all_faces,
        "max_faces":         max_faces,
        "source_face_index": int(source_face_index),
        "target_face_index": int(target_face_index),
        "enhance":           enhance,
        "enhancer_backend":  enhancer_backend,
        "enhancer_fidelity": enhancer_fidelity,
        "watermark":         watermark,
        "consent":           consent,
    }


def _image_uploader(label: str, key: str, help_text: str = "") -> Optional[bytes]:
    """
    Render a file uploader and return the raw bytes if a file was uploaded.
    """
    uploaded = st.file_uploader(
        label,
        type=SUPPORTED_FORMATS,
        key=key,
        help=help_text,
        label_visibility="collapsed",
    )
    if uploaded is not None:
        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            st.error(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {MAX_IMAGE_SIZE_MB} MB."
            )
            return None
        return uploaded.getvalue()
    return None


def _display_image(
    image_bytes: Optional[bytes],
    caption: str,
    placeholder_text: str,
) -> None:
    """Render an image or a dashed placeholder box."""
    if image_bytes is not None:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            st.image(image_bytes, caption=f"{caption} ({w}×{h})", use_column_width=True)
        except Exception:
            st.error("Could not display image.")
    else:
        st.markdown(
            f"""
            <div style="
                border: 2px dashed #555;
                border-radius: 8px;
                padding: 60px 20px;
                text-align: center;
                color: #888;
                font-size: 14px;
                min-height: 220px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                {placeholder_text}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_result_info(info: dict) -> None:
    """Render timing + per-face stats from the API response."""
    if not info:
        return

    total_ms   = info.get("total_inference_ms", 0)
    swapped    = info.get("num_faces_swapped", 0)
    failed     = info.get("num_faces_failed", 0)
    enhanced   = info.get("enhanced", False)
    watermarked = info.get("watermarked", False)
    blend      = info.get("blend_mode", "—")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Total Time", f"{total_ms:.0f} ms")
    col2.metric("✅ Faces Swapped", swapped)
    col3.metric("❌ Faces Failed", failed)
    col4.metric("✨ Enhanced", "Yes" if enhanced else "No")

    faces = info.get("faces", [])
    if faces:
        with st.expander(f"Per-face Details ({len(faces)} face{'s' if len(faces) != 1 else ''})", expanded=False):
            for face in faces:
                fi     = face.get("face_index", "?")
                ok     = face.get("success", False)
                status = face.get("status", "—")
                timing = face.get("timing") or {}
                err    = face.get("error")

                icon = "✅" if ok else "❌"
                st.write(
                    f"{icon} **Face #{fi}** — status: `{status}` | "
                    f"align: {timing.get('align_ms', 0):.1f}ms | "
                    f"inference: {timing.get('inference_ms', 0):.1f}ms | "
                    f"blend: {timing.get('blend_ms', 0):.1f}ms"
                )
                if err:
                    st.error(f"  Error: {err}")


def main() -> None:
    st.title("🔄 Face Swap")
    st.markdown(
        "Upload a **source** image (the donor face) and a **target** image "
        "(the scene to modify). The API will inject the source identity into "
        "the target face(s)."
    )
    st.divider()

    settings = _render_sidebar()

    col_src, col_tgt, col_out = st.columns(3, gap="medium")

    with col_src:
        st.subheader("📸 Source Image")
        st.caption("The face identity to inject (donor).")
        source_bytes = _image_uploader(
            "Source Image",
            key="source_upload",
            help_text="Upload the face you want to transfer.",
        )
        _display_image(source_bytes, "Source", "Drop source image here")

    with col_tgt:
        st.subheader("🎯 Target Image")
        st.caption("The scene where the face will be replaced.")
        target_bytes = _image_uploader(
            "Target Image",
            key="target_upload",
            help_text="Upload the scene / target image.",
        )
        _display_image(target_bytes, "Target", "Drop target image here")

    with col_out:
        st.subheader("✅ Output Image")
        st.caption("Swapped result will appear here.")

        result_image: Optional[Image.Image] = st.session_state.swap_result_image
        result_info:  Optional[dict]        = st.session_state.swap_result_info

        if result_image is not None:
            st.image(result_image, caption="Swapped Output", use_column_width=True)

            # Download button
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download PNG",
                data=buf.getvalue(),
                file_name="face_swap_output.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            _display_image(None, "Output", "Output will appear here after swap")

    st.divider()

    run_col, clear_col = st.columns([3, 1])

    with run_col:
        run_disabled = (source_bytes is None or target_bytes is None or not settings.get("consent", False))
        run_clicked  = st.button(
            "🔄 Run Face Swap",
            disabled=run_disabled,
            use_container_width=True,
            type="primary",
            help=(
                "Upload both images first."
                if run_disabled
                else "Run the face swap pipeline."
            ),
        )

    with clear_col:
        if st.button("🗑️ Clear Result", use_container_width=True):
            st.session_state.swap_result_image = None
            st.session_state.swap_result_info  = None
            st.rerun()

    if run_disabled and (source_bytes is None or target_bytes is None):
        missing = []
        if source_bytes is None:
            missing.append("source")
        if target_bytes is None:
            missing.append("target")
        st.info(
            f"Please upload the {' and '.join(missing)} image{'s' if len(missing) > 1 else ''} "
            "to enable the swap button."
        )

    if run_clicked and source_bytes and target_bytes:
        with st.spinner("Running face swap pipeline…"):
            result = _call_swap_api(
                source_bytes=source_bytes,
                target_bytes=target_bytes,
                source_filename="source.jpg",
                target_filename="target.jpg",
                **settings,
            )

        if result["success"] and result["image"] is not None:
            st.session_state.swap_result_image = result["image"]
            st.session_state.swap_result_info  = result["info"]
            st.success(
                f"✅ Swap complete! "
                f"{result['info'].get('num_faces_swapped', 0)} face(s) swapped in "
                f"{result['info'].get('total_inference_ms', 0):.0f} ms."
            )
            st.rerun()
        else:
            st.error(f"❌ Swap failed: {result['error']}")

    if result_info:
        st.divider()
        st.subheader("📊 Result Details")
        _render_result_info(result_info)

    with st.expander("💡 Tips for best results", expanded=False):
        st.markdown(
            """
            - **Source image**: Use a clear, front-facing photo with good lighting.
              A single face works best. Higher resolution = better embedding quality.
            - **Target image**: Can contain multiple faces. Use *Swap All Faces* to
              replace all of them, or set *Target Face #* to pick a specific one.
            - **Blend Mode**:
              - *Poisson* produces the most seamless, photorealistic result.
              - *Alpha* is faster but may show hard edges around the face boundary.
            - **Mask Feather**: Increase this to soften the blend boundary. Useful when
              there are visible seams after the swap.
            - **Enhancement**: Enable GFPGAN or CodeFormer to remove compression
              artifacts from the swapped face. Adds ~1–3s processing time.
            - **Models must be downloaded** before the API can process images.
              Run `python utils/download_models.py --minimum` to get started.
            """
        )


if __name__ == "__main__":
    main()
else:
    # When loaded as a Streamlit page via multipage app
    main()
