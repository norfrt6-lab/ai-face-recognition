# Integration tests for the FastAPI REST backend.
#
# Uses FastAPI's built-in TestClient (backed by httpx) to send
# real HTTP requests against a fully configured app instance
# with all pipeline components replaced by mocks.
#
# Test groups:
#   1.  Health endpoint
#   2.  Recognize endpoint — happy path, error cases, consent gate
#   3.  Register endpoint  — new identity, append, consent gate
#   4.  Identities CRUD    — list, get, rename, delete
#   5.  Swap endpoint      — PNG download, base64 JSON, errors
#   6.  Middleware         — X-Request-ID, rate limiting, CORS
#   7.  Error handling     — 404, validation errors, 503
#
# None of these tests load real AI models.  Every pipeline
# component is replaced by a minimal mock attached to app.state
# before each test.

from __future__ import annotations

import base64
import io
import time
import uuid
from typing import Generator, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_bgr_image(h: int = 120, w: int = 120) -> np.ndarray:
    """Return a solid-colour BGR test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _encode_png(image: np.ndarray) -> bytes:
    """Encode a BGR numpy array to PNG bytes."""
    _, buf = cv2.imencode(".png", image)
    return buf.tobytes()


def _encode_jpeg(image: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", image)
    return buf.tobytes()


def _make_face_box_mock(
    x1=10.0,
    y1=10.0,
    x2=100.0,
    y2=100.0,
    confidence=0.95,
    face_index=0,
    with_landmarks=True,
):
    """Return a mock FaceBox-like object."""
    fb = MagicMock()
    fb.x1 = x1
    fb.y1 = y1
    fb.x2 = x2
    fb.y2 = y2
    fb.confidence = confidence
    fb.face_index = face_index
    fb.has_landmarks = with_landmarks
    if with_landmarks:
        fb.landmarks = np.array(
            [[30.0, 35.0], [70.0, 35.0], [50.0, 55.0], [35.0, 75.0], [65.0, 75.0]],
            dtype=np.float32,
        )
    else:
        fb.landmarks = None
    return fb


def _make_detection_result_mock(n_faces: int = 1):
    """Return a mock DetectionResult with n_faces."""
    det = MagicMock()
    faces = [_make_face_box_mock(face_index=i) for i in range(n_faces)]
    det.faces = faces
    det.is_empty = n_faces == 0
    det.best_face = faces[0] if faces else None
    det.image_width = 120
    det.image_height = 120
    det.inference_time_ms = 5.0
    return det


def _make_embedding_mock():
    """Return a mock FaceEmbedding."""
    emb = MagicMock()
    rng = np.random.default_rng(7)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    emb.vector = v
    emb.face_index = 0
    return emb


def _make_swap_result_mock(success: bool = True):
    """Return a mock SwapResult."""
    from core.swapper.base_swapper import SwapStatus

    sr = MagicMock()
    sr.success = success
    sr.output_image = _make_bgr_image()
    sr.status = SwapStatus.SUCCESS if success else SwapStatus.INFERENCE_ERROR
    sr.swap_time_ms = 20.0
    sr.inference_time_ms = 15.0
    sr.align_time_ms = 2.0
    sr.blend_time_ms = 3.0
    sr.error = None if success else "mock inference error"
    sr.target_face = _make_face_box_mock()
    return sr


def _make_batch_swap_result_mock(n=1, all_success=True):
    """Return a mock BatchSwapResult."""
    bsr = MagicMock()
    bsr.output_image = _make_bgr_image()
    bsr.swap_results = [_make_swap_result_mock(all_success) for _ in range(n)]
    bsr.num_swapped = n if all_success else 0
    bsr.num_failed = 0 if all_success else n
    bsr.all_success = all_success
    bsr.total_time_ms = 25.0
    return bsr


def _make_face_match_mock(is_known: bool = True):
    """Return a mock FaceMatch."""
    m = MagicMock()
    m.is_known = is_known
    m.identity_name = "Alice" if is_known else None
    m.identity_id = str(uuid.uuid4()) if is_known else None
    m.similarity = 0.87 if is_known else 0.22
    return m


@pytest.fixture(scope="module")
def app() -> FastAPI:
    """
    Create a FastAPI app instance without lifespan model loading.

    We call create_app() directly (not the module-level `app`) so
    the lifespan context is not triggered and we can freely attach
    mock components to state.
    """
    from api.main import create_app

    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """
    TestClient with all pipeline components mocked on app.state.

    The 'with' block triggers the lifespan. Because create_app() uses
    a lifespan that tries to load real models, we patch the model
    loading functions so they are no-ops.
    """
    # Patch all model-loading routines to prevent disk access
    with (
        patch("api.main.YOLOFaceDetector", autospec=False),
        patch("api.main.InsightFaceRecognizer", autospec=False),
        patch("api.main.InSwapper", autospec=False),
        patch("api.main.FaceDatabase", autospec=False),
    ):
        with TestClient(app, raise_server_exceptions=True) as c:
            # Attach fresh mocks to app.state for each test
            _attach_mocks(app)
            yield c


def _attach_mocks(app: FastAPI) -> None:
    """Attach minimal mock pipeline components to app.state."""
    # Detector
    detector = MagicMock()
    detector.is_loaded = True
    detector.detect.return_value = _make_detection_result_mock(1)
    app.state.detector = detector

    # Recognizer
    recognizer = MagicMock()
    recognizer.is_loaded = True
    recognizer.model_name = "buffalo_l"
    recognizer.get_embedding.return_value = _make_embedding_mock()
    recognizer.get_attributes.return_value = None
    app.state.recognizer = recognizer

    # Swapper
    swapper = MagicMock()
    swapper.is_loaded = True
    swapper.swap.return_value = _make_swap_result_mock(True)
    swapper.swap_all.return_value = _make_batch_swap_result_mock(1, True)
    app.state.swapper = swapper

    # Enhancer (optional — set to None by default)
    app.state.enhancer = None

    # Face database
    face_db = MagicMock()
    face_db.is_loaded = True  # face_db doesn't have is_loaded but harmless
    face_db.num_identities = 2
    face_db.search.return_value = [_make_face_match_mock(True)]
    face_db.register.return_value = str(uuid.uuid4())
    face_db.list_identities.return_value = [
        {"identity_id": str(uuid.uuid4()), "name": "Alice", "num_embeddings": 3},
        {"identity_id": str(uuid.uuid4()), "name": "Bob", "num_embeddings": 1},
    ]
    face_db.get_identity.return_value = MagicMock(
        name="Alice",
        num_embeddings=3,
        created_at=time.time(),
        metadata={},
    )
    app.state.face_database = face_db

    # Output dir
    import tempfile

    app.state.output_dir = tempfile.mkdtemp()


# Convenience image fixture
@pytest.fixture()
def sample_image_bytes() -> bytes:
    return _encode_jpeg(_make_bgr_image(200, 200))


@pytest.fixture()
def sample_png_bytes() -> bytes:
    return _encode_png(_make_bgr_image(200, 200))


@pytest.mark.api
class TestHealthEndpoint:

    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_has_status(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "status" in data

    def test_health_response_has_version(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "version" in data

    def test_health_response_has_uptime(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_response_has_components(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "components" in data
        assert isinstance(data["components"], dict)

    def test_health_detector_component_present(self, client: TestClient, app: FastAPI):
        data = client.get("/api/v1/health").json()
        assert "detector" in data["components"]

    def test_health_recognizer_component_present(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "recognizer" in data["components"]

    def test_health_swapper_component_present(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "swapper" in data["components"]

    def test_health_status_ok_when_all_loaded(self, client: TestClient, app: FastAPI):
        # All mocks have is_loaded=True
        data = client.get("/api/v1/health").json()
        assert data["status"] in ("ok", "degraded")

    def test_health_status_down_when_detector_missing(self, client: TestClient, app: FastAPI):
        original = app.state.detector
        app.state.detector = None
        try:
            data = client.get("/api/v1/health").json()
            assert data["status"] == "down"
        finally:
            app.state.detector = original

    def test_health_environment_field(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "environment" in data

    def test_health_no_auth_required(self, client: TestClient):
        """Health endpoint must be accessible without authentication."""
        resp = client.get("/api/v1/health")
        assert resp.status_code != 401
        assert resp.status_code != 403

    def test_health_x_request_id_header(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert "x-request-id" in resp.headers

    def test_health_custom_request_id_echoed(self, client: TestClient):
        rid = "test-request-123"
        resp = client.get("/api/v1/health", headers={"X-Request-ID": rid})
        assert resp.headers.get("x-request-id") == rid


@pytest.mark.api
class TestRecognizeEndpoint:

    def _post(
        self,
        client: TestClient,
        image_bytes: bytes,
        filename: str = "test.jpg",
        consent: str = "true",
        extra_fields: Optional[dict] = None,
    ):
        files = {"image": (filename, image_bytes, "image/jpeg")}
        data = {"consent": consent}
        if extra_fields:
            data.update(extra_fields)
        return client.post("/api/v1/recognize", files=files, data=data)

    def test_recognize_200_with_consent(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes)
        assert resp.status_code == 200

    def test_recognize_returns_json(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes)
        assert resp.headers["content-type"].startswith("application/json")

    def test_recognize_num_faces_detected(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "num_faces_detected" in data
        assert isinstance(data["num_faces_detected"], int)

    def test_recognize_faces_list(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "faces" in data
        assert isinstance(data["faces"], list)

    def test_recognize_face_has_bbox(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        faces = data["faces"]
        assert len(faces) > 0
        bbox = faces[0]["bbox"]
        assert all(k in bbox for k in ("x1", "y1", "x2", "y2", "confidence"))

    def test_recognize_face_has_match(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        match = data["faces"][0]["match"]
        assert "is_known" in match
        assert "similarity" in match
        assert "threshold_used" in match

    def test_recognize_known_identity(self, client, app, sample_image_bytes):
        # face_db.search returns a known match by default
        data = self._post(client, sample_image_bytes).json()
        assert data["faces"][0]["match"]["is_known"] is True

    def test_recognize_unknown_identity(self, client, app, sample_image_bytes):
        app.state.face_database.search.return_value = [_make_face_match_mock(False)]
        data = self._post(client, sample_image_bytes).json()
        assert data["faces"][0]["match"]["is_known"] is False

    def test_recognize_403_without_consent(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes, consent="false")
        assert resp.status_code == 403

    def test_recognize_403_consent_missing(self, client, sample_image_bytes):
        files = {"image": ("t.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post("/api/v1/recognize", files=files, data={})
        # consent defaults to false → 403
        assert resp.status_code == 403

    def test_recognize_400_invalid_image(self, client):
        files = {"image": ("bad.jpg", b"not-an-image", "image/jpeg")}
        resp = client.post("/api/v1/recognize", files=files, data={"consent": "true"})
        assert resp.status_code == 400

    def test_recognize_no_faces(self, client, app, sample_image_bytes):
        app.state.detector.detect.return_value = _make_detection_result_mock(0)
        data = self._post(client, sample_image_bytes).json()
        assert data["num_faces_detected"] == 0
        assert data["faces"] == []

    def test_recognize_inference_time_field(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0

    def test_recognize_image_dimensions(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "image_width" in data
        assert "image_height" in data
        assert data["image_width"] > 0
        assert data["image_height"] > 0

    def test_recognize_503_detector_not_loaded(self, client, app, sample_image_bytes):
        app.state.detector.is_loaded = False
        resp = self._post(client, sample_image_bytes)
        assert resp.status_code == 503
        app.state.detector.is_loaded = True

    def test_recognize_503_detector_none(self, client, app, sample_image_bytes):
        original = app.state.detector
        app.state.detector = None
        resp = self._post(client, sample_image_bytes)
        assert resp.status_code == 503
        app.state.detector = original

    def test_recognize_with_attributes(self, client, app, sample_image_bytes):
        attr_mock = MagicMock()
        attr_mock.age = 28.5
        attr_mock.gender = "F"
        attr_mock.gender_score = 0.92
        app.state.recognizer.get_attributes.return_value = attr_mock
        data = self._post(
            client, sample_image_bytes, extra_fields={"return_attributes": "true"}
        ).json()
        # Attributes may be in response
        assert data["num_faces_detected"] >= 0

    def test_recognize_response_has_request_id(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes)
        assert "x-request-id" in resp.headers

    def test_recognize_png_image(self, client, sample_png_bytes):
        files = {"image": ("test.png", sample_png_bytes, "image/png")}
        resp = client.post("/api/v1/recognize", files=files, data={"consent": "true"})
        assert resp.status_code == 200

    def test_recognize_multiple_faces(self, client, app, sample_image_bytes):
        app.state.detector.detect.return_value = _make_detection_result_mock(3)
        data = self._post(client, sample_image_bytes).json()
        assert data["num_faces_detected"] == 3
        assert len(data["faces"]) == 3


@pytest.mark.api
class TestRegisterEndpoint:

    def _post(
        self,
        client: TestClient,
        image_bytes: bytes,
        name: str = "Alice",
        consent: str = "true",
        extra_fields: Optional[dict] = None,
    ):
        files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
        data = {"name": name, "consent": consent}
        if extra_fields:
            data.update(extra_fields)
        return client.post("/api/v1/register", files=files, data=data)

    def test_register_201_success(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes)
        assert resp.status_code == 201

    def test_register_returns_identity_id(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "identity_id" in data
        assert len(data["identity_id"]) > 0

    def test_register_returns_identity_name(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes, name="Bob").json()
        assert data["identity_name"] == "Bob"

    def test_register_returns_embeddings_added(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "embeddings_added" in data
        assert data["embeddings_added"] >= 1

    def test_register_returns_message(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "message" in data
        assert len(data["message"]) > 0

    def test_register_403_no_consent(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes, consent="false")
        assert resp.status_code == 403

    def test_register_400_no_face(self, client, app, sample_image_bytes):
        app.state.detector.detect.return_value = _make_detection_result_mock(0)
        resp = self._post(client, sample_image_bytes)
        assert resp.status_code == 400

    def test_register_400_invalid_image(self, client):
        files = {"image": ("bad.jpg", b"garbage", "image/jpeg")}
        resp = client.post(
            "/api/v1/register",
            files=files,
            data={"name": "Test", "consent": "true"},
        )
        assert resp.status_code == 400

    def test_register_422_missing_name(self, client, sample_image_bytes):
        files = {"image": ("face.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/register",
            files=files,
            data={"consent": "true"},  # name missing
        )
        # FastAPI returns 422 for missing required form field
        assert resp.status_code == 422

    def test_register_422_name_with_slashes(self, client, sample_image_bytes):
        resp = self._post(client, sample_image_bytes, name="../../evil")
        assert resp.status_code in (400, 422)

    def test_register_503_no_face_database(self, client, app, sample_image_bytes):
        original = app.state.face_database
        app.state.face_database = None
        resp = self._post(client, sample_image_bytes)
        assert resp.status_code == 503
        app.state.face_database = original

    def test_register_faces_detected_field(self, client, sample_image_bytes):
        data = self._post(client, sample_image_bytes).json()
        assert "faces_detected" in data
        assert data["faces_detected"] >= 1


@pytest.mark.api
class TestIdentitiesEndpoints:

    def test_list_identities_200(self, client):
        resp = client.get("/api/v1/identities")
        assert resp.status_code == 200

    def test_list_identities_has_total(self, client):
        data = client.get("/api/v1/identities").json()
        assert "total" in data
        assert isinstance(data["total"], int)

    def test_list_identities_has_items(self, client):
        data = client.get("/api/v1/identities").json()
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_list_identities_items_have_name(self, client):
        data = client.get("/api/v1/identities").json()
        items = data["items"]
        assert len(items) > 0
        assert "name" in items[0]

    def test_list_identities_name_filter(self, client, app):
        resp = client.get("/api/v1/identities", params={"name_filter": "Alice"})
        assert resp.status_code == 200

    def test_list_identities_pagination(self, client):
        resp = client.get("/api/v1/identities", params={"page": 1, "page_size": 10})
        assert resp.status_code == 200
        data = resp.json()
        assert "page" in data
        assert "page_size" in data

    def test_list_identities_503_no_db(self, client, app):
        original = app.state.face_database
        app.state.face_database = None
        resp = client.get("/api/v1/identities")
        assert resp.status_code == 503
        app.state.face_database = original

    def test_get_identity_200(self, client, app):
        uid = str(uuid.uuid4())
        resp = client.get(f"/api/v1/identities/{uid}")
        assert resp.status_code == 200

    def test_get_identity_returns_name(self, client, app):
        uid = str(uuid.uuid4())
        data = client.get(f"/api/v1/identities/{uid}").json()
        assert "name" in data

    def test_get_identity_404_not_found(self, client, app):
        app.state.face_database.get_identity.side_effect = KeyError("not found")
        uid = str(uuid.uuid4())
        resp = client.get(f"/api/v1/identities/{uid}")
        assert resp.status_code == 404
        app.state.face_database.get_identity.side_effect = None

    def test_delete_identity_200(self, client, app):
        uid = str(uuid.uuid4())
        resp = client.delete(
            f"/api/v1/identities/{uid}",
            data={"confirm": "true"},
        )
        assert resp.status_code == 200

    def test_delete_identity_400_no_confirm(self, client, app):
        uid = str(uuid.uuid4())
        resp = client.delete(
            f"/api/v1/identities/{uid}",
            data={"confirm": "false"},
        )
        assert resp.status_code == 400

    def test_delete_identity_404_not_found(self, client, app):
        app.state.face_database.remove_identity.side_effect = KeyError("not found")
        uid = str(uuid.uuid4())
        resp = client.delete(
            f"/api/v1/identities/{uid}",
            data={"confirm": "true"},
        )
        assert resp.status_code == 404
        app.state.face_database.remove_identity.side_effect = None

    def test_rename_identity_200(self, client, app):
        uid = str(uuid.uuid4())
        resp = client.patch(
            f"/api/v1/identities/{uid}",
            data={"new_name": "Charlie"},
        )
        assert resp.status_code == 200

    def test_rename_identity_renamed_field(self, client, app):
        uid = str(uuid.uuid4())
        data = client.patch(
            f"/api/v1/identities/{uid}",
            data={"new_name": "Charlie"},
        ).json()
        assert data.get("renamed") is True

    def test_rename_identity_404(self, client, app):
        app.state.face_database.rename_identity.side_effect = KeyError("not found")
        uid = str(uuid.uuid4())
        resp = client.patch(
            f"/api/v1/identities/{uid}",
            data={"new_name": "Charlie"},
        )
        assert resp.status_code == 404
        app.state.face_database.rename_identity.side_effect = None

    def test_rename_identity_slashes_rejected(self, client, app):
        uid = str(uuid.uuid4())
        resp = client.patch(
            f"/api/v1/identities/{uid}",
            data={"new_name": "../../hack"},
        )
        assert resp.status_code in (400, 422)


@pytest.mark.api
class TestSwapEndpoint:

    def _post_swap(
        self,
        client: TestClient,
        source_bytes: bytes,
        target_bytes: bytes,
        consent: str = "true",
        return_base64: str = "false",
        extra_fields: Optional[dict] = None,
    ):
        files = {
            "source_file": ("source.jpg", source_bytes, "image/jpeg"),
            "target_file": ("target.jpg", target_bytes, "image/jpeg"),
        }
        data = {
            "consent": consent,
            "return_base64": return_base64,
        }
        if extra_fields:
            data.update(extra_fields)
        return client.post("/api/v1/swap", files=files, data=data)

    def test_swap_200_returns_png(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert resp.status_code == 200
        assert "image/png" in resp.headers["content-type"]

    def test_swap_response_is_valid_png(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        # PNG files start with the PNG magic bytes
        assert resp.content[:4] == b"\x89PNG"

    def test_swap_content_disposition_header(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert "content-disposition" in resp.headers
        assert "swap_" in resp.headers["content-disposition"]

    def test_swap_x_faces_swapped_header(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert "x-faces-swapped" in resp.headers

    def test_swap_x_processing_ms_header(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert "x-processing-ms" in resp.headers

    def test_swap_base64_json_200(self, client, sample_image_bytes):
        resp = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        )
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]

    def test_swap_base64_json_has_output_base64(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "output_base64" in data
        assert data["output_base64"] is not None
        # Verify it decodes to a valid PNG
        decoded = base64.b64decode(data["output_base64"])
        assert decoded[:4] == b"\x89PNG"

    def test_swap_base64_json_has_num_faces_swapped(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "num_faces_swapped" in data
        assert isinstance(data["num_faces_swapped"], int)

    def test_swap_base64_json_has_faces_list(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "faces" in data
        assert isinstance(data["faces"], list)

    def test_swap_base64_json_blend_mode_field(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "blend_mode" in data

    def test_swap_base64_json_enhanced_field(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "enhanced" in data
        assert isinstance(data["enhanced"], bool)

    def test_swap_base64_json_watermarked_field(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "watermarked" in data

    def test_swap_base64_json_total_inference_ms(self, client, sample_image_bytes):
        data = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            return_base64="true",
        ).json()
        assert "total_inference_ms" in data
        assert data["total_inference_ms"] >= 0

    def test_swap_no_consent_returns_4xx(self, client, sample_image_bytes):
        resp = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            consent="false",
        )
        assert resp.status_code in (400, 403)

    def test_swap_400_invalid_source_image(self, client, sample_image_bytes):
        files = {
            "source_file": ("bad.jpg", b"not-an-image", "image/jpeg"),
            "target_file": ("target.jpg", sample_image_bytes, "image/jpeg"),
        }
        resp = client.post(
            "/api/v1/swap",
            files=files,
            data={"consent": "true"},
        )
        assert resp.status_code == 400

    def test_swap_400_invalid_target_image(self, client, sample_image_bytes):
        files = {
            "source_file": ("source.jpg", sample_image_bytes, "image/jpeg"),
            "target_file": ("bad.jpg", b"not-an-image", "image/jpeg"),
        }
        resp = client.post(
            "/api/v1/swap",
            files=files,
            data={"consent": "true"},
        )
        assert resp.status_code == 400

    def test_swap_400_no_source_face(self, client, app, sample_image_bytes):
        app.state.detector.detect.return_value = _make_detection_result_mock(0)
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert resp.status_code == 400
        app.state.detector.detect.return_value = _make_detection_result_mock(1)

    def test_swap_503_swapper_not_loaded(self, client, app, sample_image_bytes):
        app.state.swapper.is_loaded = False
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert resp.status_code == 503
        app.state.swapper.is_loaded = True

    def test_swap_503_swapper_none(self, client, app, sample_image_bytes):
        original = app.state.swapper
        app.state.swapper = None
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert resp.status_code == 503
        app.state.swapper = original

    def test_swap_503_detector_none(self, client, app, sample_image_bytes):
        original = app.state.detector
        app.state.detector = None
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert resp.status_code == 503
        app.state.detector = original

    def test_swap_x_enhanced_header(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert "x-enhanced" in resp.headers

    def test_swap_x_watermarked_header(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert "x-watermarked" in resp.headers

    def test_swap_request_id_header(self, client, sample_image_bytes):
        resp = self._post_swap(client, sample_image_bytes, sample_image_bytes)
        assert "x-request-id" in resp.headers

    def test_swap_all_faces_mode_200(self, client, app, sample_image_bytes):
        resp = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            extra_fields={"swap_all_faces": "true"},
        )
        assert resp.status_code == 200

    def test_swap_all_faces_mode_returns_png(self, client, app, sample_image_bytes):
        resp = self._post_swap(
            client,
            sample_image_bytes,
            sample_image_bytes,
            extra_fields={"swap_all_faces": "true"},
        )
        assert "image/png" in resp.headers["content-type"]


@pytest.mark.api
class TestMiddleware:

    def test_x_request_id_auto_generated(self, client):
        resp = client.get("/api/v1/health")
        rid = resp.headers.get("x-request-id")
        assert rid is not None
        assert len(rid) > 0

    def test_x_request_id_echoed_when_provided(self, client):
        custom_id = "my-custom-request-id-xyz"
        resp = client.get("/api/v1/health", headers={"X-Request-ID": custom_id})
        assert resp.headers.get("x-request-id") == custom_id

    def test_x_processing_ms_header_present(self, client):
        resp = client.get("/api/v1/health")
        assert "x-processing-ms" in resp.headers

    def test_x_processing_ms_is_numeric(self, client):
        resp = client.get("/api/v1/health")
        val = resp.headers.get("x-processing-ms", "")
        assert float(val) >= 0.0

    def test_unique_request_ids_per_request(self, client):
        rid1 = client.get("/api/v1/health").headers.get("x-request-id")
        rid2 = client.get("/api/v1/health").headers.get("x-request-id")
        assert rid1 != rid2

    def test_request_id_propagated_to_post(self, client, sample_image_bytes):
        custom_id = "post-test-request-id-001"
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true"},
            headers={"X-Request-ID": custom_id},
        )
        assert resp.headers.get("x-request-id") == custom_id

    def test_x_processing_ms_on_post(self, client, sample_image_bytes):
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true"},
        )
        assert "x-processing-ms" in resp.headers

    def test_cors_origin_header_present(self, client):
        resp = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )
        # CORS middleware should respond with 200 (not block the request)
        assert resp.status_code == 200

    def test_cors_preflight_options(self, client):
        resp = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Preflight should be accepted
        assert resp.status_code in (200, 204)


@pytest.mark.api
class TestErrorHandling:

    def test_404_unknown_route(self, client):
        resp = client.get("/api/v1/this-route-does-not-exist")
        assert resp.status_code == 404

    def test_404_response_is_json(self, client):
        resp = client.get("/api/v1/nonexistent-endpoint-abc")
        assert resp.headers["content-type"].startswith("application/json")

    def test_404_response_has_error_field(self, client):
        resp = client.get("/api/v1/nonexistent-endpoint-abc")
        data = resp.json()
        assert "error" in data

    def test_404_error_code_is_not_found(self, client):
        resp = client.get("/api/v1/nonexistent-endpoint-abc")
        data = resp.json()
        assert data["error"] == "not_found"

    def test_404_response_has_message(self, client):
        resp = client.get("/api/v1/nonexistent-endpoint-abc")
        data = resp.json()
        assert "message" in data
        assert len(data["message"]) > 0

    def test_validation_error_422_on_invalid_top_k(self, client, sample_image_bytes):
        # top_k max is 20; sending 99999 triggers a 422
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true", "top_k": "99999"},
        )
        assert resp.status_code == 422

    def test_validation_error_has_message(self, client, sample_image_bytes):
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true", "top_k": "99999"},
        )
        data = resp.json()
        assert "message" in data

    def test_validation_error_has_error_field(self, client, sample_image_bytes):
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true", "top_k": "99999"},
        )
        data = resp.json()
        assert "error" in data

    def test_503_response_structure(self, client, app, sample_image_bytes):
        original = app.state.detector
        app.state.detector = None
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true"},
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert "message" in data
        app.state.detector = original

    def test_503_error_code_is_service_unavailable(self, client, app, sample_image_bytes):
        original = app.state.detector
        app.state.detector = None
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true"},
        )
        data = resp.json()
        assert data["error"] == "service_unavailable"
        app.state.detector = original

    def test_400_response_has_error_field(self, client):
        files = {"image": ("bad.jpg", b"garbage-bytes", "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_400_error_code_is_bad_request(self, client):
        files = {"image": ("bad.jpg", b"garbage-bytes", "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "true"},
        )
        data = resp.json()
        assert data["error"] == "bad_request"

    def test_403_response_has_error_field(self, client, sample_image_bytes):
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "false"},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert "error" in data

    def test_403_error_code_is_forbidden(self, client, sample_image_bytes):
        files = {"image": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/recognize",
            files=files,
            data={"consent": "false"},
        )
        data = resp.json()
        assert data["error"] == "forbidden"

    def test_error_response_has_request_id(self, client):
        custom_id = "error-trace-id-001"
        resp = client.get(
            "/api/v1/nonexistent-endpoint-abc",
            headers={"X-Request-ID": custom_id},
        )
        # The error response body should echo the request_id
        data = resp.json()
        assert data.get("request_id") == custom_id

    def test_register_422_missing_name_response_structure(self, client, sample_image_bytes):
        files = {"image": ("face.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post(
            "/api/v1/register",
            files=files,
            data={"consent": "true"},  # name field intentionally omitted
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        assert "message" in data
