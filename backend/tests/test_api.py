"""
Backend API tests.

These tests mock the ML layer (model_service) entirely, so they run in
milliseconds and never require a trained checkpoint on disk - only
route/auth/validation logic is under test here. End-to-end behavior with
a real checkpoint is covered separately by manual smoke testing and the
ml/ package's own test suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set env vars before any app import touches get_settings().
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("ADMIN_USERNAME", "admin")
# bcrypt hash of "testpassword123"
os.environ.setdefault(
    "ADMIN_PASSWORD_HASH",
    "$2b$12$mL1pB2u1zpk1r7hkfkQxAO6bWpEwB7rHkQmz1J0iCq7hAeC0Xk4Iu",
)

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()


class FakePredictionResult:
    """Stand-in for pulmoguard.infer.PredictionResult."""

    def __init__(self, abstain: bool = False):
        self._data = {
            "predicted_class": "PNEUMONIA",
            "confidence": 0.91,
            "normalized_entropy": 0.85 if abstain else 0.10,
            "abstain": abstain,
            "class_probabilities": {"NORMAL": 0.09, "PNEUMONIA": 0.91},
            "mc_dropout_passes": 20,
        }

    def to_dict(self) -> dict:
        return dict(self._data)


class FakePredictor:
    def __init__(self, abstain: bool = False):
        self._abstain = abstain

    def predict(self, image_bytes):
        return FakePredictionResult(abstain=self._abstain)


@pytest.fixture
def client():
    """Yield a TestClient with the model service mocked out (no real checkpoint needed)."""
    with patch("app.services.model_service.load_model") as mock_load, \
         patch("app.services.model_service.is_model_loaded", return_value=True), \
         patch("app.services.model_service.get_predictor", return_value=FakePredictor()):
        mock_load.return_value = None
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as c:
            yield c


@pytest.fixture
def auth_token(client):
    """Regenerate the known bcrypt hash for 'testpassword123' at runtime to
    avoid relying on a hardcoded hash that might not match this bcrypt version."""
    import bcrypt
    from app.core import config as config_module

    real_hash = bcrypt.hashpw(b"testpassword123", bcrypt.gensalt()).decode()
    settings = config_module.get_settings()
    settings.ADMIN_PASSWORD_HASH = real_hash  # patch the cached singleton directly

    resp = client.post("/api/v1/auth/token", data={"username": "admin", "password": "testpassword123"})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def _dummy_image_file():
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("L", (32, 32), color=128).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "PulmoGuard API"


def test_liveness(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_readiness_when_model_loaded(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_login_wrong_password_returns_401(client):
    resp = client.post("/api/v1/auth/token", data={"username": "admin", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user_returns_401(client):
    resp = client.post("/api/v1/auth/token", data={"username": "nobody", "password": "x"})
    assert resp.status_code == 401


def test_login_success_returns_token(auth_token):
    assert isinstance(auth_token, str) and len(auth_token) > 10


def test_predict_without_token_returns_401(client):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("xray.jpeg", _dummy_image_file(), "image/jpeg")},
    )
    assert resp.status_code == 401


def test_predict_with_invalid_token_returns_401(client):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("xray.jpeg", _dummy_image_file(), "image/jpeg")},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert resp.status_code == 401


def test_predict_with_valid_token_returns_prediction(client, auth_token):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("xray.jpeg", _dummy_image_file(), "image/jpeg")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "PNEUMONIA"
    assert body["abstain"] is False
    assert "message" in body
    assert body["filename"] == "xray.jpeg"


def test_predict_rejects_unsupported_content_type(client, auth_token):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400


def test_predict_rejects_empty_file(client, auth_token):
    resp = client.post(
        "/api/v1/predict",
        files={"file": ("xray.jpeg", b"", "image/jpeg")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400


def test_response_includes_request_id_header(client):
    resp = client.get("/health/live")
    assert "X-Request-ID" in resp.headers
