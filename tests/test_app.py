"""HTTP tests against the FastAPI app with TestClient. Offline; need `just setup` and `just train` first."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from app import app  # noqa: E402
from src.data import CLASSES, IMAGES_DIR, sample_image_path  # noqa: E402
from src.detector import MODEL_PATH  # noqa: E402

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `just train` first")
needs_data = pytest.mark.skipif(not IMAGES_DIR.exists(), reason="run `just setup` first")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "capstone1-vehicle-detection"
    assert isinstance(body["model_loaded"], bool)


@needs_model
def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    for key in ("val_accuracy", "val_box_mean_iou", "epochs", "images_used", "wall_seconds", "mode"):
        assert key in m
    assert 0.0 <= m["val_accuracy"] <= 1.0
    assert 0.0 <= m["val_box_mean_iou"] <= 1.0


@needs_model
@needs_data
def test_predict_sample_image(client):
    path = sample_image_path()
    with path.open("rb") as f:
        r = client.post("/predict", files={"file": (path.name, f, "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["class"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    w, h = body["image_size"]
    x1, y1, x2, y2 = body["box"]
    assert len(body["box"]) == 4
    assert 0 <= x1 <= x2 <= w and 0 <= y1 <= y2 <= h
    assert len(body["top3"]) == 3


@needs_model
def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    for anchor in ('id="input"', 'id="run"', 'id="result"', 'header class="top"'):
        assert anchor in r.text


@needs_data
def test_sample_endpoint(client):
    r = client.get("/sample/00000001")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_safety_page(client):
    r = client.get("/safety")
    assert r.status_code == 200
    for table_id in ("by-year", "by-country", "by-state", "by-model", "victims", "inspection"):
        assert f'id="{table_id}"' in r.text
    assert "safety_deaths_per_year.png" in r.text
    assert "Autopilot" in r.text
    assert client.get("/static/safety_deaths_per_year.png").status_code == 200


def test_safety_json_consistency(client):
    r = client.get("/safety.json")
    assert r.status_code == 200
    body = r.json()
    s = body["summary"]
    assert s["events"] == sum(row["events"] for row in body["tables"]["by_year"])
    assert s["deaths"] == sum(row["deaths"] for row in body["tables"]["by_year"])
    assert s["verified_autopilot_deaths"] <= s["verified_plus_nhtsa_deaths"]
    assert body["inspection"]["footer_rows_dropped"] >= 10
