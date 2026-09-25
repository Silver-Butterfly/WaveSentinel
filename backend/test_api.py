from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_state_contract():
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert body["system"]["mode"] == "SIMULATION"
    assert isinstance(body["detections"], list)
    assert "settings" in body


def test_model_contract():
    r = client.get("/api/model/contracts")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "TensorRT FP32"
    assert body["input"]["shape"] == [1, 3, 640, 640]
    assert body["output"]["shape"] == [1, 8, 8400]


def test_system_status_contains_frozen_g3():
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    assert body["g3_benchmark"]["status"] == "PASS"
    assert body["g3_benchmark"]["p50_ms"] == 4.3501


def test_mission_action():
    r = client.post("/api/mission/pause")
    assert r.status_code == 200
    assert r.json()["mission"]["status"] == "PAUSED"
    client.post("/api/mission/resume")


def test_create_and_review_detection():
    payload = {
        "mission_id": "WS-TEST",
        "frame_id": "WS_TEST_001",
        "timestamp": "2026-09-19T00:00:00Z",
        "class_name": "Pipe",
        "confidence": 0.75,
        "bbox_xyxy": [1, 2, 10, 20],
        "source": "uploaded_file"
    }
    r = client.post("/api/detections", json=payload)
    assert r.status_code == 201
    created = r.json()["detection"]
    assert created["risk"] == "UNASSESSED"
    assert created["confidence"] == 75.0

    r = client.post(
        "/api/detections/WS_TEST_001/review",
        json={"status": "confirmed", "notes": "reviewed"}
    )
    assert r.status_code == 200
    assert r.json()["detection"]["review_status"] == "confirmed"


def test_telemetry_endpoint():
    r = client.get("/api/telemetry")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "simulation"
    assert "current" in body
    assert isinstance(body["log"], list)


def test_live_frame_endpoint():
    r = client.get("/api/live/frame")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "PASS"
    assert body["source"] == "simulation"
    assert body["status"] == "PASS"
    assert "telemetry" in body


def test_batch_lifecycle():
    r = client.post("/api/batch/start")
    assert r.status_code == 200
    assert r.json()["batch"]["status"] == "PROCESSING"
    r = client.get("/api/batch/status")
    assert r.status_code == 200
    assert r.json()["batch"]["totalFrames"] == 248


def test_archive_endpoint():
    r = client.get("/api/archive")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["detections"], list)
    assert isinstance(body["class_counts"], dict)
