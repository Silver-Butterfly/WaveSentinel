from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import CLASS_NAMES, CLASS_THRESHOLDS, MODEL_VERSION, EXPECTED_ENGINE_SHA256, INPUT_SIZE, NMS_IOU, engine_status, infer_image
from store import store

app = FastAPI(
    title="WaveSentinel Backend",
    version="0.1.0",
    description="Phase H backend bridge for the frozen SSS TensorRT deployment.",
)

origins = [x.strip() for x in os.getenv("WAVESENTINEL_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    status: str = Field(pattern="^(pending|confirmed|dismissed|flagged)$")
    notes: str | None = None


class DetectionRequest(BaseModel):
    mission_id: str
    frame_id: str
    timestamp: str
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    depth: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    position_source: str = "not_available"
    telemetry_source: str = "not_provided"
    motion_correction_state: str = "not_applied"
    input_quality: str = "not_evaluated"
    operator_notes: str = ""
    source: str = "uploaded_file"


@app.get("/api/health")
def health():
    es = engine_status()
    return {
        "status": "ok",
        "service": "wavesentinel-backend",
        "phase": "H",
        "mode": "SIMULATION",
        "timestamp_unix": time.time(),
        "engine": es,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }


@app.get("/api/state")
def state():
    s = store.snapshot()
    return {
        **s,
        "system": {
            "mode": "SIMULATION",
            "backend_online": True,
            "engine": engine_status(),
            "model_version": MODEL_VERSION,
        },
        "provenance": {
            "application_mode": "simulation",
            "hardware_connected": False,
            "telemetry_source": "simulation",
            "position_source": "simulated",
        },
    }


@app.get("/api/system/status")
def system_status():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "backend": "TensorRT FP32",
        "model_version": MODEL_VERSION,
        "engine": engine_status(),
        "gpu": gpu,
        "cuda": torch.version.cuda,
        "g3_benchmark": {
            "status": "PASS",
            "p50_ms": 4.3501,
            "p95_ms": 4.5374,
            "p99_ms": 4.751622,
            "mean_ms": 4.364825,
            "throughput_fps": 229.10426,
            "scope": "backend inference only; preprocessing, host-device transfer and postprocessing excluded",
        },
        "class_thresholds": CLASS_THRESHOLDS,
        "nms_iou": NMS_IOU,
    }


@app.post("/api/mission/{action}")
def mission_action(action: str):
    allowed = {"start": "ACTIVE", "resume": "ACTIVE", "pause": "PAUSED", "end": "ENDED"}
    if action not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported mission action: {action}")
    return {"status": "PASS", "mission": store.update_mission(allowed[action])}


@app.post("/api/detections", status_code=201)
def create_detection(req: DetectionRequest):
    detection = {
        "mission_id": req.mission_id,
        "frame_id": req.frame_id,
        "timestamp": req.timestamp,
        "class": req.class_name,
        "confidence": round(float(req.confidence) * 100.0 if float(req.confidence) <= 1.0 else float(req.confidence), 2),
        "risk": "UNASSESSED",
        "risk_source": "not_available",
        "depth": req.depth,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "position_source": req.position_source,
        "telemetry_source": req.telemetry_source,
        "motion_correction_state": req.motion_correction_state,
        "input_quality": req.input_quality,
        "model_version": MODEL_VERSION,
        "review_status": "pending",
        "operator_notes": req.operator_notes,
        "source": req.source,
        "bbox_xyxy": req.bbox_xyxy,
        "risk_note": "No validated risk classifier is attached to the detection.",
    }
    return {"status": "PASS", "detection": store.add_detection(detection)}


@app.post("/api/detections/{frame_id}/review")
def review_detection(frame_id: str, req: ReviewRequest):
    updated = store.review_detection(frame_id, req.status, req.notes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Detection not found.")
    return {"status": "PASS", "detection": updated}


@app.post("/api/inference/image")
async def inference_image(
    file: UploadFile = File(...),
    frame_id: str = Form("WS_UPLOAD"),
    timestamp: str = Form("not_provided"),
    range_m: str = Form("not_provided"),
    depth: str = Form("not_provided"),
    survey_line: str = Form("not_provided"),
):
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 100 MB upload limit.")
    try:
        result = infer_image(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    result["metadata"] = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "range_m": range_m,
        "depth": depth,
        "survey_line": survey_line,
    }
    return result


@app.get("/api/telemetry")
def telemetry():
    state = store.snapshot()
    return {
        "status": "PASS",
        "source": "simulation",
        "current": state.get("telemetry"),
        "log": state.get("telemetryLog", []),
    }


@app.get("/api/live/frame")
def live_frame():
    return {"status": "PASS", **store.live_frame()}


@app.post("/api/batch/start")
def batch_start():
    return {"status": "PASS", "batch": store.start_batch()}


@app.get("/api/batch/status")
def batch_status():
    return {"status": "PASS", "batch": store.batch_status()}


@app.get("/api/archive")
def archive():
    state = store.snapshot()
    return {
        "status": "PASS",
        "mission_id": state["mission"]["missionId"],
        "detections": state["detections"],
        "class_counts": state.get("classCounts", {}),
    }


@app.post("/api/settings/reset")
def reset():
    return {"status": "PASS", "state": store.reset()}


@app.get("/api/model/contracts")
def model_contracts():
    return {
        "model_version": MODEL_VERSION,
        "backend": "TensorRT FP32",
        "input": {"shape": [1, 3, INPUT_SIZE, INPUT_SIZE], "dtype": "float32"},
        "output": {"shape": [1, 8, 8400], "dtype": "float32"},
        "classes": CLASS_NAMES,
        "class_thresholds": CLASS_THRESHOLDS,
        "nms_iou": NMS_IOU,
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "risk_assessment": "not_available",
        "georeference": "simulation_or_validated_geometry_required",
    }
