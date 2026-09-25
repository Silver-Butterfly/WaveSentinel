from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path

STATE_PATH = Path(__file__).with_name("demo_state.json")


class StateStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._initial = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        self._state = copy.deepcopy(self._initial)
        self._last_tick = time.monotonic()
        self._live_index = 0
        self._batch = {
            "batchId": "BATCH-018",
            "status": "IDLE",
            "totalFrames": 248,
            "processedFrames": 0,
            "detections": 0,
            "reviewRequired": 0,
        }

    def _telemetry_from_time(self) -> dict:
        now = time.time()
        phase = now * 0.7
        heave = 0.18 + math.sin(phase) * 0.04
        pitch = 2.4 + math.cos(phase * 0.8) * 0.3
        roll = -1.1 + math.sin(phase * 0.5) * 0.3
        ts = time.strftime("%H:%M:%S", time.localtime(now))
        return {
            "heave": f"{heave:+.2f} m",
            "pitch": f"{pitch:+.1f}°",
            "roll": f"{roll:+.1f}°",
            "heading": f"{self._state['mission']['heading']:.0f}°",
            "depth": f"{self._state['mission']['depth']:.1f} m",
            "vesselSpeed": f"{self._state['mission']['vesselSpeed']:.1f} kts",
            "latency": f"{self._state['mission']['avgLatencyMs']:.2f} ms",
            "timestamp": ts,
            "position": "20.4521° N, 85.1294° E",
            "frameId": f"SIM_{self._state['mission']['framesProcessed']:06d}",
            "source": "simulation",
            "valid": True,
        }

    def tick(self):
        with self._lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self._last_tick)
            self._last_tick = now
            mission = self._state["mission"]
            if mission["status"] == "ACTIVE" and mission["framesProcessed"] < mission["framesTotal"]:
                increment = max(1, int(round(elapsed * float(mission["frameRate"]))))
                mission["framesProcessed"] = min(mission["framesTotal"], mission["framesProcessed"] + increment)
            telemetry = self._telemetry_from_time()
            self._state["telemetry"] = telemetry
            log = self._state.setdefault("telemetryLog", [])
            log.insert(0, {
                "time": telemetry["timestamp"],
                "heave": telemetry["heave"],
                "pitch": telemetry["pitch"],
                "roll": telemetry["roll"],
                "depth": telemetry["depth"],
            })
            del log[15:]
            return copy.deepcopy(self._state)

    def snapshot(self):
        return self.tick()

    def update_mission(self, status: str):
        with self._lock:
            self._state["mission"]["status"] = status
            self._last_tick = time.monotonic()
            return copy.deepcopy(self._state["mission"])

    def add_detection(self, detection: dict):
        with self._lock:
            self._state["detections"].insert(0, copy.deepcopy(detection))
            self._state["mission"]["detectionsCount"] += 1
            if detection.get("risk") in {"HIGH", "CRITICAL"}:
                self._state["mission"]["highRiskCount"] += 1
            counts = self._state.setdefault("classCounts", {})
            name = detection.get("class", "Unknown").upper().replace(" ", "_")
            counts[name] = counts.get(name, 0) + 1
            return copy.deepcopy(detection)

    def review_detection(self, frame_id: str, status: str, notes: str | None = None):
        with self._lock:
            for item in self._state["detections"]:
                if item["frame_id"] == frame_id:
                    item["review_status"] = status
                    if notes is not None:
                        item["operator_notes"] = notes
                    return copy.deepcopy(item)
        return None

    def update_settings(self, settings: dict):
        with self._lock:
            self._state["settings"].update(copy.deepcopy(settings))
            return copy.deepcopy(self._state["settings"])

    def start_batch(self):
        with self._lock:
            self._batch.update({
                "status": "PROCESSING",
                "processedFrames": 0,
                "detections": 0,
                "reviewRequired": 0,
            })
            self._batch["startedAt"] = time.time()
            return copy.deepcopy(self._batch)

    def batch_status(self):
        with self._lock:
            if self._batch["status"] == "PROCESSING":
                started = self._batch.get("startedAt", time.time())
                elapsed = max(0.0, time.time() - started)
                processed = min(self._batch["totalFrames"], int(elapsed * 140))
                self._batch["processedFrames"] = processed
                self._batch["detections"] = min(31, int(processed * 0.125))
                self._batch["reviewRequired"] = min(5, int(processed * 0.02))
                if processed >= self._batch["totalFrames"]:
                    self._batch["status"] = "COMPLETE"
                    self._batch["processedFrames"] = self._batch["totalFrames"]
                    self._batch["detections"] = 31
                    self._batch["reviewRequired"] = 5
            return copy.deepcopy(self._batch)

    def live_frame(self):
        with self._lock:
            detections = self._state.get("detections", [])
            candidates = [d for d in detections if d.get("position_source") != "not_available"]
            if not candidates:
                candidates = detections
            item = copy.deepcopy(candidates[self._live_index % len(candidates)]) if candidates else None
            self._live_index += 1
            return {
                "frame_id": self._state["telemetry"]["frameId"],
                "timestamp": self._state["telemetry"]["timestamp"],
                "source": "simulation",
                "stream_status": "STREAMING",
                "telemetry": copy.deepcopy(self._state["telemetry"]),
                "detection": item,
            }

    def reset(self):
        with self._lock:
            self._state = copy.deepcopy(self._initial)
            self._last_tick = time.monotonic()
            self._live_index = 0
            self._batch = {
                "batchId": "BATCH-018",
                "status": "IDLE",
                "totalFrames": 248,
                "processedFrames": 0,
                "detections": 0,
                "reviewRequired": 0,
            }
            return copy.deepcopy(self._state)


store = StateStore()
