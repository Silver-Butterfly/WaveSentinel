from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ENGINE_PATH = Path(os.getenv(
    "WAVESENTINEL_TRT_ENGINE",
    r"K:\Debris model\runs\phase_g2_tensorrt\best_fp32.engine",
))
EXPECTED_ENGINE_SHA256 = os.getenv(
    "WAVESENTINEL_ENGINE_SHA256",
    "5e2bfabaca634873b61242c1817323d51cdb002d63831cee1a2874fd60818067",
)
MODEL_VERSION = "Phase-D Robust-V2-HN"
INPUT_SIZE = 640
TILE_W = 1024
TILE_H = 500
OVERLAP = 0.20
STRIDE = int(round(TILE_W * (1.0 - OVERLAP)))
NMS_IOU = 0.50
MAX_DET = 300
CLASS_NAMES = {0: "Pipe", 1: "Shipwreck", 2: "Mine", 3: "Ghost Net"}
CLASS_THRESHOLDS = {0: 0.56, 1: 0.45, 2: 0.89, 3: 0.36}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def nms_classwise(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float, max_det: int):
    keep = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        while order.size:
            i = int(order[0])
            keep.append(i)
            if len(keep) >= max_det:
                break
            if order.size == 1:
                break
            overlaps = _iou_one_to_many(boxes[i], boxes[order[1:]])
            order = order[1:][overlaps < iou_thr]
    return np.array(sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:max_det], dtype=np.int64)


def decode_yolo_output(output: np.ndarray, source_shape: tuple[int, int], tile_info: dict[str, int] | None = None):
    pred = np.asarray(output[0], dtype=np.float32)
    if pred.shape[0] != 8:
        raise RuntimeError(f"Unexpected TensorRT output shape: {pred.shape}")
    pred = pred.T

    xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    classes = np.argmax(cls_scores, axis=1).astype(np.int32)
    scores = np.max(cls_scores, axis=1)

    selected = np.zeros(scores.shape[0], dtype=bool)
    thresholds = np.array([CLASS_THRESHOLDS[i] for i in classes], dtype=np.float32)
    selected = scores >= thresholds
    xywh = xywh[selected]
    scores = scores[selected]
    classes = classes[selected]

    if xywh.size == 0:
        return []

    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0

    if tile_info is not None:
        scale = TILE_W / INPUT_SIZE
        pad_x = tile_info["pad_x"]
        pad_y = tile_info["pad_y"]
        boxes *= scale
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes[:, [0, 2]] += tile_info["x0"]
        boxes[:, [1, 3]] += tile_info["y0"]

    h, w = source_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    keep = nms_classwise(boxes, scores, classes, NMS_IOU, MAX_DET)
    results = []
    for i in keep:
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        results.append({
            "class_id": int(classes[i]),
            "class": CLASS_NAMES[int(classes[i])],
            "confidence": float(scores[i]),
            "threshold": CLASS_THRESHOLDS[int(classes[i])],
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_width": max(0.0, x2 - x1),
            "bbox_height": max(0.0, y2 - y1),
            "risk": "UNASSESSED",
            "risk_source": "not_available",
        })
    return results


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Unable to decode image. Use a readable PNG/JPG/BMP/TIFF.")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _percentile_normalize(tile: np.ndarray) -> np.ndarray:
    x = tile.astype(np.float32)
    p1 = float(np.percentile(x, 1.0))
    p99 = float(np.percentile(x, 99.0))
    if p99 <= p1 + 1e-6:
        out = np.zeros_like(x, dtype=np.float32)
    else:
        out = np.clip((x - p1) / (p99 - p1), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def make_prepared_tensor(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    return np.repeat(x[None, :, :], 3, axis=0)[None].astype(np.float32)


def make_raw_tiles(gray: np.ndarray):
    h, w = gray.shape
    if h > TILE_H:
        raise ValueError(
            f"Raw SSS tiling expects a sonar strip up to {TILE_H}px high; received {h}px. "
            "Use the prepared 640×640 workflow or provide an operational SSS strip."
        )
    xs = [0] if w <= TILE_W else list(range(0, max(1, w - TILE_W + 1), STRIDE))
    if w > TILE_W and xs[-1] != w - TILE_W:
        xs.append(w - TILE_W)
    tiles = []
    for x0 in xs:
        crop = gray[:, x0:min(x0 + TILE_W, w)]
        ch, cw = crop.shape
        canvas = np.zeros((TILE_W, TILE_W), dtype=np.uint8)
        pad_x = (TILE_W - cw) // 2
        pad_y = (TILE_W - ch) // 2
        canvas[pad_y:pad_y + ch, pad_x:pad_x + cw] = _percentile_normalize(crop)
        tiles.append((make_prepared_tensor(canvas), {"x0": int(x0), "y0": 0, "pad_x": pad_x, "pad_y": pad_y}))
    return tiles


class TensorRTEngine:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        if not ENGINE_PATH.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {ENGINE_PATH}")
        actual = sha256_file(ENGINE_PATH)
        if actual != EXPECTED_ENGINE_SHA256:
            raise RuntimeError(
                f"Engine SHA256 mismatch: expected {EXPECTED_ENGINE_SHA256}, got {actual}"
            )
        import tensorrt as trt
        self.trt = trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(ENGINE_PATH.read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        if list(self.engine.get_tensor_shape(self.input_name)) != [1, 3, 640, 640]:
            raise RuntimeError("Unexpected TensorRT input contract.")
        if list(self.engine.get_tensor_shape(self.output_name)) != [1, 8, 8400]:
            raise RuntimeError("Unexpected TensorRT output contract.")
        self.lock = threading.Lock()

    def infer(self, tensor: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(np.ascontiguousarray(tensor)).cuda(non_blocking=True)
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        context = self.engine.create_execution_context()
        stream = torch.cuda.current_stream()
        context.set_tensor_address(self.input_name, x.data_ptr())
        context.set_tensor_address(self.output_name, out.data_ptr())
        t0 = time.perf_counter()
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned false.")
        stream.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out.detach().cpu().numpy(), elapsed_ms

    def infer_safe(self, tensor: np.ndarray):
        with self.lock:
            return self.infer(tensor)


_engine: TensorRTEngine | None = None
_engine_lock = threading.Lock()
_engine_hash_cache: tuple[int, int, str] | None = None


def get_engine() -> TensorRTEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = TensorRTEngine()
    return _engine


def engine_status() -> dict[str, Any]:
    global _engine_hash_cache
    present = ENGINE_PATH.exists()
    sha = None
    if present:
        stat = ENGINE_PATH.stat()
        key = (int(stat.st_mtime_ns), int(stat.st_size))
        if _engine_hash_cache is not None and _engine_hash_cache[:2] == key:
            sha = _engine_hash_cache[2]
        else:
            sha = sha256_file(ENGINE_PATH)
            _engine_hash_cache = (key[0], key[1], sha)
    return {
        "engine_present": present,
        "engine_loaded": _engine is not None,
        "engine_sha256": sha,
        "expected_sha256": EXPECTED_ENGINE_SHA256,
        "hash_match": bool(present and sha == EXPECTED_ENGINE_SHA256),
        "model_version": MODEL_VERSION,
        "backend": "TensorRT FP32",
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 8, 8400],
        "class_thresholds": CLASS_THRESHOLDS,
        "nms_iou": NMS_IOU,
    }


def infer_image(data: bytes) -> dict[str, Any]:
    gray = decode_image(data)
    h, w = gray.shape
    start = time.perf_counter()
    detections = []
    timings = []
    if (w, h) == (640, 640):
        tensor = make_prepared_tensor(gray)
        raw, infer_ms = get_engine().infer_safe(tensor)
        timings.append(infer_ms)
        detections.extend(decode_yolo_output(raw, (h, w)))
        mode = "prepared_640"
        tile_count = 1
    else:
        mode = "raw_sss_tiled"
        tiles = make_raw_tiles(gray)
        tile_count = len(tiles)
        for tensor, info in tiles:
            raw, infer_ms = get_engine().infer_safe(tensor)
            timings.append(infer_ms)
            detections.extend(decode_yolo_output(raw, (h, w), info))
        if detections:
            boxes = np.array([d["bbox_xyxy"] for d in detections], dtype=np.float32)
            scores = np.array([d["confidence"] for d in detections], dtype=np.float32)
            classes = np.array([d["class_id"] for d in detections], dtype=np.int32)
            keep = nms_classwise(boxes, scores, classes, NMS_IOU, MAX_DET)
            detections = [detections[int(i)] for i in keep]

    total_ms = (time.perf_counter() - start) * 1000.0
    for idx, d in enumerate(detections):
        d["detection_id"] = f"DET-{int(time.time() * 1000)}-{idx:03d}"
        d["source_width"] = int(w)
        d["source_height"] = int(h)
        d["tile_count"] = tile_count

    return {
        "status": "PASS",
        "source": "uploaded_file",
        "input": {
            "width": int(w),
            "height": int(h),
            "mode": mode,
            "tile_count": tile_count,
            "additional_preprocessing": mode != "prepared_640",
        },
        "detections": detections,
        "objects_count": len(detections),
        "runtime": {
            "backend": "TensorRT FP32",
            "inference_ms_sum": float(sum(timings)),
            "total_backend_ms": float(total_ms),
        },
        "model": {
            "version": MODEL_VERSION,
            "engine_sha256": EXPECTED_ENGINE_SHA256,
            "class_thresholds": CLASS_THRESHOLDS,
            "nms_iou": NMS_IOU,
        },
        "provenance": {
            "data_source": "uploaded_file",
            "telemetry_source": "not_provided",
            "position_source": "not_available",
            "risk_assessment": "not_available",
        },
    }
