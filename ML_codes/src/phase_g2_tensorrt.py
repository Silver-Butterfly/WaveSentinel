from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.2"
EXPECTED_PHASE_D_SHA256 = "276f91a207df2563a09d4a05aca8f47b8899dace2071703650acae8cb4eb891e"
EXPECTED_PHASE_G1_ONNX_SHA256 = "17db4dee6147d1eb05867cda7ca503b1133e3b6e83a45cb7c0e359cf9f78f12b"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return None


def import_status(name: str) -> dict[str, Any]:
    try:
        module = __import__(name)
        return {"present": True, "version": getattr(module, "__version__", "installed")}
    except Exception as exc:
        return {"present": False, "version": None, "error": str(exc)}


def command_info(command: str, args: list[str] | None = None) -> dict[str, Any]:
    exe = shutil.which(command)
    if not exe:
        return {"found": False, "path": None, "output": None, "returncode": None}
    argv = [exe] + (args or ["--version"])
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return {
            "found": p.returncode == 0,
            "path": exe,
            "output": (p.stdout or p.stderr).strip()[-4000:],
            "returncode": p.returncode,
        }
    except Exception as exc:
        return {"found": True, "path": exe, "output": None, "returncode": None, "error": str(exc)}


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tensorrt": import_status("tensorrt"),
        "onnx": import_status("onnx"),
        "onnxruntime": import_status("onnxruntime"),
        "ultralytics": import_status("ultralytics"),
        "torch": import_status("torch"),
    }

    try:
        import tensorrt as trt
        info["tensorrt_version"] = trt.__version__
    except Exception:
        info["tensorrt_version"] = None

    try:
        import torch
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["cuda_capability"] = (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        )
    except Exception as exc:
        info["torch_cuda_version"] = None
        info["cuda_available"] = None
        info["cuda_device"] = None
        info["cuda_capability"] = None
        info["torch_error"] = str(exc)

    info["trtexec"] = command_info("trtexec")
    info["nvcc"] = command_info("nvcc")
    info["nvidia_smi"] = command_info(
        "nvidia-smi",
        ["--query-gpu=name,driver_version", "--format=csv,noheader"],
    )
    return info


def preflight(
    checkpoint: Path,
    onnx_path: Path,
    expected_checkpoint_sha256: str | None,
    expected_onnx_sha256: str | None,
) -> dict[str, Any]:
    require_file(checkpoint, "Phase D checkpoint")
    require_file(onnx_path, "Phase G1 ONNX")

    checkpoint_sha = sha256_file(checkpoint)
    onnx_sha = sha256_file(onnx_path)
    env = environment_info()

    checks = {
        "checkpoint_sha256_match": expected_checkpoint_sha256 is None or checkpoint_sha.lower() == expected_checkpoint_sha256.lower(),
        "onnx_sha256_match": expected_onnx_sha256 is None or onnx_sha.lower() == expected_onnx_sha256.lower(),
        "tensorrt_python_present": env["tensorrt"]["present"],
        "torch_cuda_available": env["cuda_available"] is True,
    }

    status = "PASS" if all(checks.values()) else "BLOCKED"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "onnx": str(onnx_path),
        "onnx_sha256": onnx_sha,
        "expected_onnx_sha256": expected_onnx_sha256,
        "environment": env,
        "checks": checks,
        "notes": [
            "trtexec is optional for this Python API workflow.",
            "The ONNX↔TensorRT comparison uses the installed Ultralytics TensorRT backend and does not require the separate cuda Python bindings package.",
            "No source checkpoint or ONNX artifact is modified by preflight.",
        ],
    }


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gb: float = 1.0,
) -> dict[str, Any]:
    require_file(onnx_path, "ONNX model")
    if workspace_gb <= 0:
        raise ValueError("workspace_gb must be > 0")

    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError("TensorRT Python package is required.") from exc

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1024 ** 3))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    inputs = []
    outputs = []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        inputs.append({"name": t.name, "shape": list(t.shape), "dtype": str(t.dtype)})
    for i in range(network.num_outputs):
        t = network.get_output(i)
        outputs.append({"name": t.name, "shape": list(t.shape), "dtype": str(t.dtype)})

    return {
        "status": "PASS",
        "tensorrt_version": trt.__version__,
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "engine": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "build_config": {
            "precision": "fp32",
            "workspace_gb": workspace_gb,
            "strongly_typed": True,
        },
        "network": {
            "inputs": inputs,
            "outputs": outputs,
            "num_inputs": network.num_inputs,
            "num_outputs": network.num_outputs,
        },
        "note": "G2 baseline engine built as FP32. Performance benchmarking and FP16 optimization are deferred to later deployment work.",
    }


def validate_engine(engine_path: Path) -> dict[str, Any]:
    require_file(engine_path, "TensorRT engine")
    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError("TensorRT Python package is required.") from exc

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization returned None.")

    tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        tensors.append({
            "name": name,
            "mode": str(engine.get_tensor_mode(name)).split(".")[-1],
            "shape": list(engine.get_tensor_shape(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
        })

    return {
        "status": "PASS",
        "tensorrt_version": trt.__version__,
        "engine": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "num_io_tensors": engine.num_io_tensors,
        "tensors": tensors,
    }


def _prediction_payload(result: Any) -> dict[str, Any]:
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
    conf = boxes.conf.detach().cpu().tolist() if boxes is not None else []
    cls = boxes.cls.detach().cpu().tolist() if boxes is not None else []
    return {
        "boxes": xyxy,
        "confidence": conf,
        "classes": [int(x) for x in cls],
    }


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pair_predictions(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    used = set()
    pairs = []
    unmatched_a = []

    for i, (box_a, cls_a, conf_a) in enumerate(zip(a["boxes"], a["classes"], a["confidence"])):
        candidates = [
            j
            for j, cls_b in enumerate(b["classes"])
            if j not in used and cls_b == cls_a
        ]
        if not candidates:
            unmatched_a.append(i)
            continue

        j = max(candidates, key=lambda k: _iou(box_a, b["boxes"][k]))
        used.add(j)
        pairs.append({
            "class_id": cls_a,
            "iou": _iou(box_a, b["boxes"][j]),
            "confidence_abs_delta": abs(conf_a - b["confidence"][j]),
            "max_box_coord_abs_delta": max(
                abs(x - y) for x, y in zip(box_a, b["boxes"][j])
            ),
        })

    unmatched_b = [j for j in range(len(b["boxes"])) if j not in used]
    mean_iou = sum(x["iou"] for x in pairs) / len(pairs) if pairs else 0.0
    mean_conf_delta = (
        sum(x["confidence_abs_delta"] for x in pairs) / len(pairs)
        if pairs else 0.0
    )
    max_coord_delta = max(
        (x["max_box_coord_abs_delta"] for x in pairs), default=0.0
    )

    return {
        "a_count": len(a["boxes"]),
        "b_count": len(b["boxes"]),
        "paired_count": len(pairs),
        "unmatched_a": len(unmatched_a),
        "unmatched_b": len(unmatched_b),
        "mean_pair_iou": mean_iou,
        "mean_confidence_abs_delta": mean_conf_delta,
        "max_box_coord_abs_delta": max_coord_delta,
        "pairs": pairs,
        "numerically_stable": (
            len(unmatched_a) == 0
            and len(unmatched_b) == 0
        ),
    }


def compare_onnx_tensorrt(
    onnx_path: Path,
    engine_path: Path,
    reference_image: Path,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.70,
    max_det: int = 300,
) -> dict[str, Any]:
    require_file(onnx_path, "ONNX model")
    require_file(engine_path, "TensorRT engine")
    require_file(reference_image, "reference image")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for ONNX↔TensorRT comparison.") from exc

    common = {
        "source": str(reference_image),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "verbose": False,
        "save": False,
    }

    onnx_model = YOLO(str(onnx_path))
    trt_model = YOLO(str(engine_path))

    onnx_result = onnx_model.predict(device="cpu", **common)[0]
    trt_result = trt_model.predict(device="0", **common)[0]

    onnx_payload = _prediction_payload(onnx_result)
    trt_payload = _prediction_payload(trt_result)
    comparison = pair_predictions(onnx_payload, trt_payload)

    return {
        "status": "PASS" if comparison["numerically_stable"] else "REVIEW",
        "onnx": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "engine": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "reference_image": str(reference_image),
        "inference": {
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "onnx_device": "cpu",
            "tensorrt_device": "0",
        },
        "comparison": comparison,
        "note": "End-to-end Ultralytics prediction equivalence on one fixed reference image. This is a consistency check, not an accuracy evaluation.",
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase G2 TensorRT engine pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--expected-checkpoint-sha256", default=EXPECTED_PHASE_D_SHA256)
    p.add_argument("--expected-onnx-sha256", default=EXPECTED_PHASE_G1_ONNX_SHA256)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("build")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--workspace-gb", type=float, default=1.0)
    p.add_argument("--expected-onnx-sha256", default=EXPECTED_PHASE_G1_ONNX_SHA256)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("validate")
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("compare")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--reference-image", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "preflight":
        result = preflight(
            args.checkpoint,
            args.onnx,
            args.expected_checkpoint_sha256,
            args.expected_onnx_sha256,
        )
    elif args.command == "build":
        require_file(args.onnx, "ONNX model")
        actual = sha256_file(args.onnx)
        if actual.lower() != args.expected_onnx_sha256.lower():
            raise RuntimeError(
                f"ONNX SHA256 mismatch. Expected {args.expected_onnx_sha256}, got {actual}."
            )
        result = build_engine(args.onnx, args.engine, args.workspace_gb)
    elif args.command == "validate":
        result = validate_engine(args.engine)
    else:
        result = compare_onnx_tensorrt(
            args.onnx,
            args.engine,
            args.reference_image,
            args.imgsz,
            args.conf,
            args.iou,
            args.max_det,
        )

    print(json.dumps(result, indent=2))
    if args.output:
        write_json(result, args.output)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
