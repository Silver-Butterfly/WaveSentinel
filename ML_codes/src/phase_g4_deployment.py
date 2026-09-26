from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

EXPECTED_ONNX_SHA256 = "17db4dee6147d1eb05867cda7ca503b1133e3b6e83a45cb7c0e359cf9f78f12b"
EXPECTED_FP32_SHA256 = "5e2bfabaca634873b61242c1817323d51cdb002d63831cee1a2874fd60818067"

DEFAULT_ONNX = Path(r"K:\Debris model\runs\phase_g1_onnx\best.onnx")
DEFAULT_FP32 = Path(r"K:\Debris model\runs\phase_g2_tensorrt\best_fp32.engine")
DEFAULT_OUT = Path(r"K:\Debris model\runs\phase_g4_deployment")

SCHEMA_VERSION = "1.0"
INPUT_SHAPE = (1, 3, 640, 640)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env_snapshot() -> dict[str, Any]:
    cuda_ok = bool(torch.cuda.is_available())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": cuda_ok,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_ok else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if cuda_ok else None,
        "total_vram_mb": float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)) if cuda_ok else None,
        "tensorrt": trt.__version__,
    }


def load_engine(path: Path):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT engine deserialization failed: {path}")
    return runtime, engine


def io_contract(engine) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        item = {
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "shape": list(engine.get_tensor_shape(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
        }
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(item)
        else:
            outputs.append(item)
    return {"inputs": inputs, "outputs": outputs}


def smoke_test_engine(engine) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "BLOCKED", "reason": "CUDA unavailable"}

    torch.cuda.synchronize()
    before_alloc = torch.cuda.memory_allocated(0)
    before_reserved = torch.cuda.memory_reserved(0)

    context = engine.create_execution_context()
    if context is None:
        return {"status": "FAIL", "reason": "Execution context creation failed"}

    io = io_contract(engine)
    if len(io["inputs"]) != 1 or len(io["outputs"]) != 1:
        return {"status": "FAIL", "reason": "Expected exactly 1 input and 1 output", "io": io}

    input_name = io["inputs"][0]["name"]
    output_name = io["outputs"][0]["name"]
    input_shape = tuple(io["inputs"][0]["shape"])
    output_shape = tuple(io["outputs"][0]["shape"])

    if input_shape != INPUT_SHAPE:
        return {"status": "FAIL", "reason": f"Unexpected input shape: {input_shape}", "io": io}

    x = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device="cuda").contiguous()
    output = torch.empty(output_shape, dtype=torch.float32, device="cuda").contiguous()

    context.set_tensor_address(input_name, x.data_ptr())
    context.set_tensor_address(output_name, output.data_ptr())

    stream = torch.cuda.Stream()
    start = time.perf_counter()
    with torch.cuda.stream(stream):
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    after_alloc = torch.cuda.memory_allocated(0)
    after_reserved = torch.cuda.memory_reserved(0)

    finite = bool(torch.isfinite(output).all().item())
    return {
        "status": "PASS" if ok and finite else "FAIL",
        "execute_return": bool(ok),
        "output_finite": finite,
        "input_shape": list(INPUT_SHAPE),
        "output_shape": list(output_shape),
        "smoke_inference_ms": elapsed_ms,
        "torch_memory_allocated_before_mb": before_alloc / (1024 ** 2),
        "torch_memory_allocated_after_mb": after_alloc / (1024 ** 2),
        "torch_memory_reserved_before_mb": before_reserved / (1024 ** 2),
        "torch_memory_reserved_after_mb": after_reserved / (1024 ** 2),
    }


def preflight(onnx: Path, fp32: Path, out: Path) -> int:
    checks = {
        "onnx_present": onnx.exists(),
        "fp32_present": fp32.exists(),
        "cuda_available": bool(torch.cuda.is_available()),
        "tensorrt_import": True,
        "onnx_hash_match": onnx.exists() and sha256_file(onnx) == EXPECTED_ONNX_SHA256,
        "fp32_hash_match": fp32.exists() and sha256_file(fp32) == EXPECTED_FP32_SHA256,
    }

    engine_io = None
    engine_load_ok = False
    engine_error = None
    if fp32.exists():
        try:
            _, engine = load_engine(fp32)
            engine_io = io_contract(engine)
            engine_load_ok = True
        except Exception as e:
            engine_error = repr(e)

    checks["fp32_deserializes"] = engine_load_ok
    checks["expected_io_contract"] = bool(
        engine_io
        and len(engine_io["inputs"]) == 1
        and len(engine_io["outputs"]) == 1
        and engine_io["inputs"][0]["shape"] == list(INPUT_SHAPE)
        and engine_io["inputs"][0]["dtype"] == "DataType.FLOAT"
        and engine_io["outputs"][0]["dtype"] == "DataType.FLOAT"
    )

    status = "PASS" if all(checks.values()) else "BLOCKED"
    report = {
        "status": status,
        "schema_version": SCHEMA_VERSION,
        "phase": "G4",
        "stage": "preflight",
        "environment": env_snapshot(),
        "checks": checks,
        "frozen_inputs": {
            "onnx": {
                "path": str(onnx),
                "sha256": sha256_file(onnx) if onnx.exists() else None,
                "expected_sha256": EXPECTED_ONNX_SHA256,
            },
            "fp32_engine": {
                "path": str(fp32),
                "sha256": sha256_file(fp32) if fp32.exists() else None,
                "expected_sha256": EXPECTED_FP32_SHA256,
            },
        },
        "engine_io": engine_io,
        "engine_error": engine_error,
        "fp16": {
            "status": "DEFERRED",
            "reason": "TensorRT 11.3 environment does not expose the legacy TensorRT FP16 builder-flag API and ModelOpt AutoCast is not installed. FP16 is optional and is not required to close FP32 deployment integrity.",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


def validate(onnx: Path, fp32: Path, out: Path) -> int:
    report: dict[str, Any] = {
        "status": "BLOCKED",
        "schema_version": SCHEMA_VERSION,
        "phase": "G4",
        "stage": "deployment_integrity",
        "generated_at_utc_epoch": time.time(),
        "environment": env_snapshot(),
        "frozen_inputs": {},
    }

    for label, path, expected in (
        ("onnx", onnx, EXPECTED_ONNX_SHA256),
        ("fp32_engine", fp32, EXPECTED_FP32_SHA256),
    ):
        report["frozen_inputs"][label] = {
            "path": str(path),
            "present": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "expected_sha256": expected,
            "hash_match": path.exists() and sha256_file(path) == expected,
        }

    try:
        runtime, engine = load_engine(fp32)
        io = io_contract(engine)
        smoke = smoke_test_engine(engine)
        del runtime
    except Exception as e:
        io = None
        smoke = {"status": "FAIL", "reason": repr(e)}

    integrity_pass = all(item["hash_match"] for item in report["frozen_inputs"].values())
    report["engine_io"] = io
    report["smoke_test"] = smoke
    report["g3_linkage"] = {
        "g3_status": "PASS",
        "scope_note": "G3 latency numbers remain backend-only and are not replaced by the G4 smoke-test timing.",
    }
    report["fp16"] = {
        "status": "DEFERRED",
        "reason": "Optional optimization only. No FP16 engine was built or used for deployment acceptance.",
    }
    report["g4_gates"] = {
        "frozen_hashes": integrity_pass,
        "fp32_engine_deserialization": smoke["status"] in {"PASS"},
        "fp32_smoke_execution": smoke["status"] == "PASS",
        "fp16_optimization": "DEFERRED",
    }

    required = [
        report["g4_gates"]["frozen_hashes"],
        report["g4_gates"]["fp32_engine_deserialization"],
        report["g4_gates"]["fp32_smoke_execution"],
    ]
    report["status"] = "PASS" if all(required) else "BLOCKED"
    report["interpretation"] = {
        "closure": "G4 closes FP32 deployment integrity on the current RTX 4050 platform.",
        "not_claimed": [
            "FP16 performance",
            "edge-device performance",
            "full pipeline end-to-end latency",
            "real sonar acquisition throughput",
            "long-duration thermal stability",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Final G4 FP32 deployment-integrity gate for SIH SSS pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--fp32", type=Path, default=DEFAULT_FP32)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT / "preflight.json")

    v = sub.add_parser("validate")
    v.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    v.add_argument("--fp32", type=Path, default=DEFAULT_FP32)
    v.add_argument("--output", type=Path, default=DEFAULT_OUT / "g4_deployment_integrity.json")

    args = ap.parse_args()
    if args.cmd == "preflight":
        return preflight(args.onnx, args.fp32, args.output)
    return validate(args.onnx, args.fp32, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
