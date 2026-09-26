from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
EXPECTED_PHASE_D_SHA256 = "276f91a207df2563a09d4a05aca8f47b8899dace2071703650acae8cb4eb891e"
EXPECTED_PHASE_G1_ONNX_SHA256 = "17db4dee6147d1eb05867cda7ca503b1133e3b6e83a45cb7c0e359cf9f78f12b"
EXPECTED_PHASE_G2_TRT_SHA256 = "5e2bfabaca634873b61242c1817323d51cdb002d63831cee1a2874fd60818067"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def percentile_linear(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("latencies_ms must not be empty")
    total_ms = sum(latencies_ms)
    return {
        "count": float(len(latencies_ms)),
        "min_ms": min(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p50_ms": percentile_linear(latencies_ms, 50),
        "p95_ms": percentile_linear(latencies_ms, 95),
        "p99_ms": percentile_linear(latencies_ms, 99),
        "max_ms": max(latencies_ms),
        "total_ms": total_ms,
        "throughput_fps": len(latencies_ms) / (total_ms / 1000.0),
        "p50_fps": 1000.0 / percentile_linear(latencies_ms, 50),
        "p95_fps": 1000.0 / percentile_linear(latencies_ms, 95),
        "p99_fps": 1000.0 / percentile_linear(latencies_ms, 99),
    }


def import_status(name: str) -> dict[str, Any]:
    try:
        module = __import__(name)
        return {"present": True, "version": getattr(module, "__version__", "installed")}
    except Exception as exc:
        return {"present": False, "version": None, "error": str(exc)}


def command_output(command: str, args: list[str] | None = None, timeout: int = 15) -> dict[str, Any]:
    exe = shutil.which(command)
    if not exe:
        return {"found": False, "path": None, "stdout": None, "stderr": None, "returncode": None}
    argv = [exe] + (args or ["--version"])
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "found": proc.returncode == 0,
            "path": exe,
            "stdout": (proc.stdout or "").strip()[-4000:],
            "stderr": (proc.stderr or "").strip()[-4000:],
            "returncode": proc.returncode,
        }
    except Exception as exc:
        return {"found": True, "path": exe, "stdout": None, "stderr": None, "returncode": None, "error": str(exc)}


def environment_snapshot() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "phase_g3_schema": SCHEMA_VERSION,
        "onnx": import_status("onnx"),
        "onnxruntime": import_status("onnxruntime"),
        "tensorrt": import_status("tensorrt"),
        "torch": import_status("torch"),
        "cv2": import_status("cv2"),
        "psutil": import_status("psutil"),
        "nvidia_smi": command_output("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_cuda_version"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
            env["cuda_capability"] = list(torch.cuda.get_device_capability(0))
            env["cuda_total_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
    except Exception as exc:
        env["torch_error"] = str(exc)
    try:
        import tensorrt as trt
        env["tensorrt_version"] = trt.__version__
    except Exception:
        env["tensorrt_version"] = None
    try:
        import onnxruntime as ort
        env["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:
        env["onnxruntime_providers"] = []
    return env


def read_nvidia_smi_memory() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found"}
    query = [
        exe,
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
        if proc.returncode != 0:
            return {"available": False, "reason": (proc.stderr or proc.stdout).strip()[-2000:]}
        rows = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            row = {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
                "utilization_gpu_percent": float(parts[4]),
                "temperature_c": float(parts[5]),
            }
            rows.append(row)
        return {"available": True, "gpus": rows}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def select_reference_images(images_dir: Path, count: int, seed: int) -> list[Path]:
    if count <= 0:
        raise ValueError("image_count must be > 0")
    files = sorted([*images_dir.glob("*.jpg"), *images_dir.glob("*.jpeg"), *images_dir.glob("*.png")])
    if not files:
        raise FileNotFoundError(f"No supported images found in {images_dir}")
    import random

    rng = random.Random(seed)
    if count >= len(files):
        rng.shuffle(files)
        return files
    return rng.sample(files, count)


def letterbox_bgr(im: Any, new_shape: int = 640) -> Any:
    import cv2
    import numpy as np

    if im is None or im.size == 0:
        raise ValueError("Invalid image")
    if isinstance(new_shape, int):
        target_h = target_w = new_shape
    else:
        target_h, target_w = new_shape

    shape = im.shape[:2]
    r = min(target_h / shape[0], target_w / shape[1])
    new_unpad = (round(shape[1] * r), round(shape[0] * r))
    dw = target_w - new_unpad[0]
    dh = target_h - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def load_inputs(paths: list[Path], imgsz: int) -> list[Any]:
    import cv2
    import numpy as np

    inputs: list[Any] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        image = letterbox_bgr(image, imgsz)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        inputs.append(arr[None, ...])
    return inputs


def process_rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def cuda_memory_snapshot() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return {
            "available": True,
            "free_mb": round(free_bytes / (1024 ** 2), 2),
            "total_mb": round(total_bytes / (1024 ** 2), 2),
            "torch_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
            "torch_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def nvidia_process_snapshot() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found"}
    query = [
        exe,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
        if proc.returncode != 0:
            return {"available": False, "reason": (proc.stderr or proc.stdout).strip()[-2000:]}
        current_pid = str(os.getpid())
        rows = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mb": float(parts[2])})
        current = [x for x in rows if x["pid"] == current_pid]
        return {"available": True, "current_pid": current_pid, "processes": rows, "current_process": current}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def benchmark_onnx(
    onnx_path: Path,
    inputs: list[Any],
    warmup: int,
    iterations: int,
    sample_every: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    rss_before = process_rss_mb()
    gpu_before = cuda_memory_snapshot()
    smi_before = read_nvidia_smi_memory()
    proc_before = nvidia_process_snapshot()

    init_t0 = time.perf_counter_ns()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    init_ms = (time.perf_counter_ns() - init_t0) / 1e6
    input_name = session.get_inputs()[0].name

    cold_t0 = time.perf_counter_ns()
    session.run(None, {input_name: inputs[0]})
    cold_ms = (time.perf_counter_ns() - cold_t0) / 1e6

    for i in range(warmup):
        session.run(None, {input_name: inputs[i % len(inputs)]})

    rss_after_warmup = process_rss_mb()
    gpu_after_warmup = cuda_memory_snapshot()
    smi_after_warmup = read_nvidia_smi_memory()
    proc_after_warmup = nvidia_process_snapshot()

    latencies = []
    rss_samples = []
    for i in range(iterations):
        x = inputs[i % len(inputs)]
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: x})
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        latencies.append(elapsed_ms)
        if sample_every > 0 and ((i + 1) % sample_every == 0 or i == iterations - 1):
            rss = process_rss_mb()
            if rss is not None:
                rss_samples.append(rss)

    rss_after = process_rss_mb()
    gpu_after = cuda_memory_snapshot()
    smi_after = read_nvidia_smi_memory()
    proc_after = nvidia_process_snapshot()

    stats = summarize_latencies(latencies)
    return {
        "backend": "onnxruntime_cpu",
        "runtime": {"onnxruntime_version": ort.__version__, "providers": session.get_providers()},
        "initialization_ms": init_ms,
        "cold_first_inference_ms": cold_ms,
        "warmup_iterations": warmup,
        "benchmark_iterations": iterations,
        "input_count": len(inputs),
        "latency_ms": stats,
        "rss_mb": {"before_init": rss_before, "after_warmup": rss_after_warmup, "after_benchmark": rss_after, "peak_sampled": max(rss_samples) if rss_samples else None},
        "gpu_memory": {"before_init": gpu_before, "after_warmup": gpu_after, "after_benchmark": gpu_after},
        "nvidia_smi": {"gpu_before": smi_before, "gpu_after_warmup": smi_after_warmup, "gpu_after_benchmark": smi_after},
        "nvidia_process": {"before_init": proc_before, "after_warmup": proc_after_warmup, "after_benchmark": proc_after},
        "timing_scope": "backend inference only; image decode/preprocessing, file I/O, and output postprocessing excluded",
    }


def benchmark_tensorrt(
    engine_path: Path,
    inputs: list[Any],
    warmup: int,
    iterations: int,
    sample_every: int,
) -> dict[str, Any]:
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for TensorRT benchmark.")

    rss_before = process_rss_mb()
    gpu_before = cuda_memory_snapshot()
    smi_before = read_nvidia_smi_memory()
    proc_before = nvidia_process_snapshot()

    init_t0 = time.perf_counter_ns()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("TensorRT engine deserialization failed.")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT execution context creation failed.")

    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name
    if input_name is None or output_name is None:
        raise RuntimeError("Could not resolve TensorRT input/output tensor names.")

    input_shape = tuple(engine.get_tensor_shape(input_name))
    output_shape = tuple(engine.get_tensor_shape(output_name))
    if any(dim < 0 for dim in input_shape + output_shape):
        raise RuntimeError(f"Dynamic TensorRT shapes are unsupported in G3 fixed-shape benchmark: {input_shape}, {output_shape}")

    device_inputs = [torch.from_numpy(x).to(device="cuda", non_blocking=True) for x in inputs]
    output = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    input_dtype = str(engine.get_tensor_dtype(input_name))
    output_dtype = str(engine.get_tensor_dtype(output_name))
    if input_dtype != "DataType.FLOAT" or output_dtype != "DataType.FLOAT":
        raise RuntimeError(f"G3 benchmark expects FP32 engine, got input={input_dtype}, output={output_dtype}")
    init_ms = (time.perf_counter_ns() - init_t0) / 1e6

    context.set_tensor_address(input_name, int(device_inputs[0].data_ptr()))
    context.set_tensor_address(output_name, int(output.data_ptr()))
    cold_t0 = time.perf_counter_ns()
    ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT cold execute_async_v3 returned False.")
    stream.synchronize()
    cold_ms = (time.perf_counter_ns() - cold_t0) / 1e6

    for i in range(warmup):
        x = device_inputs[i % len(device_inputs)]
        context.set_tensor_address(input_name, int(x.data_ptr()))
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT warmup execution failed at iteration {i}.")
    stream.synchronize()

    rss_after_warmup = process_rss_mb()
    gpu_after_warmup = cuda_memory_snapshot()
    smi_after_warmup = read_nvidia_smi_memory()
    proc_after_warmup = nvidia_process_snapshot()

    latencies = []
    rss_samples = []
    for i in range(iterations):
        x = device_inputs[i % len(device_inputs)]
        context.set_tensor_address(input_name, int(x.data_ptr()))
        t0 = time.perf_counter_ns()
        ok = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"TensorRT benchmark execution failed at iteration {i}.")
        stream.synchronize()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        latencies.append(elapsed_ms)
        if sample_every > 0 and ((i + 1) % sample_every == 0 or i == iterations - 1):
            rss = process_rss_mb()
            if rss is not None:
                rss_samples.append(rss)

    rss_after = process_rss_mb()
    gpu_after = cuda_memory_snapshot()
    smi_after = read_nvidia_smi_memory()
    proc_after = nvidia_process_snapshot()

    stats = summarize_latencies(latencies)
    return {
        "backend": "tensorrt_gpu",
        "runtime": {"tensorrt_version": trt.__version__, "cuda_device": torch.cuda.get_device_name(0), "cuda_capability": list(torch.cuda.get_device_capability(0))},
        "engine": {"path": str(engine_path), "sha256": sha256_file(engine_path), "input": {"name": input_name, "shape": list(input_shape), "dtype": input_dtype}, "output": {"name": output_name, "shape": list(output_shape), "dtype": output_dtype}},
        "initialization_ms": init_ms,
        "cold_first_inference_ms": cold_ms,
        "warmup_iterations": warmup,
        "benchmark_iterations": iterations,
        "input_count": len(inputs),
        "latency_ms": stats,
        "rss_mb": {"before_init": rss_before, "after_warmup": rss_after_warmup, "after_benchmark": rss_after, "peak_sampled": max(rss_samples) if rss_samples else None},
        "gpu_memory": {"before_init": gpu_before, "after_warmup": gpu_after, "after_benchmark": gpu_after},
        "nvidia_smi": {"gpu_before": smi_before, "gpu_after_warmup": smi_after_warmup, "gpu_after_benchmark": smi_after},
        "nvidia_process": {"before_init": proc_before, "after_warmup": proc_after_warmup, "after_benchmark": proc_after},
        "timing_scope": "TensorRT execution only; image decode/preprocessing and host-to-device copy excluded from timed latency",
    }


def worker(args: argparse.Namespace) -> dict[str, Any]:
    env = environment_snapshot()
    image_paths = select_reference_images(args.images, args.image_count, args.seed)
    inputs = load_inputs(image_paths, args.imgsz)
    common = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "worker_pid": os.getpid(),
        "backend": args.backend,
        "env": env,
        "images": [str(x) for x in image_paths],
        "benchmark": {"imgsz": args.imgsz, "warmup": args.warmup, "iterations": args.iterations, "sample_every": args.sample_every, "seed": args.seed},
    }
    if args.backend == "onnx":
        require_file(args.onnx, "ONNX model")
        common["artifact"] = {"path": str(args.onnx), "sha256": sha256_file(args.onnx)}
        result = benchmark_onnx(args.onnx, inputs, args.warmup, args.iterations, args.sample_every)
    elif args.backend == "tensorrt":
        require_file(args.engine, "TensorRT engine")
        common["artifact"] = {"path": str(args.engine), "sha256": sha256_file(args.engine)}
        result = benchmark_tensorrt(args.engine, inputs, args.warmup, args.iterations, args.sample_every)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")
    common["result"] = result
    return common


def run_worker(script: Path, backend: str, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script),
        "worker",
        "--backend", backend,
        "--images", str(args.images),
        "--imgsz", str(args.imgsz),
        "--image-count", str(args.image_count),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--sample-every", str(args.sample_every),
        "--seed", str(args.seed),
        "--raw-output", str(output),
    ]
    if args.onnx:
        cmd.extend(["--onnx", str(args.onnx)])
    if args.engine:
        cmd.extend(["--engine", str(args.engine)])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr).strip()[-8000:]
        raise RuntimeError(f"G3 worker failed for {backend} (exit {proc.returncode}).\n{tail}")
    if not output.is_file():
        raise RuntimeError(f"G3 worker did not create output: {output}")
    return json.loads(output.read_text(encoding="utf-8"))


def compare_backends(onnx_result: dict[str, Any], trt_result: dict[str, Any]) -> dict[str, Any]:
    ort = onnx_result["result"]["latency_ms"]
    trt = trt_result["result"]["latency_ms"]
    ort_p50 = ort["p50_ms"]
    trt_p50 = trt["p50_ms"]
    return {
        "p50_speedup_x": ort_p50 / trt_p50 if trt_p50 > 0 else None,
        "p95_speedup_x": ort["p95_ms"] / trt["p95_ms"] if trt["p95_ms"] > 0 else None,
        "p99_speedup_x": ort["p99_ms"] / trt["p99_ms"] if trt["p99_ms"] > 0 else None,
        "throughput_speedup_x": trt["throughput_fps"] / ort["throughput_fps"] if ort["throughput_fps"] > 0 else None,
        "note": "Speedup compares ONNX Runtime CPU against TensorRT GPU. This is a heterogeneous backend/device comparison, not CPU-vs-CPU or GPU-vs-GPU equivalence.",
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    require_file(args.onnx, "ONNX model")
    require_file(args.engine, "TensorRT engine")
    if args.expected_onnx_sha256 and sha256_file(args.onnx).lower() != args.expected_onnx_sha256.lower():
        raise RuntimeError("ONNX SHA256 mismatch against frozen G1 artifact.")
    if args.expected_engine_sha256 and sha256_file(args.engine).lower() != args.expected_engine_sha256.lower():
        raise RuntimeError("TensorRT engine SHA256 mismatch against frozen G2 artifact.")
    if args.image_count <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("image_count > 0, warmup >= 0, iterations > 0 required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ort_path = args.output_dir / "worker_onnx.json"
    trt_path = args.output_dir / "worker_tensorrt.json"
    ort = run_worker(Path(__file__), "onnx", args, ort_path)
    trt = run_worker(Path(__file__), "tensorrt", args, trt_path)

    result = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "phase": "G3",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {
            "onnx": {"path": str(args.onnx), "sha256": sha256_file(args.onnx)},
            "tensorrt_engine": {"path": str(args.engine), "sha256": sha256_file(args.engine)},
        },
        "benchmark_config": {
            "images": str(args.images),
            "image_count": args.image_count,
            "imgsz": args.imgsz,
            "warmup_iterations": args.warmup,
            "benchmark_iterations": args.iterations,
            "rss_sample_every": args.sample_every,
            "seed": args.seed,
            "measurement_scope": "backend inference only, with separate cold initialization and first-inference timings",
        },
        "onnxruntime_cpu": ort,
        "tensorrt_gpu": trt,
        "comparison": compare_backends(ort, trt),
        "interpretation": {
            "not_measured": [
                "image decode/file I/O",
                "SSS preprocessing/normalization",
                "host-to-device input copy in timed latency",
                "Ultralytics/NMS/postprocessing",
                "FP16 TensorRT performance",
                "edge-device performance",
                "thermal throttling over prolonged mission duration",
            ],
            "recommendation": "Run the same command twice if the first run shows unusually high system load or GPU activity from other applications, then report the cleaner run with no hidden cherry-picking.",
        },
    }
    (args.output_dir / "g3_runtime_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase G3 ONNX Runtime vs TensorRT performance benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--images", type=Path, required=True, help="Directory containing prepared validation/reference images")
    common.add_argument("--imgsz", type=int, default=640)
    common.add_argument("--image-count", type=int, default=32)
    common.add_argument("--warmup", type=int, default=50)
    common.add_argument("--iterations", type=int, default=200)
    common.add_argument("--sample-every", type=int, default=10)
    common.add_argument("--seed", type=int, default=26057)
    common.add_argument("--onnx", type=Path)
    common.add_argument("--engine", type=Path)

    p = sub.add_parser("preflight")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--expected-onnx-sha256", default=EXPECTED_PHASE_G1_ONNX_SHA256)
    p.add_argument("--expected-engine-sha256", default=EXPECTED_PHASE_G2_TRT_SHA256)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("benchmark", parents=[common])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-onnx-sha256", default=EXPECTED_PHASE_G1_ONNX_SHA256)
    p.add_argument("--expected-engine-sha256", default=EXPECTED_PHASE_G2_TRT_SHA256)

    p = sub.add_parser("worker", parents=[common])
    p.add_argument("--backend", choices=["onnx", "tensorrt"], required=True)
    p.add_argument("--raw-output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "preflight":
        require_file(args.onnx, "ONNX model")
        require_file(args.engine, "TensorRT engine")
        env = environment_snapshot()
        actual_onnx = sha256_file(args.onnx)
        actual_engine = sha256_file(args.engine)
        checks = {
            "onnx_sha256_match": actual_onnx.lower() == args.expected_onnx_sha256.lower(),
            "engine_sha256_match": actual_engine.lower() == args.expected_engine_sha256.lower(),
            "onnxruntime_present": env["onnxruntime"]["present"],
            "tensorrt_present": env["tensorrt"]["present"],
            "torch_cuda_available": env.get("cuda_available") is True,
            "psutil_present": env["psutil"]["present"],
            "cv2_present": env["cv2"]["present"],
            "nvidia_smi_available": env["nvidia_smi"]["found"],
        }
        status = "PASS" if all(checks.values()) else "BLOCKED"
        result = {
            "status": status,
            "schema_version": SCHEMA_VERSION,
            "phase": "G3",
            "checkpoint_hashes": {
                "onnx": {"path": str(args.onnx), "sha256": actual_onnx, "expected": args.expected_onnx_sha256},
                "tensorrt": {"path": str(args.engine), "sha256": actual_engine, "expected": args.expected_engine_sha256},
            },
            "environment": env,
            "checks": checks,
            "notes": [
                "G3 benchmarks the frozen FP32 TensorRT engine against ONNX Runtime CPU.",
                "The timed region excludes image decoding, preprocessing, host-to-device transfer, and postprocessing.",
                "A separate cold initialization time and first-inference time are recorded.",
            ],
        }
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0 if status == "PASS" else 2

    if args.command == "worker":
        if args.onnx is None and args.backend == "onnx":
            raise RuntimeError("--onnx is required for ONNX worker")
        if args.engine is None and args.backend == "tensorrt":
            raise RuntimeError("--engine is required for TensorRT worker")
        result = worker(args)
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["result"], indent=2))
        return 0

    result = aggregate(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
