from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 1
DEFAULT_DEVICE = "0"
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.70
DEFAULT_MAX_DET = 300
EXPECTED_PHASE_D_SHA256 = "276f91a207df2563a09d4a05aca8f47b8899dace2071703650acae8cb4eb891e"


@dataclass(frozen=True)
class ExportConfig:
    imgsz: int = DEFAULT_IMGSZ
    batch: int = DEFAULT_BATCH
    device: str = DEFAULT_DEVICE
    half: bool = False
    dynamic: bool = False
    simplify: bool = False
    opset: int | None = None

    def validate(self) -> None:
        if self.imgsz <= 0:
            raise ValueError("imgsz must be > 0")
        if self.batch <= 0:
            raise ValueError("batch must be > 0")
        if self.opset is not None and self.opset < 11:
            raise ValueError("opset must be >= 11 when provided")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    for module_name, key in [("ultralytics", "ultralytics"), ("torch", "torch"), ("onnx", "onnx"), ("onnxruntime", "onnxruntime")]:
        try:
            module = __import__(module_name)
            info[key] = getattr(module, "__version__", "installed")
        except Exception:
            info[key] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda_available"] = None
        info["cuda_device"] = None
    return info


def preflight(checkpoint: Path, expected_sha256: str | None) -> dict[str, Any]:
    require_file(checkpoint, "checkpoint")
    actual_sha256 = sha256_file(checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_match": None if expected_sha256 is None else actual_sha256.lower() == expected_sha256.lower(),
        "environment": environment_info(),
    }
    missing = [k for k in ("ultralytics", "torch", "onnx", "onnxruntime") if result["environment"].get(k) is None]
    result["missing_modules"] = missing
    result["status"] = "PASS" if not missing and result["sha256_match"] is not False else "BLOCKED"
    return result


def export_onnx(checkpoint: Path, output_dir: Path, cfg: ExportConfig) -> Path:
    cfg.validate()
    require_file(checkpoint, "checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for ONNX export.") from exc

    model = YOLO(str(checkpoint))
    kwargs: dict[str, Any] = {
        "format": "onnx",
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "device": cfg.device,
        "half": cfg.half,
        "dynamic": cfg.dynamic,
        "simplify": cfg.simplify,
    }
    if cfg.opset is not None:
        kwargs["opset"] = cfg.opset

    exported = model.export(**kwargs)
    exported_path = Path(str(exported))
    if not exported_path.is_file():
        raise FileNotFoundError(f"Ultralytics reported export path but file was not found: {exported_path}")

    destination = output_dir / exported_path.name
    if exported_path.resolve() != destination.resolve():
        destination.write_bytes(exported_path.read_bytes())
    return destination


def validate_onnx(onnx_path: Path) -> dict[str, Any]:
    require_file(onnx_path, "ONNX model")
    try:
        import onnx
    except Exception as exc:
        raise RuntimeError("onnx package is required for graph validation.") from exc

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    graph_inputs = [
        {
            "name": x.name,
            "shape": [d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim],
        }
        for x in model.graph.input
    ]
    graph_outputs = [
        {
            "name": x.name,
            "shape": [d.dim_value if d.dim_value else d.dim_param for d in x.type.tensor_type.shape.dim],
        }
        for x in model.graph.output
    ]
    return {
        "status": "PASS",
        "onnx": str(onnx_path),
        "sha256": sha256_file(onnx_path),
        "opset": [int(x.version) for x in model.opset_import],
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
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


def _pair_predictions(pt: dict[str, Any], onnx: dict[str, Any]) -> dict[str, Any]:
    import math

    def iou(a: list[float], b: list[float]) -> float:
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

    used = set()
    pairs = []
    unmatched_pt = []
    for i, (box_a, cls_a, conf_a) in enumerate(zip(pt["boxes"], pt["classes"], pt["confidence"])):
        candidates = [j for j, (box_b, cls_b) in enumerate(zip(onnx["boxes"], onnx["classes"])) if j not in used and cls_b == cls_a]
        if not candidates:
            unmatched_pt.append(i)
            continue
        j = max(candidates, key=lambda k: iou(box_a, onnx["boxes"][k]))
        used.add(j)
        box_iou = iou(box_a, onnx["boxes"][j])
        conf_delta = abs(conf_a - onnx["confidence"][j])
        coord_delta = max(abs(x - y) for x, y in zip(box_a, onnx["boxes"][j]))
        pairs.append({"class_id": cls_a, "iou": box_iou, "confidence_abs_delta": conf_delta, "max_box_coord_abs_delta": coord_delta})

    unmatched_onnx = [j for j in range(len(onnx["boxes"])) if j not in used]
    mean_iou = sum(x["iou"] for x in pairs) / len(pairs) if pairs else 0.0
    mean_conf_delta = sum(x["confidence_abs_delta"] for x in pairs) / len(pairs) if pairs else 0.0
    max_coord_delta = max((x["max_box_coord_abs_delta"] for x in pairs), default=0.0)
    return {
        "pt_count": len(pt["boxes"]),
        "onnx_count": len(onnx["boxes"]),
        "paired_count": len(pairs),
        "unmatched_pt": len(unmatched_pt),
        "unmatched_onnx": len(unmatched_onnx),
        "mean_pair_iou": mean_iou,
        "mean_confidence_abs_delta": mean_conf_delta,
        "max_box_coord_abs_delta": max_coord_delta,
        "pairs": pairs,
        "numerically_stable": bool(
            len(unmatched_pt) == 0
            and len(unmatched_onnx) == 0
            and all(math.isfinite(v) for v in (mean_iou, mean_conf_delta, max_coord_delta))
        ),
    }


def compare_models(checkpoint: Path, onnx_path: Path, reference_image: Path, imgsz: int = DEFAULT_IMGSZ, device: str = DEFAULT_DEVICE) -> dict[str, Any]:
    require_file(checkpoint, "checkpoint")
    require_file(onnx_path, "ONNX model")
    require_file(reference_image, "reference image")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Ultralytics is required for PyTorch vs ONNX comparison.") from exc

    common = {
        "source": str(reference_image),
        "imgsz": imgsz,
        "conf": DEFAULT_CONF,
        "iou": DEFAULT_IOU,
        "max_det": DEFAULT_MAX_DET,
        "device": device,
        "verbose": False,
        "save": False,
    }
    pt_model = YOLO(str(checkpoint))
    onnx_model = YOLO(str(onnx_path))
    pt_result = pt_model.predict(**common)[0]
    onnx_result = onnx_model.predict(**common)[0]
    pt_payload = _prediction_payload(pt_result)
    onnx_payload = _prediction_payload(onnx_result)
    comparison = _pair_predictions(pt_payload, onnx_payload)
    return {
        "status": "PASS" if comparison["numerically_stable"] else "REVIEW",
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "reference_image": str(reference_image),
        "inference": {
            "imgsz": imgsz,
            "conf": DEFAULT_CONF,
            "iou": DEFAULT_IOU,
            "max_det": DEFAULT_MAX_DET,
            "device": device,
        },
        "comparison": comparison,
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase G1 ONNX export and validation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--expected-sha256", default=EXPECTED_PHASE_D_SHA256)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("export")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--half", action="store_true")
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--opset", type=int)
    p.add_argument("--expected-sha256", default=EXPECTED_PHASE_D_SHA256)

    p = sub.add_parser("validate")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--output", type=Path)

    p = sub.add_parser("compare")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--reference-image", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "preflight":
        result = preflight(args.checkpoint, args.expected_sha256)
    elif args.command == "export":
        actual = sha256_file(args.checkpoint)
        if actual.lower() != args.expected_sha256.lower():
            raise RuntimeError(f"Checkpoint SHA256 mismatch. Expected {args.expected_sha256}, got {actual}.")
        cfg = ExportConfig(
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
        )
        output = export_onnx(args.checkpoint, args.output_dir, cfg)
        result = {
            "status": "PASS",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": actual,
            "onnx": str(output),
            "onnx_sha256": sha256_file(output),
            "export_config": asdict(cfg),
        }
    elif args.command == "validate":
        result = validate_onnx(args.onnx)
    else:
        result = compare_models(args.checkpoint, args.onnx, args.reference_image, args.imgsz, args.device)

    print(json.dumps(result, indent=2))
    if getattr(args, "output", None):
        write_json(result, args.output)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
