from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {
    0: "pipe",
    1: "shipwreck",
    2: "mine",
    3: "ghost_net",
}

DEFAULT_THRESHOLDS = {
    "pipe": 0.74,
    "shipwreck": 0.38,
    "mine": 0.71,
    "ghost_net": 0.40,
}

VARIANTS = (
    "baseline",
    "lee",
    "clahe",
    "lee_clahe",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-only SSS preprocessing ablation")
    p.add_argument("--data", required=True, help="Dataset root containing val/images and val/labels")
    p.add_argument("--model", required=True, help="YOLO checkpoint")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--variants", default=",".join(VARIANTS), help="Comma-separated variants")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-images", type=int, default=0, help="0 = all validation images")
    return p.parse_args()


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def lee_filter(image: np.ndarray, kernel: int = 7, noise_factor: float = 1.0) -> np.ndarray:
    src = image.astype(np.float32)
    mean = cv2.boxFilter(src, -1, (kernel, kernel), normalize=True)
    sq_mean = cv2.boxFilter(src * src, -1, (kernel, kernel), normalize=True)
    local_var = np.maximum(sq_mean - mean * mean, 0.0)

    valid_var = local_var[np.isfinite(local_var)]
    noise_var = float(np.median(valid_var)) * float(noise_factor)
    if noise_var <= 1e-8:
        return image.copy()

    gain = np.maximum(local_var - noise_var, 0.0) / np.maximum(local_var, 1e-8)
    out = mean + gain * (src - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(image: np.ndarray) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return op.apply(image)


def transform(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "baseline":
        return image
    if variant == "lee":
        return lee_filter(image)
    if variant == "clahe":
        return clahe(image)
    if variant == "lee_clahe":
        return clahe(lee_filter(image))
    raise ValueError(f"Unknown variant: {variant}")


def label_boxes(path: Path, width: int, height: int) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return boxes

    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        if cls not in CLASS_NAMES:
            continue

        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append(
            {
                "class_id": cls,
                "class": CLASS_NAMES[cls],
                "bbox": [x1, y1, x2, y2],
                "line": line_no,
            }
        )
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def predict(model: YOLO, image: np.ndarray, args: argparse.Namespace) -> list[dict]:
    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        device=args.device,
        conf=0.001,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )[0]

    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, conf, cls in zip(boxes, confs, classes):
        cls = int(cls)
        if cls not in CLASS_NAMES:
            continue
        name = CLASS_NAMES[cls]
        if float(conf) < DEFAULT_THRESHOLDS[name]:
            continue
        out.append({
            "class_id": cls,
            "class": name,
            "confidence": float(conf),
            "bbox": [float(v) for v in box],
        })
    return out


def match(gt: list[dict], pred: list[dict], threshold: float) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if g["class_id"] != p["class_id"]:
                continue
            ov = iou(g["bbox"], p["bbox"])
            if ov >= threshold:
                candidates.append((ov, p["confidence"], gi, pi))

    candidates.sort(reverse=True)
    used_g = set()
    used_p = set()
    matches = []
    for ov, conf, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, ov))
    return matches, used_g, used_p


def metrics(records: list[dict]) -> dict:
    per_class = {}
    total_tp = total_fp = total_fn = 0
    matched_ious = []

    for cls_id, cls_name in CLASS_NAMES.items():
        tp = fp = fn = 0
        cls_ious = []
        for rec in records:
            gt = rec["gt"]
            pred = rec["pred"]
            gt_cls = [x for x in gt if x["class_id"] == cls_id]
            pred_cls = [x for x in pred if x["class_id"] == cls_id]
            m, used_g, used_p = match(gt_cls, pred_cls, 0.50)
            tp += len(m)
            fp += len(pred_cls) - len(used_p)
            fn += len(gt_cls) - len(used_g)
            cls_ious.extend(x[2] for x in m)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": float(np.mean(cls_ious)) if cls_ious else None,
            "matched": len(cls_ious),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        matched_ious.extend(cls_ious)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": float(np.mean(matched_ious)) if matched_ious else None,
        },
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [x for x in variants if x not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")

    data = Path(args.data).expanduser().resolve()
    image_dir = data / "val" / "images"
    label_dir = data / "val" / "labels"
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}])
    if args.max_images > 0:
        paths = paths[:args.max_images]

    model = YOLO(str(Path(args.model).expanduser().resolve()))
    summary = {
        "project": "WaveSentinel / DRISHTI-SSS",
        "purpose": "validation-only preprocessing ablation",
        "data": str(data),
        "model": str(Path(args.model).expanduser().resolve()),
        "validation_images": len(paths),
        "thresholds": DEFAULT_THRESHOLDS,
        "variants": {},
    }

    csv_rows = []
    for variant in variants:
        print(f"\n=== {variant.upper()} ===")
        records = []
        failures = []

        for idx, image_path in enumerate(paths, 1):
            try:
                raw = read_gray(image_path)
                processed = transform(raw, variant)
                gt_path = label_dir / f"{image_path.stem}.txt"
                gt = label_boxes(gt_path, raw.shape[1], raw.shape[0])
                pred = predict(model, processed, args)
                records.append({"image": image_path.name, "gt": gt, "pred": pred})
            except Exception as exc:
                failures.append({"image": image_path.name, "error": str(exc)})
            if idx % 50 == 0 or idx == len(paths):
                print(f"  {idx}/{len(paths)}")

        m = metrics(records)
        summary["variants"][variant] = {
            **m,
            "images_evaluated": len(records),
            "failures": failures,
        }

        print(json.dumps(m["overall"], indent=2))
        for cls_name, cm in m["per_class"].items():
            print(f"  {cls_name:12s} P={cm['precision']:.4f} R={cm['recall']:.4f} F1={cm['f1']:.4f} mIoU={cm['mean_iou']}")

        for cls_name, cm in m["per_class"].items():
            csv_rows.append({
                "variant": variant,
                "class": cls_name,
                "tp": cm["tp"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "precision": cm["precision"],
                "recall": cm["recall"],
                "f1": cm["f1"],
                "mean_iou": cm["mean_iou"],
            })
        csv_rows.append({
            "variant": variant,
            "class": "ALL",
            **m["overall"],
        })

    (out_dir / "ablation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "ablation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in csv_rows for k in row}))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved: {out_dir / 'ablation_report.json'}")
    print(f"Saved: {out_dir / 'ablation_metrics.csv'}")


if __name__ == "__main__":
    main()
