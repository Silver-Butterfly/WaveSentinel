from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CLASS_NAMES = {0: "pipe", 1: "shipwreck", 2: "mine", 3: "ghost_net"}

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_MODEL = r"K:\Debris model\runs\drishti_v5_ghost2x\weights\best.pt"
DEFAULT_DATA = r"D:\DATASETS\DRISHTI-SSS-TRAIN"
DEFAULT_OUTPUT = r"K:\Debris model\runs\threshold_calibration\ghost2x_val"

PRED_FLOOR = 0.001
MATCH_IOU = 0.50
NMS_IOU = 0.70
IMGSZ = 640
BATCH = 16
MAX_DET = 300
MIN_THR = 0.05
MAX_THR = 0.95
STEP = 0.01


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def read_gt(label_path, w, h):
    out = []
    if not label_path.exists():
        return out
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, bw, bh = map(float, p[1:])
        except ValueError:
            continue
        if c not in CLASS_NAMES:
            continue
        box = [
            max(0.0, min(float(w), (cx - bw / 2) * w)),
            max(0.0, min(float(h), (cy - bh / 2) * h)),
            max(0.0, min(float(w), (cx + bw / 2) * w)),
            max(0.0, min(float(h), (cy + bh / 2) * h)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        out.append({"class_id": c, "bbox": box, "line": line_no})
    return out


def greedy_match(gt, preds, threshold):
    candidates = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            if g["class_id"] != p["class_id"]:
                continue
            s = iou(g["bbox"], p["bbox"])
            if s >= threshold:
                candidates.append((s, gi, pi))
    candidates.sort(reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for s, gi, pi in candidates:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            pairs.append((gi, pi, s))
    return pairs


def evaluate(images, ground_truths, predictions, class_id, threshold, match_iou):
    gt_n = pred_n = tp = 0
    ious = []
    tp_confs = []

    for key in images:
        gt = [x for x in ground_truths[key] if x["class_id"] == class_id]
        pr = [x for x in predictions[key]
              if x["class_id"] == class_id and x["confidence"] >= threshold]

        matches = greedy_match(gt, pr, match_iou)

        gt_n += len(gt)
        pred_n += len(pr)
        tp += len(matches)
        ious.extend(s for _, _, s in matches)
        tp_confs.extend(pr[pi]["confidence"] for _, pi, _ in matches)

    fp = pred_n - tp
    fn = gt_n - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": round(threshold, 4),
        "ground_truth": gt_n,
        "predictions": pred_n,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_matched_iou": round(float(np.mean(ious)), 6) if ious else None,
        "median_matched_iou": round(float(np.median(ious)), 6) if ious else None,
        "mean_tp_confidence": round(float(np.mean(tp_confs)), 6) if tp_confs else None,
    }


def threshold_grid():
    return [round(x, 4) for x in np.arange(MIN_THR, MAX_THR + STEP / 2, STEP)]


def main():
    ap = argparse.ArgumentParser(description="Validation-only class-wise threshold calibration.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--prediction-floor", type=float, default=PRED_FLOOR)
    ap.add_argument("--match-iou", type=float, default=MATCH_IOU)
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU)
    ap.add_argument("--max-det", type=int, default=MAX_DET)
    ap.add_argument("--min-threshold", type=float, default=MIN_THR)
    ap.add_argument("--max-threshold", type=float, default=MAX_THR)
    ap.add_argument("--step", type=float, default=STEP)
    args = ap.parse_args()

    if args.split == "test":
        print("WARNING: test split selected. For proper calibration, use --split val.")

    model_path = Path(args.model).expanduser().resolve()
    root = Path(args.data).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    images_dir = root / args.split / "images"
    labels_dir = root / args.split / "labels"

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweeps").mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
    if not paths:
        raise RuntimeError(f"No images found in {images_dir}")

    started = utc_now()
    model = YOLO(str(model_path))

    gts = {}
    preds = {}
    missing_labels = []

    print("=" * 72)
    print("DRISHTI-SSS VALIDATION THRESHOLD CALIBRATION")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(f"Split: {args.split}")
    print(f"Images: {len(paths)}")
    print(f"Prediction floor: {args.prediction_floor}")
    print(f"Match IoU: {args.match_iou}")
    print()

    for start in range(0, len(paths), args.batch):
        batch_paths = paths[start:start + args.batch]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz,
            device=args.device,
            conf=args.prediction_floor,
            iou=args.nms_iou,
            max_det=args.max_det,
            batch=args.batch,
            verbose=False,
        )

        for path, result in zip(batch_paths, results):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Could not read {path}")
            h, w = image.shape[:2]

            label_path = labels_dir / f"{path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(str(label_path))

            key = str(path)
            gts[key] = read_gt(label_path, w, h)
            preds[key] = []

            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, c in zip(boxes, confs, classes):
                    if int(c) not in CLASS_NAMES:
                        continue
                    preds[key].append({
                        "class_id": int(c),
                        "confidence": float(conf),
                        "bbox": [float(v) for v in box],
                    })

        print(f"  {min(start + args.batch, len(paths))}/{len(paths)}")

    if missing_labels:
        raise RuntimeError("Missing labels found. Calibration aborted.")

    all_images = list(gts.keys())
    thresholds = [round(float(x), 4)
                  for x in np.arange(args.min_threshold,
                                     args.max_threshold + args.step / 2,
                                     args.step)]

    sweeps = {}
    selected = {}

    for class_id, class_name in CLASS_NAMES.items():
        rows = []
        for t in thresholds:
            rows.append(evaluate(
                all_images, gts, preds, class_id, t, args.match_iou
            ))

        best = max(
            rows,
            key=lambda r: (r["f1"], r["recall"], r["precision"], r["threshold"])
        )
        sweeps[class_name] = rows
        selected[class_name] = best

        with (out / "sweeps" / f"{class_name}_sweep.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"{class_name:12s} -> threshold={best['threshold']:.2f} "
            f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}"
        )

    calibrated = {
        name: selected[name]["threshold"] for name in CLASS_NAMES.values()
    }

    overall_gt = sum(selected[n]["ground_truth"] for n in CLASS_NAMES.values())
    overall_tp = sum(selected[n]["true_positives"] for n in CLASS_NAMES.values())
    overall_fp = sum(selected[n]["false_positives"] for n in CLASS_NAMES.values())
    overall_fn = sum(selected[n]["false_negatives"] for n in CLASS_NAMES.values())

    overall_p = overall_tp / (overall_tp + overall_fp) if overall_tp + overall_fp else 0.0
    overall_r = overall_tp / (overall_tp + overall_fn) if overall_tp + overall_fn else 0.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if overall_p + overall_r else 0.0
    )

    report = {
        "project": "AI-Powered Underwater Marine Debris & Anomaly Detection System",
        "purpose": "Validation-only class-wise confidence threshold calibration.",
        "started_at": started,
        "finished_at": utc_now(),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "data": {
            "root": str(root),
            "split": args.split,
            "images": len(paths),
            "ground_truth_objects": sum(len(v) for v in gts.values()),
        },
        "configuration": {
            "imgsz": args.imgsz,
            "device": args.device,
            "batch": args.batch,
            "prediction_floor": args.prediction_floor,
            "nms_iou": args.nms_iou,
            "match_iou": args.match_iou,
            "max_det": args.max_det,
            "threshold_range": [
                args.min_threshold,
                args.max_threshold,
                args.step,
            ],
        },
        "selection_rule": (
            "Select the class-wise threshold maximizing validation F1. "
            "Ties prefer higher recall, then precision, then higher threshold."
        ),
        "calibrated_thresholds": calibrated,
        "selected_validation_metrics": {
            "overall": {
                "ground_truth": overall_gt,
                "true_positives": overall_tp,
                "false_positives": overall_fp,
                "false_negatives": overall_fn,
                "precision": round(overall_p, 6),
                "recall": round(overall_r, 6),
                "f1": round(overall_f1, 6),
            },
            "classes": selected,
        },
        "sweeps": sweeps,
        "missing_label_files": missing_labels,
        "warning": (
            "Calibration must use validation data. After thresholds are frozen, "
            "evaluate the sealed test split without further threshold tuning."
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "ultralytics": getattr(__import__("ultralytics"), "__version__", "unknown"),
        },
    }

    (out / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out / "calibrated_thresholds.json").write_text(
        json.dumps({
            "model": str(model_path),
            "split": args.split,
            "match_iou": args.match_iou,
            "thresholds": calibrated,
        }, indent=2),
        encoding="utf-8",
    )

    txt = [
        "DRISHTI-SSS VALIDATION THRESHOLD CALIBRATION",
        "=" * 72,
        f"Model: {model_path}",
        f"Split: {args.split}",
        f"Images: {len(paths)}",
        f"GT objects: {sum(len(v) for v in gts.values())}",
        f"Match IoU: {args.match_iou:.2f}",
        "",
        "CALIBRATED THRESHOLDS",
        "-" * 72,
    ]

    for name in CLASS_NAMES.values():
        row = selected[name]
        txt.append(
            f"{name:12s} threshold={row['threshold']:.2f} "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f}"
        )

    txt.extend([
        "",
        "OVERALL AT SELECTED THRESHOLDS",
        "-" * 72,
        f"Precision: {overall_p:.4f}",
        f"Recall:    {overall_r:.4f}",
        f"F1:        {overall_f1:.4f}",
        "",
        "Freeze these thresholds, then run the sealed test set.",
    ])

    (out / "calibration_report.txt").write_text(
        "\n".join(txt),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("CALIBRATION COMPLETE")
    print("=" * 72)
    print("Thresholds:", json.dumps(calibrated))
    print(f"Validation P/R/F1: {overall_p:.4f} / {overall_r:.4f} / {overall_f1:.4f}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
