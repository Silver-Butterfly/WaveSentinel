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
import ultralytics
from ultralytics import YOLO


CLASS_NAMES = {0: "pipe", 1: "shipwreck", 2: "mine", 3: "ghost_net"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_MODEL = (
    r"K:\Debris model\runs\drishti_phase_d_robust_v2_hn"
    r"\weights\best.pt"
)
DEFAULT_DATA = r"D:\DATASETS\DRISHTI-SSS-ROBUST-V2-HN"
DEFAULT_OUTPUT = Path(r"K:\Debris model\runs\drishti_phase_d_robust_v2_hn") / "threshold_calibration"

IMGSZ = 640
BATCH = 16
DEVICE = "0"
PRED_FLOOR = 0.001
NMS_IOU = 0.70
MATCH_IOU = 0.50
MAX_DET = 300
MIN_THR = 0.05
MAX_THR = 0.95
STEP = 0.01
EXPECTED_VAL_IMAGES = 780


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def read_gt(label_path: Path, width: int, height: int):
    out = []
    if not label_path.exists():
        raise FileNotFoundError(f"Missing validation label: {label_path}")

    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        p = line.split()
        if len(p) != 5:
            continue

        try:
            class_id = int(float(p[0]))
            cx, cy, bw, bh = map(float, p[1:])
        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        box = [
            max(0.0, min(float(width), (cx - bw / 2) * width)),
            max(0.0, min(float(height), (cy - bh / 2) * height)),
            max(0.0, min(float(width), (cx + bw / 2) * width)),
            max(0.0, min(float(height), (cy + bh / 2) * height)),
        ]

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        out.append(
            {
                "class_id": class_id,
                "bbox": box,
                "line": line_no,
            }
        )

    return out


def greedy_match(gt, preds, match_iou):
    candidates = []

    for gi, g in enumerate(gt):
        for pi, p in enumerate(preds):
            if g["class_id"] != p["class_id"]:
                continue

            score = iou(g["bbox"], p["bbox"])
            if score >= match_iou:
                candidates.append((score, gi, pi))

    candidates.sort(reverse=True)

    used_gt = set()
    used_pred = set()
    pairs = []

    for score, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi, score))

    return pairs


def evaluate_class(
    keys,
    ground_truths,
    predictions,
    class_id,
    threshold,
    match_iou,
):
    gt_count = 0
    pred_count = 0
    tp = 0
    matched_ious = []
    tp_conf = []

    for key in keys:
        gt = [
            item
            for item in ground_truths[key]
            if item["class_id"] == class_id
        ]

        preds = [
            item
            for item in predictions[key]
            if item["class_id"] == class_id
            and item["confidence"] >= threshold
        ]

        matches = greedy_match(gt, preds, match_iou)

        gt_count += len(gt)
        pred_count += len(preds)
        tp += len(matches)

        for _, pred_index, score in matches:
            matched_ious.append(score)
            tp_conf.append(preds[pred_index]["confidence"])

    fp = pred_count - tp
    fn = gt_count - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": round(threshold, 4),
        "ground_truth": gt_count,
        "predictions": pred_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            float(np.mean(matched_ious)) if matched_ious else None
        ),
        "median_matched_iou": (
            float(np.median(matched_ious)) if matched_ious else None
        ),
        "mean_tp_confidence": (
            float(np.mean(tp_conf)) if tp_conf else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase D validation-only class-wise threshold calibration."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--prediction-floor", type=float, default=PRED_FLOOR)
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU)
    parser.add_argument("--match-iou", type=float, default=MATCH_IOU)
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--min-threshold", type=float, default=MIN_THR)
    parser.add_argument("--max-threshold", type=float, default=MAX_THR)
    parser.add_argument("--step", type=float, default=STEP)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_root = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    val_images = data_root / "val" / "images"
    val_labels = data_root / "val" / "labels"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not val_images.is_dir():
        raise FileNotFoundError(val_images)

    if not val_labels.is_dir():
        raise FileNotFoundError(val_labels)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output}"
        )

    image_paths = sorted(
        p for p in val_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if len(image_paths) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL_IMAGES} validation images, "
            f"found {len(image_paths)}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    output.mkdir(parents=True, exist_ok=True)
    (output / "sweeps").mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PHASE D | VALIDATION-ONLY THRESHOLD CALIBRATION")
    print("=" * 78)
    print(f"Model        : {model_path}")
    print(f"Model SHA256 : {sha256_file(model_path)}")
    print(f"Validation   : {data_root / 'val'}")
    print(f"Images       : {len(image_paths)}")
    print(f"GPU          : {torch.cuda.get_device_name(int(args.device))}")
    print(f"Prediction floor: {args.prediction_floor}")
    print(f"Match IoU       : {args.match_iou}")
    print(f"Threshold range : {args.min_threshold}-{args.max_threshold}")
    print("SEALED TEST    : NOT TOUCHED")
    print("=" * 78)

    model = YOLO(str(model_path))

    ground_truths = {}
    predictions = {}

    for start in range(0, len(image_paths), args.batch):
        batch_paths = image_paths[start:start + args.batch]

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
                raise RuntimeError(f"Could not read image: {path}")

            height, width = image.shape[:2]
            key = str(path)

            ground_truths[key] = read_gt(
                val_labels / f"{path.stem}.txt",
                width,
                height,
            )

            predictions[key] = []

            if result.boxes is None or not len(result.boxes):
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confs, classes):
                class_id = int(class_id)

                if class_id not in CLASS_NAMES:
                    continue

                predictions[key].append(
                    {
                        "class_id": class_id,
                        "confidence": float(confidence),
                        "bbox": [float(v) for v in box],
                    }
                )

        print(
            f"Prediction pass: "
            f"{min(start + args.batch, len(image_paths))}/{len(image_paths)}"
        )

    keys = list(ground_truths.keys())
    grid = [
        round(float(x), 4)
        for x in np.arange(
            args.min_threshold,
            args.max_threshold + args.step / 2,
            args.step,
        )
    ]

    selected = {}
    sweeps = {}

    for class_id, class_name in CLASS_NAMES.items():
        rows = []

        for threshold in grid:
            rows.append(
                evaluate_class(
                    keys,
                    ground_truths,
                    predictions,
                    class_id,
                    threshold,
                    args.match_iou,
                )
            )

        best = max(
            rows,
            key=lambda row: (
                row["f1"],
                row["recall"],
                row["precision"],
                row["threshold"],
            ),
        )

        selected[class_name] = best
        sweeps[class_name] = rows

        sweep_path = output / "sweeps" / f"{class_name}_sweep.csv"

        with sweep_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"{class_name:12s} "
            f"threshold={best['threshold']:.2f} "
            f"P={best['precision']:.4f} "
            f"R={best['recall']:.4f} "
            f"F1={best['f1']:.4f}"
        )

    thresholds = {
        name: selected[name]["threshold"]
        for name in CLASS_NAMES.values()
    }

    overall_gt = sum(
        selected[name]["ground_truth"] for name in CLASS_NAMES.values()
    )
    overall_tp = sum(
        selected[name]["true_positives"] for name in CLASS_NAMES.values()
    )
    overall_fp = sum(
        selected[name]["false_positives"] for name in CLASS_NAMES.values()
    )
    overall_fn = sum(
        selected[name]["false_negatives"] for name in CLASS_NAMES.values()
    )

    overall_precision = (
        overall_tp / (overall_tp + overall_fp)
        if overall_tp + overall_fp
        else 0.0
    )
    overall_recall = (
        overall_tp / (overall_tp + overall_fn)
        if overall_tp + overall_fn
        else 0.0
    )
    overall_f1 = (
        2.0 * overall_precision * overall_recall
        / (overall_precision + overall_recall)
        if overall_precision + overall_recall
        else 0.0
    )

    report = {
        "phase": "D",
        "stage": "validation_threshold_calibration",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "dataset": str(data_root),
        "split": "val",
        "validation_images": len(image_paths),
        "sealed_test_touched": False,
        "configuration": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
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
            "Maximize class-wise validation F1; ties prefer higher recall, "
            "then precision, then higher threshold."
        ),
        "calibrated_thresholds": thresholds,
        "selected_validation_metrics": {
            "overall": {
                "ground_truth": overall_gt,
                "true_positives": overall_tp,
                "false_positives": overall_fp,
                "false_negatives": overall_fn,
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
            },
            "classes": selected,
        },
        "sweeps": sweeps,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(int(args.device)),
        },
        "generated_at": utc_now(),
    }

    (output / "calibration_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    (output / "calibrated_thresholds.json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "model_sha256": sha256_file(model_path),
                "split": "val",
                "match_iou": args.match_iou,
                "thresholds": thresholds,
                "sealed_test_touched": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = [
        "PHASE D | VALIDATION THRESHOLD CALIBRATION",
        "=" * 78,
        f"Model: {model_path}",
        f"Validation images: {len(image_paths)}",
        "",
        "CALIBRATED THRESHOLDS",
        "-" * 78,
    ]

    for name in CLASS_NAMES.values():
        row = selected[name]
        summary.append(
            f"{name:12s} threshold={row['threshold']:.2f} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f}"
        )

    summary.extend(
        [
            "",
            "OVERALL AT SELECTED THRESHOLDS",
            "-" * 78,
            f"Precision: {overall_precision:.4f}",
            f"Recall:    {overall_recall:.4f}",
            f"F1:        {overall_f1:.4f}",
            "",
            "Thresholds are frozen for the next evaluation stage.",
            "SEALED TEST: NOT TOUCHED",
        ]
    )

    (output / "calibration_report.txt").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
