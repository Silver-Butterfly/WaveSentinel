from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CLASS_NAMES = {
    0: "pipe",
    1: "shipwreck",
    2: "mine",
    3: "ghost_net",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

DEFAULT_MODEL = r"K:\Debris model\runs\drishti_v5_ghost2x\weights\best.pt"
DEFAULT_DATA = r"D:\DATASETS\DRISHTI-SSS-TRAIN"
DEFAULT_OUTPUT = r"K:\Debris model\runs\phase1_stress_test_v1"

THRESHOLDS = {
    "pipe": 0.56,
    "shipwreck": 0.45,
    "mine": 0.89,
    "ghost_net": 0.36,
}

MATCH_IOU = 0.50
MODEL_IOU = 0.70
MODEL_CONF_FLOOR = 0.001
MAX_DET = 300
BATCH_SIZE = 16
SEED = 26057

STRESS_LEVELS = (
    "clean",
    "mild_speckle",
    "moderate_speckle",
    "strong_speckle",
    "speckle_plus_gain",
    "speckle_plus_dropout",
    "severe_combined",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise RuntimeError(f"Unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def xyxy_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_class_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []

    for class_id in sorted({int(d["class_id"]) for d in detections}):
        class_dets = [
            d for d in detections if int(d["class_id"]) == class_id
        ]
        class_dets.sort(
            key=lambda d: (float(d["confidence"]), box_area(d["bbox_xyxy"])),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        for candidate in class_dets:
            if all(
                xyxy_iou(candidate["bbox_xyxy"], chosen["bbox_xyxy"])
                < iou_threshold
                for chosen in selected
            ):
                selected.append(candidate)

        kept.extend(selected)

    return kept


def parse_yolo_label(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing label: {path}")

    boxes: list[dict[str, Any]] = []

    for line_no, line in enumerate(
        path.read_text(errors="replace").splitlines(), 1
    ):
        text = line.strip()
        if not text:
            continue

        parts = text.split()
        if len(parts) != 5:
            continue

        try:
            cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
        except ValueError:
            continue

        if cls not in CLASS_NAMES:
            continue

        x1 = (xc - w / 2.0) * width
        y1 = (yc - h / 2.0) * height
        x2 = (xc + w / 2.0) * width
        y2 = (yc + h / 2.0) * height

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        if box_area([x1, y1, x2, y2]) <= 0:
            continue

        boxes.append({
            "class_id": cls,
            "class": CLASS_NAMES[cls],
            "bbox_xyxy": [x1, y1, x2, y2],
            "label_line": line_no,
        })

    return boxes


def match_predictions(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, float, int, int]] = []

    for gt_i, gt in enumerate(ground_truth):
        for pred_i, pred in enumerate(predictions):
            if int(gt["class_id"]) != int(pred["class_id"]):
                continue

            iou = xyxy_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou >= MATCH_IOU:
                candidates.append(
                    (iou, float(pred["confidence"]), gt_i, pred_i)
                )

    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []

    for iou, conf, gt_i, pred_i in candidates:
        if gt_i in used_gt or pred_i in used_pred:
            continue

        used_gt.add(gt_i)
        used_pred.add(pred_i)
        matches.append({
            "gt_index": gt_i,
            "pred_index": pred_i,
            "iou": iou,
            "confidence": conf,
        })

    return matches, used_gt, used_pred


def multiplicative_speckle(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    noise = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=x.shape)
    y = np.clip(x * noise, 0.0, 1.0)
    return np.rint(y * 255.0).astype(np.uint8)


def gain_variation(image: np.ndarray, gain_std: float, bias_std: float, rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    gain = float(rng.normal(1.0, gain_std))
    bias = float(rng.normal(0.0, bias_std))
    y = np.clip(x * gain + bias, 0.0, 1.0)
    return np.rint(y * 255.0).astype(np.uint8)


def acoustic_attenuation(
    image: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape
    y0 = int(rng.integers(max(0, int(0.10 * h)), max(1, int(0.65 * h))))
    y1 = int(rng.integers(max(y0 + 1, int(0.35 * h)), h))
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, :] = 1.0

    kernel = max(7, (w // 35) | 1)
    blur = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    attenuation = 1.0 - np.clip(blur * strength, 0.0, 0.85)

    x = image.astype(np.float32) * attenuation
    return np.clip(x, 0, 255).astype(np.uint8)


def stripe_dropout(
    image: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = image.copy().astype(np.float32)
    h, w = x.shape

    total = max(1, int(w * fraction))
    remaining = total

    while remaining > 0:
        width = int(
            min(
                remaining,
                max(3, int(rng.integers(3, max(4, w // 25 + 1)))),
            )
        )
        start = int(rng.integers(0, max(1, w - width + 1)))

        mode = rng.choice(["zero", "attenuate"])
        if mode == "zero":
            x[:, start:start + width] = 0
        else:
            x[:, start:start + width] *= float(rng.uniform(0.05, 0.30))

        remaining -= width

    return np.clip(x, 0, 255).astype(np.uint8)


def resolution_degrade(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    h, w = image.shape
    nh = max(16, int(round(h * scale)))
    nw = max(16, int(round(w * scale)))

    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def corrupt_image(
    image: np.ndarray,
    variant: str,
    seed: int,
) -> np.ndarray:
    if variant == "clean":
        return image.copy()

    rng = np.random.default_rng(seed)
    out = image.copy()

    if variant == "mild_speckle":
        return multiplicative_speckle(out, 0.12, rng)

    if variant == "moderate_speckle":
        return multiplicative_speckle(out, 0.24, rng)

    if variant == "strong_speckle":
        return multiplicative_speckle(out, 0.42, rng)

    if variant == "speckle_plus_gain":
        out = multiplicative_speckle(out, 0.24, rng)
        out = gain_variation(out, 0.18, 0.05, rng)
        return out

    if variant == "speckle_plus_dropout":
        out = multiplicative_speckle(out, 0.30, rng)
        out = acoustic_attenuation(out, 0.65, rng)
        out = stripe_dropout(out, 0.06, rng)
        return resolution_degrade(out, 0.70)

    if variant == "severe_combined":
        out = multiplicative_speckle(out, 0.48, rng)
        out = gain_variation(out, 0.25, 0.08, rng)
        out = acoustic_attenuation(out, 0.80, rng)
        out = stripe_dropout(out, 0.10, rng)
        return resolution_degrade(out, 0.55)

    raise ValueError(f"Unknown stress variant: {variant}")


def predict_batch(
    model: YOLO,
    images: list[np.ndarray],
    device: str,
) -> list[list[dict[str, Any]]]:
    results = model.predict(
        source=images,
        imgsz=640,
        device=device,
        conf=MODEL_CONF_FLOOR,
        iou=MODEL_IOU,
        max_det=MAX_DET,
        verbose=False,
        batch=min(BATCH_SIZE, len(images)),
    )

    all_predictions: list[list[dict[str, Any]]] = []

    for image, result in zip(images, results):
        h, w = image.shape[:2]
        detections: list[dict[str, Any]] = []

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confs, classes):
                class_id = int(class_id)
                if class_id not in CLASS_NAMES:
                    continue

                cls_name = CLASS_NAMES[class_id]
                if float(confidence) < THRESHOLDS[cls_name]:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box]
                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))

                if box_area([x1, y1, x2, y2]) <= 0:
                    continue

                detections.append({
                    "class_id": class_id,
                    "class": cls_name,
                    "confidence": float(confidence),
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

        all_predictions.append(greedy_class_nms(detections, 0.50))

    return all_predictions


def evaluate_variant(
    names: list[str],
    images: list[np.ndarray],
    labels: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    stats = {
        name: {
            "ground_truth": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "matched_ious": [],
        }
        for name in CLASS_NAMES.values()
    }

    total_failures = 0

    for gt_boxes, preds in zip(labels, predictions):
        matches, used_gt, used_pred = match_predictions(gt_boxes, preds)

        for gt in gt_boxes:
            stats[gt["class"]]["ground_truth"] += 1

        for match in matches:
            gt = gt_boxes[match["gt_index"]]
            name = gt["class"]
            stats[name]["true_positives"] += 1
            stats[name]["matched_ious"].append(float(match["iou"]))

        for idx, pred in enumerate(preds):
            if idx not in used_pred:
                stats[pred["class"]]["false_positives"] += 1

        for idx, gt in enumerate(gt_boxes):
            if idx not in used_gt:
                stats[gt["class"]]["false_negatives"] += 1

    total_gt = sum(v["ground_truth"] for v in stats.values())
    total_tp = sum(v["true_positives"] for v in stats.values())
    total_fp = sum(v["false_positives"] for v in stats.values())
    total_fn = sum(v["false_negatives"] for v in stats.values())

    def metrics(tp: int, fp: int, fn: int, ious: list[float]) -> dict[str, Any]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else None,
        }

    result = {
        "stress_variant": names[0] if len(names) == 1 else None,
        "overall": metrics(total_tp, total_fp, total_fn, [
            iou for v in stats.values() for iou in v["matched_ious"]
        ]),
        "per_class": {},
        "failures": total_failures,
    }

    for name, value in stats.items():
        result["per_class"][name] = metrics(
            value["true_positives"],
            value["false_positives"],
            value["false_negatives"],
            value["matched_ious"],
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only SSS robustness stress test."
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--variants",
        default=",".join(STRESS_LEVELS),
        help="Comma-separated stress variants.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data).expanduser().resolve()
    val_images_dir = data_root / "val" / "images"
    val_labels_dir = data_root / "val" / "labels"
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not val_images_dir.exists():
        raise FileNotFoundError(f"Validation images not found: {val_images_dir}")
    if not val_labels_dir.exists():
        raise FileNotFoundError(f"Validation labels not found: {val_labels_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(STRESS_LEVELS))
    if unknown:
        raise ValueError(
            f"Unknown variants: {unknown}. Allowed: {', '.join(STRESS_LEVELS)}"
        )

    ensure_dir(output_dir)

    image_paths = sorted(
        p for p in val_images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError("No validation images found.")

    names: list[str] = []
    original_images: list[np.ndarray] = []
    ground_truth: list[list[dict[str, Any]]] = []

    print("=" * 78)
    print("DRISHTI-SSS ROBUSTNESS STRESS TEST")
    print("=" * 78)
    print(f"Validation images : {len(image_paths)}")
    print(f"Model             : {model_path}")
    print(f"Device             : {args.device}")
    print(f"Thresholds         : {json.dumps(THRESHOLDS)}")
    print(f"Stress variants    : {', '.join(variants)}")
    print("Evaluation          : validation only")
    print("Sealed test         : NOT TOUCHED")
    print()

    for path in image_paths:
        names.append(path.name)
        image = read_grayscale(path)
        original_images.append(image)
        ground_truth.append(
            parse_yolo_label(
                val_labels_dir / f"{path.stem}.txt",
                image.shape[1],
                image.shape[0],
            )
        )

    model = YOLO(str(model_path))

    report: dict[str, Any] = {
        "project": "WaveSentinel / DRISHTI-SSS",
        "purpose": "validation-only robustness stress test",
        "started_at": utc_now(),
        "data": str(data_root),
        "model": str(model_path),
        "validation_images": len(image_paths),
        "seed": args.seed,
        "thresholds": THRESHOLDS,
        "match_iou": MATCH_IOU,
        "model_iou": MODEL_IOU,
        "model_conf_floor": MODEL_CONF_FLOOR,
        "stress_definition": {
            "mild_speckle": "multiplicative log-normal speckle, sigma=0.12",
            "moderate_speckle": "multiplicative log-normal speckle, sigma=0.24",
            "strong_speckle": "multiplicative log-normal speckle, sigma=0.42",
            "speckle_plus_gain": "moderate speckle + global gain/bias variation",
            "speckle_plus_dropout": "stronger speckle + attenuation + sparse stripe dropout + 0.70 resolution factor",
            "severe_combined": "strong speckle + gain/bias + attenuation + 10% stripe dropout + 0.55 resolution factor",
        },
        "variants": {},
    }

    csv_path = output_dir / "stress_metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "variant",
            "class",
            "gt",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou",
        ])

        for variant in variants:
            print(f"[{variant}] generating stress set...")

            stressed_images = [
                corrupt_image(
                    image,
                    variant,
                    args.seed + (i * 10007),
                )
                for i, image in enumerate(original_images)
            ]

            print(f"[{variant}] inference...")
            predictions: list[list[dict[str, Any]]] = []

            for start in range(0, len(stressed_images), BATCH_SIZE):
                batch = stressed_images[start:start + BATCH_SIZE]
                predictions.extend(
                    predict_batch(model, batch, args.device)
                )
                done = min(start + BATCH_SIZE, len(stressed_images))
                print(f"\r  {done}/{len(stressed_images)}", end="")

            print()

            result = evaluate_variant(
                [variant],
                stressed_images,
                ground_truth,
                predictions,
            )
            result["stress_variant"] = variant
            report["variants"][variant] = result

            print(
                f"  P={result['overall']['precision']:.4f} "
                f"R={result['overall']['recall']:.4f} "
                f"F1={result['overall']['f1']:.4f} "
                f"mIoU={result['overall']['mean_iou']:.4f}"
            )

            for class_name, values in result["per_class"].items():
                writer.writerow([
                    variant,
                    class_name,
                    values["ground_truth"] if "ground_truth" in values else "",
                    values["tp"],
                    values["fp"],
                    values["fn"],
                    values["precision"],
                    values["recall"],
                    values["f1"],
                    values["mean_iou"],
                ])

    clean_f1 = report["variants"].get("clean", {}).get("overall", {}).get("f1")
    clean_recall = report["variants"].get("clean", {}).get("overall", {}).get("recall")

    for variant, values in report["variants"].items():
        overall = values["overall"]
        if clean_f1 is not None:
            overall["f1_delta_vs_clean"] = overall["f1"] - clean_f1
        if clean_recall is not None:
            overall["recall_delta_vs_clean"] = overall["recall"] - clean_recall

    report["finished_at"] = utc_now()

    report_path = output_dir / "stress_test_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STRESS TEST COMPLETE")
    print("=" * 78)
    print(f"Report : {report_path}")
    print(f"CSV    : {csv_path}")
    print()
    print("Interpretation:")
    print("  clean = current model baseline on validation")
    print("  stressed variants = synthetic robustness probes")
    print("  no stressed variant is used for threshold tuning")
    print("  sealed test was not touched")


if __name__ == "__main__":
    main()
