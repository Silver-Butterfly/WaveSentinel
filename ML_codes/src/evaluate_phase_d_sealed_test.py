from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


# ============================================================
# PHASE D | SEALED TEST EVALUATION
# Frozen thresholds. Do not modify after this run.
# ============================================================

DATASET = Path(r"D:\DATASETS\DRISHTI-SSS-ROBUST-V2-HN")
MODEL = Path(
    r"K:\Debris model\runs\drishti_phase_d_robust_v2_hn\weights\best.pt"
)
OUTPUT = Path(
    r"K:\Debris model\runs\drishti_phase_d_robust_v2_hn\sealed_test_evaluation"
)

EXPECTED_TEST_IMAGES = 779
IMG_SIZE = 640
BATCH = 16
DEVICE = 0
MODEL_CONF_FLOOR = 0.001
MODEL_IOU = 0.70
MATCH_IOU = 0.50
MAX_DET = 300

FROZEN_THRESHOLDS = {
    "pipe": 0.56,
    "shipwreck": 0.45,
    "mine": 0.89,
    "ghost_net": 0.36,
}

CLASS_NAMES = {
    0: "pipe",
    1: "shipwreck",
    2: "mine",
    3: "ghost_net",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def parse_yolo_label(
    label_path: Path,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing test label: {label_path}")

    boxes: list[dict[str, Any]] = []

    for line_no, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Malformed YOLO label {label_path}:{line_no}: {line}"
            )

        cls = int(float(parts[0]))
        if cls not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class {cls} in {label_path}:{line_no}"
            )

        xc, yc, bw, bh = map(float, parts[1:])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height

        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Non-positive box in {label_path}:{line_no}: {line}"
            )

        boxes.append(
            {
                "class_id": cls,
                "class": CLASS_NAMES[cls],
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )

    return boxes


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def xyxy_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    denom = box_area(a) + box_area(b) - inter
    return inter / denom if denom > 0 else 0.0


def classwise_nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []

    for class_id in CLASS_NAMES:
        candidates = [
            d for d in detections if int(d["class_id"]) == class_id
        ]
        candidates.sort(
            key=lambda d: float(d["confidence"]),
            reverse=True,
        )

        while candidates:
            best = candidates.pop(0)
            kept.append(best)

            remaining = []
            for candidate in candidates:
                iou = xyxy_iou(
                    best["bbox_xyxy"],
                    candidate["bbox_xyxy"],
                )
                if iou < iou_threshold:
                    remaining.append(candidate)

            candidates = remaining

    kept.sort(key=lambda d: float(d["confidence"]), reverse=True)
    return kept


def match_predictions(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, float, int, int]] = []

    for gt_i, gt in enumerate(ground_truth):
        for pred_i, pred in enumerate(predictions):
            if int(gt["class_id"]) != int(pred["class_id"]):
                continue

            iou = xyxy_iou(
                gt["bbox_xyxy"],
                pred["bbox_xyxy"],
            )
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
        matches.append(
            {
                "gt_index": gt_i,
                "pred_index": pred_i,
                "iou": iou,
                "confidence": conf,
            }
        )

    return matches, used_gt, used_pred


def evaluate(
    labels: list[list[dict[str, Any]]],
    predictions: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    stats = {
        name: {
            "ground_truth": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
        }
        for name in CLASS_NAMES.values()
    }

    for gt_boxes, preds in zip(labels, predictions):
        matches, used_gt, used_pred = match_predictions(gt_boxes, preds)

        for gt in gt_boxes:
            stats[gt["class"]]["ground_truth"] += 1

        for match in matches:
            gt = gt_boxes[match["gt_index"]]
            name = gt["class"]
            stats[name]["tp"] += 1
            stats[name]["matched_ious"].append(float(match["iou"]))

        for pred_i, pred in enumerate(preds):
            if pred_i not in used_pred:
                stats[pred["class"]]["fp"] += 1

        for gt_i, gt in enumerate(gt_boxes):
            if gt_i not in used_gt:
                stats[gt["class"]]["fn"] += 1

    per_class: dict[str, Any] = {}

    for name, s in stats.items():
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[name] = {
            "ground_truth": s["ground_truth"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": (
                float(np.mean(s["matched_ious"]))
                if s["matched_ious"]
                else 0.0
            ),
        }

    total_gt = sum(v["ground_truth"] for v in per_class.values())
    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    all_ious = []
    for s in stats.values():
        all_ious.extend(s["matched_ious"])

    return {
        "ground_truth": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "per_class": per_class,
    }


def predict(
    model: YOLO,
    images: list[np.ndarray],
) -> list[list[dict[str, Any]]]:
    results = model.predict(
        source=images,
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=MODEL_CONF_FLOOR,
        iou=MODEL_IOU,
        max_det=MAX_DET,
        verbose=False,
        batch=min(BATCH, len(images)),
    )

    outputs: list[list[dict[str, Any]]] = []

    for image, result in zip(images, results):
        h, w = image.shape[:2]
        detections: list[dict[str, Any]] = []

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = (
                result.boxes.cls.detach().cpu().numpy().astype(int)
            )

            for box, confidence, class_id in zip(
                boxes,
                confs,
                classes,
            ):
                class_id = int(class_id)

                if class_id not in CLASS_NAMES:
                    continue

                class_name = CLASS_NAMES[class_id]
                threshold = FROZEN_THRESHOLDS[class_name]

                if float(confidence) < threshold:
                    continue

                x1, y1, x2, y2 = map(float, box)

                x1 = max(0.0, min(float(w), x1))
                y1 = max(0.0, min(float(h), y1))
                x2 = max(0.0, min(float(w), x2))
                y2 = max(0.0, min(float(h), y2))

                if box_area([x1, y1, x2, y2]) <= 0:
                    continue

                detections.append(
                    {
                        "class_id": class_id,
                        "class": class_name,
                        "confidence": float(confidence),
                        "threshold": threshold,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )

        outputs.append(classwise_nms(detections, 0.50))

    return outputs


def count_images(folder: Path) -> int:
    return sum(
        1
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    print("=" * 78)
    print("PHASE D | SEALED TEST EVALUATION")
    print("=" * 78)
    print("This run uses the frozen validation-calibrated thresholds.")
    print("NO threshold tuning. NO model changes. NO test-set selection.")
    print()

    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    if not MODEL.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL}")

    test_images_dir = DATASET / "test" / "images"
    test_labels_dir = DATASET / "test" / "labels"

    if not test_images_dir.is_dir():
        raise FileNotFoundError(test_images_dir)

    if not test_labels_dir.is_dir():
        raise FileNotFoundError(test_labels_dir)

    actual_count = count_images(test_images_dir)

    if actual_count != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"SEALED TEST COUNT MISMATCH: expected "
            f"{EXPECTED_TEST_IMAGES}, found {actual_count}"
        )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists. Refusing to overwrite sealed result: {OUTPUT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Expected GPU 0.")

    OUTPUT.mkdir(parents=True, exist_ok=False)

    model_hash = sha256_file(MODEL)

    image_paths = sorted(
        p for p in test_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Dataset      : {DATASET}")
    print(f"Checkpoint   : {MODEL}")
    print(f"SHA256       : {model_hash}")
    print(f"TEST images  : {len(image_paths)}")
    print(f"GPU          : {torch.cuda.get_device_name(DEVICE)}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA         : {torch.version.cuda}")
    print(f"Ultralytics  : {ultralytics.__version__}")
    print()
    print("Frozen thresholds:")
    for name in CLASS_NAMES.values():
        print(f"  {name:12s} = {FROZEN_THRESHOLDS[name]:.2f}")
    print()
    print("Evaluation configuration:")
    print(f"  imgsz      = {IMG_SIZE}")
    print(f"  batch      = {BATCH}")
    print(f"  device     = {DEVICE}")
    print(f"  conf floor = {MODEL_CONF_FLOOR}")
    print(f"  NMS IoU    = {MODEL_IOU}")
    print(f"  match IoU  = {MATCH_IOU}")
    print()
    print("SEALED TEST: FINAL EVALUATION")
    print("=" * 78)

    started_at = utc_now()

    model = YOLO(str(MODEL))

    images: list[np.ndarray] = []
    labels: list[list[dict[str, Any]]] = []

    for path in image_paths:
        image = read_grayscale(path)
        images.append(image)
        labels.append(
            parse_yolo_label(
                test_labels_dir / f"{path.stem}.txt",
                image.shape[1],
                image.shape[0],
            )
        )

    predictions: list[list[dict[str, Any]]] = []

    for start in range(0, len(images), BATCH):
        batch = images[start:start + BATCH]
        predictions.extend(predict(model, batch))
        done = min(start + BATCH, len(images))
        print(f"\rInference: {done}/{len(images)}", end="")

    print()

    metrics = evaluate(labels, predictions)
    finished_at = utc_now()

    report = {
        "phase": "D",
        "stage": "sealed_test_evaluation",
        "purpose": (
            "Final held-out TEST evaluation after validation-only "
            "threshold calibration and validation-only robustness stress test."
        ),
        "dataset": str(DATASET),
        "model": str(MODEL),
        "model_sha256": model_hash,
        "split": "test",
        "sealed_test": True,
        "test_images": len(image_paths),
        "frozen_thresholds": FROZEN_THRESHOLDS,
        "configuration": {
            "imgsz": IMG_SIZE,
            "batch": BATCH,
            "device": DEVICE,
            "prediction_confidence_floor": MODEL_CONF_FLOOR,
            "nms_iou": MODEL_IOU,
            "match_iou": MATCH_IOU,
            "max_det": MAX_DET,
            "post_threshold_classwise_nms_iou": 0.50,
        },
        "overall": {
            k: metrics[k]
            for k in (
                "ground_truth",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_iou",
            )
        },
        "per_class": metrics["per_class"],
        "integrity": {
            "thresholds_source": "frozen from VAL-only calibration",
            "test_used_for_threshold_tuning": False,
            "test_used_for_model_selection": False,
            "training_performed_during_evaluation": False,
            "source_dataset_modified": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(DEVICE),
        },
        "started_at": started_at,
        "finished_at": finished_at,
    }

    report_path = OUTPUT / "phase_d_sealed_test_evaluation.json"
    summary_path = OUTPUT / "phase_d_sealed_test_evaluation.txt"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "=" * 78,
        "PHASE D | SEALED TEST EVALUATION COMPLETE",
        "=" * 78,
        f"Checkpoint : {MODEL}",
        f"SHA256     : {model_hash}",
        f"TEST images: {len(image_paths)}",
        "",
        "FROZEN THRESHOLDS",
        "-" * 78,
    ]

    for name in CLASS_NAMES.values():
        lines.append(
            f"{name:12s} threshold={FROZEN_THRESHOLDS[name]:.2f}"
        )

    lines.extend(
        [
            "",
            "OVERALL",
            "-" * 78,
            f"Ground truth : {metrics['ground_truth']}",
            f"TP           : {metrics['tp']}",
            f"FP           : {metrics['fp']}",
            f"FN           : {metrics['fn']}",
            f"Precision    : {metrics['precision']:.6f}",
            f"Recall       : {metrics['recall']:.6f}",
            f"F1           : {metrics['f1']:.6f}",
            f"Mean IoU     : {metrics['mean_iou']:.6f}",
            "",
            "PER CLASS",
            "-" * 78,
        ]
    )

    for name in CLASS_NAMES.values():
        row = metrics["per_class"][name]
        lines.append(
            f"{name:12s} "
            f"P={row['precision']:.6f} "
            f"R={row['recall']:.6f} "
            f"F1={row['f1']:.6f} "
            f"mIoU={row['mean_iou']:.6f} "
            f"GT={row['ground_truth']} "
            f"TP={row['tp']} "
            f"FP={row['fp']} "
            f"FN={row['fn']}"
        )

    lines.extend(
        [
            "",
            "INTEGRITY",
            "-" * 78,
            "Threshold tuning on TEST        : NO",
            "Model selection on TEST         : NO",
            "Training during evaluation      : NO",
            "Source dataset modified         : NO",
            "TEST status                     : SEALED / FINAL",
            "",
            f"Report: {report_path}",
            "=" * 78,
        ]
    )

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
