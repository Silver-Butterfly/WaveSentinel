from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
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
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_MODEL = r"K:\Debris model\runs\drishti_v5_ghost2x\weights\best.pt"
DEFAULT_OUTPUT = r"K:\Debris model\runs\inference"

# Frozen validation-calibrated operating thresholds.
# Selected on the validation split by maximizing class-wise F1 at IoU=0.50.
# These are operating thresholds, not calibrated probability estimates.
DEFAULT_CLASS_THRESHOLDS = {
    "pipe": 0.74,
    "shipwreck": 0.38,
    "mine": 0.71,
    "ghost_net": 0.40,
}

# Documented Phase-1 SSS geometry.
TILE_WIDTH = 1024
TILE_HEIGHT = 500
OVERLAP = 0.20
STRIDE = int(round(TILE_WIDTH * (1.0 - OVERLAP)))
MODEL_CANVAS = 1024
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(value: str | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_CLASS_THRESHOLDS)

    if not value:
        return thresholds

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--thresholds must be valid JSON, for example "
            '\'{"pipe":0.25,"shipwreck":0.25,"mine":0.25,"ghost_net":0.40}\''
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("--thresholds must decode to a JSON object.")

    for name, threshold in parsed.items():
        if name not in DEFAULT_CLASS_THRESHOLDS:
            raise ValueError(
                f"Unknown class in --thresholds: {name!r}. "
                f"Expected: {', '.join(DEFAULT_CLASS_THRESHOLDS)}"
            )

        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold for {name} must be in [0, 1].")

        thresholds[name] = threshold

    return thresholds


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise RuntimeError(f"Unsupported image shape for {path}: {image.shape}")


def percentile_normalize_to_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got {gray.shape}")

    arr = gray.astype(np.float32)
    finite = np.isfinite(arr)

    if not np.any(finite):
        raise ValueError("Image contains no finite pixel values.")

    values = arr[finite]
    lo = float(np.percentile(values, LOW_PERCENTILE))
    hi = float(np.percentile(values, HIGH_PERCENTILE))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Invalid normalization percentiles.")

    if hi <= lo:
        mn = float(values.min())
        mx = float(values.max())
        if mx <= mn:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = mn, mx

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 255.0, 0.0, 255.0)
    return np.rint(arr).astype(np.uint8)


def tile_starts(full_size: int, tile_size: int, stride: int) -> list[int]:
    if full_size <= tile_size:
        return [0]

    starts: list[int] = []
    pos = 0

    while pos + tile_size < full_size:
        starts.append(pos)
        pos += stride

    final_start = full_size - tile_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return sorted(set(starts))


def pad_to_canvas(tile: np.ndarray, target: int = MODEL_CANVAS) -> tuple[np.ndarray, int, int]:
    h, w = tile.shape[:2]

    if h > target or w > target:
        raise ValueError(f"Tile {w}x{h} exceeds model canvas {target}x{target}.")

    canvas = np.zeros((target, target), dtype=np.uint8)
    left = (target - w) // 2
    top = (target - h) // 2
    canvas[top:top + h, left:left + w] = tile
    return canvas, left, top


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0.0 else 0.0


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def clip_box(box: list[float], width: float, height: float) -> list[float]:
    return [
        max(0.0, min(float(width), float(box[0]))),
        max(0.0, min(float(height), float(box[1]))),
        max(0.0, min(float(width), float(box[2]))),
        max(0.0, min(float(height), float(box[3]))),
    ]


def greedy_class_nms(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    if not detections:
        return []

    kept: list[dict[str, Any]] = []

    for class_id in sorted({int(d["class_id"]) for d in detections}):
        class_dets = [d for d in detections if int(d["class_id"]) == class_id]
        class_dets.sort(
            key=lambda d: (float(d["confidence"]), box_area(d["bbox_xyxy"])),
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        for candidate in class_dets:
            if all(
                xyxy_iou(candidate["bbox_xyxy"], current["bbox_xyxy"]) < iou_threshold
                for current in selected
            ):
                selected.append(candidate)

        kept.extend(selected)

    kept.sort(key=lambda d: (int(d["class_id"]), -float(d["confidence"])))
    return kept


def build_detection(
    class_id: int,
    confidence: float,
    bbox: list[float],
    raw_width: int,
    raw_height: int,
    threshold: float,
    tile_index: int | None = None,
    tile_x0: int | None = None,
    tile_y0: int | None = None,
) -> dict[str, Any]:
    name = CLASS_NAMES[int(class_id)]

    return {
        "class_id": int(class_id),
        "class": name,
        "confidence": round(float(confidence), 6),
        "threshold": round(float(threshold), 6),
        "bbox_xyxy": [round(float(v), 2) for v in bbox],
        "bbox_width": round(max(0.0, bbox[2] - bbox[0]), 2),
        "bbox_height": round(max(0.0, bbox[3] - bbox[1]), 2),
        "tile_index": tile_index,
        "tile_x0": tile_x0,
        "tile_y0": tile_y0,
        "source_width": int(raw_width),
        "source_height": int(raw_height),
    }


def class_threshold(class_id: int, thresholds: dict[str, float]) -> float:
    name = CLASS_NAMES.get(int(class_id))
    return float(thresholds[name]) if name is not None else 1.0


def source_image_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    paths = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise RuntimeError(f"No supported images found under: {source}")

    return paths


def safe_output_stem(path: Path, source: Path) -> str:
    if source.is_dir():
        try:
            rel = path.relative_to(source)
            text = str(rel).replace("\\", "/")
        except ValueError:
            text = path.name
    else:
        text = path.name

    stem = Path(text.replace("/", "__")).stem

    if len(stem) > 160:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        stem = f"{stem[:145]}__{digest}"

    return stem


def run_prepared(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    device: str,
    iou: float,
    max_det: int,
    thresholds: dict[str, float],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    gray = read_grayscale(image_path)
    h, w = gray.shape[:2]

    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        conf=0.001,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )[0]

    detections: list[dict[str, Any]] = []

    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confs, classes):
            class_id = int(class_id)
            if class_id not in CLASS_NAMES:
                continue

            threshold = class_threshold(class_id, thresholds)
            if float(confidence) < threshold:
                continue

            bbox = clip_box([float(v) for v in box], w, h)
            if box_area(bbox) <= 0.0:
                continue

            detections.append(
                build_detection(
                    class_id=class_id,
                    confidence=float(confidence),
                    bbox=bbox,
                    raw_width=w,
                    raw_height=h,
                    threshold=threshold,
                )
            )

    # Ultralytics has already performed NMS within this image. This second
    # class-aware pass applies the same explicit reporting rule uniformly.
    detections = greedy_class_nms(detections, 0.50)

    metadata = {
        "input_mode": "prepared",
        "image_width": int(w),
        "image_height": int(h),
        "additional_preprocessing": False,
    }

    return gray, detections, metadata


def run_raw_sss(
    model: YOLO,
    image_path: Path,
    imgsz: int,
    device: str,
    iou: float,
    max_det: int,
    thresholds: dict[str, float],
    merge_iou: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    # This implements the documented Phase-1 percentile normalization and
    # tiling geometry. It deliberately does not invent Lee/CLAHE processing.
    gray = read_grayscale(image_path)
    normalized = percentile_normalize_to_uint8(gray)

    raw_h, raw_w = normalized.shape[:2]
    x_starts = tile_starts(raw_w, TILE_WIDTH, STRIDE)
    y_starts = tile_starts(raw_h, TILE_HEIGHT, STRIDE)

    all_detections: list[dict[str, Any]] = []
    tile_count = 0

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + TILE_WIDTH, raw_w)
            y1 = min(y0 + TILE_HEIGHT, raw_h)

            crop = normalized[y0:y1, x0:x1]

            if crop.shape != (TILE_HEIGHT, TILE_WIDTH):
                tile = np.zeros((TILE_HEIGHT, TILE_WIDTH), dtype=np.uint8)
                tile[:crop.shape[0], :crop.shape[1]] = crop
            else:
                tile = crop

            canvas, pad_left, pad_top = pad_to_canvas(tile, MODEL_CANVAS)
            tile_count += 1

            result = model.predict(
                source=canvas,
                imgsz=imgsz,
                device=device,
                conf=0.001,
                iou=iou,
                max_det=max_det,
                verbose=False,
            )[0]

            if result.boxes is None or not len(result.boxes):
                continue

            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confs, classes):
                class_id = int(class_id)
                if class_id not in CLASS_NAMES:
                    continue

                threshold = class_threshold(class_id, thresholds)
                if float(confidence) < threshold:
                    continue

                local_box = [
                    float(box[0]) - pad_left,
                    float(box[1]) - pad_top,
                    float(box[2]) - pad_left,
                    float(box[3]) - pad_top,
                ]

                local_box = clip_box(
                    local_box,
                    width=min(TILE_WIDTH, x1 - x0),
                    height=min(TILE_HEIGHT, y1 - y0),
                )

                if box_area(local_box) <= 0.0:
                    continue

                source_box = [
                    local_box[0] + x0,
                    local_box[1] + y0,
                    local_box[2] + x0,
                    local_box[3] + y0,
                ]
                source_box = clip_box(source_box, raw_w, raw_h)

                if box_area(source_box) <= 0.0:
                    continue

                all_detections.append(
                    build_detection(
                        class_id=class_id,
                        confidence=float(confidence),
                        bbox=source_box,
                        raw_width=raw_w,
                        raw_height=raw_h,
                        threshold=threshold,
                        tile_index=tile_count - 1,
                        tile_x0=x0,
                        tile_y0=y0,
                    )
                )

    # Adjacent SSS tiles overlap by design, so duplicate detections must be
    # merged after restoring them to the original source coordinates.
    final_detections = greedy_class_nms(all_detections, merge_iou)

    metadata = {
        "input_mode": "raw_sss",
        "image_width": int(raw_w),
        "image_height": int(raw_h),
        "tile_width": TILE_WIDTH,
        "tile_height": TILE_HEIGHT,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "model_canvas": [MODEL_CANVAS, MODEL_CANVAS],
        "x_tile_starts": x_starts,
        "y_tile_starts": y_starts,
        "tile_count": tile_count,
        "normalization": {
            "method": "percentile_clip",
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
            "output_dtype": "uint8",
        },
        "lee_filter": False,
        "clahe": False,
    }

    return normalized, final_detections, metadata


def annotate_image(image_gray: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        label = f'{det["class"]} {det["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            image,
            label,
            (max(0, x1), max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_image",
        "mode",
        "class_id",
        "class",
        "confidence",
        "threshold",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "tile_index",
        "tile_x0",
        "tile_y0",
        "source_width",
        "source_height",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            bbox = record["bbox_xyxy"]
            writer.writerow(
                {
                    "source_image": record["source_image"],
                    "mode": record["mode"],
                    "class_id": record["class_id"],
                    "class": record["class"],
                    "confidence": record["confidence"],
                    "threshold": record["threshold"],
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "bbox_width": record["bbox_width"],
                    "bbox_height": record["bbox_height"],
                    "tile_index": record["tile_index"],
                    "tile_x0": record["tile_x0"],
                    "tile_y0": record["tile_y0"],
                    "source_width": record["source_width"],
                    "source_height": record["source_height"],
                }
            )


def build_summary(image_records: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    counts = Counter()
    confidences = {name: [] for name in CLASS_NAMES.values()}
    images_with_detections = 0

    for record in image_records:
        detections = record["detections"]
        if detections:
            images_with_detections += 1

        for det in detections:
            name = det["class"]
            counts[name] += 1
            confidences[name].append(float(det["confidence"]))

    classes: dict[str, Any] = {}
    for class_id, name in CLASS_NAMES.items():
        values = confidences[name]
        classes[name] = {
            "class_id": class_id,
            "count": int(counts[name]),
            "threshold": float(thresholds[name]),
            "mean_confidence": round(float(np.mean(values)), 6) if values else None,
            "median_confidence": round(float(np.median(values)), 6) if values else None,
            "max_confidence": round(float(np.max(values)), 6) if values else None,
        }

    return {
        "images_processed": len(image_records),
        "images_with_detections": images_with_detections,
        "images_without_detections": len(image_records) - images_with_detections,
        "total_final_detections": int(sum(counts.values())),
        "class_counts": {name: int(counts[name]) for name in CLASS_NAMES.values()},
        "classes": classes,
    }



def yolo_labels_to_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []

    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line_number, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height

        box = clip_box([x1, y1, x2, y2], width, height)

        if box_area(box) <= 0.0:
            continue

        boxes.append(
            {
                "class_id": class_id,
                "class": CLASS_NAMES[class_id],
                "bbox_xyxy": box,
                "label_line": line_number,
            }
        )

    return boxes


def greedy_match_ground_truth(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    match_iou: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, float, int, int]] = []

    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if int(gt["class_id"]) != int(pred["class_id"]):
                continue

            overlap = xyxy_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])

            if overlap >= match_iou:
                candidates.append(
                    (
                        overlap,
                        float(pred["confidence"]),
                        gt_index,
                        pred_index,
                    )
                )

    candidates.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []

    for overlap, confidence, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "iou": float(overlap),
                "confidence": float(confidence),
            }
        )

    return matches, used_gt, used_pred


def best_overlap(
    gt: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> tuple[float, int | None]:
    best_iou = 0.0
    best_index: int | None = None

    for index, prediction in enumerate(predictions):
        overlap = xyxy_iou(gt["bbox_xyxy"], prediction["bbox_xyxy"])
        if overlap > best_iou:
            best_iou = overlap
            best_index = index

    return best_iou, best_index


def draw_evaluation_image(
    gray: np.ndarray,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matched_gt: set[int],
    matched_predictions: set[int],
) -> np.ndarray:
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for index, gt in enumerate(ground_truth):
        x1, y1, x2, y2 = [int(round(v)) for v in gt["bbox_xyxy"]]
        label = f'GT {gt["class"]}'
        thickness = 2 if index in matched_gt else 3

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            thickness,
        )
        cv2.putText(
            image,
            label,
            (max(0, x1), max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for index, pred in enumerate(predictions):
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox_xyxy"]]
        label = f'P {pred["class"]} {pred["confidence"]:.2f}'
        thickness = 1 if index in matched_predictions else 3

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            thickness,
        )
        cv2.putText(
            image,
            label,
            (max(0, x1), min(image.shape[0] - 5, y2 + 17)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return image


def evaluate_predictions(
    image_records: list[dict[str, Any]],
    source: Path,
    labels_dir: Path,
    match_iou: float,
    output_dir: Path,
    gallery_top: int,
) -> dict[str, Any]:
    failure_root = output_dir / "evaluation"
    false_negative_dir = failure_root / "false_negatives"
    localization_dir = failure_root / "localization_failures"
    false_positive_dir = failure_root / "false_positives"
    confusion_dir = failure_root / "confusions"

    for directory in (
        false_negative_dir,
        localization_dir,
        false_positive_dir,
        confusion_dir,
    ):
        ensure_dir(directory)

    stats = {
        name: {
            "ground_truth": 0,
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "matched_ious": [],
            "tp_confidences": [],
        }
        for name in CLASS_NAMES.values()
    }

    confusion_counter: Counter[tuple[str, str]] = Counter()
    image_results: list[dict[str, Any]] = []
    failures_by_type: dict[str, list[tuple[float, str, np.ndarray]]] = {
        "false_negatives": [],
        "localization_failures": [],
        "false_positives": [],
        "confusions": [],
    }

    evaluated_images = 0
    missing_labels: list[str] = []

    for record in image_records:
        image_path = Path(record["source_image"])
        label_path = labels_dir / f"{image_path.stem}.txt"

        gray = read_grayscale(image_path)
        height, width = gray.shape[:2]
        ground_truth = yolo_labels_to_boxes(label_path, width, height)
        predictions = record["detections"]

        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue

        evaluated_images += 1

        for gt in ground_truth:
            stats[gt["class"]]["ground_truth"] += 1

        matches, used_gt, used_predictions = greedy_match_ground_truth(
            ground_truth,
            predictions,
            match_iou,
        )

        for match in matches:
            gt = ground_truth[match["gt_index"]]
            prediction = predictions[match["pred_index"]]
            name = gt["class"]

            stats[name]["true_positives"] += 1
            stats[name]["matched_ious"].append(match["iou"])
            stats[name]["tp_confidences"].append(match["confidence"])

        image_eval: dict[str, Any] = {
            "source_image": str(image_path),
            "label_path": str(label_path),
            "ground_truth_count": len(ground_truth),
            "prediction_count": len(predictions),
            "matches": [],
            "false_negatives": [],
            "false_positives": [],
            "localization_failures": [],
            "confusions": [],
        }

        for match in matches:
            gt = ground_truth[match["gt_index"]]
            prediction = predictions[match["pred_index"]]
            image_eval["matches"].append(
                {
                    "class": gt["class"],
                    "confidence": round(match["confidence"], 6),
                    "iou": round(match["iou"], 6),
                }
            )

        for gt_index, gt in enumerate(ground_truth):
            if gt_index in used_gt:
                continue

            name = gt["class"]
            stats[name]["false_negatives"] += 1

            same_class_predictions = [
                prediction
                for prediction in predictions
                if int(prediction["class_id"]) == int(gt["class_id"])
            ]
            same_class_iou, same_class_index = best_overlap(
                gt,
                same_class_predictions,
            )

            any_iou, any_index = best_overlap(gt, predictions)
            failure_type = "no_prediction"
            best_prediction = None

            if same_class_index is not None:
                best_prediction = same_class_predictions[same_class_index]
                if same_class_iou > 0.0:
                    failure_type = "localization_failure"

            if any_index is not None and predictions[any_index]["class_id"] != gt["class_id"]:
                if any_iou >= 0.20:
                    predicted_class = predictions[any_index]["class"]
                    confusion_counter[(name, predicted_class)] += 1
                    failure_type = "class_confusion"

            item = {
                "class": name,
                "bbox_xyxy": [round(v, 2) for v in gt["bbox_xyxy"]],
                "failure_type": failure_type,
                "best_same_class_iou": round(same_class_iou, 6),
                "best_any_class_iou": round(any_iou, 6),
                "best_same_class_confidence": (
                    round(float(best_prediction["confidence"]), 6)
                    if best_prediction is not None
                    else None
                ),
            }
            image_eval["false_negatives"].append(item)

            ranking = 1.0 - same_class_iou
            failures_by_type["false_negatives"].append(
                (
                    ranking,
                    str(image_path),
                    draw_evaluation_image(
                        gray,
                        ground_truth,
                        predictions,
                        used_gt,
                        used_predictions,
                    ),
                )
            )

            if failure_type == "localization_failure":
                failures_by_type["localization_failures"].append(
                    (
                        ranking,
                        str(image_path),
                        draw_evaluation_image(
                            gray,
                            ground_truth,
                            predictions,
                            used_gt,
                            used_predictions,
                        ),
                    )
                )

            if failure_type == "class_confusion":
                failures_by_type["confusions"].append(
                    (
                        any_iou,
                        str(image_path),
                        draw_evaluation_image(
                            gray,
                            ground_truth,
                            predictions,
                            used_gt,
                            used_predictions,
                        ),
                    )
                )

        for pred_index, prediction in enumerate(predictions):
            if pred_index in used_predictions:
                continue

            name = prediction["class"]
            stats[name]["false_positives"] += 1

            best_gt_iou, best_gt_index = best_overlap(
                prediction,
                ground_truth,
            )
            best_gt_class = (
                ground_truth[best_gt_index]["class"]
                if best_gt_index is not None
                else None
            )

            image_eval["false_positives"].append(
                {
                    "class": name,
                    "confidence": round(float(prediction["confidence"]), 6),
                    "bbox_xyxy": [round(v, 2) for v in prediction["bbox_xyxy"]],
                    "best_gt_class": best_gt_class,
                    "best_gt_iou": round(best_gt_iou, 6),
                }
            )

            failures_by_type["false_positives"].append(
                (
                    float(prediction["confidence"]),
                    str(image_path),
                    draw_evaluation_image(
                        gray,
                        ground_truth,
                        predictions,
                        used_gt,
                        used_predictions,
                    ),
                )
            )

        image_results.append(image_eval)

    class_report: dict[str, Any] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious: list[float] = []

    for name in CLASS_NAMES.values():
        s = stats[name]
        tp = int(s["true_positives"])
        fp = int(s["false_positives"])
        fn = int(s["false_negatives"])

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(s["matched_ious"])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        class_report[name] = {
            "class_id": next(
                class_id
                for class_id, class_name in CLASS_NAMES.items()
                if class_name == name
            ),
            "ground_truth": s["ground_truth"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision_at_threshold": round(precision, 6),
            "recall_at_threshold": round(recall, 6),
            "f1_at_threshold": round(f1, 6),
            "mean_matched_iou": (
                round(float(np.mean(s["matched_ious"])), 6)
                if s["matched_ious"]
                else None
            ),
            "median_matched_iou": (
                round(float(np.median(s["matched_ious"])), 6)
                if s["matched_ious"]
                else None
            ),
            "mean_tp_confidence": (
                round(float(np.mean(s["tp_confidences"])), 6)
                if s["tp_confidences"]
                else None
            ),
            "median_tp_confidence": (
                round(float(np.median(s["tp_confidences"])), 6)
                if s["tp_confidences"]
                else None
            ),
        }

    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    overall_f1 = (
        2.0 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall
        else 0.0
    )

    report = {
        "evaluation": "ground_truth_comparison",
        "source": str(source),
        "labels_dir": str(labels_dir),
        "match_iou": match_iou,
        "evaluated_images": evaluated_images,
        "total_source_images": len(image_records),
        "missing_label_files": missing_labels,
        "overall": {
            "ground_truth": sum(s["ground_truth"] for s in stats.values()),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision_at_threshold": round(overall_precision, 6),
            "recall_at_threshold": round(overall_recall, 6),
            "f1_at_threshold": round(overall_f1, 6),
            "mean_matched_iou": (
                round(float(np.mean(all_ious)), 6) if all_ious else None
            ),
            "median_matched_iou": (
                round(float(np.median(all_ious)), 6) if all_ious else None
            ),
        },
        "classes": class_report,
        "confusions": [
            {
                "ground_truth": gt,
                "prediction": prediction,
                "count": count,
            }
            for (gt, prediction), count in confusion_counter.most_common()
        ],
        "images": image_results,
    }

    (failure_root / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "DRISHTI-SSS GROUND-TRUTH EVALUATION",
        "=" * 64,
        f"Images evaluated: {evaluated_images}/{len(image_records)}",
        f"Match IoU: {match_iou:.2f}",
        "",
        "OVERALL",
        "-" * 64,
        f"GT:        {report['overall']['ground_truth']}",
        f"TP:        {total_tp}",
        f"FP:        {total_fp}",
        f"FN:        {total_fn}",
        f"Precision: {overall_precision:.4f}",
        f"Recall:    {overall_recall:.4f}",
        f"F1:        {overall_f1:.4f}",
        f"Mean IoU:  {report['overall']['mean_matched_iou']}",
        "",
        "PER CLASS",
        "-" * 64,
    ]

    for name in CLASS_NAMES.values():
        s = class_report[name]
        lines.append(
            f"{name:12s} GT={s['ground_truth']:4d} "
            f"TP={s['true_positives']:4d} "
            f"FP={s['false_positives']:4d} "
            f"FN={s['false_negatives']:4d} "
            f"P={s['precision_at_threshold']:.4f} "
            f"R={s['recall_at_threshold']:.4f} "
            f"F1={s['f1_at_threshold']:.4f} "
            f"IoU={s['mean_matched_iou']}"
        )

    lines.extend(["", "CONFUSIONS", "-" * 64])

    if confusion_counter:
        for (gt, prediction), count in confusion_counter.most_common():
            lines.append(f"{gt:12s} -> {prediction:12s}: {count}")
    else:
        lines.append("None")

    if missing_labels:
        lines.extend(["", "MISSING LABEL FILES", "-" * 64])
        lines.extend(missing_labels)

    (failure_root / "evaluation.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    def save_gallery(
        items: list[tuple[float, str, np.ndarray]],
        directory: Path,
        reverse: bool,
    ) -> None:
        if not items:
            return

        items = sorted(items, key=lambda x: x[0], reverse=reverse)[:gallery_top]

        for number, (_, image_path, image) in enumerate(items, 1):
            digest = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:8]
            destination = directory / f"{number:03d}_{digest}.jpg"
            cv2.imwrite(str(destination), image)

    save_gallery(failures_by_type["false_negatives"], false_negative_dir, True)
    save_gallery(failures_by_type["localization_failures"], localization_dir, True)
    save_gallery(failures_by_type["false_positives"], false_positive_dir, True)
    save_gallery(failures_by_type["confusions"], confusion_dir, True)

    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference and reporting pipeline for the DRISHTI-SSS detector."
    )

    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Image or directory. prepared = already training-compatible images; "
            "raw_sss = raw SSS input using documented P1/P99 + tile/pad geometry."
        ),
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["prepared", "raw_sss"], default="prepared")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--thresholds",
        default=None,
        help=(
            "JSON class thresholds. Default: "
            + json.dumps(DEFAULT_CLASS_THRESHOLDS, separators=(",", ":"))
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compare predictions against YOLO ground-truth labels.",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help=(
            "Directory containing YOLO .txt labels. Defaults to a sibling "
            "'labels' directory next to the source image directory."
        ),
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help="IoU required for a same-class TP. Default: 0.50",
    )
    parser.add_argument(
        "--gallery-top",
        type=int,
        default=30,
        help="Maximum failure examples saved per gallery. Default: 30",
    )

    args = parser.parse_args()

    if not 0.0 < args.iou <= 1.0:
        parser.error("--iou must be in (0, 1].")
    if not 0.0 < args.merge_iou <= 1.0:
        parser.error("--merge-iou must be in (0, 1].")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive.")
    if args.max_det <= 0:
        parser.error("--max-det must be positive.")
    if not 0.0 < args.match_iou <= 1.0:
        parser.error("--match-iou must be in (0, 1].")
    if args.gallery_top < 0:
        parser.error("--gallery-top must be >= 0.")

    return args


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    thresholds = parse_thresholds(args.thresholds)

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"Output directory is not empty: {output}\n"
            "Use --overwrite or select another --output path."
        )

    ensure_dir(output)
    annotated_dir = output / "annotated"
    reports_dir = output / "reports"
    ensure_dir(annotated_dir)
    ensure_dir(reports_dir)

    paths = source_image_paths(source)
    model = YOLO(str(model_path))
    started_at = utc_now()

    print("=" * 72)
    print("DRISHTI-SSS INFERENCE + REPORTING PIPELINE")
    print("=" * 72)
    print(f"Started:    {started_at}")
    print(f"Model:      {model_path}")
    print(f"Source:     {source}")
    print(f"Mode:       {args.mode}")
    print(f"Images:     {len(paths)}")
    print(f"Image size: {args.imgsz}")
    print(f"Device:     {args.device}")
    print(f"Model IoU:  {args.iou}")
    print(f"Merge IoU:  {args.merge_iou}")
    print(f"Thresholds: {json.dumps(thresholds)}")

    if args.mode == "raw_sss":
        print()
        print(
            "RAW SSS MODE: P1/P99 normalization + 1024x500 tiling + "
            "centered 1024x1024 padding. No Lee/CLAHE is invented here."
        )

    image_records: list[dict[str, Any]] = []
    csv_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, image_path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {image_path.name}")

        try:
            if args.mode == "prepared":
                gray, detections, inference_metadata = run_prepared(
                    model=model,
                    image_path=image_path,
                    imgsz=args.imgsz,
                    device=args.device,
                    iou=args.iou,
                    max_det=args.max_det,
                    thresholds=thresholds,
                )
            else:
                gray, detections, inference_metadata = run_raw_sss(
                    model=model,
                    image_path=image_path,
                    imgsz=args.imgsz,
                    device=args.device,
                    iou=args.iou,
                    max_det=args.max_det,
                    thresholds=thresholds,
                    merge_iou=args.merge_iou,
                )

            output_stem = safe_output_stem(image_path, source)
            annotated_path = annotated_dir / f"{output_stem}.png"
            annotated = annotate_image(gray, detections)

            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"Could not write annotated image: {annotated_path}")

            record = {
                "source_image": str(image_path),
                "source_name": image_path.name,
                "mode": args.mode,
                "image_sha256": sha256_file(image_path),
                "image_width": int(gray.shape[1]),
                "image_height": int(gray.shape[0]),
                "detection_count": len(detections),
                "detections": detections,
                "annotated_image": str(annotated_path),
                "inference": inference_metadata,
            }
            image_records.append(record)

            for det in detections:
                row = dict(det)
                row["source_image"] = str(image_path)
                row["mode"] = args.mode
                csv_records.append(row)

            print(f"  Final detections: {len(detections)}")

        except Exception as exc:
            failures.append({"source_image": str(image_path), "error": str(exc)})
            print(f"  ERROR: {exc}")

    summary = build_summary(image_records, thresholds)
    summary["failed_images"] = len(failures)
    summary["failures"] = failures

    evaluation_report = None

    if args.evaluate:
        if args.mode != "prepared":
            raise RuntimeError(
                "Ground-truth evaluation currently requires --mode prepared "
                "because the labels are defined in prepared-image coordinates."
            )

        if not source.is_dir():
            raise RuntimeError(
                "--evaluate expects --source to be the image directory containing "
                "the evaluated dataset."
            )

        labels_dir = (
            Path(args.labels_dir).expanduser().resolve()
            if args.labels_dir
            else source.parent / "labels"
        )

        if not labels_dir.exists():
            raise FileNotFoundError(f"Ground-truth labels directory not found: {labels_dir}")

        print()
        print("GROUND-TRUTH EVALUATION")
        print(f"Labels:     {labels_dir}")
        print(f"Match IoU:  {args.match_iou}")

        evaluation_report = evaluate_predictions(
            image_records=image_records,
            source=source,
            labels_dir=labels_dir,
            match_iou=args.match_iou,
            output_dir=output,
            gallery_top=args.gallery_top,
        )

    detections_report = {
        "generated_at": utc_now(),
        "model": str(model_path),
        "mode": args.mode,
        "evaluation_enabled": bool(args.evaluate),
        "images": image_records,
    }

    run_metadata = {
        "project": "AI-Powered Underwater Marine Debris & Anomaly Detection System",
        "pipeline": "infer_sss.py",
        "started_at": started_at,
        "finished_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "ultralytics_version": getattr(__import__("ultralytics"), "__version__", "unknown"),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "source": str(source),
        "mode": args.mode,
        "configuration": {
            "imgsz": args.imgsz,
            "device": args.device,
            "iou": args.iou,
            "merge_iou": args.merge_iou,
            "max_det": args.max_det,
            "class_thresholds": thresholds,
            "evaluation_enabled": bool(args.evaluate),
            "evaluation_match_iou": args.match_iou if args.evaluate else None,
            "evaluation_gallery_top": args.gallery_top if args.evaluate else None,
        },
        "preprocessing": (
            {
                "tile_width": TILE_WIDTH,
                "tile_height": TILE_HEIGHT,
                "overlap": OVERLAP,
                "stride": STRIDE,
                "model_canvas": [MODEL_CANVAS, MODEL_CANVAS],
                "normalization": "P1/P99 clipping -> [0,255] uint8",
                "lee_filter": False,
                "clahe": False,
            }
            if args.mode == "raw_sss"
            else {
                "input_assumption": "already training-compatible",
                "additional_preprocessing": False,
            }
        ),
        "class_names": CLASS_NAMES,
        "thresholds_note": (
            "Operating thresholds are configurable values, not calibrated probability claims."
        ),
    }

    (reports_dir / "detections.json").write_text(
        json.dumps(detections_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(csv_records, reports_dir / "detections.csv")
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if evaluation_report is not None:
        summary["evaluation"] = {
            "overall": evaluation_report["overall"],
            "classes": evaluation_report["classes"],
            "missing_label_files": evaluation_report["missing_label_files"],
        }
        (reports_dir / "summary_with_evaluation.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (reports_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)
    print(f"Images processed:    {summary['images_processed']}")
    print(f"Images with results: {summary['images_with_detections']}")
    print(f"Final detections:    {summary['total_final_detections']}")
    print(f"Failed images:       {summary['failed_images']}")
    print()
    for name in CLASS_NAMES.values():
        print(f"  {name:12s}: {summary['class_counts'][name]:5d}")
    print()
    print(f"Annotated: {annotated_dir}")
    print(f"JSON:      {reports_dir / 'detections.json'}")
    print(f"CSV:       {reports_dir / 'detections.csv'}")
    print(f"Summary:   {reports_dir / 'summary.json'}")
    if evaluation_report is not None:
        print(f"Evaluation: {output / 'evaluation' / 'evaluation.json'}")
        print(f"Eval TXT:   {output / 'evaluation' / 'evaluation.txt'}")
        print(
            f"Eval P/R/F1: {evaluation_report['overall']['precision_at_threshold']:.4f} / "
            f"{evaluation_report['overall']['recall_at_threshold']:.4f} / "
            f"{evaluation_report['overall']['f1_at_threshold']:.4f}"
        )
    print(f"Metadata:  {reports_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
