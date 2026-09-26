from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


CLASS_NAMES = {
    0: "pipe",
    1: "shipwreck",
    2: "mine",
    3: "ghost_net",
}

THRESHOLDS = {
    "pipe": 0.74,
    "shipwreck": 0.38,
    "mine": 0.71,
    "ghost_net": 0.40,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Mine from the original clean Ghost2X training set, not the robustness-expanded set.
DEFAULT_DATA = r"D:\DATASETS\DRISHTI-SSS-GHOST2X"
DEFAULT_MODEL = r"K:\Debris model\runs\drishti_v5_ghost2x\weights\best.pt"
DEFAULT_OUTPUT = r"K:\Debris model\runs\hard_negative_mining_v2"

MODEL_CONF = 0.001
MODEL_IOU = 0.70
MAX_DET = 300
IMG_SIZE = 640


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise RuntimeError(f"Unsupported image shape {image.shape}: {path}")

    if gray.dtype != "uint8":
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    return gray


def load_empty_label_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    candidates: list[Path] = []

    for image_path in list_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for {image_path}: {label_path}"
            )

        if not label_path.read_text(errors="replace").strip():
            candidates.append(image_path)

    return candidates


def annotate(image, detections):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_xyxy"]]
        label = f'{det["class"]} {det["confidence"]:.3f}'

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            1,
        )

        top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(
            canvas,
            (x1, top),
            (x1 + tw + 6, top + th + baseline + 6),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            canvas,
            label,
            (x1 + 3, top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def extract_detections(result):
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, confidence, class_id in zip(boxes, confs, classes):
        class_id = int(class_id)

        if class_id not in CLASS_NAMES:
            continue

        class_name = CLASS_NAMES[class_id]
        confidence = float(confidence)

        if confidence < MODEL_CONF:
            continue

        detections.append({
            "class_id": class_id,
            "class": class_name,
            "confidence": confidence,
            "operating_threshold": THRESHOLDS[class_name],
            "above_operating_threshold": confidence >= THRESHOLDS[class_name],
            "bbox_xyxy": [float(v) for v in box],
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Memory-safe hard-negative candidate mining from clean "
            "empty-label training images."
        )
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--candidate-conf",
        type=float,
        default=0.15,
        help="Lowest candidate confidence saved for review.",
    )
    parser.add_argument(
        "--operating-only",
        action="store_true",
        help="Only save detections at or above their frozen class threshold.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMG_SIZE,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    if not 0.0 <= args.candidate_conf < 1.0:
        raise ValueError("--candidate-conf must be in [0,1).")
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive.")

    data_root = Path(args.data).expanduser().resolve()
    images_dir = data_root / "train" / "images"
    labels_dir = data_root / "train" / "labels"
    model_path = Path(args.model).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {output_root}. "
                "Use --overwrite or choose a new output path."
            )
        shutil.rmtree(output_root)

    ensure_dir(output_root)
    candidate_dir = output_root / "candidate_images"
    ensure_dir(candidate_dir)

    # This intentionally mines from the clean Ghost2X source rather than
    # the robustness-expanded dataset. Augmentation artifacts should not
    # become hard negatives.
    candidates = load_empty_label_images(images_dir, labels_dir)

    if not candidates:
        raise RuntimeError("No empty-label training images were found.")

    model = YOLO(str(model_path))

    print("=" * 78)
    print("DRISHTI-SSS HARD-NEGATIVE MINING V2")
    print("=" * 78)
    print(f"Dataset         : {data_root}")
    print(f"Model           : {model_path}")
    print(f"Empty-label imgs: {len(candidates)}")
    print(f"Candidate conf  : {args.candidate_conf}")
    print(f"Operating only  : {args.operating_only}")
    print(f"Image size      : {args.imgsz}")
    print(f"Device          : {args.device}")
    print()
    print("Memory-safe mode: inference is ONE IMAGE AT A TIME.")
    print("Candidates are REVIEW ONLY and are not added automatically.")
    print()

    records: list[dict[str, Any]] = []
    class_counts = defaultdict(int)

    for index, image_path in enumerate(candidates, 1):
        print(f"\r[{index}/{len(candidates)}] {image_path.name}", end="")

        try:
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                conf=MODEL_CONF,
                iou=MODEL_IOU,
                max_det=MAX_DET,
                verbose=False,
                batch=1,
            )[0]

            detections = extract_detections(result)

            filtered: list[dict[str, Any]] = []
            for det in detections:
                threshold = (
                    THRESHOLDS[det["class"]]
                    if args.operating_only
                    else args.candidate_conf
                )
                if det["confidence"] >= threshold:
                    filtered.append(det)

            if filtered:
                for det in filtered:
                    class_counts[det["class"]] += 1

                records.append({
                    "source_image": str(image_path),
                    "source_name": image_path.name,
                    "image_sha256": sha256_file(image_path),
                    "detection_count": len(filtered),
                    "max_confidence": max(d["confidence"] for d in filtered),
                    "above_operating_threshold": any(
                        d["above_operating_threshold"] for d in filtered
                    ),
                    "detections": filtered,
                    "candidate_status": "REVIEW_REQUIRED",
                })

        except Exception as exc:
            records.append({
                "source_image": str(image_path),
                "source_name": image_path.name,
                "image_sha256": None,
                "detection_count": 0,
                "max_confidence": None,
                "above_operating_threshold": False,
                "detections": [],
                "candidate_status": "ERROR",
                "error": str(exc),
            })

        # Explicit cleanup after every image to keep the 6 GB GPU stable.
        del result
        if index % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print()
    records.sort(
        key=lambda r: (
            r["candidate_status"] == "REVIEW_REQUIRED",
            r["max_confidence"] or -1.0,
        ),
        reverse=True,
    )

    review_records = [
        r for r in records
        if r["candidate_status"] == "REVIEW_REQUIRED"
    ][: args.max_candidates]

    for rank, record in enumerate(review_records, 1):
        source = Path(record["source_image"])
        image = read_image(source)
        annotated_image = annotate(image, record["detections"])

        out_name = f"{rank:04d}__{source.stem}.png"
        out_path = candidate_dir / out_name

        if not cv2.imwrite(str(out_path), annotated_image):
            raise RuntimeError(f"Could not write {out_path}")

        record["review_rank"] = rank
        record["review_image"] = str(out_path)

    report = {
        "created_at": utc_now(),
        "purpose": "Hard-negative candidate mining from clean empty-label training images",
        "dataset": str(data_root),
        "model": str(model_path),
        "candidate_confidence": args.candidate_conf,
        "operating_only": args.operating_only,
        "image_size": args.imgsz,
        "max_candidates": args.max_candidates,
        "frozen_thresholds": THRESHOLDS,
        "empty_label_images_scanned": len(candidates),
        "candidate_images_found": len(review_records),
        "candidate_detection_counts": dict(sorted(class_counts.items())),
        "memory_safe": True,
        "input_policy": "clean source training set only; do not mine robustness-expanded copies",
        "records": review_records,
        "review_rule": (
            "A candidate is not automatically a false positive. "
            "Manually inspect the sonar image before adding it to V2."
        ),
    }

    report_path = output_root / "hard_negative_candidates.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = output_root / "hard_negative_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "source_name",
            "max_confidence",
            "detection_count",
            "classes",
            "above_operating_threshold",
            "review_image",
        ])

        for record in review_records:
            classes = "|".join(
                f'{d["class"]}:{d["confidence"]:.3f}'
                for d in record["detections"]
            )
            writer.writerow([
                record["review_rank"],
                record["source_name"],
                f'{record["max_confidence"]:.6f}',
                record["detection_count"],
                classes,
                record["above_operating_threshold"],
                record["review_image"],
            ])

    print("=" * 78)
    print("MINING COMPLETE")
    print("=" * 78)
    print(f"Empty-label images scanned : {len(candidates)}")
    print(f"Candidate images found     : {len(review_records)}")
    print(f"Candidate images saved     : {candidate_dir}")
    print(f"JSON                       : {report_path}")
    print(f"CSV                        : {csv_path}")
    print()
    print("No candidates were added to the training dataset.")


if __name__ == "__main__":
    main()
