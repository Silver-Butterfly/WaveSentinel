"""
PHASE 1 | EXP 1
V3 SEALED TEST EVALUATION

One-shot evaluation of the trained V3 Pipeline Detector on the permanently
held-out SubPipeMini2 acquisition group.

This script:
- verifies the V3 dataset/checkpoint artifacts before evaluation
- evaluates split="test" exactly once
- does not tune thresholds on the sealed test
- does not train or resume training
- does not modify source or preprocessed data
- saves a JSON evaluation report
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


DATASET_ROOT = Path(
    r"K:\Debris model\experiments\exp01_subpipe_pipeline_robust_v3\dataset"
)
CHECKPOINT = Path(
    r"K:\Debris model\experiments\exp01_subpipe_pipeline_robust_v3-2\weights\best.pt"
)
RUN_ROOT = CHECKPOINT.parent.parent
OUTPUT_DIR = RUN_ROOT / "final_evaluation"
REPORT_PATH = OUTPUT_DIR / "exp01_v3_final_sealed_test_evaluation.json"

DATA_YAML = DATASET_ROOT / "subpipe_pipeline.yaml"
TEST_IMAGES = DATASET_ROOT / "images" / "test"
TEST_LABELS = DATASET_ROOT / "labels" / "test"

EXPECTED_TEST_TILES = 96
EXPECTED_TEST_SOURCE_IMAGES = 28
EXPECTED_TEST_ACQUISITION_GROUP = 1

IMGSZ = 1024
DEVICE = 0
CONF = 0.001
IOU = 0.7


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_split_manifest():
    candidates = [
        DATASET_ROOT.parent / "split_manifest.json",
        DATASET_ROOT / "split_manifest.json",
        RUN_ROOT / "split_manifest.json",
        RUN_ROOT / "dataset_split_manifest.json",
        RUN_ROOT / "v3_split_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in RUN_ROOT.glob("**/*.json"):
        name = p.name.lower()
        if "split" in name or "manifest" in name:
            return p
    return None


def inspect_test_artifact():
    errors = []
    warnings = {}

    required = [
        DATASET_ROOT,
        CHECKPOINT,
        DATA_YAML,
        TEST_IMAGES,
        TEST_LABELS,
    ]

    for p in required:
        if not p.exists():
            errors.append(f"Missing required artifact: {p}")

    if errors:
        return errors, warnings, {}

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )
    labels = sorted(
        p for p in TEST_LABELS.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    )

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:
        errors.append(f"{len(missing_labels)} test images have no label file")
    if orphan_labels:
        errors.append(f"{len(orphan_labels)} orphan test label files")

    if len(images) != EXPECTED_TEST_TILES:
        errors.append(
            f"Expected {EXPECTED_TEST_TILES} sealed test tiles, found {len(images)}"
        )

    invalid_records = 0
    out_of_bounds = 0
    object_count = 0
    positive_tiles = 0
    background_tiles = 0

    for label_path in labels:
        has_object = False
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue

            has_object = True
            parts = line.split()
            if len(parts) != 5:
                invalid_records += 1
                continue

            try:
                cls, xc, yc, w, h = map(float, parts)
            except ValueError:
                invalid_records += 1
                continue

            object_count += 1

            if cls != 0:
                invalid_records += 1

            if w <= 0 or h <= 0:
                invalid_records += 1

            if not (
                -1e-6 <= xc <= 1 + 1e-6
                and -1e-6 <= yc <= 1 + 1e-6
                and -1e-6 <= w <= 1 + 1e-6
                and -1e-6 <= h <= 1 + 1e-6
            ):
                out_of_bounds += 1

            if (
                xc - w / 2 < -1e-6
                or xc + w / 2 > 1 + 1e-6
                or yc - h / 2 < -1e-6
                or yc + h / 2 > 1 + 1e-6
            ):
                out_of_bounds += 1

        if has_object:
            positive_tiles += 1
        else:
            background_tiles += 1

    if invalid_records:
        errors.append(f"Invalid test label records: {invalid_records}")
    if out_of_bounds:
        errors.append(f"Out-of-bounds test boxes: {out_of_bounds}")

    manifest_path = discover_split_manifest()
    manifest = load_json(manifest_path) if manifest_path else None

    manifest_summary = None
    if manifest:
        manifest_summary = {
            "path": str(manifest_path),
            "method": manifest.get("method"),
            "validation_fraction": manifest.get("validation_fraction"),
            "development_group": manifest.get("development_group"),
            "sealed_test_group": manifest.get("sealed_test_group"),
            "train_source_units": manifest.get("train_source_units"),
            "val_source_units": manifest.get("val_source_units"),
            "test_source_units": manifest.get("test_source_units"),
            "test_time_min": manifest.get("test_time_min"),
            "test_time_max": manifest.get("test_time_max"),
        }

    if background_tiles == 0:
        warnings["background_tiles"] = "No background-candidate tiles found."

    return errors, warnings, {
        "tiles": len(images),
        "labels": len(labels),
        "positive_tiles": positive_tiles,
        "background_tiles": background_tiles,
        "objects": object_count,
        "source_images_expected": EXPECTED_TEST_SOURCE_IMAGES,
        "acquisition_group_expected": EXPECTED_TEST_ACQUISITION_GROUP,
        "invalid_label_records": invalid_records,
        "out_of_bounds_boxes": out_of_bounds,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "split_manifest": manifest_summary,
    }


def main():
    print("=" * 78)
    print("PHASE 1 | EXP 1 FINAL SEALED TEST EVALUATION")
    print("SubPipeMini2 Pipeline Detector | Robust V3")
    print("=" * 78)

    errors, warnings, artifact = inspect_test_artifact()

    print("\nArtifact alignment:")
    print(f"  Dataset:    {DATASET_ROOT}")
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  YAML:       {DATA_YAML}")

    print("\nSealed-test artifact:")
    for key, value in artifact.items():
        print(f"  {key}: {value}")

    if errors:
        print("\nBLOCKING ARTIFACT ERRORS:")
        for error in errors:
            print(f"  ERROR: {error}")
        print("\nTEST WAS NOT EVALUATED.")
        sys.exit(2)

    print("\nWarnings:")
    if warnings:
        for key, value in warnings.items():
            print(f"  WARNING: {key}: {value}")
    else:
        print("  None")

    print("\nEvaluation policy:")
    print("  - One standard sealed-test evaluation")
    print("  - No threshold tuning on test")
    print("  - No training/resume")
    print("  - No dataset modification")
    print(f"  - imgsz={IMGSZ}, conf={CONF}, iou={IOU}")

    checkpoint_hash = sha256_file(CHECKPOINT)
    print(f"\nCheckpoint SHA256: {checkpoint_hash}")

    print("\nLoading best.pt...")
    model = YOLO(str(CHECKPOINT))

    print("\nRunning sealed test...")
    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        plots=True,
        save_json=True,
        verbose=True,
    )

    box = getattr(results, "box", None)

    def metric(name):
        if box is None:
            return None
        try:
            return float(getattr(box, name))
        except Exception:
            return None

    metrics = {
        "precision": metric("mp"),
        "recall": metric("mr"),
        "map50": metric("map50"),
        "map50_95": metric("map"),
        "map75": metric("map75"),
    }

    per_class_ap50 = None
    try:
        if box is not None and getattr(box, "ap50", None) is not None:
            per_class_ap50 = [float(x) for x in box.ap50.tolist()]
    except Exception:
        pass

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "exp01_subpipe_pipeline_robust_v3",
        "dataset": "SubPipeMini2",
        "modality": "Side-Scan Sonar (SSS) only",
        "task": "Pipeline object detection",
        "model": "YOLO11s",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_root": str(DATASET_ROOT),
        "data_yaml": str(DATA_YAML),
        "test_status": "SEALED_HELD_OUT_TEST",
        "test_acquisition_group": EXPECTED_TEST_ACQUISITION_GROUP,
        "test_artifact": artifact,
        "evaluation_settings": {
            "imgsz": IMGSZ,
            "device": DEVICE,
            "conf": CONF,
            "iou": IOU,
            "threshold_tuning_on_test": False,
        },
        "metrics": metrics,
        "per_class_ap50": per_class_ap50,
        "warnings": warnings,
        "test_evaluation_performed": True,
        "source_dataset_modified": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL SEALED TEST RESULT")
    print("=" * 78)
    for key, value in metrics.items():
        print(f"  {key:12s}: {value}")

    print("\nReport:")
    print(f"  {REPORT_PATH}")

    print("\nSEALED TEST EVALUATION COMPLETE.")
    print("No source/preprocessed dataset was modified.")


if __name__ == "__main__":
    main()
