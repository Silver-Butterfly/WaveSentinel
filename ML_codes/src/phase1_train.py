"""
EXP 1: SubPipeMini2 Pipeline Detector Baseline
Project: AI-Powered Underwater Marine Debris & Anomaly Detection System
Scope: SSS-only

CORRECTED TRAINING DESIGN
-------------------------
- Dataset: SubPipeMini2 only
- Task: Pipeline object detection
- Class 0: Pipeline
- Input: 1024 x 1024
- Representations: PBM_HF + PBM_LF
- Existing acquisition split is preserved:
      acquisition group 0 -> development pool
      acquisition group 1 -> final held-out test
- A VALIDATION split is created ONLY from acquisition group 0.
- Validation is source-level, not tile-level, to prevent overlapping tiles
  from the same source image appearing in both train and validation.
- Acquisition group 1 is NEVER used during training or model selection.
- AI4Shipwrecks is NOT used in Exp 1.
- Source datasets and preprocessing outputs are never modified.

WHY A VALIDATION SET IS CREATED
--------------------------------
Ultralytics requires a validation dataset for detector training/model
selection. We therefore create a source-level validation subset from the
existing TRAIN acquisition group only.

This does NOT alter the original preprocessing split manifest. It creates a
separate experiment-local training dataset.

No separate manual feature extraction is required. The pretrained YOLO
backbone performs feature extraction internally.

BEFORE RUNNING
--------------
    pip install ultralytics

RUN
---
    python phase1_train.py
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSED_ROOT = Path(r"K:\Debris model\preprocessing_output")
SPLIT_ROOT = PREPROCESSED_ROOT / "splits"
SPLIT_MANIFEST = SPLIT_ROOT / "split_manifest.json"
SANITY_REPORT = SPLIT_ROOT / "final_pretraining_sanity_check.json"

EXPERIMENT_ROOT = Path(r"K:\Debris model\experiments")

# New explicit run name so the failed YAML-only attempt is preserved.
EXP_NAME = "exp01_subpipe_pipeline_baseline_v2"
RUN_ROOT = EXPERIMENT_ROOT / EXP_NAME

# Baseline model.
MODEL_CHECKPOINT = "yolo11n.pt"

# Model/training parameters.
IMG_SIZE = 1024
EPOCHS = 100
BATCH = 8
WORKERS = 4
SEED = 42

# Source-level validation fraction from the EXISTING training acquisition group.
# This is not taken from the final test acquisition group.
VAL_FRACTION = 0.15

# Background candidates are used as provisional negatives.
# The audit explicitly states they are NOT confirmed negatives.
USE_BACKGROUND_CANDIDATES = True

# Conservative sonar augmentation.
AUGMENT = True
DEGREES = 0.0
TRANSLATE = 0.05
SCALE = 0.20
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0

# Early stopping is based on the validation split from acquisition group 0.
PATIENCE = 20

# Final evaluation is performed once, after training, on acquisition group 1.
EVALUATE_FINAL_TEST = True


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite: {dst}")
    shutil.copy2(src, dst)


def make_unique_name(stem: str, suffix: str, used: set[str]) -> str:
    candidate = f"{stem}{suffix}"
    if candidate not in used:
        used.add(candidate)
        return candidate

    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def validate_manifest_record(record: dict):
    required = [
        "split",
        "output_image",
        "output_label",
        "tile_id",
        "record_type",
        "source_image",
        "acquisition_group",
    ]
    missing = [x for x in required if x not in record]
    if missing:
        raise ValueError(
            f"Manifest record missing fields {missing}: {record}"
        )


# ============================================================
# SOURCE-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

def make_source_level_dev_split(
    records: list[dict],
) -> tuple[dict[str, list[dict]], dict]:
    """
    Create train/validation from the EXISTING acquisition-group-0 training
    records.

    Critical property:
        all tiles originating from one source_image stay in exactly one of
        train or validation.

    The final acquisition-group-1 test records are never passed here.
    """

    train_records = [
        r for r in records
        if r["split"] == "train"
    ]
    test_records = [
        r for r in records
        if r["split"] == "test"
    ]

    train_groups = {r["acquisition_group"] for r in train_records}
    test_groups = {r["acquisition_group"] for r in test_records}

    if train_groups != {0}:
        raise RuntimeError(
            f"Expected existing training records to belong only to "
            f"acquisition group 0, got {sorted(train_groups)}"
        )

    if test_groups != {1}:
        raise RuntimeError(
            f"Expected existing test records to belong only to "
            f"acquisition group 1, got {sorted(test_groups)}"
        )

    # Group all development tiles by source image.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in train_records:
        by_source[record["source_image"]].append(record)

    sources = sorted(by_source)

    if len(sources) < 10:
        raise RuntimeError(
            f"Too few source images for a source-level validation split: "
            f"{len(sources)}"
        )

    # Deterministic hash ranking avoids tile-level randomness and remains
    # reproducible across runs.
    ranked_sources = sorted(
        sources,
        key=lambda s: sha256_text(f"{SEED}:{s}")
    )

    val_count = max(1, int(round(len(ranked_sources) * VAL_FRACTION)))
    val_sources = set(ranked_sources[:val_count])

    if len(val_sources) >= len(ranked_sources):
        raise RuntimeError("Validation split consumed all development sources.")

    dev_train_sources = set(ranked_sources) - val_sources

    selected = {
        "train": [
            r for source in sorted(dev_train_sources)
            for r in by_source[source]
        ],
        "val": [
            r for source in sorted(val_sources)
            for r in by_source[source]
        ],
        "test": test_records,
    }

    # Hard leakage checks.
    train_source_set = {
        r["source_image"] for r in selected["train"]
    }
    val_source_set = {
        r["source_image"] for r in selected["val"]
    }
    test_source_set = {
        r["source_image"] for r in selected["test"]
    }

    if train_source_set & val_source_set:
        raise RuntimeError("Source leakage between train and validation.")

    if train_source_set & test_source_set:
        raise RuntimeError("Source leakage between train and test.")

    if val_source_set & test_source_set:
        raise RuntimeError("Source leakage between validation and test.")

    # Acquisition-group checks.
    if {
        r["acquisition_group"] for r in selected["train"]
    } != {0}:
        raise RuntimeError("Train split contains a non-development acquisition group.")

    if {
        r["acquisition_group"] for r in selected["val"]
    } != {0}:
        raise RuntimeError("Validation split contains a non-development group.")

    if {
        r["acquisition_group"] for r in selected["test"]
    } != {1}:
        raise RuntimeError("Test split does not remain acquisition group 1.")

    split_info = {
        "method": "source_level_split_within_existing_training_acquisition_group",
        "seed": SEED,
        "validation_fraction": VAL_FRACTION,
        "development_acquisition_group": 0,
        "final_test_acquisition_group": 1,
        "source_counts": {
            "train": len(train_source_set),
            "val": len(val_source_set),
            "test": len(test_source_set),
        },
        "tile_counts": {
            "train": len(selected["train"]),
            "val": len(selected["val"]),
            "test": len(selected["test"]),
        },
        "train_validation_source_overlap": 0,
        "train_test_source_overlap": 0,
        "validation_test_source_overlap": 0,
        "test_is_sealed": True,
    }

    return selected, split_info


# ============================================================
# BUILD EXPERIMENT-LOCAL YOLO DATASET
# ============================================================

def build_yolo_dataset(
    manifest: dict,
) -> tuple[Path, dict]:
    """
    Creates a completely separate experiment-local YOLO dataset.

    Nothing under preprocessing_output or the original dataset is changed.
    """

    dataset_root = RUN_ROOT / "dataset"

    if dataset_root.exists():
        raise FileExistsError(
            f"Experiment dataset already exists:\n{dataset_root}\n\n"
            "This script refuses to overwrite an existing experiment. "
            "Delete/rename the incomplete v2 experiment directory before "
            "rerunning, if necessary."
        )

    subpipe = manifest.get("subpipe", [])
    if not subpipe:
        raise ValueError("SubPipe manifest is empty.")

    selected, split_info = make_source_level_dev_split(subpipe)

    # Apply background-candidate policy independently to each split.
    filtered: dict[str, list[dict]] = {}

    for split, records in selected.items():
        filtered[split] = []

        for record in records:
            record_type = record["record_type"]

            if record_type == "positive_tile":
                filtered[split].append(record)

            elif record_type == "background_candidate_tile":
                if USE_BACKGROUND_CANDIDATES:
                    filtered[split].append(record)

            else:
                raise ValueError(
                    f"Unexpected record_type {record_type} "
                    f"for tile {record['tile_id']}"
                )

    counts = {
        split: dict(Counter(
            r["record_type"] for r in filtered[split]
        ))
        for split in filtered
    }

    # Every selected split must contain data.
    for split in ("train", "val", "test"):
        if not filtered[split]:
            raise RuntimeError(f"No records selected for {split}.")

    # Copy images and labels.
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        used_image_names: set[str] = set()

        for record in filtered[split]:
            image_src = Path(record["output_image"])
            label_src = Path(record["output_label"])

            if not image_src.exists():
                raise FileNotFoundError(image_src)

            if not label_src.exists():
                raise FileNotFoundError(label_src)

            # Use tile ID to preserve traceability while guaranteeing
            # unique filenames inside each experiment split.
            stem = str(record["tile_id"])
            image_name = make_unique_name(
                stem,
                image_src.suffix.lower(),
                used_image_names,
            )
            label_name = Path(image_name).with_suffix(".txt").name

            safe_copy(image_src, image_dir / image_name)
            safe_copy(label_src, label_dir / label_name)

        print(
            f"{split.upper()}: copied {len(filtered[split])} images"
        )

    # Ultralytics requires train + val. Test is kept as a separate field and
    # is only accessed explicitly after training.
    dataset_yaml = dataset_root / "subpipe_pipeline.yaml"

    yaml_text = f"""path: {dataset_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: Pipeline
"""

    dataset_yaml.write_text(yaml_text, encoding="utf-8")

    dataset_stats = {
        "selected_records": {
            split: len(filtered[split])
            for split in ("train", "val", "test")
        },
        "record_types": counts,
        "source_level_split": split_info,
        "background_candidates_used": USE_BACKGROUND_CANDIDATES,
        "background_candidates_are_confirmed_negatives": False,
    }

    return dataset_yaml, dataset_stats


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("EXP 1: SUBPIPEMINI2 PIPELINE DETECTOR BASELINE")
    print("=" * 78)

    # --------------------------------------------------------
    # Confirm data gate
    # --------------------------------------------------------

    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing split manifest:\n{SPLIT_MANIFEST}"
        )

    if not SANITY_REPORT.exists():
        raise FileNotFoundError(
            f"Missing final pre-training sanity-check report:\n{SANITY_REPORT}"
        )

    sanity = load_json(SANITY_REPORT)
    gate = sanity.get("training_gate")

    print(f"Previous training gate: {gate}")

    if gate != "PASS":
        raise RuntimeError(
            f"Training blocked because final sanity-check gate is {gate}."
        )

    manifest = load_json(SPLIT_MANIFEST)

    # --------------------------------------------------------
    # Strict taxonomy confirmation
    # --------------------------------------------------------

    all_categories = {
        ann.get("category_name")
        for record in manifest.get("subpipe", [])
        for ann in record.get("annotations", [])
    }

    if all_categories - {"Pipeline"}:
        raise RuntimeError(
            "Unexpected SubPipe detector categories found: "
            f"{sorted(all_categories)}"
        )

    # --------------------------------------------------------
    # Build source-level train/validation/test experiment dataset
    # --------------------------------------------------------

    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_yaml, dataset_stats = build_yolo_dataset(manifest)

    config = {
        "experiment": "Exp 1",
        "run_name": EXP_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "SSS-only",
        "dataset": "SubPipeMini2",
        "task": "object_detection",
        "class_mapping": {
            "0": "Pipeline"
        },
        "ai4shipwrecks_used": False,

        "preprocessed_root": str(PREPROCESSED_ROOT),
        "split_manifest": str(SPLIT_MANIFEST),
        "dataset_yaml": str(dataset_yaml),

        "model": {
            "checkpoint": MODEL_CHECKPOINT,
            "input_size": IMG_SIZE,
        },

        "training": {
            "epochs": EPOCHS,
            "batch": BATCH,
            "workers": WORKERS,
            "seed": SEED,
            "patience": PATIENCE,
            "validation_source": (
                "15% source-level subset of existing acquisition group 0 "
                "training sources"
            ),
            "final_test_source": (
                "existing acquisition group 1, completely held out"
            ),
        },

        "feature_extraction": {
            "manual_feature_extraction": False,
            "description": (
                "Feature extraction is performed internally by the pretrained "
                "YOLO backbone during training/inference."
            ),
        },

        "background_candidate_policy": {
            "use_background_candidates": USE_BACKGROUND_CANDIDATES,
            "interpretation": (
                "Empty-label background_candidate_tile records are used as "
                "provisional negatives. They are not confirmed ground-truth "
                "negatives."
            ),
        },

        "augmentation": {
            "enabled": AUGMENT,
            "degrees": DEGREES,
            "translate": TRANSLATE,
            "scale": SCALE,
            "shear": SHEAR,
            "perspective": PERSPECTIVE,
            "flipud": FLIPUD,
            "fliplr": FLIPLR,
            "mosaic": MOSAIC,
            "mixup": MIXUP,
            "copy_paste": COPY_PASTE,
        },

        "dataset_selection": dataset_stats,

        "evaluation_policy": {
            "validation_used_for_model_selection": True,
            "final_test_used_after_training_only": True,
            "test_split": "acquisition_group_1",
            "test_is_deployment_estimate": False,
            "test_warning": (
                "The final test contains fewer than 30 source images and "
                "should be interpreted as a held-out acquisition-window "
                "result, not a statistically strong deployment estimate."
            ),
        },

        "immutability": {
            "source_datasets_modified": False,
            "preprocessing_outputs_modified": False,
        },
    }

    config_path = RUN_ROOT / "experiment_config.json"
    write_json(config_path, config)

    print("\nExperiment configuration saved:")
    print(config_path)

    print("\nSelected data:")
    print(json.dumps(dataset_stats, indent=2))

    # --------------------------------------------------------
    # Load Ultralytics
    # --------------------------------------------------------

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run:\n"
            "    pip install ultralytics"
        ) from exc

    print("\nLoading pretrained detector:")
    print(f"  {MODEL_CHECKPOINT}")

    model = YOLO(MODEL_CHECKPOINT)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\nStarting training...")
    print("  Train: acquisition group 0 development sources")
    print("  Val:   source-level subset of acquisition group 0")
    print("  Test:  acquisition group 1, SEALED until training finishes")
    print("  AI4:   NOT USED")

    train_results = model.train(
        data=str(dataset_yaml),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        seed=SEED,

        # Ultralytics will use images/val for validation/model selection.
        val=True,

        project=str(EXPERIMENT_ROOT),
        name=EXP_NAME,
        exist_ok=False,

        # Conservative sonar augmentation.
        augment=AUGMENT,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,

        # Early stopping based on validation performance.
        patience=PATIENCE,

        # Save metrics/plots/checkpoints.
        plots=True,
        save=True,
        verbose=True,
    )

    print("\nTraining completed.")

    # --------------------------------------------------------
    # Locate checkpoint
    # --------------------------------------------------------

    weight_candidates = [
        RUN_ROOT / "weights" / "best.pt",
        RUN_ROOT / "weights" / "last.pt",
    ]

    best_weight = next(
        (p for p in weight_candidates if p.exists()),
        None,
    )

    if best_weight is None:
        raise FileNotFoundError(
            "Training finished but best.pt/last.pt was not found under:\n"
            f"{RUN_ROOT / 'weights'}"
        )

    print(f"Best checkpoint: {best_weight}")

    write_json(
        RUN_ROOT / "training_complete.json",
        {
            "training_complete": True,
            "checkpoint": str(best_weight),
            "validation_was_used": True,
            "final_test_evaluation_pending": EVALUATE_FINAL_TEST,
        },
    )

    # --------------------------------------------------------
    # FINAL SEALED TEST EVALUATION
    # --------------------------------------------------------

    if EVALUATE_FINAL_TEST:
        print("\n" + "=" * 78)
        print("FINAL HELD-OUT TEST EVALUATION")
        print("=" * 78)
        print(
            "This is the first time acquisition group 1 is being used."
        )

        best_model = YOLO(str(best_weight))

        metrics = best_model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=IMG_SIZE,
            batch=BATCH,
            workers=WORKERS,
            plots=True,
            project=str(EXPERIMENT_ROOT),
            name=f"{EXP_NAME}_test",
            exist_ok=False,
            verbose=True,
        )

        metric_summary = {}

        try:
            metric_summary["map50"] = float(metrics.box.map50)
            metric_summary["map50_95"] = float(metrics.box.map)
            metric_summary["precision"] = float(metrics.box.mp)
            metric_summary["recall"] = float(metrics.box.mr)
        except Exception as exc:
            metric_summary["metric_extraction_error"] = str(exc)

        evaluation_report = {
            "experiment": "Exp 1",
            "run_name": EXP_NAME,
            "evaluation_split": "test",
            "test_acquisition_group": 1,
            "test_is_held_out": True,
            "test_was_not_used_for_training": True,
            "test_was_not_used_for_model_selection": True,
            "checkpoint": str(best_weight),
            "metrics": metric_summary,
            "interpretation_warning": (
                "The SubPipe test split is a short held-out acquisition "
                "window with fewer than 30 source images. Treat these "
                "metrics as a held-out acquisition-window result, not as "
                "a statistically strong estimate of deployment performance."
            ),
            "background_candidate_warning": (
                "Test background_candidate_tile samples are not confirmed "
                "ground-truth negatives."
                if USE_BACKGROUND_CANDIDATES
                else "Background candidates were excluded."
            ),
        }

        evaluation_path = RUN_ROOT / "final_test_evaluation.json"
        write_json(evaluation_path, evaluation_report)

        print("\nFinal test evaluation:")
        print(json.dumps(evaluation_report, indent=2))

    print("\n" + "=" * 78)
    print("EXP 1 COMPLETE")
    print("=" * 78)
    print(f"Experiment directory: {RUN_ROOT}")
    print(f"Configuration:        {config_path}")
    print(f"Checkpoint:           {best_weight}")
    print(
        "\nNext step: inspect validation curves and the final held-out "
        "test metrics before changing the architecture or adding AI4."
    )


if __name__ == "__main__":
    main()
