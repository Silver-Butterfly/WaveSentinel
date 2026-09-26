"""
PHASE 1 | EXP 1 ROBUST PIPELINE DETECTOR
SubPipeMini2 SSS-only

PURPOSE
-------
Train a stronger, more acquisition-robust pipeline detector while preserving
the already completed sealed test result as a final benchmark.

Why this is a new experiment:
- v2 used a hash-selected source-level validation split from one long
  acquisition window.
- v2 training was heavily represented by HF tiles while the sealed test is
  LF-heavy.
- v2 used YOLO11n.
- v2 did not explicitly balance HF/LF exposure.

v3 changes ONLY the development training procedure:
1. Validation is a contiguous temporal block at the END of acquisition group 0.
2. HF/LF source timestamps are kept together.
3. LF training tiles are oversampled to balance HF/LF exposure.
4. YOLO11s replaces YOLO11n.
5. Mild intensity/scale augmentation is used. No photographic augmentation.
6. Acquisition group 1 remains completely sealed and is NOT used for model
   selection.
7. No source or preprocessing files are modified.

IMPORTANT
---------
This script does NOT evaluate the sealed test set. Model selection happens
using validation only. Run the final test evaluator exactly once after the
model is selected.

Run:
    python train_exp1_pipeline_robust_v3.py
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
from ultralytics import YOLO

PREPROCESSED_ROOT = Path(r"K:\Debris model\preprocessing_output")
SPLIT_MANIFEST = PREPROCESSED_ROOT / "splits" / "split_manifest.json"

EXPERIMENT_ROOT = Path(r"K:\Debris model\experiments")
EXP_NAME = "exp01_subpipe_pipeline_robust_v3"
RUN_ROOT = EXPERIMENT_ROOT / EXP_NAME
DATASET_ROOT = RUN_ROOT / "dataset"

MODEL_CHECKPOINT = "yolo11s.pt"

IMG_SIZE = 1024
EPOCHS = 100
BATCH = 4
WORKERS = 4
SEED = 42
PATIENCE = 20

# Temporal validation from the END of acquisition group 0.
# This makes validation a real temporal holdout instead of another random
# sample from the same highly redundant sequence.
VAL_FRACTION = 0.20

# Keep all available development records, including provisional background
# candidates. They are explicitly documented as not confirmed negatives.
USE_BACKGROUND_CANDIDATES = True

# Representation balancing: oversample the representation with fewer training
# tiles until HF/LF tile exposure is approximately equal.
BALANCE_REPRESENTATIONS = True

# Conservative sonar augmentation.
DEGREES = 0.0
TRANSLATE = 0.05
SCALE = 0.15
SHEAR = 0.0
PERSPECTIVE = 0.0
FLIPUD = 0.0
FLIPLR = 0.5
MOSAIC = 0.0
MIXUP = 0.0
COPY_PASTE = 0.0

# Mild value perturbation for acquisition/intensity robustness.
HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.10

random.seed(SEED)
np.random.seed(SEED)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_timestamp(record: dict) -> float:
    value = record.get("source_timestamp")
    if value is None:
        value = Path(record["source_image"]).stem
    return float(value)


def source_group_key(record: dict) -> str:
    # HF/LF frames captured at the same timestamp are treated as one source
    # unit for validation assignment.
    return f"{source_timestamp(record):.3f}"


def validate_manifest(subpipe: list[dict]):
    if not subpipe:
        raise RuntimeError("SubPipe manifest is empty.")

    categories = {
        ann.get("category_name")
        for r in subpipe
        for ann in r.get("annotations", [])
    }
    if categories - {"Pipeline"}:
        raise RuntimeError(f"Unexpected categories: {categories}")

    groups = {
        r.get("acquisition_group")
        for r in subpipe
        if r.get("record_type") != "unannotated_source_image"
    }
    if groups - {0, 1}:
        raise RuntimeError(f"Unexpected acquisition groups: {groups}")


def build_temporal_split(subpipe: list[dict]):
    """
    Existing acquisition group 1 stays sealed.

    Within acquisition group 0, the latest contiguous 20% of unique source
    timestamps becomes validation. HF/LF records sharing a timestamp stay
    together.
    """
    dev = [
        r for r in subpipe
        if r.get("acquisition_group") == 0
        and r.get("record_type") in {"positive_tile", "background_candidate_tile"}
    ]
    test = [
        r for r in subpipe
        if r.get("acquisition_group") == 1
        and r.get("record_type") in {"positive_tile", "background_candidate_tile"}
    ]

    by_source = defaultdict(list)
    for r in dev:
        by_source[source_group_key(r)].append(r)

    timestamps = sorted(by_source, key=float)
    if len(timestamps) < 10:
        raise RuntimeError(
            f"Too few development source timestamps: {len(timestamps)}"
        )

    val_count = max(1, int(round(len(timestamps) * VAL_FRACTION)))
    val_keys = set(timestamps[-val_count:])
    train_keys = set(timestamps[:-val_count])

    train = [r for k in timestamps if k in train_keys for r in by_source[k]]
    val = [r for k in timestamps if k in val_keys for r in by_source[k]]

    train_sources = {source_group_key(r) for r in train}
    val_sources = {source_group_key(r) for r in val}
    test_sources = {source_group_key(r) for r in test}

    if train_sources & val_sources:
        raise RuntimeError("Temporal train/val source overlap.")
    if train_sources & test_sources:
        raise RuntimeError("Train/test source overlap.")
    if val_sources & test_sources:
        raise RuntimeError("Validation/test source overlap.")

    if {r.get("acquisition_group") for r in train} != {0}:
        raise RuntimeError("Train acquisition-group invariant failed.")
    if {r.get("acquisition_group") for r in val} != {0}:
        raise RuntimeError("Validation acquisition-group invariant failed.")
    if {r.get("acquisition_group") for r in test} != {1}:
        raise RuntimeError("Test acquisition-group invariant failed.")

    return train, val, test, {
        "method": "contiguous_temporal_holdout_within_acquisition_group_0",
        "validation_fraction": VAL_FRACTION,
        "development_group": 0,
        "sealed_test_group": 1,
        "train_source_units": len(train_sources),
        "val_source_units": len(val_sources),
        "test_source_units": len(test_sources),
        "train_time_min": min(map(source_timestamp, train)),
        "train_time_max": max(map(source_timestamp, train)),
        "val_time_min": min(map(source_timestamp, val)),
        "val_time_max": max(map(source_timestamp, val)),
        "test_time_min": min(map(source_timestamp, test)),
        "test_time_max": max(map(source_timestamp, test)),
    }


def filter_records(records: list[dict]):
    out = []
    for r in records:
        rt = r["record_type"]
        if rt == "positive_tile":
            out.append(r)
        elif rt == "background_candidate_tile" and USE_BACKGROUND_CANDIDATES:
            out.append(r)
        elif rt not in {"positive_tile", "background_candidate_tile"}:
            raise RuntimeError(f"Unexpected record type: {rt}")
    return out


def balance_representation(records: list[dict]):
    if not BALANCE_REPRESENTATIONS:
        return records, {"enabled": False}

    by_rep = defaultdict(list)
    for r in records:
        by_rep[r["representation"]].append(r)

    counts = {k: len(v) for k, v in by_rep.items()}
    if set(counts) != {"PBM_HF", "PBM_LF"}:
        raise RuntimeError(f"Expected HF/LF only, got {counts}")

    target = max(counts.values())
    balanced = []

    for rep in ("PBM_HF", "PBM_LF"):
        items = by_rep[rep]
        if not items:
            raise RuntimeError(f"No records for {rep}")
        balanced.extend(items)
        needed = target - len(items)
        if needed > 0:
            # Deterministic cyclic oversampling. We do not alter source files.
            for i in range(needed):
                balanced.append(items[i % len(items)])

    # Stable ordering keeps reproducibility.
    balanced = sorted(
        balanced,
        key=lambda r: (
            source_timestamp(r),
            r["representation"],
            int(r.get("tile_index", 0)),
            r["tile_id"],
        ),
    )

    return balanced, {
        "enabled": True,
        "original_counts": counts,
        "balanced_counts": {
            "PBM_HF": sum(r["representation"] == "PBM_HF" for r in balanced),
            "PBM_LF": sum(r["representation"] == "PBM_LF" for r in balanced),
        },
    }


def unique_name(base: str, suffix: str, used: set[str], repeat_index: int):
    name = f"{base}{suffix}"
    if name not in used:
        used.add(name)
        return name

    candidate = f"{base}_rep{repeat_index:03d}{suffix}"
    while candidate in used:
        repeat_index += 1
        candidate = f"{base}_rep{repeat_index:03d}{suffix}"
    used.add(candidate)
    return candidate


def copy_dataset_records(records: list[dict], split: str, allow_repeat_names: bool):
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    used = set()
    copied = 0

    for idx, r in enumerate(records):
        src_img = Path(r["output_image"])
        src_lbl = Path(r["output_label"])
        if not src_img.exists():
            raise FileNotFoundError(src_img)
        if not src_lbl.exists():
            raise FileNotFoundError(src_lbl)

        base = r["tile_id"]
        image_name = unique_name(
            base,
            src_img.suffix.lower(),
            used,
            idx if allow_repeat_names else 0,
        )
        label_name = Path(image_name).with_suffix(".txt").name

        dst_img = image_dir / image_name
        dst_lbl = label_dir / label_name

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        copied += 1

    return copied


def write_yaml():
    yaml_path = DATASET_ROOT / "subpipe_pipeline.yaml"
    yaml_path.write_text(
        f"""path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: Pipeline
""",
        encoding="utf-8",
    )
    return yaml_path


def main():
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(SPLIT_MANIFEST)
    if RUN_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment:\n{RUN_ROOT}"
        )

    manifest = load_json(SPLIT_MANIFEST)
    subpipe = manifest.get("subpipe", [])
    validate_manifest(subpipe)

    train_raw, val_raw, test_raw, split_info = build_temporal_split(subpipe)

    train = filter_records(train_raw)
    val = filter_records(val_raw)
    test = filter_records(test_raw)

    train, balance_info = balance_representation(train)

    if not train or not val or not test:
        raise RuntimeError("One of train/val/test is empty.")

    # Test inventory is recorded but never opened for training/model selection.
    test_source_images = {r["source_image"] for r in test}

    RUN_ROOT.mkdir(parents=True, exist_ok=False)

    train_count = copy_dataset_records(train, "train", allow_repeat_names=True)
    val_count = copy_dataset_records(val, "val", allow_repeat_names=False)
    test_count = copy_dataset_records(test, "test", allow_repeat_names=False)
    yaml_path = write_yaml()

    config = {
        "experiment": EXP_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "SSS-only",
        "dataset": "SubPipeMini2",
        "task": "object_detection",
        "class_mapping": {"0": "Pipeline"},
        "model_checkpoint": MODEL_CHECKPOINT,
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "batch": BATCH,
        "workers": WORKERS,
        "seed": SEED,
        "patience": PATIENCE,
        "validation": split_info,
        "representation_balance": balance_info,
        "background_candidates": {
            "used": USE_BACKGROUND_CANDIDATES,
            "confirmed_negatives": False,
        },
        "augmentation": {
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
            "hsv_h": HSV_H,
            "hsv_s": HSV_S,
            "hsv_v": HSV_V,
        },
        "dataset_counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "test_source_images": len(test_source_images),
        },
        "test_policy": (
            "Acquisition group 1 is sealed and is not used for training, "
            "checkpoint selection, augmentation decisions, or threshold tuning."
        ),
    }

    config_path = RUN_ROOT / "training_config.json"
    save_json(config_path, config)

    print("=" * 78)
    print("PHASE 1 | EXP 1 ROBUST PIPELINE DETECTOR v3")
    print("=" * 78)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Model:   {MODEL_CHECKPOINT}")
    print("\nTemporal split:")
    print(json.dumps(split_info, indent=2))
    print("\nRepresentation balance:")
    print(json.dumps(balance_info, indent=2))
    print("\nDataset counts:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count} (SEALED)")

    model = YOLO(MODEL_CHECKPOINT)

    print("\nStarting v3 training...")
    results = model.train(
        data=str(yaml_path),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        seed=SEED,
        val=True,
        project=str(EXPERIMENT_ROOT),
        name=EXP_NAME,
        exist_ok=False,
        device=0,
        augment=True,
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
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        patience=PATIENCE,
        plots=True,
        save=True,
        verbose=True,
    )

    # Find actual Ultralytics checkpoint location.
    candidates = [
        RUN_ROOT / "weights" / "best.pt",
        RUN_ROOT / "weights" / "last.pt",
        EXPERIMENT_ROOT / f"{EXP_NAME}-2" / "weights" / "best.pt",
        EXPERIMENT_ROOT / f"{EXP_NAME}-2" / "weights" / "last.pt",
    ]
    best = next((p for p in candidates if p.exists() and p.name == "best.pt"), None)
    if best is None:
        raise FileNotFoundError(
            "v3 training finished but best.pt was not found. "
            f"Checked: {candidates}"
        )

    result = {
        "training_complete": True,
        "experiment": EXP_NAME,
        "checkpoint": str(best),
        "dataset": str(DATASET_ROOT),
        "test_evaluation_performed": False,
        "next_step": (
            "Inspect validation performance and select the model before "
            "running one final sealed-test evaluation."
        ),
    }
    save_json(RUN_ROOT / "training_complete.json", result)

    print("\n" + "=" * 78)
    print("V3 TRAINING COMPLETE")
    print("=" * 78)
    print(f"Best checkpoint: {best}")
    print("SEALED TEST WAS NOT EVALUATED.")
    print("No source/preprocessed dataset was modified.")


if __name__ == "__main__":
    main()
