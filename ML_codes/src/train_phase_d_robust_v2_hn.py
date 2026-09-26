from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# PHASE D FINAL TRAINING CONFIG
# ============================================================

DATASET = Path(r"D:\DATASETS\DRISHTI-SSS-ROBUST-V2-HN")
DATA_YAML = DATASET / "drishti.yaml"

BASE_MODEL = Path(
    r"K:\Debris model\runs\drishti_v5_ghost2x\weights\best.pt"
)

PROJECT = Path(r"K:\Debris model\runs")
RUN_NAME = "drishti_phase_d_robust_v2_hn"

IMGSZ = 640
EPOCHS = 120
BATCH = 16
DEVICE = 0
WORKERS = 8
SEED = 26057
PATIENCE = 30

EXPECTED_COUNTS = {
    "train": 5636,
    "val": 780,
    "test": 779,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_count(split: str) -> int:
    image_dir = DATASET / split / "images"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(
        1
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def preflight() -> None:
    print("=" * 72)
    print("PHASE D FINAL TRAINING PRE-FLIGHT")
    print("=" * 72)

    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {BASE_MODEL}")

    run_dir = PROJECT / RUN_NAME
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Choose a new RUN_NAME rather than overwriting an experiment."
        )

    for split, expected in EXPECTED_COUNTS.items():
        count = image_count(split)
        print(f"{split:>5}: {count}")
        if count != expected:
            raise RuntimeError(
                f"{split} count mismatch: expected {expected}, found {count}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training run is configured for GPU 0."
        )

    gpu_name = torch.cuda.get_device_name(DEVICE)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.version.cuda}")
    print(f"Ultralytics: {__import__('ultralytics').__version__}")
    print(f"Dataset: {DATASET}")
    print(f"Checkpoint: {BASE_MODEL}")
    print(f"Checkpoint SHA256: {sha256(BASE_MODEL)}")

    print("\nTraining configuration:")
    print(f"  imgsz       = {IMGSZ}")
    print(f"  epochs      = {EPOCHS}")
    print(f"  batch       = {BATCH}")
    print(f"  device      = {DEVICE}")
    print(f"  workers     = {WORKERS}")
    print(f"  seed        = {SEED}")
    print(f"  patience    = {PATIENCE}")
    print("  deterministic = True")

    print("\nPhase D policy:")
    print("  Robust-V2-HN offline robustness data is already built.")
    print("  The core Ultralytics training recipe is kept consistent")
    print("  with the earlier V1/Ghost2X training for a fair comparison.")
    print("  VAL and TEST are not modified by this script.")
    print("=" * 72)


def main() -> None:
    preflight()

    PROJECT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BASE_MODEL))

    train_args = {
        "data": str(DATA_YAML),
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "project": str(PROJECT),
        "name": RUN_NAME,
        "seed": SEED,
        "deterministic": True,
        "patience": PATIENCE,
        "pretrained": False,
        "cache": False,
        "amp": True,
        "plots": True,
        "save": True,
        "verbose": True,
        "exist_ok": False,
    }

    print("\nStarting Phase D training...")
    print("Do not use TEST for model selection during this run.\n")

    results = model.train(**train_args)

    save_dir = Path(results.save_dir)

    metadata = {
        "phase": "D",
        "experiment": RUN_NAME,
        "dataset": str(DATASET),
        "dataset_yaml": str(DATA_YAML),
        "base_model": str(BASE_MODEL),
        "base_model_sha256": sha256(BASE_MODEL),
        "expected_counts": EXPECTED_COUNTS,
        "training": {
            "imgsz": IMGSZ,
            "epochs": EPOCHS,
            "batch": BATCH,
            "device": DEVICE,
            "workers": WORKERS,
            "seed": SEED,
            "deterministic": True,
            "patience": PATIENCE,
            "cache": False,
            "amp": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(DEVICE),
            "ultralytics": __import__("ultralytics").__version__,
        },
        "post_training_policy": {
            "val_selection": True,
            "threshold_calibration": "VAL only",
            "sealed_test": "after VAL/calibration/stress-test",
        },
    }

    metadata_path = save_dir / "phase_d_training_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("PHASE D TRAINING COMPLETE")
    print("=" * 72)
    print(f"Run directory : {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last checkpoint: {save_dir / 'weights' / 'last.pt'}")
    print(f"Metadata       : {metadata_path}")
    print("\nNext step: evaluate the new model on VAL only.")
    print("=" * 72)


if __name__ == "__main__":
    main()
