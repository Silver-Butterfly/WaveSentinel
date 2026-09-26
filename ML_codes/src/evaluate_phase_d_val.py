from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


# ============================================================
# PHASE D | VALIDATION EVALUATION ONLY
# ============================================================

DATASET = Path(r"D:\DATASETS\DRISHTI-SSS-ROBUST-V2-HN")
DATA_YAML = DATASET / "drishti.yaml"
MODEL = Path(
    r"K:\Debris model\runs\drishti_phase_d_robust_v2_hn\weights\best.pt"
)

OUTPUT = Path(
    r"K:\Debris model\runs\drishti_phase_d_robust_v2_hn\val_evaluation"
)

SPLIT = "val"
IMGSZ = 640
BATCH = 16
DEVICE = 0
CONF = 0.001
IOU = 0.70
MAX_DET = 300

EXPECTED_VAL_IMAGES = 780

CLASS_NAMES = {
    0: "pipe",
    1: "shipwreck",
    2: "mine",
    3: "ghost_net",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_images(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def preflight() -> None:
    print("=" * 78)
    print("PHASE D | VALIDATION EVALUATION")
    print("=" * 78)

    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not MODEL.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL}")

    val_images = DATASET / "val" / "images"
    val_labels = DATASET / "val" / "labels"

    if not val_images.is_dir():
        raise FileNotFoundError(val_images)

    if not val_labels.is_dir():
        raise FileNotFoundError(val_labels)

    actual = count_images(val_images)
    if actual != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"VAL image count mismatch: expected {EXPECTED_VAL_IMAGES}, found {actual}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This evaluation expects GPU 0.")

    print(f"Dataset   : {DATASET}")
    print(f"Checkpoint: {MODEL}")
    print(f"SHA256    : {sha256(MODEL)}")
    print(f"VAL images: {actual}")
    print(f"GPU       : {torch.cuda.get_device_name(DEVICE)}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"CUDA      : {torch.version.cuda}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print()
    print("Evaluation configuration:")
    print(f"  split    = {SPLIT}")
    print(f"  imgsz    = {IMGSZ}")
    print(f"  batch    = {BATCH}")
    print(f"  device   = {DEVICE}")
    print(f"  conf     = {CONF}")
    print(f"  iou      = {IOU}")
    print(f"  max_det  = {MAX_DET}")
    print()
    print("SEALED TEST: NOT TOUCHED")
    print("=" * 78)


def main() -> None:
    preflight()

    OUTPUT.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(MODEL))

    metrics = model.val(
        data=str(DATA_YAML),
        split=SPLIT,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        plots=True,
        save_json=False,
        verbose=True,
        project=str(OUTPUT.parent),
        name=OUTPUT.name,
        exist_ok=False,
    )

    box = metrics.box
    names = getattr(metrics, "names", CLASS_NAMES)

    per_class = {}

    for class_id, class_name in CLASS_NAMES.items():
        try:
            p = float(box.p[class_id])
            r = float(box.r[class_id])
            f1 = float(box.f1[class_id])
            ap50 = float(box.ap50[class_id])
            ap5095 = float(box.ap[class_id])

            per_class[class_name] = {
                "class_id": class_id,
                "precision": p,
                "recall": r,
                "f1": f1,
                "mAP50": ap50,
                "mAP50_95": ap5095,
            }
        except (IndexError, TypeError):
            per_class[class_name] = {
                "class_id": class_id,
                "precision": None,
                "recall": None,
                "f1": None,
                "mAP50": None,
                "mAP50_95": None,
            }

    report = {
        "phase": "D",
        "stage": "validation_evaluation",
        "dataset": str(DATASET),
        "model": str(MODEL),
        "model_sha256": sha256(MODEL),
        "split": SPLIT,
        "validation_images": EXPECTED_VAL_IMAGES,
        "sealed_test_touched": False,
        "configuration": {
            "imgsz": IMGSZ,
            "batch": BATCH,
            "device": DEVICE,
            "confidence_floor": CONF,
            "nms_iou": IOU,
            "max_det": MAX_DET,
        },
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
        },
        "per_class": per_class,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(DEVICE),
        },
    }

    report_path = OUTPUT / "phase_d_val_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    summary_lines = [
        "=" * 78,
        "PHASE D | VALIDATION EVALUATION COMPLETE",
        "=" * 78,
        f"Model: {MODEL}",
        f"VAL images: {EXPECTED_VAL_IMAGES}",
        "",
        "OVERALL",
        f"Precision : {box.mp:.6f}",
        f"Recall    : {box.mr:.6f}",
        f"mAP50     : {box.map50:.6f}",
        f"mAP50-95  : {box.map:.6f}",
        "",
        "PER CLASS",
    ]

    for name in CLASS_NAMES.values():
        row = per_class[name]
        summary_lines.append(
            f"{name:12s} "
            f"P={row['precision']:.6f} "
            f"R={row['recall']:.6f} "
            f"F1={row['f1']:.6f} "
            f"mAP50={row['mAP50']:.6f} "
            f"mAP50-95={row['mAP50_95']:.6f}"
        )

    summary_lines.extend([
        "",
        "Sealed TEST: NOT TOUCHED",
        f"Report: {report_path}",
        "=" * 78,
    ])

    print("\n".join(summary_lines))
    (OUTPUT / "phase_d_val_evaluation.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
