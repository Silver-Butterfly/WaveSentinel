from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SOURCE_DEFAULT = r"D:\DATASETS\DRISHTI-SSS-GHOST2X"
OUTPUT_DEFAULT = r"D:\DATASETS\DRISHTI-SSS-ROBUST-V2"
SEED_DEFAULT = 26057

# Additional training images as a fraction of the clean source training set.
# The original clean images are always retained.
ROBUST_FRACTION_DEFAULT = 0.30

# Distribution across generated robustness variants.
VARIANT_WEIGHTS = {
    "mild_speckle": 0.35,
    "moderate_speckle": 0.25,
    "shadow_gain": 0.15,
    "dropout_resolution": 0.15,
    "combined_moderate": 0.10,
}


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


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise RuntimeError(f"Unsupported image shape {image.shape}: {path}")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


def multiplicative_speckle(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    noise = rng.lognormal(
        mean=-0.5 * sigma * sigma,
        sigma=sigma,
        size=x.shape,
    )
    y = np.clip(x * noise, 0.0, 1.0)
    return np.rint(y * 255.0).astype(np.uint8)


def gain_and_contrast(
    image: np.ndarray,
    gain_std: float,
    bias_std: float,
    gamma_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    x = image.astype(np.float32) / 255.0

    gain = float(rng.normal(1.0, gain_std))
    bias = float(rng.normal(0.0, bias_std))
    gamma = float(rng.uniform(gamma_range[0], gamma_range[1]))

    x = np.clip(x * gain + bias, 0.0, 1.0)
    x = np.power(x, gamma)

    return np.rint(np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def attenuation_band(
    image: np.ndarray,
    strength_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape

    # Smooth horizontal band to mimic localized acoustic attenuation.
    y0_low = max(0, int(0.10 * h))
    y0_high = max(y0_low + 1, int(0.60 * h))
    y0 = int(rng.integers(y0_low, y0_high))

    min_h = max(8, int(0.12 * h))
    max_h = max(min_h + 1, int(0.45 * h))
    band_h = int(rng.integers(min_h, max_h))

    mask = np.zeros((h, w), dtype=np.float32)
    y1 = min(h, y0 + band_h)
    mask[y0:y1, :] = 1.0

    kernel = max(7, ((w // 30) * 2 + 1))
    smooth = cv2.GaussianBlur(mask, (kernel, kernel), 0)

    strength = float(rng.uniform(*strength_range))
    factor = 1.0 - np.clip(smooth * strength, 0.0, 0.90)

    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def sparse_stripe_dropout(
    image: np.ndarray,
    fraction_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape
    x = image.astype(np.float32).copy()

    target_fraction = float(rng.uniform(*fraction_range))
    target_width = max(1, int(round(w * target_fraction)))
    removed = 0

    attempts = 0
    max_attempts = 30

    while removed < target_width and attempts < max_attempts:
        attempts += 1

        width_max = max(4, w // 30)
        width = int(rng.integers(3, width_max + 1))
        start = int(rng.integers(0, max(1, w - width + 1)))

        mode = rng.choice(("attenuate", "zero"))
        if mode == "zero":
            x[:, start:start + width] = 0.0
        else:
            x[:, start:start + width] *= float(rng.uniform(0.10, 0.35))

        removed += width

    return np.clip(x, 0, 255).astype(np.uint8)


def resolution_degrade(
    image: np.ndarray,
    scale_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape
    scale = float(rng.uniform(*scale_range))

    small_w = max(16, int(round(w * scale)))
    small_h = max(16, int(round(h * scale)))

    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def mild_blur(
    image: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if float(rng.random()) >= probability:
        return image

    kernel = int(rng.choice((3, 5)))
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def apply_variant(
    image: np.ndarray,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    operations: list[str] = []

    if variant == "mild_speckle":
        image = multiplicative_speckle(image, 0.10, rng)
        operations.append("multiplicative_speckle_sigma_0.10")
        image = mild_blur(image, 0.20, rng)
        operations.append("optional_blur_p0.20")
        return image, operations

    if variant == "moderate_speckle":
        image = multiplicative_speckle(image, 0.20, rng)
        operations.append("multiplicative_speckle_sigma_0.20")
        image = gain_and_contrast(
            image,
            gain_std=0.08,
            bias_std=0.025,
            gamma_range=(0.92, 1.08),
            rng=rng,
        )
        operations.append("mild_gain_bias_gamma")
        return image, operations

    if variant == "shadow_gain":
        image = gain_and_contrast(
            image,
            gain_std=0.14,
            bias_std=0.04,
            gamma_range=(0.88, 1.12),
            rng=rng,
        )
        operations.append("gain_bias_gamma_variation")

        image = attenuation_band(
            image,
            strength_range=(0.30, 0.60),
            rng=rng,
        )
        operations.append("acoustic_attenuation_band")

        return image, operations

    if variant == "dropout_resolution":
        image = multiplicative_speckle(image, 0.18, rng)
        operations.append("multiplicative_speckle_sigma_0.18")

        image = sparse_stripe_dropout(
            image,
            fraction_range=(0.02, 0.06),
            rng=rng,
        )
        operations.append("sparse_stripe_dropout_2_to_6pct")

        image = resolution_degrade(
            image,
            scale_range=(0.70, 0.88),
            rng=rng,
        )
        operations.append("resolution_factor_0.70_to_0.88")

        return image, operations

    if variant == "combined_moderate":
        image = multiplicative_speckle(image, 0.24, rng)
        operations.append("multiplicative_speckle_sigma_0.24")

        image = gain_and_contrast(
            image,
            gain_std=0.10,
            bias_std=0.03,
            gamma_range=(0.90, 1.10),
            rng=rng,
        )
        operations.append("moderate_gain_bias_gamma")

        image = attenuation_band(
            image,
            strength_range=(0.20, 0.45),
            rng=rng,
        )
        operations.append("mild_acoustic_attenuation")

        image = sparse_stripe_dropout(
            image,
            fraction_range=(0.01, 0.04),
            rng=rng,
        )
        operations.append("sparse_stripe_dropout_1_to_4pct")

        return image, operations

    raise ValueError(f"Unknown variant: {variant}")


def choose_variant_counts(n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    remaining = n

    ordered = list(VARIANT_WEIGHTS.items())
    for index, (variant, weight) in enumerate(ordered):
        if index == len(ordered) - 1:
            counts[variant] = remaining
        else:
            count = int(round(n * weight))
            count = min(count, remaining)
            counts[variant] = count
            remaining -= count

    return counts


def copy_clean_split(source_root: Path, output_root: Path, split: str) -> tuple[list[Path], list[Path]]:
    src_images = source_root / split / "images"
    src_labels = source_root / split / "labels"

    out_images = output_root / split / "images"
    out_labels = output_root / split / "labels"

    ensure_dir(out_images)
    ensure_dir(out_labels)

    image_paths = list_images(src_images)
    copied: list[Path] = []

    for src in image_paths:
        dst = out_images / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

        label = src_labels / f"{src.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing label for {src}: {label}")

        shutil.copy2(label, out_labels / label.name)

    return image_paths, copied


def write_yaml(output_root: Path) -> None:
    yaml = """path: D:/DATASETS/DRISHTI-SSS-ROBUST-V2
train: train/images
val: val/images
test: test/images

names:
  0: pipe
  1: shipwreck
  2: mine
  3: ghost_net
"""
    (output_root / "drishti.yaml").write_text(yaml, encoding="utf-8")


def build_augmented_train(
    source_root: Path,
    output_root: Path,
    fraction: float,
    seed: int,
) -> dict:
    src_images_dir = source_root / "train" / "images"
    src_labels_dir = source_root / "train" / "labels"

    out_images_dir = output_root / "train" / "images"
    out_labels_dir = output_root / "train" / "labels"
    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    source_images = list_images(src_images_dir)
    if not source_images:
        raise RuntimeError(f"No training images found in {src_images_dir}")

    rng = random.Random(seed)

    generated_total = int(round(len(source_images) * fraction))
    generated_total = max(0, generated_total)
    variant_counts = choose_variant_counts(generated_total)

    selected: list[tuple[Path, str, int]] = []

    for variant, count in variant_counts.items():
        if count <= 0:
            continue
        candidates = source_images.copy()
        rng.shuffle(candidates)

        for index in range(count):
            src = candidates[index % len(candidates)]
            variant_seed = seed + len(selected) * 104729
            selected.append((src, variant, variant_seed))

    rng.shuffle(selected)

    manifest_rows = []
    class_counter = Counter()
    generated_counter = Counter()

    for idx, (src, variant, variant_seed) in enumerate(selected):
        label_path = src_labels_dir / f"{src.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        image = read_gray(src)
        transformed, operations = apply_variant(image, variant, variant_seed)

        out_name = f"{src.stem}__robust_{idx + 1:05d}.png"
        out_path = out_images_dir / out_name
        label_out = out_labels_dir / f"{Path(out_name).stem}.txt"

        if not cv2.imwrite(str(out_path), transformed):
            raise RuntimeError(f"Could not write {out_path}")

        shutil.copy2(label_path, label_out)

        class_ids = []
        for line in label_path.read_text(errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_ids.append(int(parts[0]))
                except ValueError:
                    pass

        for class_id in set(class_ids):
            class_counter[class_id] += 1

        generated_counter[variant] += 1

        manifest_rows.append({
            "generated_image": str(out_path),
            "source_image": str(src),
            "source_sha256": sha256_file(src),
            "variant": variant,
            "seed": variant_seed,
            "operations": operations,
            "label": str(label_out),
        })

    manifest_path = output_root / "robustness_augmentation_manifest.json"
    manifest_payload = {
        "created_at": utc_now(),
        "source": str(source_root),
        "output": str(output_root),
        "seed": seed,
        "source_train_images": len(source_images),
        "generated_robust_images": len(selected),
        "robust_fraction": fraction,
        "variant_weights": VARIANT_WEIGHTS,
        "variant_counts": generated_counter,
        "generated_class_presence": {
            str(k): v for k, v in sorted(class_counter.items())
        },
        "records": manifest_rows,
    }

    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )

    csv_path = output_root / "robustness_augmentation_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "generated_image",
                "source_image",
                "source_sha256",
                "variant",
                "seed",
                "operations",
                "label",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({
                **row,
                "operations": "|".join(row["operations"]),
            })

    return {
        "source_train_images": len(source_images),
        "generated_robust_images": len(selected),
        "variant_counts": dict(generated_counter),
        "generated_class_presence": dict(class_counter),
        "manifest_json": str(manifest_path),
        "manifest_csv": str(csv_path),
    }


def count_images(path: Path) -> int:
    return len(list_images(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DRISHTI-SSS robust V2 training dataset."
    )
    parser.add_argument("--source", default=SOURCE_DEFAULT)
    parser.add_argument("--output", default=OUTPUT_DEFAULT)
    parser.add_argument("--fraction", type=float, default=ROBUST_FRACTION_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.fraction <= 1.0:
        raise ValueError("--fraction must be between 0 and 1.")

    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    if output_root.exists():
        existing = list(output_root.iterdir())
        if existing and not args.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {output_root}\n"
                "Use --overwrite or choose a new output path."
            )
        if args.overwrite:
            shutil.rmtree(output_root)

    ensure_dir(output_root)

    print("=" * 78)
    print("DRISHTI-SSS ROBUST V2 DATASET BUILDER")
    print("=" * 78)
    print(f"Source dataset : {source_root}")
    print(f"Output dataset : {output_root}")
    print(f"Robust fraction: {args.fraction:.2%}")
    print(f"Seed           : {args.seed}")
    print()
    print("IMPORTANT")
    print("  - VAL is copied unchanged.")
    print("  - TEST is copied unchanged.")
    print("  - Existing clean TRAIN images are retained.")
    print("  - Labels are copied unchanged because all transforms are photometric.")
    print()

    # Clean split copies.
    _, train_copied = copy_clean_split(source_root, output_root, "train")
    _, val_copied = copy_clean_split(source_root, output_root, "val")
    _, test_copied = copy_clean_split(source_root, output_root, "test")

    print(f"Clean train images copied: {len(train_copied)}")
    print(f"Clean val images copied  : {len(val_copied)}")
    print(f"Clean test images copied : {len(test_copied)}")
    print()

    result = build_augmented_train(
        source_root=source_root,
        output_root=output_root,
        fraction=args.fraction,
        seed=args.seed,
    )

    write_yaml(output_root)

    total_train = count_images(output_root / "train" / "images")
    total_val = count_images(output_root / "val" / "images")
    total_test = count_images(output_root / "test" / "images")

    summary = {
        "created_at": utc_now(),
        "source": str(source_root),
        "output": str(output_root),
        "seed": args.seed,
        "robust_fraction": args.fraction,
        "train_images_total": total_train,
        "val_images_total": total_val,
        "test_images_total": total_test,
        **result,
        "rules": {
            "validation_untouched": True,
            "test_untouched": True,
            "clean_train_retained": True,
            "labels_reused_without_geometric_change": True,
        },
    }

    summary_path = output_root / "dataset_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("ROBUST V2 DATASET READY")
    print("=" * 78)
    print(f"Train: {total_train}")
    print(f"Val  : {total_val}")
    print(f"Test : {total_test}")
    print(f"Generated robustness images: {result['generated_robust_images']}")
    print(f"Summary: {summary_path}")
    print(f"YAML   : {output_root / 'drishti.yaml'}")
    print()
    print("Variant counts:")
    for name, count in result["variant_counts"].items():
        print(f"  {name:22s} {count}")


if __name__ == "__main__":
    main()
