from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


QUALITY_SCRIPT = Path(r"K:\Debris model\src\sss_input_quality.py")
SOURCE = Path(r"D:\DATASETS\DRISHTI-SSS-GHOST2X\train\images")
OUTPUT = Path(r"K:\Debris model\runs\phase_f1_quality_v2_calibration")
SAMPLE_SIZE = 600
SEED = 26057

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_base_module():
    if not QUALITY_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing base quality module: {QUALITY_SCRIPT}")

    spec = importlib.util.spec_from_file_location("sss_input_quality", QUALITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load base quality module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["sss_input_quality"] = module
    spec.loader.exec_module(module)
    return module


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def speckle(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    noise = rng.lognormal(
        mean=-0.5 * sigma * sigma,
        sigma=sigma,
        size=x.shape,
    )
    return np.rint(np.clip(x * noise, 0, 1) * 255).astype(np.uint8)


def stripe_dropout(
    image: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    x = image.astype(np.float32).copy()
    h, w = x.shape
    target = max(1, int(round(w * fraction)))
    remaining = target

    while remaining > 0:
        width = min(
            remaining,
            max(3, int(rng.integers(3, max(4, w // 25 + 1)))),
        )
        start = int(rng.integers(0, max(1, w - width + 1)))

        if rng.random() < 0.5:
            x[:, start:start + width] = 0
        else:
            x[:, start:start + width] *= float(rng.uniform(0.05, 0.30))

        remaining -= width

    return np.clip(x, 0, 255).astype(np.uint8)


def attenuation(
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
    factor = 1.0 - np.clip(blur * strength, 0.0, 0.85)

    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def resolution_degrade(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape
    nh = max(16, int(round(h * scale)))
    nw = max(16, int(round(w * scale)))
    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def corrupt(image: np.ndarray, variant: str, seed: int) -> np.ndarray:
    if variant == "clean":
        return image.copy()

    rng = np.random.default_rng(seed)
    out = image.copy()

    if variant == "mild_speckle":
        return speckle(out, 0.12, rng)

    if variant == "moderate_speckle":
        return speckle(out, 0.24, rng)

    if variant == "strong_speckle":
        return speckle(out, 0.42, rng)

    if variant == "speckle_plus_dropout":
        out = speckle(out, 0.30, rng)
        out = attenuation(out, 0.65, rng)
        out = stripe_dropout(out, 0.06, rng)
        return resolution_degrade(out, 0.70)

    if variant == "severe_combined":
        out = speckle(out, 0.48, rng)
        gain = float(rng.normal(1.0, 0.25))
        bias = float(rng.normal(0.0, 0.08))
        x = out.astype(np.float32) / 255.0
        out = np.rint(np.clip(x * gain + bias, 0, 1) * 255).astype(np.uint8)
        out = attenuation(out, 0.80, rng)
        out = stripe_dropout(out, 0.10, rng)
        return resolution_degrade(out, 0.55)

    raise ValueError(f"Unknown variant: {variant}")


def long_low_energy_runs(
    values: np.ndarray,
    relative_threshold: float,
    min_run_fraction: float,
) -> tuple[float, int]:
    median = float(np.median(values))
    threshold = max(1e-5, median * relative_threshold)
    low = values <= threshold

    max_run = 0
    current = 0
    for flag in low:
        if flag:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    max_fraction = max_run / max(1, len(values))
    return max_fraction, max_run


def dropout_features(image: np.ndarray) -> dict[str, float]:
    x = image.astype(np.float32) / 255.0

    row_mean = x.mean(axis=1)
    col_mean = x.mean(axis=0)

    row_cv = float(np.std(row_mean) / (np.mean(row_mean) + 1e-6))
    col_cv = float(np.std(col_mean) / (np.mean(col_mean) + 1e-6))

    row_long_frac, _ = long_low_energy_runs(row_mean, 0.25, 0.05)
    col_long_frac, _ = long_low_energy_runs(col_mean, 0.25, 0.05)

    # Local block energy exposes narrow scanline gaps that global statistics hide.
    grid_rows = 16
    grid_cols = 16
    small = cv2.resize(
        x,
        (grid_cols, grid_rows),
        interpolation=cv2.INTER_AREA,
    )
    block_mean = small.mean()

    if block_mean > 0:
        block_low = float(
            np.mean(small <= max(0.015, block_mean * 0.20))
        )
        block_p01 = float(np.percentile(small, 1))
        block_p99 = float(np.percentile(small, 99))
    else:
        block_low = 1.0
        block_p01 = 0.0
        block_p99 = 0.0

    # Measure abrupt scanline energy changes.
    col_diff = np.abs(np.diff(col_mean))
    row_diff = np.abs(np.diff(row_mean))

    col_jump = float(
        np.percentile(col_diff, 99) /
        (np.median(col_diff) + 1e-6)
    )
    row_jump = float(
        np.percentile(row_diff, 99) /
        (np.median(row_diff) + 1e-6)
    )

    return {
        "row_energy_cv": row_cv,
        "column_energy_cv": col_cv,
        "row_long_low_energy_run_fraction": row_long_frac,
        "column_long_low_energy_run_fraction": col_long_frac,
        "low_energy_block_fraction": block_low,
        "block_p01": block_p01,
        "block_p99": block_p99,
        "column_jump_ratio_p99": col_jump,
        "row_jump_ratio_p99": row_jump,
    }


def analyze(
    image: np.ndarray,
    base_quality: Any,
) -> dict[str, Any]:
    base = base_quality.calculate_quality(
        image,
        low_dynamic_range=base_quality.DEFAULT_LOW_DYNAMIC_RANGE,
        low_std=base_quality.DEFAULT_LOW_STD,
        near_black_frac=base_quality.DEFAULT_NEAR_BLACK_FRAC,
        near_white_frac=base_quality.DEFAULT_NEAR_WHITE_FRAC,
        saturation_frac=base_quality.DEFAULT_SATURATION_FRAC,
    )

    drop = dropout_features(image)

    # Conservative diagnostic score focused on structural dropout evidence.
    evidence = 0.0
    flags: list[str] = []

    if drop["column_long_low_energy_run_fraction"] >= 0.05:
        evidence += 2.5
        flags.append("long_column_low_energy_run")

    if drop["row_long_low_energy_run_fraction"] >= 0.05:
        evidence += 2.5
        flags.append("long_row_low_energy_run")

    if drop["low_energy_block_fraction"] >= 0.08:
        evidence += 2.0
        flags.append("low_energy_blocks")

    if drop["column_jump_ratio_p99"] >= 8.0:
        evidence += 1.5
        flags.append("column_energy_discontinuity")

    if drop["row_jump_ratio_p99"] >= 8.0:
        evidence += 1.5
        flags.append("row_energy_discontinuity")

    # The base image statistics remain useful diagnostics but do not gate
    # normal inputs in V2 because the V1 calibration showed poor separation.
    status = "NORMAL"
    if evidence >= 4.0:
        status = "DROP_OUT_SUSPECTED"
    elif evidence >= 2.5:
        status = "DEGRADED_STRUCTURE"

    return {
        "status": status,
        "structural_score": round(evidence, 3),
        **{
            f"base_{k}": v
            for k, v in base.items()
            if k not in {"flags", "status"}
        },
        **{k: round(v, 6) for k, v in drop.items()},
        "flags": flags,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = np.asarray(
        [r["structural_score"] for r in records],
        dtype=np.float64,
    )

    return {
        "count": len(records),
        "status_counts": {
            "NORMAL": sum(r["status"] == "NORMAL" for r in records),
            "DEGRADED_STRUCTURE": sum(
                r["status"] == "DEGRADED_STRUCTURE"
                for r in records
            ),
            "DROP_OUT_SUSPECTED": sum(
                r["status"] == "DROP_OUT_SUSPECTED"
                for r in records
            ),
        },
        "structural_score": {
            "min": float(scores.min()),
            "p10": float(np.percentile(scores, 10)),
            "median": float(np.percentile(scores, 50)),
            "p90": float(np.percentile(scores, 90)),
            "p99": float(np.percentile(scores, 99)),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")

    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source not found: {SOURCE}")

    base_quality = load_base_module()

    paths = sorted(
        p for p in SOURCE.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise RuntimeError("No images found.")

    rng = np.random.default_rng(SEED)
    if len(paths) > SAMPLE_SIZE:
        selected = sorted(
            rng.choice(
                np.asarray(paths, dtype=object),
                size=SAMPLE_SIZE,
                replace=False,
            ).tolist()
        )
    else:
        selected = paths

    OUTPUT.mkdir(parents=True, exist_ok=False)

    variants = [
        "clean",
        "mild_speckle",
        "moderate_speckle",
        "strong_speckle",
        "speckle_plus_dropout",
        "severe_combined",
    ]

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for variant in variants:
        records: list[dict[str, Any]] = []

        print("=" * 78)
        print(f"VARIANT: {variant}")
        print("=" * 78)

        for index, path in enumerate(selected):
            image = read_gray(path)
            test_image = corrupt(
                image,
                variant,
                SEED + index * 10007,
            )

            metrics = analyze(test_image, base_quality)
            record = {
                "variant": variant,
                "source_name": path.name,
                **metrics,
            }
            records.append(record)
            all_records.append(record)

            print(
                f"\r{index + 1}/{len(selected)} "
                f"{record['status']:20s} "
                f"score={record['structural_score']:5.2f}",
                end="",
            )

        print()
        summaries[variant] = summarize(records)

        s = summaries[variant]
        print(
            f"{variant:22s} "
            f"NORMAL={s['status_counts']['NORMAL']:4d} "
            f"DEGRADED={s['status_counts']['DEGRADED_STRUCTURE']:4d} "
            f"DROPOUT={s['status_counts']['DROP_OUT_SUSPECTED']:4d} "
            f"median={s['structural_score']['median']:.2f} "
            f"p99={s['structural_score']['p99']:.2f}"
        )

    report = {
        "phase": "F1",
        "stage": "quality_v2_calibration",
        "source": str(SOURCE),
        "sample_size": len(selected),
        "seed": SEED,
        "variants": variants,
        "summaries": summaries,
        "decision": (
            "V1 global quality-score gate is not suitable for production. "
            "V2 separates structural dropout evidence from generic image "
            "quality statistics. Production thresholds remain unfrozen."
        ),
    }

    (OUTPUT / "quality_v2_calibration_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    with (OUTPUT / "quality_v2_records.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "variant",
            "source_name",
            "status",
            "structural_score",
            "base_quality_score",
            "base_p1_p99_range",
            "base_std",
            "row_energy_cv",
            "column_energy_cv",
            "row_long_low_energy_run_fraction",
            "column_long_low_energy_run_fraction",
            "low_energy_block_fraction",
            "block_p01",
            "block_p99",
            "column_jump_ratio_p99",
            "row_jump_ratio_p99",
            "flags",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in all_records:
            row = {
                key: record.get(key)
                for key in fieldnames
            }
            row["flags"] = "|".join(record["flags"])
            writer.writerow(row)

    print()
    print("=" * 78)
    print("PHASE F1 | QUALITY V2 CALIBRATION COMPLETE")
    print("=" * 78)

    for variant in variants:
        s = summaries[variant]
        print(
            f"{variant:22s} "
            f"NORMAL={s['status_counts']['NORMAL']:4d} "
            f"DEGRADED={s['status_counts']['DEGRADED_STRUCTURE']:4d} "
            f"DROPOUT={s['status_counts']['DROP_OUT_SUSPECTED']:4d} "
            f"median={s['structural_score']['median']:.2f} "
            f"p99={s['structural_score']['p99']:.2f}"
        )

    print()
    print(f"Output: {OUTPUT}")
    print("No production quality threshold has been frozen.")


if __name__ == "__main__":
    main()
