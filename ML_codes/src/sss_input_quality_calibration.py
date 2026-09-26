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
CLEAN_SOURCE = Path(r"D:\DATASETS\DRISHTI-SSS-GHOST2X\train\images")
OUTPUT = Path(r"K:\Debris model\runs\phase_f_f1_quality_calibration")
SAMPLE_SIZE = 600
SEED = 26057

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_quality_module():
    if not QUALITY_SCRIPT.is_file():
        raise FileNotFoundError(f"Quality script not found: {QUALITY_SCRIPT}")

    spec = importlib.util.spec_from_file_location("sss_input_quality", QUALITY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load quality script.")

    module = importlib.util.module_from_spec(spec)
    sys.modules["sss_input_quality"] = module
    spec.loader.exec_module(module)
    return module


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


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
    return np.rint(np.clip(x * noise, 0, 1) * 255.0).astype(np.uint8)


def acoustic_attenuation(
    image: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = image.shape
    y0 = int(
        rng.integers(
            max(0, int(0.10 * h)),
            max(1, int(0.65 * h)),
        )
    )
    y1 = int(
        rng.integers(
            max(y0 + 1, int(0.35 * h)),
            h,
        )
    )

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, :] = 1.0

    kernel = max(7, (w // 35) | 1)
    blur = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    attenuation = 1.0 - np.clip(
        blur * strength,
        0.0,
        0.85,
    )

    x = image.astype(np.float32) * attenuation
    return np.clip(x, 0, 255).astype(np.uint8)


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
        width = int(
            min(
                remaining,
                max(
                    3,
                    int(rng.integers(3, max(4, w // 25 + 1))),
                ),
            )
        )
        start = int(rng.integers(0, max(1, w - width + 1)))

        if rng.random() < 0.5:
            x[:, start:start + width] = 0
        else:
            x[:, start:start + width] *= float(
                rng.uniform(0.05, 0.30)
            )

        remaining -= width

    return np.clip(x, 0, 255).astype(np.uint8)


def resolution_degrade(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    h, w = image.shape
    nh = max(16, int(round(h * scale)))
    nw = max(16, int(round(w * scale)))

    small = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )


def corrupt(
    image: np.ndarray,
    variant: str,
    seed: int,
) -> np.ndarray:
    if variant == "clean":
        return image.copy()

    rng = np.random.default_rng(seed)
    out = image.copy()

    if variant == "mild_speckle":
        return multiplicative_speckle(out, 0.12, rng)

    if variant == "moderate_speckle":
        return multiplicative_speckle(out, 0.24, rng)

    if variant == "strong_speckle":
        return multiplicative_speckle(out, 0.42, rng)

    if variant == "speckle_plus_dropout":
        out = multiplicative_speckle(out, 0.30, rng)
        out = acoustic_attenuation(out, 0.65, rng)
        out = stripe_dropout(out, 0.06, rng)
        return resolution_degrade(out, 0.70)

    if variant == "severe_combined":
        out = multiplicative_speckle(out, 0.48, rng)

        gain = float(rng.normal(1.0, 0.25))
        bias = float(rng.normal(0.0, 0.08))
        x = out.astype(np.float32) / 255.0
        out = np.rint(
            np.clip(x * gain + bias, 0, 1) * 255.0
        ).astype(np.uint8)

        out = acoustic_attenuation(out, 0.80, rng)
        out = stripe_dropout(out, 0.10, rng)
        return resolution_degrade(out, 0.55)

    raise ValueError(f"Unknown variant: {variant}")


def quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def summarize_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r["quality_score"]) for r in records]

    return {
        "count": len(records),
        "quality_score": quantiles(scores),
        "status_counts": {
            "GOOD": sum(r["status"] == "GOOD" for r in records),
            "DEGRADED": sum(r["status"] == "DEGRADED" for r in records),
            "BLOCK": sum(r["status"] == "BLOCK" for r in records),
        },
        "degraded_or_block_rate": float(
            np.mean(
                [
                    r["status"] != "GOOD"
                    for r in records
                ]
            )
        ),
        "block_rate": float(
            np.mean(
                [
                    r["status"] == "BLOCK"
                    for r in records
                ]
            )
        ),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT}"
        )

    if not CLEAN_SOURCE.is_dir():
        raise FileNotFoundError(
            f"Clean source not found: {CLEAN_SOURCE}"
        )

    quality = load_quality_module()

    paths = sorted(
        p for p in CLEAN_SOURCE.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise RuntimeError("No clean images found.")

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

    config = {
        "clean_source": str(CLEAN_SOURCE),
        "sample_size": len(selected),
        "requested_sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "variants": variants,
        "quality_script": str(QUALITY_SCRIPT),
        "purpose": (
            "Calibrate/characterize the input-quality heuristic using "
            "clean prepared SSS imagery and controlled synthetic degradations."
        ),
        "guardrails": [
            "TRAIN only; VAL and TEST are not used.",
            "No model inference.",
            "No threshold tuning against detection ground truth.",
            "Synthetic variants are probes, not claims about real sonar severity.",
        ],
    }

    (OUTPUT / "calibration_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for variant in variants:
        print("=" * 78)
        print(f"VARIANT: {variant}")
        print("=" * 78)

        records: list[dict[str, Any]] = []

        for index, path in enumerate(selected):
            image = read_gray(path)
            variant_image = corrupt(
                image,
                variant,
                SEED + index * 10007,
            )

            metrics = quality.calculate_quality(
                variant_image,
                low_dynamic_range=quality.DEFAULT_LOW_DYNAMIC_RANGE,
                low_std=quality.DEFAULT_LOW_STD,
                near_black_frac=quality.DEFAULT_NEAR_BLACK_FRAC,
                near_white_frac=quality.DEFAULT_NEAR_WHITE_FRAC,
                saturation_frac=quality.DEFAULT_SATURATION_FRAC,
            )

            record = {
                "variant": variant,
                "source_name": path.name,
                **metrics,
            }

            records.append(record)
            all_records.append(record)

            print(
                f"\r{index + 1}/{len(selected)} "
                f"{record['status']:8s} "
                f"score={record['quality_score']:6.2f}",
                end="",
            )

        print()

        summaries[variant] = summarize_variant(records)

        s = summaries[variant]
        print(
            f"{variant:22s} "
            f"GOOD={s['status_counts']['GOOD']:4d} "
            f"DEGRADED={s['status_counts']['DEGRADED']:4d} "
            f"BLOCK={s['status_counts']['BLOCK']:4d} "
            f"non-good={s['degraded_or_block_rate'] * 100:6.2f}% "
            f"block={s['block_rate'] * 100:6.2f}% "
            f"median_score={s['quality_score']['median']:6.2f}"
        )

    calibration_report = {
        "summaries": summaries,
        "interpretation": {
            "clean_false_positive_rate": (
                "Use the clean variant to quantify how often the "
                "heuristic rejects normal prepared SSS inputs."
            ),
            "degradation_detection": (
                "Use non-good/block rates across synthetic variants "
                "to see whether the heuristic separates degraded inputs "
                "from clean inputs."
            ),
            "threshold_decision": (
                "Do not freeze production thresholds from this file alone. "
                "First inspect the distributions and representative images."
            ),
        },
    }

    (OUTPUT / "quality_calibration_report.json").write_text(
        json.dumps(calibration_report, indent=2),
        encoding="utf-8",
    )

    with (OUTPUT / "quality_calibration_records.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "variant",
            "source_name",
            "status",
            "quality_score",
            "mean",
            "std",
            "p1_p99_range",
            "normalized_entropy",
            "edge_density",
            "near_black_fraction",
            "near_white_fraction",
            "row_dropout_fraction",
            "column_dropout_fraction",
            "row_low_energy_fraction",
            "column_low_energy_fraction",
            "flags",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in all_records:
            row = dict(record)
            row["flags"] = "|".join(record["flags"])
            writer.writerow(row)

    print()
    print("=" * 78)
    print("PHASE F1 | QUALITY HEURISTIC CALIBRATION COMPLETE")
    print("=" * 78)

    for variant in variants:
        s = summaries[variant]
        print(
            f"{variant:22s} "
            f"GOOD={s['status_counts']['GOOD']:4d} "
            f"DEGRADED={s['status_counts']['DEGRADED']:4d} "
            f"BLOCK={s['status_counts']['BLOCK']:4d} "
            f"non-good={s['degraded_or_block_rate'] * 100:6.2f}% "
            f"median={s['quality_score']['median']:6.2f}"
        )

    print()
    print(f"Output: {OUTPUT}")
    print("Production thresholds are NOT frozen by this calibration.")


if __name__ == "__main__":
    main()
