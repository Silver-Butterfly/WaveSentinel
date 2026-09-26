from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Engineering defaults. These are intentionally exposed as configuration,
# not presented as experimentally calibrated thresholds.
DEFAULT_LOW_DYNAMIC_RANGE = 35.0
DEFAULT_LOW_STD = 12.0
DEFAULT_NEAR_BLACK_FRAC = 0.18
DEFAULT_NEAR_WHITE_FRAC = 0.18
DEFAULT_DROPOUT_ROW_FRAC = 0.65
DEFAULT_DROPOUT_COL_FRAC = 0.15
DEFAULT_SATURATION_FRAC = 0.35


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def percentile_range(image: np.ndarray) -> float:
    p1, p99 = np.percentile(image, [1, 99])
    return float(p99 - p1)


def normalized_entropy(image: np.ndarray) -> float:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return entropy / 8.0


def edge_density(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 50, 150)
    return float(np.mean(edges > 0))


def fraction_near(image: np.ndarray, threshold: int, high: bool) -> float:
    if high:
        return float(np.mean(image >= threshold))
    return float(np.mean(image <= threshold))


def dropout_metrics(
    image: np.ndarray,
    row_threshold: float = DEFAULT_DROPOUT_ROW_FRAC,
    col_threshold: float = DEFAULT_DROPOUT_COL_FRAC,
) -> dict[str, float]:
    x = image.astype(np.float32) / 255.0

    row_energy = x.mean(axis=1)
    col_energy = x.mean(axis=0)

    row_dropout = float(np.mean(row_energy <= (1.0 - row_threshold)))
    col_dropout = float(np.mean(col_energy <= col_threshold))

    # Detect unusually long, low-energy rows/columns.
    row_med = float(np.median(row_energy))
    col_med = float(np.median(col_energy))

    row_low_fraction = (
        float(np.mean(row_energy <= max(0.01, row_med * 0.25)))
        if row_med > 0
        else 1.0
    )
    col_low_fraction = (
        float(np.mean(col_energy <= max(0.01, col_med * 0.25)))
        if col_med > 0
        else 1.0
    )

    return {
        "row_dropout_fraction": row_dropout,
        "column_dropout_fraction": col_dropout,
        "row_low_energy_fraction": row_low_fraction,
        "column_low_energy_fraction": col_low_fraction,
    }


def calculate_quality(
    image: np.ndarray,
    low_dynamic_range: float,
    low_std: float,
    near_black_frac: float,
    near_white_frac: float,
    saturation_frac: float,
) -> dict[str, Any]:
    mean = float(image.mean())
    std = float(image.std())
    dynamic_range = percentile_range(image)
    entropy = normalized_entropy(image)
    edges = edge_density(image)

    near_black = fraction_near(image, 8, high=False)
    near_white = fraction_near(image, 247, high=True)
    extreme = near_black + near_white

    dropout = dropout_metrics(image)

    flags: list[str] = []

    if dynamic_range < low_dynamic_range:
        flags.append("low_dynamic_range")

    if std < low_std:
        flags.append("low_global_variance")

    if near_black > near_black_frac:
        flags.append("high_near_black_fraction")

    if near_white > near_white_frac:
        flags.append("high_near_white_fraction")

    if extreme > saturation_frac:
        flags.append("high_extreme_intensity_fraction")

    if dropout["row_low_energy_fraction"] >= 0.10:
        flags.append("row_dropout_pattern")

    if dropout["column_low_energy_fraction"] >= 0.10:
        flags.append("column_dropout_pattern")

    # Heuristic quality score. This is an engineering gate, not a calibrated
    # probability and must be tuned against representative sonar captures.
    penalties = 0.0

    if dynamic_range < low_dynamic_range:
        penalties += 25.0
    elif dynamic_range < low_dynamic_range * 1.5:
        penalties += 10.0

    if std < low_std:
        penalties += 20.0
    elif std < low_std * 1.5:
        penalties += 8.0

    if near_black > near_black_frac:
        penalties += 10.0

    if near_white > near_white_frac:
        penalties += 10.0

    if extreme > saturation_frac:
        penalties += 15.0

    if dropout["row_low_energy_fraction"] >= 0.10:
        penalties += 15.0

    if dropout["column_low_energy_fraction"] >= 0.10:
        penalties += 15.0

    bonus = min(10.0, entropy * 10.0 + min(edges * 100.0, 10.0))
    quality_score = max(0.0, min(100.0, 100.0 - penalties + bonus))

    if quality_score >= 75.0 and not flags:
        status = "GOOD"
    elif quality_score >= 50.0:
        status = "DEGRADED"
    else:
        status = "BLOCK"

    return {
        "status": status,
        "quality_score": round(quality_score, 3),
        "mean": round(mean, 3),
        "std": round(std, 3),
        "p1_p99_range": round(dynamic_range, 3),
        "normalized_entropy": round(entropy, 5),
        "edge_density": round(edges, 5),
        "near_black_fraction": round(near_black, 5),
        "near_white_fraction": round(near_white, 5),
        **{k: round(v, 5) for k, v in dropout.items()},
        "flags": flags,
    }


def analyze_path(path: Path, **kwargs: Any) -> dict[str, Any]:
    image = read_gray(path)
    metrics = calculate_quality(image, **kwargs)
    return {
        "source_name": path.name,
        "source": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineering prototype for SSS input-quality and dropout gating."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-dynamic-range", type=float, default=DEFAULT_LOW_DYNAMIC_RANGE)
    parser.add_argument("--low-std", type=float, default=DEFAULT_LOW_STD)
    parser.add_argument("--near-black-frac", type=float, default=DEFAULT_NEAR_BLACK_FRAC)
    parser.add_argument("--near-white-frac", type=float, default=DEFAULT_NEAR_WHITE_FRAC)
    parser.add_argument("--saturation-frac", type=float, default=DEFAULT_SATURATION_FRAC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )

    output.mkdir(parents=True, exist_ok=False)

    if source.is_file():
        paths = [source]
    else:
        paths = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not paths:
        raise RuntimeError("No supported images found.")

    config = {
        "low_dynamic_range": args.low_dynamic_range,
        "low_std": args.low_std,
        "near_black_frac": args.near_black_frac,
        "near_white_frac": args.near_white_frac,
        "saturation_frac": args.saturation_frac,
        "quality_status": {
            "GOOD": "Operational candidate",
            "DEGRADED": "Input may be used with a warning",
            "BLOCK": "Do not trust inference without review",
        },
        "note": (
            "Heuristic engineering gate. Thresholds are not experimentally "
            "calibrated probabilities and must be validated on representative "
            "real sonar streams before operational use."
        ),
    }

    records = []
    for index, path in enumerate(paths, start=1):
        record = analyze_path(
            path,
            low_dynamic_range=args.low_dynamic_range,
            low_std=args.low_std,
            near_black_frac=args.near_black_frac,
            near_white_frac=args.near_white_frac,
            saturation_frac=args.saturation_frac,
        )
        records.append(record)
        print(
            f"\rProcessed {index}/{len(paths)} | "
            f"{record['status']:8s} | "
            f"score={record['quality_score']:6.2f}",
            end="",
        )

    print()

    summary = {
        "images_analyzed": len(records),
        "good": sum(r["status"] == "GOOD" for r in records),
        "degraded": sum(r["status"] == "DEGRADED" for r in records),
        "block": sum(r["status"] == "BLOCK" for r in records),
        "mean_quality_score": float(
            np.mean([r["quality_score"] for r in records])
        ),
    }

    (output / "quality_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    (output / "quality_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (output / "quality_records.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    csv_path = output / "quality_records.csv"
    fieldnames = [
        "source_name",
        "source",
        "width",
        "height",
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

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["flags"] = "|".join(record["flags"])
            writer.writerow(row)

    print("=" * 78)
    print("SSS INPUT QUALITY / DROPOUT ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Images analyzed : {summary['images_analyzed']}")
    print(f"GOOD            : {summary['good']}")
    print(f"DEGRADED        : {summary['degraded']}")
    print(f"BLOCK           : {summary['block']}")
    print(f"Mean score      : {summary['mean_quality_score']:.2f}")
    print(f"Output          : {output}")
    print()
    print("This module is a heuristic engineering prototype.")
    print("It is not a calibrated sonar-quality probability model.")


if __name__ == "__main__":
    main()
