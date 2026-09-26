import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import math
import csv

CLASS_NAMES = {
    0: "pipe",
    1: "shipwreck",
    2: "mine",
    3: "ghost_net",
}

# Area fractions are based on normalized YOLO width * height.
BUCKETS = [
    ("tiny", 0.0, 0.005),
    ("small", 0.005, 0.02),
    ("medium", 0.02, 0.05),
    ("large", 0.05, float("inf")),
]


def bucket(area):
    for name, lo, hi in BUCKETS:
        if lo <= area < hi:
            return name
    return "large"


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def analyze_split(split_dir):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    per_class_areas = defaultdict(list)
    per_class_wh = defaultdict(list)
    source_counts = Counter()
    image_counts = Counter()
    total_objects = 0

    label_files = sorted(labels_dir.glob("*.txt"))

    for label_path in label_files:
        image_path = images_dir / (label_path.stem + ".jpg")
        if not image_path.exists():
            # Collision-proof V5 output may preserve png/jpeg suffixes.
            matches = list(images_dir.glob(label_path.stem + ".*"))
            if matches:
                image_path = matches[0]
            else:
                continue

        try:
            with Image.open(image_path) as im:
                iw, ih = im.size
        except Exception:
            continue

        # V5 filenames encode the source family before the double underscore.
        stem = label_path.stem.split("__", 1)[0]
        source = stem.split("_", 1)[0] if "_" in stem else stem

        image_has_class = set()

        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                _, _, w, h = map(float, parts[1:])
            except ValueError:
                continue
            if cls not in CLASS_NAMES:
                continue

            area = w * h
            per_class_areas[cls].append(area)
            per_class_wh[cls].append((w, h))
            image_has_class.add(cls)
            total_objects += 1

        if image_has_class:
            for cls in image_has_class:
                image_counts[cls] += 1
                source_counts[(source, cls)] += 1
        else:
            source_counts[(source, -1)] += 1

    return {
        "objects": total_objects,
        "per_class_areas": {str(k): v for k, v in per_class_areas.items()},
        "per_class_wh": {str(k): v for k, v in per_class_wh.items()},
        "image_counts": {str(k): v for k, v in image_counts.items()},
        "source_counts": {f"{s}|{c}": v for (s, c), v in source_counts.items()},
    }


def summarize_class(areas):
    counts = Counter(bucket(a) for a in areas)
    return {
        "objects": len(areas),
        "tiny": counts["tiny"],
        "small": counts["small"],
        "medium": counts["medium"],
        "large": counts["large"],
        "tiny_pct": round(100 * counts["tiny"] / len(areas), 2) if areas else 0,
        "small_pct": round(100 * counts["small"] / len(areas), 2) if areas else 0,
        "medium_pct": round(100 * counts["medium"] / len(areas), 2) if areas else 0,
        "large_pct": round(100 * counts["large"] / len(areas), 2) if areas else 0,
        "p01": percentile(areas, 1),
        "p05": percentile(areas, 5),
        "p25": percentile(areas, 25),
        "median": percentile(areas, 50),
        "p75": percentile(areas, 75),
        "p95": percentile(areas, 95),
        "p99": percentile(areas, 99),
    }


def main():
    ap = argparse.ArgumentParser(description="Compare DRISHTI-SSS V5 object-size distributions across train/val/test.")
    ap.add_argument("--root", required=True, help="DRISHTI-SSS-SPLIT-V5 root")
    ap.add_argument("--output", default=None, help="Output directory; defaults to <root>/size_analysis")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output) if args.output else root / "size_analysis"
    out.mkdir(parents=True, exist_ok=True)

    raw = {}
    summaries = {}

    for split in ("train", "val", "test"):
        result = analyze_split(root / split)
        raw[split] = result
        summaries[split] = {}
        for cls in CLASS_NAMES:
            summaries[split][cls] = summarize_class(
                result["per_class_areas"].get(str(cls), [])
            )

    # Pairwise distribution differences using bucket percentages.
    comparisons = {}
    for cls in CLASS_NAMES:
        comparisons[cls] = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            sa = summaries[a][cls]
            sb = summaries[b][cls]
            comparisons[cls][f"{a}_vs_{b}"] = {
                "tiny_pct_delta": round(sa["tiny_pct"] - sb["tiny_pct"], 2),
                "small_pct_delta": round(sa["small_pct"] - sb["small_pct"], 2),
                "medium_pct_delta": round(sa["medium_pct"] - sb["medium_pct"], 2),
                "large_pct_delta": round(sa["large_pct"] - sb["large_pct"], 2),
                "median_area_ratio": (
                    round(sa["median"] / sb["median"], 4)
                    if sa["median"] and sb["median"] else None
                ),
            }

    report = {
        "dataset": root.name,
        "bucket_definition": {
            "tiny": "<0.5% image area",
            "small": "0.5-2% image area",
            "medium": "2-5% image area",
            "large": ">=5% image area",
        },
        "classes": CLASS_NAMES,
        "splits": summaries,
        "comparisons": comparisons,
    }

    (out / "size_distribution_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # CSV for easy spreadsheet inspection.
    with (out / "size_distribution_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "class", "objects",
            "tiny", "small", "medium", "large",
            "tiny_pct", "small_pct", "medium_pct", "large_pct",
            "p01", "p05", "p25", "median", "p75", "p95", "p99"
        ])
        for split in ("train", "val", "test"):
            for cls, name in CLASS_NAMES.items():
                s = summaries[split][cls]
                writer.writerow([
                    split, name, s["objects"],
                    s["tiny"], s["small"], s["medium"], s["large"],
                    s["tiny_pct"], s["small_pct"], s["medium_pct"], s["large_pct"],
                    s["p01"], s["p05"], s["p25"], s["median"],
                    s["p75"], s["p95"], s["p99"]
                ])

    lines = []
    lines.append("DRISHTI-SSS V5 OBJECT-SIZE DISTRIBUTION ANALYSIS")
    lines.append("=" * 58)
    lines.append("Area buckets: tiny <0.5%, small 0.5-2%, medium 2-5%, large >=5%")
    lines.append("")

    for cls, name in CLASS_NAMES.items():
        lines.append(f"[{name}]")
        for split in ("train", "val", "test"):
            s = summaries[split][cls]
            lines.append(
                f"  {split:5s}: n={s['objects']:4d} | "
                f"tiny {s['tiny_pct']:6.2f}% | small {s['small_pct']:6.2f}% | "
                f"medium {s['medium_pct']:6.2f}% | large {s['large_pct']:6.2f}% | "
                f"median={s['median']:.8f}" if s["median"] is not None else
                f"  {split:5s}: n={s['objects']:4d} | no objects"
            )
        lines.append("")

    lines.append("TRAIN vs TEST DELTAS")
    lines.append("-" * 58)
    for cls, name in CLASS_NAMES.items():
        d = comparisons[cls]["train_vs_test"]
        lines.append(
            f"{name:10s}: tiny {d['tiny_pct_delta']:+.2f} pp, "
            f"small {d['small_pct_delta']:+.2f} pp, "
            f"medium {d['medium_pct_delta']:+.2f} pp, "
            f"large {d['large_pct_delta']:+.2f} pp, "
            f"median area ratio={d['median_area_ratio']}"
        )

    (out / "size_distribution_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(json.dumps({
        "status": "ok",
        "output": str(out),
        "files": [
            str(out / "size_distribution_report.json"),
            str(out / "size_distribution_summary.csv"),
            str(out / "size_distribution_summary.txt"),
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
