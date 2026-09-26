"""
PHASE-1 / PHASE-2 DATA + V3 FAILURE DIAGNOSTICS
================================================

Purpose
-------
1. Diagnose why exp01_subpipe_pipeline_robust_v3 collapsed on the sealed
   SubPipe acquisition-group test.
2. Audit the three additional datasets without modifying them:
      - AI4Shipwrecks
      - SeabedObjects-KLSG
      - SCTD
3. Produce machine-readable JSON + human-readable TXT reports.
4. Do NOT train, tune thresholds, alter datasets, or evaluate the sealed
   test in a way that changes its status.

IMPORTANT:
- The SubPipe sealed test remains FROZEN.
- Additional datasets are DEVELOPMENT DATA ONLY until explicitly audited.
- This script is intentionally format-agnostic for KLSG/SCTD and reports
  what it can verify instead of inventing taxonomy/annotation semantics.

Run from the user's project environment, e.g.
    python phase1_v3_and_newdata_diagnostic.py

Adjust paths in CONFIG if needed.
"""

from __future__ import annotations

import json
import math
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CONFIG = {
    "subpipe_preprocessed": Path(r"K:\Debris model\preprocessing_output"),
    "v3_dataset": Path(
        r"K:\Debris model\experiments\exp01_subpipe_pipeline_robust_v3\dataset"
    ),
    "v3_run": Path(
        r"K:\Debris model\experiments\exp01_subpipe_pipeline_robust_v3-2"
    ),
    "v3_prediction_dir": Path(r"K:\Debris model\src\runs\detect\val-2"),

    "ai4": Path(r"D:\ML Lab\Project Debris Dataset\AI4Shipwrecks"),
    "klsg": Path(r"D:\ML Lab\Project Debris Dataset\SeabedObjects-KLSG"),
    "sctd": Path(r"D:\ML Lab\Project Debris Dataset\SCTD"),

    "output_dir": Path(
        r"K:\Debris model\experiments\phase1_diagnostics_2026_09"
    ),
}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTS = {".txt", ".json", ".xml", ".csv", ".yaml", ".yml", ".pbm", ".bpm"}
SONAR_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pbm", ".bpm"}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def now():
    return datetime.now().isoformat(timespec="seconds")


def safe_rel(path, root):
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_inventory(root):
    if not root.exists():
        return {
            "exists": False,
            "root": str(root),
            "files": 0,
            "extensions": {},
        }

    counts = Counter()
    files = 0
    total_bytes = 0
    for p in root.rglob("*"):
        if p.is_file():
            files += 1
            counts[p.suffix.lower() or "<no_extension>"] += 1
            try:
                total_bytes += p.stat().st_size
            except Exception:
                pass

    return {
        "exists": True,
        "root": str(root),
        "files": files,
        "bytes": total_bytes,
        "extensions": dict(sorted(counts.items())),
    }


def sample_text(path, n=8000):
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:n]
    except Exception:
        return ""


def looks_like_yolo_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        vals = [float(x) for x in parts]
    except Exception:
        return False
    return all(math.isfinite(x) for x in vals)


def parse_yolo_file(path):
    valid = 0
    invalid = 0
    classes = Counter()
    boxes = []

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {"valid": 0, "invalid": 0, "classes": {}, "boxes": []}

    for line in lines:
        if not line.strip():
            continue
        if not looks_like_yolo_line(line):
            invalid += 1
            continue
        c, x, y, w, h = map(float, line.split())
        valid += 1
        classes[str(int(c)) if c.is_integer() else str(c)] += 1
        boxes.append((c, x, y, w, h))

    return {
        "valid": valid,
        "invalid": invalid,
        "classes": dict(classes),
        "boxes": boxes,
    }


def image_stats(path):
    if Image is None:
        return {"readable": None, "error": "Pillow not installed"}

    try:
        with Image.open(path) as im:
            arr = np.asarray(im) if np is not None else None
            result = {
                "readable": True,
                "format": im.format,
                "mode": im.mode,
                "width": im.width,
                "height": im.height,
            }
            if arr is not None:
                a = arr.astype(np.float32)
                result.update({
                    "min": float(a.min()),
                    "max": float(a.max()),
                    "mean": float(a.mean()),
                    "std": float(a.std()),
                    "p01": float(np.percentile(a, 1)),
                    "p50": float(np.percentile(a, 50)),
                    "p99": float(np.percentile(a, 99)),
                })
            return result
    except Exception as e:
        return {"readable": False, "error": repr(e)}


def collect_images(root, limit=None):
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
            if limit and len(out) >= limit:
                break
    return out


def filename_tokens(path):
    s = path.stem.lower()
    return [x for x in re.split(r"[_\-. ]+", s) if x]


def infer_role_from_path(path):
    s = str(path).lower()
    if any(x in s for x in ["train", "training"]):
        return "train"
    if any(x in s for x in ["test", "testing", "eval", "evaluation"]):
        return "test"
    if any(x in s for x in ["val", "valid", "validation"]):
        return "validation"
    return "unknown"


def discover_annotation_neighbors(img):
    candidates = []
    stem = img.stem

    for ext in [".txt", ".json", ".xml", ".csv", ".yaml", ".yml", ".png", ".bmp", ".tif", ".tiff"]:
        candidates.append(img.with_suffix(ext))

    # common label directories
    for parent in [img.parent, img.parent / "labels", img.parent.parent / "labels"]:
        for ext in [".txt", ".json", ".xml", ".csv"]:
            candidates.append(parent / (stem + ext))

    found = [p for p in candidates if p.exists() and p.is_file()]
    # de-duplicate
    return list(dict.fromkeys(found))


# ---------------------------------------------------------------------
# V3 sealed-test artifact inspection
# ---------------------------------------------------------------------

def inspect_v3():
    out = {
        "timestamp": now(),
        "dataset": str(CONFIG["v3_dataset"]),
        "run": str(CONFIG["v3_run"]),
        "prediction_dir": str(CONFIG["v3_prediction_dir"]),
    }

    ds = CONFIG["v3_dataset"]
    pred = CONFIG["v3_prediction_dir"]
    run = CONFIG["v3_run"]

    out["dataset_inventory"] = file_inventory(ds)
    out["run_inventory"] = file_inventory(run)
    out["prediction_inventory"] = file_inventory(pred)

    # Test image/label counts and YOLO statistics
    test_images = []
    test_labels = []

    for d in [ds / "images" / "test", ds / "test" / "images"]:
        if d.exists():
            test_images.extend([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    for d in [ds / "labels" / "test", ds / "test" / "labels"]:
        if d.exists():
            test_labels.extend([p for p in d.rglob("*.txt") if p.is_file()])

    test_images = sorted(set(test_images))
    test_labels = sorted(set(test_labels))

    label_by_stem = defaultdict(list)
    for p in test_labels:
        label_by_stem[p.stem].append(p)

    test_objects = 0
    invalid = 0
    oob = 0
    positive_tiles = 0
    negative_tiles = 0
    class_counts = Counter()
    box_areas = []
    truncated = 0

    for img in test_images:
        labs = label_by_stem.get(img.stem, [])
        if not labs:
            negative_tiles += 1
            continue

        parsed = parse_yolo_file(labs[0])
        invalid += parsed["invalid"]
        n = parsed["valid"]

        if n:
            positive_tiles += 1
        else:
            negative_tiles += 1

        test_objects += n
        class_counts.update(parsed["classes"])

        for c, x, y, w, h in parsed["boxes"]:
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                oob += 1
            area = max(0.0, w) * max(0.0, h)
            box_areas.append(area)
            if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
                truncated += 1

    out["sealed_test_artifact"] = {
        "images": len(test_images),
        "labels": len(test_labels),
        "positive_tiles": positive_tiles,
        "background_tiles": negative_tiles,
        "objects": test_objects,
        "invalid_label_records": invalid,
        "out_of_bounds_boxes": oob,
        "partial_or_truncated_boxes": truncated,
        "classes": dict(class_counts),
        "box_area_median": statistics.median(box_areas) if box_areas else None,
        "box_area_mean": statistics.mean(box_areas) if box_areas else None,
    }

    # Try to find an evaluation JSON anywhere under run
    eval_jsons = list(run.rglob("*.json")) if run.exists() else []
    out["evaluation_jsons"] = [str(x) for x in eval_jsons[:50]]

    # Try to read common Ultralytics results.csv
    csvs = list(run.rglob("results.csv")) if run.exists() else []
    out["results_csvs"] = [str(x) for x in csvs[:20]]

    # Prediction image/label counts, if present
    if pred.exists():
        pred_images = [
            p for p in pred.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        out["prediction_images"] = len(pred_images)

        # Common label output folders
        pred_label_dirs = [pred / "labels", pred / "predictions", pred / "labels" / "test"]
        pfiles = []
        for d in pred_label_dirs:
            if d.exists():
                pfiles.extend(d.rglob("*.txt"))
        out["prediction_label_files"] = len(set(pfiles))

    return out


# ---------------------------------------------------------------------
# Generic dataset audit
# ---------------------------------------------------------------------

def audit_dataset(root, name, sample_limit=200):
    result = {
        "name": name,
        "root": str(root),
        "timestamp": now(),
        "inventory": file_inventory(root),
        "exists": root.exists(),
    }

    if not root.exists():
        result["verdict"] = "MISSING_PATH"
        return result

    all_files = [p for p in root.rglob("*") if p.is_file()]

    # directory roles
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    role_counts = Counter(infer_role_from_path(p) for p in dirs)
    result["directory_role_hints"] = dict(role_counts)

    # likely README / metadata files
    docs = [
        p for p in all_files
        if p.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml", ".xml"}
        and any(k in p.name.lower() for k in [
            "readme", "license", "meta", "description", "label", "class",
            "annotation", "dataset", "info", "config"
        ])
    ]
    result["metadata_candidates"] = [str(x) for x in docs[:40]]

    # image audit sample
    imgs = collect_images(root, limit=sample_limit)
    result["sample_image_count"] = len(imgs)

    readable = 0
    unreadable = 0
    dims = Counter()
    modes = Counter()
    means = []
    stds = []
    percentiles = []

    for p in imgs:
        st = image_stats(p)
        if st.get("readable"):
            readable += 1
            dims[f'{st["width"]}x{st["height"]}'] += 1
            modes[st["mode"]] += 1
            if "mean" in st:
                means.append(st["mean"])
            if "std" in st:
                stds.append(st["std"])
            if "p01" in st:
                percentiles.append((st["p01"], st["p50"], st["p99"]))
        else:
            unreadable += 1

    result["sample_images"] = {
        "readable": readable,
        "unreadable": unreadable,
        "dimensions": dict(dims),
        "modes": dict(modes),
        "mean_pixel_mean": statistics.mean(means) if means else None,
        "mean_pixel_std": statistics.mean(stds) if stds else None,
        "p01_median": statistics.median(x[0] for x in percentiles) if percentiles else None,
        "p50_median": statistics.median(x[1] for x in percentiles) if percentiles else None,
        "p99_median": statistics.median(x[2] for x in percentiles) if percentiles else None,
    }

    # annotation format discovery
    ext_counts = Counter(p.suffix.lower() for p in all_files)
    result["annotation_format_hints"] = {
        ext: ext_counts.get(ext, 0)
        for ext in sorted(LABEL_EXTS)
        if ext_counts.get(ext, 0)
    }

    # YOLO parse audit when TXT files look like YOLO
    txts = [p for p in all_files if p.suffix.lower() == ".txt"]
    yolo_files = 0
    yolo_valid = 0
    yolo_invalid = 0
    yolo_classes = Counter()
    yolo_boxes = 0
    yolo_oob = 0

    for p in txts[:5000]:
        parsed = parse_yolo_file(p)
        if parsed["valid"] or parsed["invalid"]:
            # treat as YOLO-like only if at least one valid record
            if parsed["valid"]:
                yolo_files += 1
                yolo_valid += parsed["valid"]
                yolo_invalid += parsed["invalid"]
                yolo_classes.update(parsed["classes"])
                yolo_boxes += len(parsed["boxes"])
                for _, x, y, w, h in parsed["boxes"]:
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        yolo_oob += 1

    result["yolo_hypothesis"] = {
        "files_with_valid_yolo_records_sampled": yolo_files,
        "valid_records": yolo_valid,
        "invalid_records": yolo_invalid,
        "objects": yolo_boxes,
        "classes": dict(yolo_classes),
        "out_of_bounds": yolo_oob,
    }

    # Pairing heuristic for common image/label stem structures
    stem_images = Counter(p.stem.lower() for p in all_files if p.suffix.lower() in IMAGE_EXTS)
    stem_txt = Counter(p.stem.lower() for p in txts)
    common_stems = set(stem_images) & set(stem_txt)

    result["stem_pairing"] = {
        "unique_image_stems": len(stem_images),
        "unique_txt_stems": len(stem_txt),
        "common_image_txt_stems": len(common_stems),
    }

    # Show top-level structure without assuming semantics
    top_dirs = sorted({
        p.relative_to(root).parts[0]
        for p in all_files
        if p.relative_to(root).parts
    })
    result["top_level_entries"] = top_dirs[:100]

    # Text excerpts for human inspection
    excerpts = {}
    for p in docs[:10]:
        txt = sample_text(p, 5000)
        if txt.strip():
            excerpts[safe_rel(p, root)] = txt
    result["metadata_excerpts"] = excerpts

    # Conservative verdict
    warnings = []

    if readable == 0 and imgs:
        warnings.append("No readable sampled images.")
    if yolo_invalid:
        warnings.append("YOLO-like text files contain invalid records.")
    if yolo_oob:
        warnings.append("YOLO-like records contain out-of-bounds values.")
    if len(common_stems) == 0 and imgs:
        warnings.append("No simple image/TXT stem pairing found. Annotation format may differ.")
    if len(set(dims)) > 5:
        warnings.append("Many image dimensions detected. Inspect whether these represent different acquisition/processing regimes.")

    result["warnings"] = warnings
    result["verdict"] = "REQUIRES_FORMAT_SPECIFIC_REVIEW"

    return result


# ---------------------------------------------------------------------
# Compare datasets using sampled image statistics
# ---------------------------------------------------------------------

def compare_dataset_stats(audits):
    rows = []
    for a in audits:
        s = a.get("sample_images", {})
        rows.append({
            "dataset": a["name"],
            "sample_images": a.get("sample_image_count"),
            "readable": s.get("readable"),
            "dimensions": s.get("dimensions"),
            "modes": s.get("modes"),
            "mean_pixel_mean": s.get("mean_pixel_mean"),
            "mean_pixel_std": s.get("mean_pixel_std"),
            "p01_median": s.get("p01_median"),
            "p50_median": s.get("p50_median"),
            "p99_median": s.get("p99_median"),
        })
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    outdir = CONFIG["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": now(),
        "purpose": "V3 sealed-test failure diagnosis + audit of AI4Shipwrecks, SeabedObjects-KLSG, SCTD",
        "guardrails": [
            "sealed SubPipe test remains frozen",
            "no training performed",
            "no threshold tuning performed",
            "no source dataset modified",
            "AI4Shipwrecks terrain extras are not automatically included",
            "dataset semantics are not inferred when evidence is insufficient",
        ],
    }

    print("=" * 78)
    print("PHASE-1 V3 FAILURE + ADDITIONAL DATASET DIAGNOSTIC")
    print("=" * 78)

    print("\n[1/5] Inspecting V3 sealed-test artifacts...")
    v3 = inspect_v3()
    report["v3"] = v3

    print("\nV3 sealed-test artifact:")
    print(json.dumps(v3.get("sealed_test_artifact", {}), indent=2))

    print("\n[2/5] Auditing AI4Shipwrecks...")
    ai4 = audit_dataset(CONFIG["ai4"], "AI4Shipwrecks")
    report["datasets"] = {"AI4Shipwrecks": ai4}
    print(json.dumps({
        "inventory": ai4["inventory"],
        "sample_images": ai4.get("sample_images"),
        "annotation_format_hints": ai4.get("annotation_format_hints"),
        "yolo_hypothesis": ai4.get("yolo_hypothesis"),
        "warnings": ai4.get("warnings"),
    }, indent=2))

    print("\n[3/5] Auditing SeabedObjects-KLSG...")
    klsg = audit_dataset(CONFIG["klsg"], "SeabedObjects-KLSG")
    report["datasets"]["SeabedObjects-KLSG"] = klsg
    print(json.dumps({
        "inventory": klsg["inventory"],
        "sample_images": klsg.get("sample_images"),
        "annotation_format_hints": klsg.get("annotation_format_hints"),
        "yolo_hypothesis": klsg.get("yolo_hypothesis"),
        "warnings": klsg.get("warnings"),
    }, indent=2))

    print("\n[4/5] Auditing SCTD...")
    sctd = audit_dataset(CONFIG["sctd"], "SCTD")
    report["datasets"]["SCTD"] = sctd
    print(json.dumps({
        "inventory": sctd["inventory"],
        "sample_images": sctd.get("sample_images"),
        "annotation_format_hints": sctd.get("annotation_format_hints"),
        "yolo_hypothesis": sctd.get("yolo_hypothesis"),
        "warnings": sctd.get("warnings"),
    }, indent=2))

    print("\n[5/5] Building cross-dataset comparison...")
    report["comparison"] = compare_dataset_stats(list(report["datasets"].values()))

    # Human-readable report
    txt = []
    txt.append("PHASE-1 V3 FAILURE + ADDITIONAL DATASET DIAGNOSTIC")
    txt.append("=" * 78)
    txt.append(f"Generated: {report['generated_at']}")
    txt.append("")
    txt.append("DECISION GUARDRAILS")
    txt.append("- Sealed SubPipe test remains frozen.")
    txt.append("- No training or threshold tuning was performed.")
    txt.append("- No source dataset was modified.")
    txt.append("- New datasets are development candidates only until audited.")
    txt.append("")

    st = v3.get("sealed_test_artifact", {})
    txt.append("V3 SEALED TEST")
    txt.append("-" * 40)
    for k, v in st.items():
        txt.append(f"{k}: {v}")
    txt.append("")

    for name, a in report["datasets"].items():
        txt.append(name)
        txt.append("-" * 40)
        txt.append(f"Path: {a['root']}")
        txt.append(f"Exists: {a['exists']}")
        txt.append(f"Total files: {a['inventory'].get('files')}")
        txt.append(f"Extensions: {a['inventory'].get('extensions')}")
        txt.append(f"Sample images: {a.get('sample_image_count')}")
        txt.append(f"Readable sample: {a.get('sample_images', {}).get('readable')}")
        txt.append(f"Dimensions: {a.get('sample_images', {}).get('dimensions')}")
        txt.append(f"Modes: {a.get('sample_images', {}).get('modes')}")
        txt.append(f"Annotation hints: {a.get('annotation_format_hints')}")
        txt.append(f"YOLO hypothesis: {a.get('yolo_hypothesis')}")
        txt.append(f"Warnings: {a.get('warnings')}")
        txt.append("Top-level entries:")
        for x in a.get("top_level_entries", []):
            txt.append(f"  - {x}")
        txt.append("Metadata candidates:")
        for x in a.get("metadata_candidates", [])[:20]:
            txt.append(f"  - {x}")
        txt.append("")

    json_path = outdir / "phase1_v3_and_newdata_diagnostic.json"
    txt_path = outdir / "phase1_v3_and_newdata_diagnostic.txt"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    txt_path.write_text("\n".join(txt), encoding="utf-8")

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)
    print(f"JSON report: {json_path}")
    print(f"TXT report : {txt_path}")
    print("\nIMPORTANT: This script does NOT train a model and does NOT alter data.")


if __name__ == "__main__":
    main()
