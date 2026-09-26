"""
COMPREHENSIVE SSS DATASET FORENSIC AUDIT
========================================

Audits the four datasets for Phase-1/Phase-2 planning without modifying
any source data and without training a model.

Datasets:
  1. SubPipeMiniSSS
  2. AI4Shipwrecks
  3. SeabedObjects-KLSG
  4. SCTD

Run:
    python comprehensive_sss_dataset_audit.py

IMPORTANT:
- No source files are modified.
- No annotations are converted.
- No model is trained.
- No sealed SubPipe test metrics are changed.
- Taxonomy is reported from files/metadata; it is NOT invented.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image



DATASETS = {
    "SubPipeMiniSSS": Path(r"K:\Debris model\SubPipeMiniSSS"),
    "AI4Shipwrecks": Path(r"K:\Debris model\AI4Shipwrecks"),
    "SeabedObjects-KLSG": Path(r"D:\ML Lab\Project Debris Dataset\SeabedObjects-KLSG"),
    "SCTD": Path(r"D:\ML Lab\Project Debris Dataset\SCTD"),
}

OUTPUT = Path(r"K:\Debris model\experiments\comprehensive_sss_dataset_audit")

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pbm", ".bpm"
}

ANNOTATION_EXTS = {
    ".txt", ".xml", ".json", ".csv", ".yaml", ".yml", ".pbm", ".bpm"
}

TEXT_EXTS = {
    ".txt", ".md", ".json", ".yaml", ".yml", ".xml", ".csv"
}



def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_rel(path, root):
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def files_under(root):
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def text(path, limit=20000):
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def likely_metadata(path):
    n = path.name.lower()
    keys = [
        "readme", "dataset", "metadata", "annotation", "label",
        "class", "category", "description", "license", "config",
        "info", "split", "train", "test", "valid"
    ]
    return any(k in n for k in keys)


def role_hint(path):
    s = str(path).lower()
    if re.search(r"(^|[/\\])(train|training)([/\\]|$)", s):
        return "train"
    if re.search(r"(^|[/\\])(val|valid|validation)([/\\]|$)", s):
        return "validation"
    if re.search(r"(^|[/\\])(test|testing|eval|evaluation)([/\\]|$)", s):
        return "test"
    return "unknown"



def image_info(path):
    try:
        with Image.open(path) as im:
            arr = np.asarray(im)
            a = arr.astype(np.float32)

            return {
                "readable": True,
                "format": im.format,
                "mode": im.mode,
                "width": int(im.width),
                "height": int(im.height),
                "channels": (
                    1 if arr.ndim == 2
                    else int(arr.shape[2])
                ),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "p01": float(np.percentile(a, 1)),
                "p05": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
            }
    except Exception as e:
        return {
            "readable": False,
            "error": repr(e),
        }


def audit_images(root, images, stats_sample_size=40):
    dims = Counter()
    modes = Counter()
    formats = Counter()
    roles = Counter()
    suffixes = Counter()

    for p in images:
        roles[role_hint(p)] += 1
        suffixes[p.suffix.lower()] += 1

    ordered = sorted(images)
    if len(ordered) <= stats_sample_size:
        sample = ordered
    else:
        idx = np.linspace(0, len(ordered) - 1, stats_sample_size, dtype=int)
        sample = [ordered[int(i)] for i in sorted(set(idx))]

    readable = 0
    unreadable = []
    means, stds, p01, p50, p99 = [], [], [], [], []

    for p in sample:
        info = image_info(p)
        if not info["readable"]:
            unreadable.append(safe_rel(p, root))
            continue

        readable += 1
        key = f'{info["width"]}x{info["height"]}'
        dims[key] += 1
        modes[info["mode"]] += 1
        formats[str(info["format"])] += 1

        means.append(info["mean"])
        stds.append(info["std"])
        p01.append(info["p01"])
        p50.append(info["p50"])
        p99.append(info["p99"])

    return {
        "total_images": len(images),
        "readability_checked_on_sample": len(sample),
        "readable": readable,
        "unreadable": len(unreadable),
        "readable_sample": readable,
        "unreadable_sample": len(unreadable),
        "unreadable_examples": unreadable[:20],
        "dimensions": dict(dims),
        "sample_dimensions": dict(dims),
        "modes": dict(modes),
        "sample_modes": dict(modes),
        "formats": dict(formats),
        "sample_formats": dict(formats),
        "extension_counts": dict(suffixes),
        "directory_role_hints": dict(roles),
        "pixel_statistics": {
            "mean_of_means": statistics.mean(means) if means else None,
            "median_of_means": statistics.median(means) if means else None,
            "mean_std": statistics.mean(stds) if stds else None,
            "median_std": statistics.median(stds) if stds else None,
            "median_p01": statistics.median(p01) if p01 else None,
            "median_p50": statistics.median(p50) if p50 else None,
            "median_p99": statistics.median(p99) if p99 else None,
        },
        "pixel_statistics_sample_only": {
            "mean_of_means": statistics.mean(means) if means else None,
            "median_of_means": statistics.median(means) if means else None,
            "mean_std": statistics.mean(stds) if stds else None,
            "median_std": statistics.median(stds) if stds else None,
            "median_p01": statistics.median(p01) if p01 else None,
            "median_p50": statistics.median(p50) if p50 else None,
            "median_p99": statistics.median(p99) if p99 else None,
        },
    }



def parse_yolo(path):
    valid = []
    invalid = []

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()
    except Exception as e:
        return valid, [f"READ_ERROR: {e}"]

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            invalid.append({
                "line": i,
                "text": line,
                "reason": "expected_5_fields",
            })
            continue

        try:
            vals = list(map(float, parts))
        except Exception:
            invalid.append({
                "line": i,
                "text": line,
                "reason": "non_numeric",
            })
            continue

        c, x, y, w, h = vals

        if not all(math.isfinite(v) for v in vals):
            invalid.append({
                "line": i,
                "text": line,
                "reason": "non_finite",
            })
            continue

        valid.append((c, x, y, w, h))

    return valid, invalid


def audit_yolo(root, txts):
    candidate_files = 0
    valid_records = 0
    invalid_records = 0
    classes = Counter()
    oob = 0
    zero_geometry = 0
    area = []

    examples = []

    for p in txts:
        valid, invalid = parse_yolo(p)

        if not valid and not invalid:
            continue

        if valid:
            candidate_files += 1

        valid_records += len(valid)
        invalid_records += len(invalid)

        for c, x, y, w, h in valid:
            classes[str(int(c)) if float(c).is_integer() else str(c)] += 1

            if not (
                0 <= x <= 1 and
                0 <= y <= 1 and
                0 <= w <= 1 and
                0 <= h <= 1
            ):
                oob += 1

            if w <= 0 or h <= 0:
                zero_geometry += 1

            area.append(max(0, w) * max(0, h))

        if invalid and len(examples) < 20:
            examples.append({
                "file": safe_rel(p, root),
                "invalid": invalid[:5],
            })

    return {
        "candidate_yolo_files": candidate_files,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "classes": dict(classes),
        "out_of_bounds_records": oob,
        "zero_geometry_records": zero_geometry,
        "box_area": {
            "median": statistics.median(area) if area else None,
            "mean": statistics.mean(area) if area else None,
            "min": min(area) if area else None,
            "max": max(area) if area else None,
        },
        "invalid_examples": examples,
    }



def local_tag(tag):
    return tag.split("}")[-1].lower()


def xml_objects(path):
    """
    Extracts common object-detection XML structures conservatively.
    Does not convert or modify anything.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return {
            "readable": False,
            "error": repr(e),
            "objects": [],
            "root_tag": None,
        }

    objects = []

    for obj in root.iter():
        if local_tag(obj.tag) not in {"object", "item", "annotation", "target"}:
            continue

        name = None
        bbox = None

        for child in obj.iter():
            tag = local_tag(child.tag)

            if tag in {"name", "class", "label", "category"} and child.text:
                if name is None:
                    name = child.text.strip()

        coords = {}
        for child in obj.iter():
            tag = local_tag(child.tag)
            if tag in {"xmin", "xmax", "ymin", "ymax"} and child.text:
                try:
                    coords[tag] = float(child.text.strip())
                except Exception:
                    pass

        if len(coords) == 4:
            bbox = coords

        objects.append({
            "label": name,
            "bbox": bbox,
        })

    return {
        "readable": True,
        "root_tag": local_tag(root.tag),
        "objects": objects,
    }


def audit_xml(root, xmls):
    readable = 0
    unreadable = 0
    total_objects = 0
    classes = Counter()
    bbox_objects = 0
    no_label = 0
    no_bbox = 0
    examples = []

    for p in xmls:
        result = xml_objects(p)

        if not result["readable"]:
            unreadable += 1
            continue

        readable += 1

        for obj in result["objects"]:
            total_objects += 1

            label = obj.get("label")
            if label:
                classes[label] += 1
            else:
                no_label += 1

            if obj.get("bbox"):
                bbox_objects += 1
            else:
                no_bbox += 1

        if result["objects"] and len(examples) < 10:
            examples.append({
                "file": safe_rel(p, root),
                "root_tag": result["root_tag"],
                "objects": result["objects"][:10],
            })

    return {
        "xml_files": len(xmls),
        "readable": readable,
        "unreadable": unreadable,
        "objects": total_objects,
        "classes": dict(classes),
        "objects_with_bbox": bbox_objects,
        "objects_without_bbox": no_bbox,
        "objects_without_label": no_label,
        "examples": examples,
    }



def json_summary(path):
    try:
        obj = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        )
    except Exception as e:
        return {
            "readable": False,
            "error": repr(e),
        }

    def summarize(x, depth=0):
        if depth > 3:
            return type(x).__name__

        if isinstance(x, dict):
            return {
                "type": "dict",
                "keys": list(x.keys())[:100],
                "children": {
                    str(k): summarize(v, depth + 1)
                    for k, v in list(x.items())[:20]
                },
            }

        if isinstance(x, list):
            return {
                "type": "list",
                "length": len(x),
                "first": summarize(x[0], depth + 1) if x else None,
            }

        return {
            "type": type(x).__name__,
            "value": str(x)[:500],
        }

    return {
        "readable": True,
        "summary": summarize(obj),
    }


def audit_json(root, jsons):
    out = {}
    for p in jsons[:30]:
        out[safe_rel(p, root)] = json_summary(p)
    return out


def mask_info(path):
    try:
        with Image.open(path) as im:
            arr = np.asarray(im)
            uniq = np.unique(arr)
            return {
                "readable": True,
                "mode": im.mode,
                "width": im.width,
                "height": im.height,
                "unique_values_sample": [
                    int(x) if np.issubdtype(type(x), np.integer)
                    else float(x)
                    for x in uniq[:100]
                ],
                "unique_count": int(len(uniq)),
                "nonzero_fraction": float(np.mean(arr != 0)),
            }
    except Exception as e:
        return {"readable": False, "error": repr(e)}



def pair_by_stem(images, annotations):
    im = defaultdict(list)
    an = defaultdict(list)

    for p in images:
        im[p.stem.lower()].append(p)

    for p in annotations:
        an[p.stem.lower()].append(p)

    common = set(im) & set(an)
    image_only = set(im) - set(an)
    annotation_only = set(an) - set(im)

    return {
        "unique_image_stems": len(im),
        "unique_annotation_stems": len(an),
        "common_stems": len(common),
        "image_only_stems": len(image_only),
        "annotation_only_stems": len(annotation_only),
        "image_only_examples": sorted(image_only)[:50],
        "annotation_only_examples": sorted(annotation_only)[:50],
    }


def audit_one(name, root):
    result = {
        "name": name,
        "root": str(root),
        "exists": root.exists(),
        "generated_at": iso_now(),
    }

    if not root.exists():
        result["verdict"] = "PATH_NOT_FOUND"
        return result

    all_files = files_under(root)
    images = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTS
    ]

    txts = [
        p for p in all_files
        if p.suffix.lower() == ".txt"
    ]

    xmls = [
        p for p in all_files
        if p.suffix.lower() == ".xml"
    ]

    jsons = [
        p for p in all_files
        if p.suffix.lower() == ".json"
    ]

    masks = [
        p for p in all_files
        if p.suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}
        and any(
            k in p.name.lower()
            for k in ["mask", "label", "annotation", "segmentation"]
        )
    ]

    ext = Counter(p.suffix.lower() or "<none>" for p in all_files)

    result["inventory"] = {
        "files": len(all_files),
        "bytes": sum(
            p.stat().st_size for p in all_files
            if p.exists()
        ),
        "extensions": dict(sorted(ext.items())),
    }

    result["structure"] = {
        "top_level_entries": sorted({
            p.relative_to(root).parts[0]
            for p in all_files
            if p.relative_to(root).parts
        }),
        "directory_role_hints": dict(
            Counter(role_hint(p) for p in root.rglob("*") if p.is_dir())
        ),
    }

    metadata = [
        p for p in all_files
        if p.suffix.lower() in TEXT_EXTS and likely_metadata(p)
    ]

    result["metadata_candidates"] = [
        safe_rel(p, root) for p in metadata[:100]
    ]

    metadata_excerpts = {}
    for p in metadata[:15]:
        sample_text = text(p, 8000)
        if sample_text.strip():
            metadata_excerpts[safe_rel(p, root)] = sample_text
    result["metadata_excerpts"] = metadata_excerpts

    result["images"] = audit_images(root, images)

    result["yolo"] = audit_yolo(root, txts)
    result["xml"] = audit_xml(root, xmls)
    result["json"] = audit_json(root, jsons)

    annotation_candidates = txts + xmls + jsons

    result["stem_pairing"] = pair_by_stem(
        images,
        annotation_candidates
    )

    mask_results = []
    for p in sorted(masks)[:40]:
        info = mask_info(p)
        if info["readable"]:
            mask_results.append(info)

    result["mask_audit"] = {
        "candidate_mask_files_sampled": min(len(masks), 40),
        "readable_sampled": sum(
            1 for x in mask_results if x["readable"]
        ),
        "nonzero_fraction_median": (
            statistics.median(
                x["nonzero_fraction"]
                for x in mask_results
            )
            if mask_results else None
        ),
        "unique_value_patterns": Counter(
            str(x["unique_values_sample"])
            for x in mask_results
        ).most_common(20),
    }

    result["candidate_groups"] = sorted({
        safe_rel(p.parent, root).split("\\")[0]
        for p in images
        if safe_rel(p, root)
    })[:200]

    warnings = []

    if not images:
        warnings.append("No image files discovered.")

    if result["images"].get("unreadable", 0):
        warnings.append("Unreadable images detected in the bounded image sample.")

    if result["xml"]["unreadable"]:
        warnings.append("Unreadable XML files detected.")

    if result["yolo"]["invalid_records"]:
        warnings.append("Invalid YOLO-like records detected.")

    if result["yolo"]["out_of_bounds_records"]:
        warnings.append("Out-of-bounds YOLO-like records detected.")

    if (
        images and annotation_candidates
        and result["stem_pairing"]["common_stems"] == 0
    ):
        warnings.append(
            "No simple image/annotation stem pairing. "
            "Dataset likely uses another association mechanism."
        )

    if len(result["images"].get("dimensions", {})) > 10:
        warnings.append(
            "Highly heterogeneous image dimensions. "
            "Determine whether these represent native frames, crops, "
            "patches, or mixed source material before preprocessing."
        )

    result["warnings"] = warnings
    result["verdict"] = "REQUIRES_SEMANTIC_REVIEW"

    return result



def cross_compare(results):
    rows = []

    for name, r in results.items():
        img = r.get("images", {})
        px = img.get("pixel_statistics", {})
        y = r.get("yolo", {})
        x = r.get("xml", {})

        rows.append({
            "dataset": name,
            "exists": r.get("exists"),
            "files": r.get("inventory", {}).get("files"),
            "images": img.get("total_images"),
            "readable_images": img.get("readable"),
            "dimensions": img.get("dimensions"),
            "modes": img.get("modes"),
            "mean_pixel_mean": px.get("mean_of_means"),
            "median_pixel_mean": px.get("median_of_means"),
            "median_p50": px.get("median_p50"),
            "median_p99": px.get("median_p99"),
            "yolo_objects": y.get("valid_records"),
            "yolo_classes": y.get("classes"),
            "xml_objects": x.get("objects"),
            "xml_classes": x.get("classes"),
        })

    return rows



def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    report = {
        "title": "Comprehensive SSS Dataset Forensic Audit",
        "generated_at": iso_now(),
        "scope": {
            "datasets": {
                k: str(v) for k, v in DATASETS.items()
            },
            "no_training": True,
            "no_threshold_tuning": True,
            "no_source_modification": True,
            "no_annotation_conversion": True,
            "sealed_subpipe_test_untouched": True,
        },
        "datasets": {},
    }

    print("=" * 80)
    print("COMPREHENSIVE SSS DATASET FORENSIC AUDIT")
    print("=" * 80)

    for i, (name, root) in enumerate(DATASETS.items(), 1):
        print(f"\n[{i}/{len(DATASETS)}] {name}", flush=True)
        print(f"Root: {root}", flush=True)

        r = audit_one(name, root)
        report["datasets"][name] = r

        if not r["exists"]:
            print("  STATUS: PATH NOT FOUND", flush=True)
            continue

        print(
            f"  files={r['inventory']['files']} | "
            f"images={r['images']['total_images']} | "
            f"readable_sample={r['images']['readable']}/{r['images']['readability_checked_on_sample']}"
        )

        print(
            f"  YOLO objects={r['yolo']['valid_records']} | "
            f"XML objects={r['xml']['objects']} | "
            f"XML classes={r['xml']['classes']}"
        )

        print(f"  warnings={len(r['warnings'])}")

    report["cross_dataset_comparison"] = cross_compare(
        report["datasets"]
    )

    decision = {
        "ready_for_training": False,
        "reason": (
            "All datasets must pass semantic compatibility, annotation "
            "quality, SSS verification, and split-design review before "
            "unified training."
        ),
        "next_required_step": (
            "Review dataset-specific annotation/class outputs and "
            "representative images, then define a common taxonomy and "
            "dataset-specific split policy."
        ),
    }

    report["decision"] = decision

    json_path = OUTPUT / "comprehensive_sss_dataset_audit.json"
    txt_path = OUTPUT / "comprehensive_sss_dataset_audit.txt"

    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8"
    )

    lines = []
    lines.append("COMPREHENSIVE SSS DATASET FORENSIC AUDIT")
    lines.append("=" * 80)
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("GUARDRAILS")
    lines.append("- No training.")
    lines.append("- No threshold tuning.")
    lines.append("- No source dataset modification.")
    lines.append("- No annotation conversion.")
    lines.append("- Pixel statistics are computed on bounded deterministic samples, not every image.")
    lines.append("- SubPipe sealed test remains untouched.")
    lines.append("")

    for name, r in report["datasets"].items():
        lines.append(name)
        lines.append("-" * 80)

        if not r["exists"]:
            lines.append(f"PATH NOT FOUND: {r['root']}")
            lines.append("")
            continue

        lines.append(f"Root: {r['root']}")
        lines.append(f"Files: {r['inventory']['files']}")
        lines.append(f"Images: {r['images']['total_images']}")
        lines.append(f"Readable images: {r['images']['readable']}")
        lines.append(f"Dimensions: {r['images']['dimensions']}")
        lines.append(f"Modes: {r['images']['modes']}")
        lines.append(f"Extensions: {r['inventory']['extensions']}")
        lines.append("")
        lines.append(f"YOLO: {r['yolo']}")
        lines.append("")
        lines.append(f"XML: {r['xml']}")
        lines.append("")
        lines.append(
            f"Stem pairing: {r['stem_pairing']}"
        )
        lines.append("")
        lines.append(
            f"Mask audit: {r['mask_audit']}"
        )
        lines.append("")
        lines.append("Warnings:")
        for w in r["warnings"]:
            lines.append(f"  - {w}")
        lines.append("")
        lines.append("Metadata candidates:")
        for x in r["metadata_candidates"][:30]:
            lines.append(f"  - {x}")
        lines.append("")
        lines.append("Metadata excerpts:")
        for k, v in list(r["metadata_excerpts"].items())[:5]:
            lines.append(f"\n--- {k} ---\n{v[:4000]}")
        lines.append("")

    lines.append("CROSS-DATASET COMPARISON")
    lines.append("-" * 80)
    for row in report["cross_dataset_comparison"]:
        lines.append(json.dumps(row, default=str))

    lines.append("")
    lines.append("DECISION")
    lines.append("-" * 80)
    lines.append(json.dumps(decision, indent=2))

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")
    print("\nTRAINING APPROVAL: NOT GRANTED BY THIS SCRIPT")


if __name__ == "__main__":
    main()
