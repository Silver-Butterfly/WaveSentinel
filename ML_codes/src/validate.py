"""

Checks:
- exact filesystem image/label accounting
- expected split counts from split_report.json
- readable images and image dimensions
- missing/orphan labels
- duplicate image stems
- duplicate image hashes
- cross-split image hash/stem leakage
- source-family and pipe-acquisition-group leakage
- YOLO label syntax
- allowed classes 0-4, with class 0 reported separately as excluded
- finite normalized coordinates
- zero/negative/oversized boxes
- boxes extending outside image bounds
- boundary-touching boxes, reported but NOT treated as invalid
- tiny boxes, reported but NOT automatically treated as invalid
- duplicate label rows
- per-class object counts
- per-source counts
- deterministic JSON + TXT reports

The validator never modifies the dataset.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SPLITS = ("train", "val", "test")
ALLOWED_CLASSES = {0, 1, 2, 3, 4}
TARGET_CLASSES = {1, 2, 3, 4}
EXCLUDED_CLASSES = {0}

BOUNDARY_EPS = 1e-6
OUT_OF_BOUNDS_EPS = 1e-6
TINY_AREA_THRESHOLD = 1e-5

IMAGE_DIR_NAME = "images"
LABEL_DIR_NAME = "labels"

SOURCE_PATTERNS = [
    ("pipe", re.compile(r"(^|[_\-.])pipe([_\-.]|$)", re.I)),
    ("mine", re.compile(r"(^|[_\-.])mine(_?cylinder)?([_\-.]|$)", re.I)),
    ("synth", re.compile(r"(synthetic[_-]?ghost[_-]?net|synth|ghost[_-]?net)", re.I)),
    ("bg", re.compile(r"(^|[_\-.])(bg|background)([_\-.]|$)", re.I)),
    ("wreckA", re.compile(r"wreck[_-]?a", re.I)),
    ("wreckR", re.compile(r"wreck[_-]?r", re.I)),
]

PIPE_GROUP_RE = re.compile(r"^pipe_([^_]+)", re.I)


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def source_family(name):
    for source, pattern in SOURCE_PATTERNS:
        if pattern.search(name):
            return source
    return "other"


def acquisition_group(name):
    m = PIPE_GROUP_RE.match(name)
    if m:
        return f"pipe:{m.group(1)}"
    return f"{source_family(name)}:{Path(name).stem}"


def parse_label_file(path):
    errors = []
    rows = []
    duplicate_rows = []
    seen = set()

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except Exception as e:
            return [], [f"cannot decode label file: {e}"], []
    except Exception as e:
        return [], [f"cannot read label file: {e}"], []

    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()

        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append({
                "line": line_no,
                "reason": f"expected 5 fields, got {len(parts)}",
                "text": line,
            })
            continue

        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except Exception as e:
            errors.append({
                "line": line_no,
                "reason": f"non-numeric label: {e}",
                "text": line,
            })
            continue

        xc, yc, w, h = vals

        if cls not in ALLOWED_CLASSES:
            errors.append({
                "line": line_no,
                "reason": f"class {cls} outside allowed range 0-4",
                "text": line,
            })
            continue

        if not all(math.isfinite(v) for v in vals):
            errors.append({
                "line": line_no,
                "reason": "NaN or infinite coordinate",
                "text": line,
            })
            continue

        if w <= 0 or h <= 0:
            errors.append({
                "line": line_no,
                "reason": "width/height must be > 0",
                "text": line,
            })
            continue

        if w > 1 or h > 1:
            errors.append({
                "line": line_no,
                "reason": "width/height > 1",
                "text": line,
            })
            continue

        if xc < 0 or xc > 1 or yc < 0 or yc > 1:
            errors.append({
                "line": line_no,
                "reason": "center coordinate outside [0,1]",
                "text": line,
            })
            continue

        box = (cls, xc, yc, w, h)

        if box in seen:
            duplicate_rows.append({
                "line": line_no,
                "text": line,
            })
        seen.add(box)
        rows.append({
            "line": line_no,
            "class": cls,
            "xc": xc,
            "yc": yc,
            "w": w,
            "h": h,
            "text": line,
        })

    return rows, errors, duplicate_rows


def box_flags(row):
    xc, yc, w, h = row["xc"], row["yc"], row["w"], row["h"]

    left = xc - w / 2
    right = xc + w / 2
    top = yc - h / 2
    bottom = yc + h / 2

    boundary = (
        abs(left) <= BOUNDARY_EPS
        or abs(right - 1) <= BOUNDARY_EPS
        or abs(top) <= BOUNDARY_EPS
        or abs(bottom - 1) <= BOUNDARY_EPS
    )

    out = (
        left < -OUT_OF_BOUNDS_EPS
        or right > 1 + OUT_OF_BOUNDS_EPS
        or top < -OUT_OF_BOUNDS_EPS
        or bottom > 1 + OUT_OF_BOUNDS_EPS
    )

    tiny = (w * h) < TINY_AREA_THRESHOLD

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "boundary": boundary,
        "out_of_bounds": out,
        "tiny": tiny,
        "area": w * h,
    }


def discover_files(directory):
    if not directory.exists():
        return [], []

    files = [p for p in directory.iterdir() if p.is_file()]
    return sorted(files, key=lambda p: p.name.lower())


def inspect_split(root, split):
    split_root = root / split
    image_dir = split_root / IMAGE_DIR_NAME
    label_dir = split_root / LABEL_DIR_NAME

    all_image_dir_files = discover_files(image_dir)
    all_label_dir_files = discover_files(label_dir)

    images = []
    non_images = []

    for p in all_image_dir_files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                width, height = im.size
                mode = im.mode
            images.append({
                "path": p,
                "name": p.name,
                "stem": p.stem,
                "sha256": sha256_file(p),
                "width": width,
                "height": height,
                "mode": mode,
            })
        except Exception as e:
            non_images.append({
                "name": p.name,
                "reason": str(e),
            })

    label_map = defaultdict(list)
    for p in all_label_dir_files:
        label_map[p.stem].append(p)

    duplicate_label_stems = {
        stem: [p.name for p in paths]
        for stem, paths in label_map.items()
        if len(paths) > 1
    }

    image_stems = Counter(x["stem"] for x in images)
    duplicate_image_stems = {
        stem: count for stem, count in image_stems.items() if count > 1
    }

    image_names = {x["stem"] for x in images}
    label_stems = set(label_map)

    missing_labels = sorted(image_names - label_stems)
    orphan_labels = sorted(label_stems - image_names)

    result = {
        "split": split,
        "filesystem_image_files": len(all_image_dir_files),
        "readable_images": len(images),
        "non_image_or_read_failures": non_images,
        "filesystem_label_files": len(all_label_dir_files),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "duplicate_image_stems": duplicate_image_stems,
        "duplicate_label_stems": duplicate_label_stems,
        "images": [],
        "dimensions": Counter(),
        "modes": Counter(),
        "objects": 0,
        "empty_images": 0,
        "classes": Counter(),
        "excluded_class_0_rows": 0,
        "excluded_class_0_images": 0,
        "sources": Counter(),
        "groups": Counter(),
        "boundary_boxes": 0,
        "out_of_bounds_boxes": 0,
        "tiny_boxes": 0,
        "duplicate_label_rows": 0,
        "label_error_images": 0,
        "label_errors": [],
        "box_errors": [],
        "duplicate_rows": [],
        "hashes": {},
        "stem_to_hash": {},
    }

    for img in images:
        stem = img["stem"]
        source = source_family(img["name"])
        group = acquisition_group(img["name"])

        result["dimensions"][f'{img["width"]}x{img["height"]}'] += 1
        result["modes"][img["mode"]] += 1
        result["sources"][source] += 1
        result["groups"][group] += 1
        result["hashes"][stem] = img["sha256"]
        result["stem_to_hash"][stem] = img["sha256"]

        label_paths = label_map.get(stem, [])

        item = {
            "image": img["name"],
            "stem": stem,
            "sha256": img["sha256"],
            "source": source,
            "group": group,
            "width": img["width"],
            "height": img["height"],
            "label_file": label_paths[0].name if len(label_paths) == 1 else None,
            "objects": 0,
            "classes": Counter(),
            "boundary_boxes": 0,
            "out_of_bounds_boxes": 0,
            "tiny_boxes": 0,
            "errors": [],
        }

        if len(label_paths) != 1:
            if not label_paths:
                item["errors"].append("missing label")
            else:
                item["errors"].append("duplicate label files")
            result["label_error_images"] += 1
            result["images"].append(item)
            continue

        rows, errors, duplicate_rows = parse_label_file(label_paths[0])

        if errors:
            result["label_error_images"] += 1
            item["errors"].extend(errors)
            result["label_errors"].append({
                "image": img["name"],
                "label": label_paths[0].name,
                "errors": errors,
            })

        if duplicate_rows:
            result["duplicate_label_rows"] += len(duplicate_rows)
            result["duplicate_rows"].append({
                "image": img["name"],
                "label": label_paths[0].name,
                "rows": duplicate_rows,
            })

        if not rows:
            result["empty_images"] += 1

        for row in rows:
            flags = box_flags(row)

            item["objects"] += 1
            item["classes"][str(row["class"])] += 1
            result["objects"] += 1
            result["classes"][row["class"]] += 1

            if row["class"] == 0:
                result["excluded_class_0_rows"] += 1
                result["excluded_class_0_images"] += 1

            if flags["boundary"]:
                item["boundary_boxes"] += 1
                result["boundary_boxes"] += 1

            if flags["out_of_bounds"]:
                item["out_of_bounds_boxes"] += 1
                result["out_of_bounds_boxes"] += 1
                result["box_errors"].append({
                    "split": split,
                    "image": img["name"],
                    "label": label_paths[0].name,
                    "line": row["line"],
                    "class": row["class"],
                    "xc": row["xc"],
                    "yc": row["yc"],
                    "w": row["w"],
                    "h": row["h"],
                    "left": flags["left"],
                    "right": flags["right"],
                    "top": flags["top"],
                    "bottom": flags["bottom"],
                    "reason": "box extends outside image bounds",
                })

            if flags["tiny"]:
                item["tiny_boxes"] += 1
                result["tiny_boxes"] += 1

        result["images"].append(item)

    result["dimensions"] = dict(result["dimensions"])
    result["modes"] = dict(result["modes"])
    result["classes"] = {str(k): v for k, v in sorted(result["classes"].items())}
    result["sources"] = dict(sorted(result["sources"].items()))
    result["groups"] = dict(sorted(result["groups"].items()))

    return result


def duplicate_groups(mapping):
    groups = defaultdict(list)
    for key, value in mapping.items():
        groups[value].append(key)
    return {
        value: sorted(keys)
        for value, keys in groups.items()
        if len(keys) > 1
    }


def cross_split_checks(split_results):
    hash_to_locations = defaultdict(list)
    stem_to_locations = defaultdict(list)
    group_to_locations = defaultdict(list)

    for split, result in split_results.items():
        for item in result["images"]:
            hash_to_locations[item["sha256"]].append(f"{split}/{item['image']}")
            stem_to_locations[item["stem"]].append(f"{split}/{item['image']}")
            group_to_locations[item["group"]].append(f"{split}/{item['image']}")

    def only_cross_split(groups):
        out = {}
        for key, locations in groups.items():
            splits = {x.split("/", 1)[0] for x in locations}
            if len(splits) > 1:
                out[key] = sorted(locations)
        return out

    return {
        "cross_split_hash_overlap": only_cross_split(hash_to_locations),
        "cross_split_stem_overlap": only_cross_split(stem_to_locations),
        "cross_split_group_overlap": only_cross_split(group_to_locations),
        "duplicate_hashes_within_any_split": {
            split: duplicate_groups({
                item["image"]: item["sha256"]
                for item in result["images"]
            })
            for split, result in split_results.items()
        },
    }


def load_expected_counts(root):
    report_path = root / "split_report.json"
    if not report_path.exists():
        return None, "split_report.json not found"

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"could not read split_report.json: {e}"

    expected = {}
    targets = data.get("targets", {})
    actual = data.get("actual", {})

    for split in SPLITS:
        value = actual.get(split, targets.get(split))
        if isinstance(value, int):
            expected[split] = value

    return expected, None


def build_verdict(root, splits, cross, expected, expected_error):
    failures = []
    warnings = []

    if expected_error:
        warnings.append(expected_error)

    for split, r in splits.items():
        if expected and split in expected:
            if r["readable_images"] != expected[split]:
                failures.append(
                    f"{split}: expected {expected[split]} images from split_report.json, "
                    f"found {r['readable_images']}"
                )

        if r["filesystem_image_files"] != r["readable_images"]:
            failures.append(
                f"{split}: {r['filesystem_image_files']} files exist in images/ but "
                f"only {r['readable_images']} are readable images"
            )

        if r["missing_labels"]:
            failures.append(f"{split}: {len(r['missing_labels'])} missing labels")

        if r["orphan_labels"]:
            failures.append(f"{split}: {len(r['orphan_labels'])} orphan labels")

        if r["duplicate_image_stems"]:
            failures.append(f"{split}: duplicate image stems detected")

        if r["duplicate_label_stems"]:
            failures.append(f"{split}: duplicate label stems detected")

        if r["label_error_images"]:
            failures.append(f"{split}: {r['label_error_images']} images have label errors")

        if r["out_of_bounds_boxes"]:
            failures.append(
                f"{split}: {r['out_of_bounds_boxes']} boxes extend outside image bounds"
            )

    if cross["cross_split_hash_overlap"]:
        failures.append("cross-split exact image hash leakage detected")

    if cross["cross_split_stem_overlap"]:
        failures.append("cross-split filename-stem leakage detected")

    if cross["cross_split_group_overlap"]:
        failures.append("cross-split acquisition-group leakage detected")

    for split, dupes in cross["duplicate_hashes_within_any_split"].items():
        if dupes:
            failures.append(f"{split}: duplicate image hashes detected")

    total_excluded = sum(r["excluded_class_0_rows"] for r in splits.values())
    if total_excluded:
        warnings.append(
            f"class 0 is present in {total_excluded} label rows. "
            "It is excluded from the target classes, but must be removed/remapped "
            "before training if the final model uses only classes 1-4."
        )

    total_tiny = sum(r["tiny_boxes"] for r in splits.values())
    if total_tiny:
        warnings.append(
            f"{total_tiny} tiny boxes detected (area < {TINY_AREA_THRESHOLD:g}); "
            "inspect them, but they are not automatically invalid."
        )

    total_boundary = sum(r["boundary_boxes"] for r in splits.values())
    if total_boundary:
        warnings.append(
            f"{total_boundary} boundary-touching boxes detected. "
            "These are valid clipped/tiled boxes and are not treated as failures."
        )

    return {
        "status": "FAIL" if failures else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "failures": failures,
        "warnings": warnings,
    }


def make_summary(data):
    lines = []
    lines.append("DRISHTI-SSS V5 DATA VALIDITY AUDIT")
    lines.append("=" * 72)
    lines.append(f"ROOT: {data['root']}")
    lines.append(f"STATUS: {data['verdict']['status']}")
    lines.append("")

    if data["verdict"]["failures"]:
        lines.append("FAILURES")
        for x in data["verdict"]["failures"]:
            lines.append(f"  - {x}")
        lines.append("")

    if data["verdict"]["warnings"]:
        lines.append("WARNINGS / REVIEW ITEMS")
        for x in data["verdict"]["warnings"]:
            lines.append(f"  - {x}")
        lines.append("")

    for split in SPLITS:
        r = data["splits"][split]
        lines.append(f"[{split.upper()}]")
        lines.append(
            f"  filesystem image files: {r['filesystem_image_files']}"
        )
        lines.append(f"  readable images:         {r['readable_images']}")
        lines.append(f"  label files:             {r['filesystem_label_files']}")
        lines.append(f"  objects:                 {r['objects']}")
        lines.append(f"  empty images:            {r['empty_images']}")
        lines.append(f"  label-error images:      {r['label_error_images']}")
        lines.append(f"  out-of-bounds boxes:     {r['out_of_bounds_boxes']}")
        lines.append(f"  boundary boxes:          {r['boundary_boxes']}")
        lines.append(f"  tiny boxes:              {r['tiny_boxes']}")
        lines.append(f"  duplicate label rows:    {r['duplicate_label_rows']}")
        lines.append(f"  class 0 rows:            {r['excluded_class_0_rows']}")
        lines.append(f"  dimensions:              {r['dimensions']}")
        lines.append(f"  sources:                 {r['sources']}")
        lines.append(f"  classes:                 {r['classes']}")
        lines.append("")

    c = data["cross_split"]
    lines.append("[CROSS-SPLIT INTEGRITY]")
    lines.append(f"  hash overlaps:  {len(c['cross_split_hash_overlap'])}")
    lines.append(f"  stem overlaps:  {len(c['cross_split_stem_overlap'])}")
    lines.append(f"  group overlaps: {len(c['cross_split_group_overlap'])}")
    lines.append("")

    if data["label_errors"]:
        lines.append("[LABEL ERRORS]")
        for x in data["label_errors"][:50]:
            lines.append(f"  {x['split']}/{x['image']}:")
            for e in x["errors"]:
                lines.append(f"    {e}")
        if len(data["label_errors"]) > 50:
            lines.append(f"  ... {len(data['label_errors']) - 50} more")
        lines.append("")

    if data["box_errors"]:
        lines.append("[OUT-OF-BOUNDS BOXES]")
        for x in data["box_errors"][:100]:
            lines.append(
                f"  {x['split']}/{x['image']} line {x['line']} "
                f"class={x['class']} "
                f"box=({x['xc']:.8f},{x['yc']:.8f},{x['w']:.8f},{x['h']:.8f}) "
                f"edges=({x['left']:.8f},{x['top']:.8f},"
                f"{x['right']:.8f},{x['bottom']:.8f})"
            )
        if len(data["box_errors"]) > 100:
            lines.append(f"  ... {len(data['box_errors']) - 100} more")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="DRISHTI split root")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"ERROR: root does not exist: {root}")

    expected, expected_error = load_expected_counts(root)

    split_results = {}
    all_label_errors = []
    all_box_errors = []

    for split in SPLITS:
        r = inspect_split(root, split)
        split_results[split] = r

        for item in r["label_errors"]:
            item["split"] = split
            all_label_errors.append(item)

        all_box_errors.extend(r["box_errors"])

    cross = cross_split_checks(split_results)

    data = {
        "root": str(root),
        "expected_counts": expected,
        "expected_counts_error": expected_error,
        "splits": split_results,
        "cross_split": cross,
        "label_errors": all_label_errors,
        "box_errors": all_box_errors,
    }

    data["verdict"] = build_verdict(
        root, split_results, cross, expected, expected_error
    )

    # JSON-safe conversion.
    json_path = root / "data_validity_report_v5.json"
    txt_path = root / "data_validity_summary_v5.txt"

    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    txt_path.write_text(
        make_summary(data),
        encoding="utf-8",
    )

    print(make_summary(data))
    print("")
    print(f"JSON report: {json_path}")
    print(f"TXT report:  {txt_path}")

    raise SystemExit(1 if data["verdict"]["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
