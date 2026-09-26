"""
AI-Powered Underwater Marine Debris & Anomaly Detection System
Phase 1 - Preprocessing Characterization

DATASETS
--------
1. SubPipeMini2
2. AI4Shipwrecks

IMPORTANT
---------
- SSS imagery only.
- No FLS.
- No MBES.
- No dataset modification.
- No training.
- This script only audits/characterizes the existing data.
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import re
import statistics

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SUBPIPE_ROOT = Path(
    r"K:\Debris model\SubPipeMiniSSS"
)

AI4_ROOT = Path(
    r"K:\Debris model\AI4Shipwrecks\AI4Shipwrecks"
)

OUTPUT_DIR = SUBPIPE_ROOT / "audit_output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_EXTENSIONS = {
    ".pbm",
    ".bpm",
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def percentile(values, q):
    values = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not values:
        return None

    return float(np.percentile(values, q))


def median(values):
    values = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not values:
        return None

    return float(np.median(values))


def mean(values):
    values = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]

    if not values:
        return None

    return float(np.mean(values))


def save_json(data, path):
    with path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def collect_files(root):
    return [
        p
        for p in root.rglob("*")
        if p.is_file()
    ]


def is_under_audit_output(path, root):
    try:
        relative_parts = {
            part.lower()
            for part in path.relative_to(root).parts
        }

        return "audit_output" in relative_parts

    except Exception:
        return False


def extract_timestamp_from_name(path):
    """
    Extract the final numeric timestamp-like token from filename.

    SubPipe filenames contain Unix-style floating timestamps.
    """

    stem = path.stem

    matches = re.findall(
        r"(?<!\d)(\d{9,}(?:\.\d+)?)(?!\d)",
        stem
    )

    if not matches:
        return None

    try:
        return float(matches[-1])
    except Exception:
        return None


def image_info(path):
    record = {
        "path": str(path),
        "readable": False,
    }

    try:
        with Image.open(path) as img:

            record["readable"] = True
            record["width"] = int(img.width)
            record["height"] = int(img.height)
            record["mode"] = img.mode

            arr = np.asarray(img)

            record["dtype"] = str(arr.dtype)
            record["channels"] = (
                1
                if arr.ndim == 2
                else int(arr.shape[2])
            )

            record["min"] = float(arr.min())
            record["max"] = float(arr.max())
            record["mean"] = float(arr.mean())
            record["std"] = float(arr.std())

    except Exception as e:

        record["error"] = str(e)

    return record


# ============================================================
# STEP 1
# SUBPIPE INVENTORY
# ============================================================

def inventory_subpipe():
    print("\n[1/7] Inventorying SubPipeMini2...")

    files = collect_files(SUBPIPE_ROOT)

    extension_counts = Counter()

    for path in files:

        if is_under_audit_output(
            path,
            SUBPIPE_ROOT
        ):
            continue

        extension_counts[
            path.suffix.lower()
        ] += 1

    result = {
        "root": str(SUBPIPE_ROOT),
        "file_count": int(
            sum(extension_counts.values())
        ),
        "extensions": dict(
            sorted(
                extension_counts.items()
            )
        ),
    }

    print(
        f"  Files discovered: "
        f"{result['file_count']}"
    )

    for ext, count in sorted(
        extension_counts.items()
    ):
        print(
            f"    {ext or '[no extension]'}: "
            f"{count}"
        )

    return result


# ============================================================
# STEP 2
# SUBPIPE IMAGE CHARACTERIZATION
# ============================================================

def characterize_subpipe_images():
    print("\n[2/7] Characterizing SubPipe SSS images...")

    files = collect_files(SUBPIPE_ROOT)

    image_files = []

    for path in files:

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if is_under_audit_output(
            path,
            SUBPIPE_ROOT
        ):
            continue

        image_files.append(path)

    image_files.sort()

    records = []

    representation_counts = Counter()

    dimension_counts = Counter()

    timestamp_counts = Counter()

    for idx, path in enumerate(
        image_files,
        start=1
    ):

        if idx % 100 == 0:
            print(
                f"  Images processed: "
                f"{idx}/{len(image_files)}"
            )

        info = image_info(path)

        info["relative_path"] = str(
            path.relative_to(
                SUBPIPE_ROOT
            )
        )

        info["timestamp"] = (
            extract_timestamp_from_name(
                path
            )
        )

        suffix = path.suffix.lower()

        if suffix == ".pbm":

            if info.get("width") == 5000:
                representation = "PBM_HF"

            elif info.get("width") == 2500:
                representation = "PBM_LF"

            else:
                representation = "PBM_OTHER"

        elif suffix == ".bpm":

            representation = "BPM"

        else:

            representation = suffix.upper()

        info["representation"] = representation

        records.append(info)

        representation_counts[
            representation
        ] += 1

        if info.get("readable"):

            dimension_counts[
                (
                    info.get("width"),
                    info.get("height")
                )
            ] += 1

        if info.get("timestamp") is not None:
            timestamp_counts[
                representation
            ] += 1

    readable = [
        r
        for r in records
        if r.get("readable")
    ]

    result = {
        "records": records,
        "summary": {
            "total_images": len(records),
            "readable_images": len(readable),
            "unreadable_images": (
                len(records) - len(readable)
            ),
            "representations": dict(
                representation_counts
            ),
            "dimensions": {
                f"{w}x{h}": count
                for (w, h), count
                in dimension_counts.items()
            },
            "mean_intensity": mean(
                [r.get("mean") for r in readable]
            ),
            "median_intensity": median(
                [r.get("mean") for r in readable]
            ),
            "median_image_std": median(
                [r.get("std") for r in readable]
            ),
        },
    }

    print(
        f"  Total SSS images: "
        f"{result['summary']['total_images']}"
    )

    print(
        f"  Readable: "
        f"{result['summary']['readable_images']}"
    )

    print(
        f"  Representations: "
        f"{dict(representation_counts)}"
    )

    print(
        f"  Dimensions: "
        f"{dict(dimension_counts)}"
    )

    return result


# ============================================================
# STEP 3
# COCO DISCOVERY + CHARACTERIZATION
# ============================================================

def discover_coco_jsons(root):
    candidates = []

    for path in root.rglob("*.json"):

        if is_under_audit_output(
            path,
            root
        ):
            continue

        try:

            with path.open(
                "r",
                encoding="utf-8"
            ) as f:
                obj = json.load(f)

        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        images = obj.get("images")
        annotations = obj.get("annotations")
        categories = obj.get("categories")

        if not isinstance(images, list):
            continue

        if not isinstance(annotations, list):
            continue

        if not isinstance(categories, list):
            continue

        if not images or not categories:
            continue

        if not all(
            isinstance(x, dict)
            for x in images
        ):
            continue

        if not all(
            isinstance(x, dict)
            for x in annotations
        ):
            continue

        if not all(
            isinstance(x, dict)
            for x in categories
        ):
            continue

        candidates.append(
            (
                path,
                obj
            )
        )

    return candidates


def characterize_coco():
    print(
        "\n[3/7] Discovering and characterizing "
        "SubPipe COCO annotations..."
    )

    candidates = discover_coco_jsons(
        SUBPIPE_ROOT
    )

    print(
        f"  COCO files found: "
        f"{len(candidates)}"
    )

    datasets = []

    for path, obj in candidates:

        print(
            f"  COCO candidate: {path}"
        )

        images = obj.get(
            "images",
            []
        )

        annotations = obj.get(
            "annotations",
            []
        )

        categories = obj.get(
            "categories",
            []
        )

        category_map = {
            cat.get("id"): cat.get("name")
            for cat in categories
        }

        image_map = {
            img.get("id"): img
            for img in images
        }

        anns_by_image = defaultdict(list)

        invalid_boxes = []

        for ann in annotations:

            image_id = ann.get(
                "image_id"
            )

            bbox = ann.get(
                "bbox"
            )

            if image_id not in image_map:
                continue

            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                invalid_boxes.append(ann)
                continue

            x, y, w, h = bbox

            if (
                not all(
                    isinstance(
                        v,
                        (int, float)
                    )
                    for v in bbox
                )
                or w <= 0
                or h <= 0
            ):
                invalid_boxes.append(ann)
                continue

            anns_by_image[
                image_id
            ].append(ann)

        annotated_image_count = sum(
            1
            for image_id in image_map
            if anns_by_image.get(
                image_id
            )
        )

        unannotated_image_count = (
            len(image_map)
            - annotated_image_count
        )

        bbox_widths = []
        bbox_heights = []
        bbox_areas = []

        annotation_records = []

        for ann in annotations:

            bbox = ann.get("bbox")

            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
            ):
                continue

            x, y, w, h = bbox

            if (
                not all(
                    isinstance(
                        v,
                        (int, float)
                    )
                    for v in bbox
                )
                or w <= 0
                or h <= 0
            ):
                continue

            bbox_widths.append(w)
            bbox_heights.append(h)
            bbox_areas.append(
                w * h
            )

            annotation_records.append(
                {
                    "image_id": ann.get(
                        "image_id"
                    ),
                    "category_id": ann.get(
                        "category_id"
                    ),
                    "category_name": category_map.get(
                        ann.get("category_id")
                    ),
                    "bbox": bbox,
                    "area": ann.get(
                        "area"
                    ),
                }
            )

        dataset = {
            "file": str(path),
            "image_count": len(images),
            "annotation_count": len(annotations),
            "category_count": len(categories),
            "categories": category_map,
            "annotated_images": (
                annotated_image_count
            ),
            "unannotated_images": (
                unannotated_image_count
            ),
            "invalid_bbox_count": len(
                invalid_boxes
            ),
            "bbox_statistics": {
                "width_median": median(
                    bbox_widths
                ),
                "height_median": median(
                    bbox_heights
                ),
                "area_median": median(
                    bbox_areas
                ),
                "width_mean": mean(
                    bbox_widths
                ),
                "height_mean": mean(
                    bbox_heights
                ),
            },
            "annotation_records": (
                annotation_records
            ),
        }

        datasets.append(dataset)

    result = {
        "datasets": datasets,
        "summary": {
            "files_found": len(
                datasets
            ),
            "total_images": sum(
                d["image_count"]
                for d in datasets
            ),
            "total_annotations": sum(
                d["annotation_count"]
                for d in datasets
            ),
        },
    }

    return result


# ============================================================
# STEP 4
# YOLO CHARACTERIZATION
# ============================================================

def characterize_yolo():
    print(
        "\n[4/7] Characterizing SubPipe YOLO labels..."
    )

    files = collect_files(
        SUBPIPE_ROOT
    )

    txt_files = []

    for path in files:

        if path.suffix.lower() != ".txt":
            continue

        if is_under_audit_output(
            path,
            SUBPIPE_ROOT
        ):
            continue

        txt_files.append(path)

    records = []

    class_counts = Counter()

    object_count = 0
    empty_count = 0
    invalid_count = 0

    widths = []
    heights = []
    areas = []

    for path in sorted(txt_files):

        try:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        except Exception as e:

            records.append(
                {
                    "path": str(path),
                    "error": str(e),
                }
            )

            continue

        stripped = text.strip()

        if not stripped:

            empty_count += 1

            records.append(
                {
                    "path": str(path),
                    "empty": True,
                    "objects": 0,
                }
            )

            continue

        lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip()
        ]

        file_objects = 0
        file_invalid = 0

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                file_invalid += 1
                continue

            try:

                class_id = int(
                    float(parts[0])
                )

                x_center = float(
                    parts[1]
                )

                y_center = float(
                    parts[2]
                )

                width = float(
                    parts[3]
                )

                height = float(
                    parts[4]
                )

            except Exception:

                file_invalid += 1
                continue

            # Reject obvious non-label metadata files.
            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):

                file_invalid += 1
                continue

            class_counts[
                class_id
            ] += 1

            object_count += 1
            file_objects += 1

            widths.append(width)
            heights.append(height)

            areas.append(
                width * height
            )

        if file_invalid > 0:
            invalid_count += file_invalid

        records.append(
            {
                "path": str(path),
                "empty": (
                    file_objects == 0
                ),
                "objects": file_objects,
                "invalid_records": file_invalid,
            }
        )

    result = {
        "records": records,
        "summary": {
            "label_files": len(txt_files),
            "object_count": object_count,
            "empty_files": empty_count,
            "invalid_records": invalid_count,
            "class_counts": dict(
                class_counts
            ),
            "bbox_statistics_normalized": {
                "width_median": median(
                    widths
                ),
                "height_median": median(
                    heights
                ),
                "area_median": median(
                    areas
                ),
            },
        },
    }

    print(
        f"  Label files: "
        f"{len(txt_files)}"
    )

    print(
        f"  Objects: "
        f"{object_count}"
    )

    print(
        f"  Empty files: "
        f"{empty_count}"
    )

    print(
        f"  Invalid records: "
        f"{invalid_count}"
    )

    print(
        f"  Classes: "
        f"{dict(class_counts)}"
    )

    return result


# ============================================================
# STEP 5
# AI4SHIPWRECKS CHARACTERIZATION
# ============================================================

def characterize_ai4shipwrecks():
    print(
        "\n[5/7] Characterizing "
        "AI4Shipwrecks masks..."
    )

    records = []

    splits = [
        "train",
        "test",
    ]

    for split in splits:

        image_dir = (
            AI4_ROOT
            / split
            / "images"
        )

        label_dir = (
            AI4_ROOT
            / split
            / "labels"
        )

        if not image_dir.exists():
            print(
                f"  WARNING: Missing "
                f"{image_dir}"
            )
            continue

        if not label_dir.exists():
            print(
                f"  WARNING: Missing "
                f"{label_dir}"
            )
            continue

        image_files = sorted(
            p
            for p in image_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                == ".png"
            )
        )

        label_files = sorted(
            p
            for p in label_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                == ".png"
            )
        )

        image_map = {
            p.stem: p
            for p in image_files
        }

        label_map = {
            p.stem: p
            for p in label_files
        }

        all_stems = sorted(
            set(image_map)
            | set(label_map)
        )

        print(
            f"\n  Split: {split}"
        )

        print(
            f"    Images discovered: "
            f"{len(image_files)}"
        )

        print(
            f"    Labels discovered: "
            f"{len(label_files)}"
        )

        print(
            f"    Unique stems: "
            f"{len(all_stems)}"
        )

        for stem in all_stems:

            image_path = image_map.get(
                stem
            )

            label_path = label_map.get(
                stem
            )

            record = {
                "dataset": (
                    "AI4Shipwrecks"
                ),
                "split": split,
                "stem": stem,
                "image_path": (
                    str(
                        image_path.relative_to(
                            AI4_ROOT
                        )
                    )
                    if image_path
                    else None
                ),
                "mask_path": (
                    str(
                        label_path.relative_to(
                            AI4_ROOT
                        )
                    )
                    if label_path
                    else None
                ),
                "image_exists": (
                    image_path is not None
                ),
                "mask_exists": (
                    label_path is not None
                ),
                "paired": (
                    image_path is not None
                    and label_path is not None
                ),
                "image_readable": False,
                "mask_readable": False,
            }

            # ------------------------------
            # IMAGE
            # ------------------------------

            if image_path is not None:

                try:

                    with Image.open(
                        image_path
                    ) as image:

                        record[
                            "image_readable"
                        ] = True

                        record[
                            "image_width"
                        ] = int(image.width)

                        record[
                            "image_height"
                        ] = int(image.height)

                        record[
                            "image_mode"
                        ] = image.mode

                except Exception as e:

                    record[
                        "image_error"
                    ] = str(e)

            # ------------------------------
            # MASK
            # ------------------------------

            if label_path is not None:

                try:

                    with Image.open(
                        label_path
                    ) as mask:

                        record[
                            "mask_readable"
                        ] = True

                        record[
                            "mask_width"
                        ] = int(mask.width)

                        record[
                            "mask_height"
                        ] = int(mask.height)

                        record[
                            "mask_mode"
                        ] = mask.mode

                        mask_array = np.asarray(
                            mask
                        )

                        unique_values = (
                            np.unique(
                                mask_array
                            )
                        )

                        record[
                            "mask_unique_values"
                        ] = [
                            int(v)
                            for v
                            in unique_values
                        ]

                        foreground = (
                            mask_array > 0
                        )

                        foreground_pixels = int(
                            np.count_nonzero(
                                foreground
                            )
                        )

                        total_pixels = int(
                            mask_array.size
                        )

                        record[
                            "mask_foreground_pixels"
                        ] = (
                            foreground_pixels
                        )

                        record[
                            "mask_total_pixels"
                        ] = total_pixels

                        record[
                            "mask_empty"
                        ] = (
                            foreground_pixels
                            == 0
                        )

                        record[
                            "mask_foreground_fraction"
                        ] = (
                            foreground_pixels
                            / total_pixels
                            if total_pixels
                            else 0.0
                        )

                        # ----------------------
                        # BOUNDING BOX
                        # ----------------------

                        if foreground_pixels > 0:

                            ys, xs = np.where(
                                foreground
                            )

                            x_min = int(
                                xs.min()
                            )

                            x_max = int(
                                xs.max()
                            )

                            y_min = int(
                                ys.min()
                            )

                            y_max = int(
                                ys.max()
                            )

                            bbox_width = (
                                x_max
                                - x_min
                                + 1
                            )

                            bbox_height = (
                                y_max
                                - y_min
                                + 1
                            )

                            record[
                                "bbox_x_min"
                            ] = x_min

                            record[
                                "bbox_y_min"
                            ] = y_min

                            record[
                                "bbox_x_max"
                            ] = x_max

                            record[
                                "bbox_y_max"
                            ] = y_max

                            record[
                                "bbox_width"
                            ] = bbox_width

                            record[
                                "bbox_height"
                            ] = bbox_height

                            record[
                                "bbox_area"
                            ] = (
                                bbox_width
                                * bbox_height
                            )

                            record[
                                "bbox_center_x"
                            ] = (
                                x_min
                                + x_max
                            ) / 2.0

                            record[
                                "bbox_center_y"
                            ] = (
                                y_min
                                + y_max
                            ) / 2.0

                            record[
                                "bbox_center_x_norm"
                            ] = (
                                (
                                    x_min
                                    + x_max
                                )
                                / 2.0
                                / mask.width
                            )

                            record[
                                "bbox_center_y_norm"
                            ] = (
                                (
                                    y_min
                                    + y_max
                                )
                                / 2.0
                                / mask.height
                            )

                        else:

                            record[
                                "bbox_x_min"
                            ] = None

                            record[
                                "bbox_y_min"
                            ] = None

                            record[
                                "bbox_x_max"
                            ] = None

                            record[
                                "bbox_y_max"
                            ] = None

                            record[
                                "bbox_width"
                            ] = None

                            record[
                                "bbox_height"
                            ] = None

                            record[
                                "bbox_area"
                            ] = None

                            record[
                                "bbox_center_x"
                            ] = None

                            record[
                                "bbox_center_y"
                            ] = None

                            record[
                                "bbox_center_x_norm"
                            ] = None

                            record[
                                "bbox_center_y_norm"
                            ] = None

                except Exception as e:

                    record[
                        "mask_error"
                    ] = str(e)

            records.append(record)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    paired = sum(
        r.get("paired", False)
        for r in records
    )

    readable_images = sum(
        r.get(
            "image_readable",
            False
        )
        for r in records
    )

    readable_masks = sum(
        r.get(
            "mask_readable",
            False
        )
        for r in records
    )

    empty_masks = sum(
        r.get(
            "mask_empty",
            False
        )
        for r in records
        if r.get("mask_readable")
    )

    nonempty_masks = (
        readable_masks
        - empty_masks
    )

    dimension_mismatches = sum(
        1
        for r in records
        if (
            r.get("image_readable")
            and r.get("mask_readable")
            and (
                r.get("image_width")
                != r.get("mask_width")
                or
                r.get("image_height")
                != r.get("mask_height")
            )
        )
    )

    train_records = [
        r
        for r in records
        if r.get("split") == "train"
    ]

    test_records = [
        r
        for r in records
        if r.get("split") == "test"
    ]

    result = {
        "image_records": records,
        "mask_records": records,
        "summary": {
            "total_records": len(records),
            "paired_records": paired,
            "readable_images": readable_images,
            "readable_masks": readable_masks,
            "empty_masks": empty_masks,
            "nonempty_masks": nonempty_masks,
            "dimension_mismatches": (
                dimension_mismatches
            ),
            "train": {
                "images": len(
                    train_records
                ),
                "empty_masks": sum(
                    r.get(
                        "mask_empty",
                        False
                    )
                    for r in train_records
                ),
                "nonempty_masks": sum(
                    not r.get(
                        "mask_empty",
                        True
                    )
                    for r in train_records
                    if r.get(
                        "mask_readable"
                    )
                ),
            },
            "test": {
                "images": len(
                    test_records
                ),
                "empty_masks": sum(
                    r.get(
                        "mask_empty",
                        False
                    )
                    for r in test_records
                ),
                "nonempty_masks": sum(
                    not r.get(
                        "mask_empty",
                        True
                    )
                    for r in test_records
                    if r.get(
                        "mask_readable"
                    )
                ),
            },
        },
    }

    print(
        "\n  AI4Shipwrecks "
        "characterization summary"
    )

    print(
        f"    Records: "
        f"{len(records)}"
    )

    print(
        f"    Complete pairs: "
        f"{paired}"
    )

    print(
        f"    Readable images: "
        f"{readable_images}"
    )

    print(
        f"    Readable masks: "
        f"{readable_masks}"
    )

    print(
        f"    Empty masks: "
        f"{empty_masks}"
    )

    print(
        f"    Non-empty masks: "
        f"{nonempty_masks}"
    )

    print(
        f"    Dimension mismatches: "
        f"{dimension_mismatches}"
    )

    return result


# ============================================================
# STEP 6
# SUBPIPE TIMESTAMP / ACQUISITION CHARACTERIZATION
# ============================================================

def characterize_subpipe_timestamps(
    image_result
):
    print(
        "\n[6/7] Characterizing "
        "SubPipe timestamp groups..."
    )

    records = image_result[
        "records"
    ]

    groups = defaultdict(list)

    for record in records:

        timestamp = record.get(
            "timestamp"
        )

        representation = record.get(
            "representation"
        )

        if timestamp is None:
            continue

        groups[
            representation
        ].append(timestamp)

    group_records = []

    for representation, timestamps in sorted(
        groups.items()
    ):

        timestamps = sorted(
            timestamps
        )

        if not timestamps:
            continue

        diffs = np.diff(
            timestamps
        )

        positive_diffs = [
            float(x)
            for x in diffs
            if x > 0
        ]

        group_records.append(
            {
                "representation": (
                    representation
                ),
                "count": len(
                    timestamps
                ),
                "start_timestamp": (
                    float(timestamps[0])
                ),
                "end_timestamp": (
                    float(timestamps[-1])
                ),
                "duration_seconds": (
                    float(
                        timestamps[-1]
                        - timestamps[0]
                    )
                ),
                "median_interval_seconds": (
                    median(
                        positive_diffs
                    )
                ),
                "p95_interval_seconds": (
                    percentile(
                        positive_diffs,
                        95
                    )
                ),
                "max_interval_seconds": (
                    max(
                        positive_diffs
                    )
                    if positive_diffs
                    else None
                ),
            }
        )

    # Candidate acquisition grouping:
    #
    # A new group is started when the timestamp gap
    # is substantially larger than the normal frame interval.
    #
    # This is deliberately described as a CANDIDATE
    # acquisition grouping, not a confirmed survey/site/pass ID.

    candidate_groups = []

    for representation, timestamps in sorted(
        groups.items()
    ):

        timestamps = sorted(
            timestamps
        )

        if len(timestamps) < 2:
            continue

        diffs = np.diff(
            timestamps
        )

        positive_diffs = [
            float(x)
            for x in diffs
            if x > 0
        ]

        baseline = (
            median(
                positive_diffs
            )
            if positive_diffs
            else None
        )

        if baseline is None:
            continue

        threshold = max(
            baseline * 10.0,
            5.0
        )

        current = [
            timestamps[0]
        ]

        for previous, current_timestamp in zip(
            timestamps[:-1],
            timestamps[1:]
        ):

            gap = (
                current_timestamp
                - previous
            )

            if gap > threshold:

                candidate_groups.append(
                    {
                        "representation": (
                            representation
                        ),
                        "candidate_group": len(
                            candidate_groups
                        ),
                        "count": len(
                            current
                        ),
                        "start_timestamp": (
                            float(
                                current[0]
                            )
                        ),
                        "end_timestamp": (
                            float(
                                current[-1]
                            )
                        ),
                        "duration_seconds": (
                            float(
                                current[-1]
                                - current[0]
                            )
                        ),
                    }
                )

                current = [
                    current_timestamp
                ]

            else:

                current.append(
                    current_timestamp
                )

        if current:

            candidate_groups.append(
                {
                    "representation": (
                        representation
                    ),
                    "candidate_group": len(
                        candidate_groups
                    ),
                    "count": len(
                        current
                    ),
                    "start_timestamp": (
                        float(
                            current[0]
                        )
                    ),
                    "end_timestamp": (
                        float(
                            current[-1]
                        )
                    ),
                    "duration_seconds": (
                        float(
                            current[-1]
                            - current[0]
                        )
                    ),
                }
            )

    print(
        f"  Timestamp groups characterized: "
        f"{len(group_records)}"
    )

    print(
        f"  Candidate acquisition groups: "
        f"{len(candidate_groups)}"
    )

    return {
        "representation_groups": (
            group_records
        ),
        "candidate_acquisition_groups": (
            candidate_groups
        ),
        "interpretation": (
            "Timestamp-derived groups are "
            "candidate acquisition groups only. "
            "They are not confirmed survey, site, "
            "or pass identities."
        ),
    }


# ============================================================
# STEP 7
# FINAL REPORT
# ============================================================

def build_final_report(
    inventory,
    images,
    coco,
    yolo,
    ai4,
    timestamps
):
    report = {
        "project": (
            "AI-Powered Underwater Marine "
            "Debris & Anomaly Detection System"
        ),
        "phase": "Phase 1",
        "scope": {
            "sonar_modality": (
                "Side-Scan Sonar (SSS) only"
            ),
            "FLS": "OUT OF SCOPE",
            "MBES": "OUT OF SCOPE",
            "training_performed": False,
            "dataset_modified": False,
        },
        "datasets": {
            "SubPipeMini2": {
                "root": str(
                    SUBPIPE_ROOT
                ),
                "inventory": inventory,
                "images": images[
                    "summary"
                ],
                "coco": coco,
                "yolo": yolo,
                "timestamps": timestamps,
            },
            "AI4Shipwrecks": {
                "root": str(
                    AI4_ROOT
                ),
                "characterization": ai4[
                    "summary"
                ],
            },
        },
        "preprocessing_status": {
            "status": (
                "CHARACTERIZATION COMPLETE"
            ),
            "training_authorized": False,
            "reason": (
                "This script characterizes "
                "dataset structure and annotation "
                "properties only. It does not "
                "authorize model training."
            ),
        },
        "important_interpretation_boundaries": [
            (
                "AI4Shipwrecks is a segmentation "
                "dataset with binary masks."
            ),
            (
                "SubPipeMini2 contains pipeline "
                "object-detection annotations."
            ),
            (
                "AI4Shipwrecks labels must not "
                "be fabricated into marine-debris "
                "or tire classes."
            ),
            (
                "SubPipe unannotated images should "
                "not automatically be interpreted "
                "as confirmed negatives without "
                "annotation-semantic verification."
            ),
            (
                "Timestamp-derived groups are "
                "candidate acquisition groups, "
                "not confirmed survey/site/pass IDs."
            ),
            (
                "AI4Shipwrecks extras/terrain is "
                "not included in train/test "
                "characterization."
            ),
        ],
    }

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "PHASE 1 PREPROCESSING CHARACTERIZATION"
    )
    print(
        "SSS-ONLY | NO TRAINING | NO DATA MODIFICATION"
    )
    print("=" * 72)

    print(
        f"\nSubPipe root:\n"
        f"  {SUBPIPE_ROOT}"
    )

    print(
        f"\nAI4Shipwrecks root:\n"
        f"  {AI4_ROOT}"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    inventory = inventory_subpipe()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    images = characterize_subpipe_images()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    coco = characterize_coco()

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    yolo = characterize_yolo()

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    ai4 = characterize_ai4shipwrecks()

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    timestamps = characterize_subpipe_timestamps(
        images
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    final_report = build_final_report(
        inventory,
        images,
        coco,
        yolo,
        ai4,
        timestamps
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    save_json(
        inventory,
        OUTPUT_DIR
        / "subpipe_inventory.json"
    )

    save_json(
        images,
        OUTPUT_DIR
        / "subpipe_image_characterization.json"
    )

    save_json(
        coco,
        OUTPUT_DIR
        / "subpipe_coco_characterization.json"
    )

    save_json(
        yolo,
        OUTPUT_DIR
        / "subpipe_yolo_characterization.json"
    )

    save_json(
        ai4,
        OUTPUT_DIR
        / "ai4shipwrecks_characterization.json"
    )

    save_json(
        timestamps,
        OUTPUT_DIR
        / "subpipe_timestamp_characterization.json"
    )

    save_json(
        final_report,
        OUTPUT_DIR
        / "preprocessing_characterization_report.json"
    )

    # --------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print(
        "CHARACTERIZATION COMPLETE"
    )
    print("=" * 72)

    print(
        "\nSubPipeMini2"
    )

    print(
        f"  SSS image representations: "
        f"{images['summary']['total_images']}"
    )

    print(
        f"  COCO files: "
        f"{coco['summary']['files_found']}"
    )

    print(
        f"  COCO images: "
        f"{coco['summary']['total_images']}"
    )

    print(
        f"  COCO annotations: "
        f"{coco['summary']['total_annotations']}"
    )

    print(
        f"  YOLO objects: "
        f"{yolo['summary']['object_count']}"
    )

    print(
        "\nAI4Shipwrecks"
    )

    print(
        f"  Records: "
        f"{ai4['summary']['total_records']}"
    )

    print(
        f"  Complete pairs: "
        f"{ai4['summary']['paired_records']}"
    )

    print(
        f"  Readable masks: "
        f"{ai4['summary']['readable_masks']}"
    )

    print(
        f"  Empty masks: "
        f"{ai4['summary']['empty_masks']}"
    )

    print(
        f"  Non-empty masks: "
        f"{ai4['summary']['nonempty_masks']}"
    )

    print(
        "\nTraining authorization:"
    )

    print(
        "  NOT AUTHORIZED"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()