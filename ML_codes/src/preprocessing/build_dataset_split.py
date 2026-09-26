"""
Phase-1 Leakage-Safe Split Manifest Builder
Project: AI-Powered Underwater Marine Debris & Anomaly Detection System

Purpose:
    Build the final data split manifest AFTER preprocessing.

LOCKED STRATEGY FOR CURRENT PHASE:
    SubPipeMini2:
        - PBM_HF + PBM_LF are kept together by acquisition window.
        - Two timestamp-defined acquisition windows are used:
              Window 0 = long acquisition window -> TRAIN
              Window 1 = short acquisition window -> TEST
        - No random image/tile split.
        - No validation split is fabricated from the same acquisition.
        - The timestamp windows remain documented as timestamp-derived
          acquisition windows, NOT confirmed survey/site/pass identities.

    AI4Shipwrecks:
        - Preserve the dataset's existing train/test split.
        - Do not mix AI4 train/test.

Important:
    This script does NOT modify source datasets.
    This script does NOT modify preprocessed images or labels.
    This script does NOT train a model.

Input:
    K:\\Debris model\\preprocessing_output

Output:
    K:\\Debris model\\preprocessing_output\\splits\\
        split_manifest.json
        split_summary.json
        acquisition_windows.json
        train/
            images/
            labels/
        test/
            images/
            labels/
        ai4shipwrecks/
            train/
            test/

The split folders contain COPIES of references via symlinks where supported;
if symlinks are unavailable, the script copies files. The original
preprocessing output remains untouched.

Run:
    python preprocessing\\build_split_manifest.py
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSED_ROOT = Path(r"K:\Debris model\preprocessing_output")
OUTPUT_ROOT = PREPROCESSED_ROOT / "splits"

SUBPIPE_MANIFEST = PREPROCESSED_ROOT / "subpipe" / "combined_manifest.json"
AI4_MANIFEST = PREPROCESSED_ROOT / "ai4shipwrecks" / "manifest.json"

# The audited timestamp structure contains:
#
# Long window:
#   HF: 1693569220.750 -> 1693570226.000
#   LF: 1693569219.799 -> 1693570229.000
#
# Short window:
#   HF: 1693573388.869 -> 1693573392.819
#   LF: 1693573348.860 -> 1693573392.819
#
# The gap is thousands of seconds, so a 30-minute threshold is a
# conservative separator for these two observed windows.
ACQUISITION_GAP_SECONDS = 1800.0

# Current operational split:
#   first/long acquisition window -> train
#   second/short acquisition window -> test
#
# This is intentionally NOT 70/15/15 because we only have two
# defensible timestamp-derived windows at present.
TRAIN_GROUP = 0
TEST_GROUP = 1

# AI4 existing dataset split is retained.
AI4_SPLITS = {"train", "test"}

# Do not include unannotated SubPipe source-image records as training negatives.
INCLUDE_UNANNOTATED_SUBPIPE = False

# We do not copy the generated dataset into another huge tree unless requested.
# Instead, the manifest is the authoritative split definition.
CREATE_SPLIT_FILE_COPIES = False


# ============================================================
# UTILITIES
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_timestamp(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_subpipe_timestamp(record):
    """
    Prefer explicit source_timestamp.
    Fall back to source_file_name stem.
    """
    ts = parse_timestamp(record.get("source_timestamp"))
    if ts is not None:
        return ts

    source_file_name = record.get("source_file_name", "")
    stem = Path(source_file_name).stem
    return parse_timestamp(stem)


def classify_timestamp_window(timestamp, boundaries):
    """
    boundaries = list of (start, end, group_id)
    """
    for start, end, group_id in boundaries:
        if start <= timestamp <= end:
            return group_id

    raise ValueError(
        f"Timestamp {timestamp} does not fall into any acquisition window."
    )


# ============================================================
# SUBPIPE ACQUISITION WINDOWS
# ============================================================

def collect_subpipe_source_timestamps(records):
    """
    Collect unique source-image timestamps from positive/background tile records.

    Unannotated source-image records are retained separately but are not
    part of the training/test tile manifest unless explicitly enabled.
    """
    by_representation = defaultdict(list)

    for r in records:
        if r.get("record_type") == "unannotated_source_image":
            continue

        ts = extract_subpipe_timestamp(r)
        if ts is None:
            raise ValueError(
                f"Could not extract source timestamp from record: {r}"
            )

        rep = r.get("representation")
        if rep not in {"PBM_HF", "PBM_LF"}:
            raise ValueError(
                f"Unexpected SubPipe representation: {rep}"
            )

        by_representation[rep].append(ts)

    unique = {
        rep: sorted(set(values))
        for rep, values in by_representation.items()
    }

    return unique


def infer_acquisition_windows(unique_timestamps):
    """
    Infer broad timestamp windows using a large gap.

    We deliberately use the broad windows visible in the audited data,
    not the earlier fragmented 16-group characterization.

    A window is a temporal connected component where consecutive source
    timestamps differ by <= ACQUISITION_GAP_SECONDS.

    HF and LF timestamps are merged before grouping so that the two
    representations of the same acquisition receive the same group.
    """
    all_timestamps = sorted(
        set(
            unique_timestamps.get("PBM_HF", [])
            + unique_timestamps.get("PBM_LF", [])
        )
    )

    if not all_timestamps:
        raise ValueError("No SubPipe timestamps found.")

    windows = []
    start = all_timestamps[0]
    previous = all_timestamps[0]

    for ts in all_timestamps[1:]:
        gap = ts - previous

        if gap > ACQUISITION_GAP_SECONDS:
            windows.append(
                {
                    "group_id": len(windows),
                    "start_timestamp": start,
                    "end_timestamp": previous,
                    "duration_seconds": previous - start,
                }
            )
            start = ts

        previous = ts

    windows.append(
        {
            "group_id": len(windows),
            "start_timestamp": start,
            "end_timestamp": previous,
            "duration_seconds": previous - start,
        }
    )

    return windows


# ============================================================
# SPLIT ASSIGNMENT
# ============================================================

def assign_subpipe_records(records, windows):
    """
    Assign every generated SubPipe tile to exactly one split.

    IMPORTANT:
        HF/LF source timestamps are assigned using the same temporal
        acquisition window. This prevents representation leakage.
    """
    boundaries = [
        (
            w["start_timestamp"],
            w["end_timestamp"],
            w["group_id"],
        )
        for w in windows
    ]

    assigned = []
    unannotated = []

    for r in records:
        record_type = r.get("record_type")

        if record_type == "unannotated_source_image":
            unannotated.append(
                {
                    **r,
                    "split": "excluded_unannotated",
                    "split_reason": (
                        "No canonical COCO annotation; excluded from "
                        "training/test manifest."
                    ),
                }
            )
            continue

        if record_type not in {
            "positive_tile",
            "background_candidate_tile",
        }:
            raise ValueError(
                f"Unexpected SubPipe record type: {record_type}"
            )

        ts = extract_subpipe_timestamp(r)
        group_id = classify_timestamp_window(ts, boundaries)

        if group_id == TRAIN_GROUP:
            split = "train"
        elif group_id == TEST_GROUP:
            split = "test"
        else:
            raise ValueError(
                f"Unexpected acquisition group {group_id}. "
                "Current strategy expects exactly two groups."
            )

        assigned.append(
            {
                **r,
                "acquisition_group": group_id,
                "acquisition_group_label": (
                    "long_timestamp_window"
                    if group_id == TRAIN_GROUP
                    else "short_timestamp_window"
                ),
                "split": split,
                "split_assignment_basis": "timestamp_acquisition_window",
            }
        )

    return assigned, unannotated


def validate_subpipe_split(records):
    errors = []

    seen_source_by_split = defaultdict(set)

    for r in records:
        split = r.get("split")
        source = r.get("source_image")

        if split not in {"train", "test"}:
            errors.append(
                f"Invalid split {split} for {r.get('tile_id')}"
            )

        if not source:
            errors.append(
                f"Missing source image for {r.get('tile_id')}"
            )
        else:
            seen_source_by_split[split].add(source)

        if not Path(r["output_image"]).exists():
            errors.append(
                f"Missing preprocessed image: {r['output_image']}"
            )

        if not Path(r["output_label"]).exists():
            errors.append(
                f"Missing preprocessed label: {r['output_label']}"
            )

    overlap = (
        seen_source_by_split["train"]
        & seen_source_by_split["test"]
    )

    if overlap:
        errors.append(
            "LEAKAGE: source images occur in both train and test. "
            f"Examples: {sorted(overlap)[:10]}"
        )

    return errors


# ============================================================
# AI4 SPLIT ASSIGNMENT
# ============================================================

def assign_ai4_records(records):
    assigned = []

    for r in records:
        split = r.get("split")

        if split not in AI4_SPLITS:
            raise ValueError(
                f"Unexpected AI4 split {split} in record {r}"
            )

        assigned.append(
            {
                **r,
                "split_assignment_basis": "AI4_original_dataset_split",
            }
        )

    return assigned


def validate_ai4_split(records):
    errors = []

    source_by_split = defaultdict(set)

    for r in records:
        split = r["split"]
        source = r["source_image"]

        source_by_split[split].add(source)

        if not Path(r["output_image"]).exists():
            errors.append(
                f"Missing AI4 preprocessed image: {r['output_image']}"
            )

        if not Path(r["output_mask"]).exists():
            errors.append(
                f"Missing AI4 preprocessed mask: {r['output_mask']}"
            )

    overlap = source_by_split["train"] & source_by_split["test"]

    if overlap:
        errors.append(
            "LEAKAGE: AI4 source images occur in both original train/test."
        )

    return errors


# ============================================================
# OPTIONAL SPLIT DIRECTORY COPIES
# ============================================================

def create_link_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)

    if dst.exists():
        return "exists"

    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
        return "copy"


def materialize_subpipe_split_files(records):
    """
    Optional convenience output.

    By default disabled because the manifest is sufficient and avoids
    duplicating thousands of files.
    """
    results = defaultdict(int)

    for r in records:
        split = r["split"]

        src_img = Path(r["output_image"])
        src_lbl = Path(r["output_label"])

        dst_img = OUTPUT_ROOT / "subpipe" / split / "images" / src_img.name
        dst_lbl = OUTPUT_ROOT / "subpipe" / split / "labels" / src_lbl.name

        results[create_link_or_copy(src_img, dst_img)] += 1
        results[create_link_or_copy(src_lbl, dst_lbl)] += 1

    return dict(results)


# ============================================================
# REPORTING
# ============================================================

def summarize_subpipe(records):
    summary = {
        "total_records": len(records),
        "by_split": defaultdict(int),
        "by_split_and_representation": defaultdict(int),
        "by_split_and_record_type": defaultdict(int),
        "by_split_and_source_image": defaultdict(set),
        "positive_annotation_records": defaultdict(int),
        "partial_bbox_records": defaultdict(int),
    }

    for r in records:
        split = r["split"]
        rep = r["representation"]
        record_type = r["record_type"]

        summary["by_split"][split] += 1
        summary["by_split_and_representation"][(split, rep)] += 1
        summary["by_split_and_record_type"][(split, record_type)] += 1
        summary["by_split_and_source_image"][split].add(
            r["source_image"]
        )

        if record_type == "positive_tile":
            summary["positive_annotation_records"][split] += 1

            if any(
                ann.get("partially_truncated", False)
                for ann in r.get("annotations", [])
            ):
                summary["partial_bbox_records"][split] += 1

    return {
        "total_records": summary["total_records"],
        "by_split": dict(summary["by_split"]),
        "by_split_and_representation": {
            f"{split}/{rep}": count
            for (split, rep), count
            in summary["by_split_and_representation"].items()
        },
        "by_split_and_record_type": {
            f"{split}/{rtype}": count
            for (split, rtype), count
            in summary["by_split_and_record_type"].items()
        },
        "unique_source_images_by_split": {
            split: len(values)
            for split, values
            in summary["by_split_and_source_image"].items()
        },
        "positive_tile_records_by_split": dict(
            summary["positive_annotation_records"]
        ),
        "tiles_with_partial_bbox_by_split": dict(
            summary["partial_bbox_records"]
        ),
    }


def summarize_ai4(records):
    result = {
        "total_tiles": len(records),
        "by_split": defaultdict(int),
        "by_split_and_type": defaultdict(int),
        "unique_source_images_by_split": defaultdict(set),
        "empty_mask_tiles_by_split": defaultdict(int),
        "nonempty_mask_tiles_by_split": defaultdict(int),
    }

    for r in records:
        split = r["split"]
        result["by_split"][split] += 1
        result["by_split_and_type"][(split, r["record_type"])] += 1
        result["unique_source_images_by_split"][split].add(
            r["source_image"]
        )

        if r.get("empty_mask"):
            result["empty_mask_tiles_by_split"][split] += 1
        else:
            result["nonempty_mask_tiles_by_split"][split] += 1

    return {
        "total_tiles": result["total_tiles"],
        "by_split": dict(result["by_split"]),
        "by_split_and_type": {
            f"{split}/{rtype}": count
            for (split, rtype), count
            in result["by_split_and_type"].items()
        },
        "unique_source_images_by_split": {
            split: len(values)
            for split, values
            in result["unique_source_images_by_split"].items()
        },
        "empty_mask_tiles_by_split": dict(
            result["empty_mask_tiles_by_split"]
        ),
        "nonempty_mask_tiles_by_split": dict(
            result["nonempty_mask_tiles_by_split"]
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("PHASE-1 LEAKAGE-SAFE SPLIT MANIFEST BUILDER")
    print("=" * 72)
    print(f"Started: {utc_now()}")
    print(f"Preprocessed root: {PREPROCESSED_ROOT}")
    print(f"Output root:       {OUTPUT_ROOT}")
    print()

    if not PREPROCESSED_ROOT.exists():
        raise FileNotFoundError(
            f"Preprocessed root does not exist: {PREPROCESSED_ROOT}"
        )

    if not SUBPIPE_MANIFEST.exists():
        raise FileNotFoundError(
            f"SubPipe manifest not found: {SUBPIPE_MANIFEST}"
        )

    if not AI4_MANIFEST.exists():
        raise FileNotFoundError(
            f"AI4 manifest not found: {AI4_MANIFEST}"
        )

    with SUBPIPE_MANIFEST.open("r", encoding="utf-8") as f:
        subpipe_records = json.load(f)

    with AI4_MANIFEST.open("r", encoding="utf-8") as f:
        ai4_records = json.load(f)

    # --------------------------------------------------------
    # 1. Infer broad acquisition windows
    # --------------------------------------------------------
    print("[1/5] Inferring SubPipe acquisition windows...")

    timestamps = collect_subpipe_source_timestamps(subpipe_records)
    windows = infer_acquisition_windows(timestamps)

    if len(windows) != 2:
        raise RuntimeError(
            f"Expected exactly 2 broad acquisition windows under the "
            f"current locked strategy, found {len(windows)}.\n"
            f"Windows: {windows}\n\n"
            f"STOPPING instead of inventing a split."
        )

    # Add representation counts.
    for window in windows:
        start = window["start_timestamp"]
        end = window["end_timestamp"]

        window["representation_source_timestamps"] = {}

        for rep, values in timestamps.items():
            inside = [
                x for x in values
                if start <= x <= end
            ]
            window["representation_source_timestamps"][rep] = {
                "unique_timestamps": len(inside),
                "start_timestamp": min(inside) if inside else None,
                "end_timestamp": max(inside) if inside else None,
            }

        window["assigned_split"] = (
            "train" if window["group_id"] == TRAIN_GROUP else "test"
        )

    save_json(
        {
            "method": "merged_HF_LF_timestamp_windows",
            "gap_threshold_seconds": ACQUISITION_GAP_SECONDS,
            "interpretation": (
                "Timestamp-derived acquisition windows only. "
                "They are not claimed to be confirmed survey/site/pass identities."
            ),
            "windows": windows,
        },
        OUTPUT_ROOT / "acquisition_windows.json",
    )

    print(f"  Broad acquisition windows: {len(windows)}")

    for w in windows:
        print(
            f"  Group {w['group_id']}: "
            f"{w['start_timestamp']:.3f} -> "
            f"{w['end_timestamp']:.3f} "
            f"({w['duration_seconds']:.3f}s) "
            f"-> {w['assigned_split']}"
        )

    # --------------------------------------------------------
    # 2. Assign SubPipe
    # --------------------------------------------------------
    print("\n[2/5] Assigning SubPipe tiles...")

    subpipe_assigned, unannotated = assign_subpipe_records(
        subpipe_records,
        windows,
    )

    subpipe_errors = validate_subpipe_split(subpipe_assigned)

    if subpipe_errors:
        raise RuntimeError(
            "SubPipe split validation failed:\n"
            + "\n".join(f"  - {x}" for x in subpipe_errors[:50])
        )

    print(f"  Assigned tiles:       {len(subpipe_assigned)}")
    print(f"  Excluded source imgs:  {len(unannotated)}")

    # --------------------------------------------------------
    # 3. Assign AI4
    # --------------------------------------------------------
    print("\n[3/5] Preserving AI4 original train/test split...")

    ai4_assigned = assign_ai4_records(ai4_records)

    ai4_errors = validate_ai4_split(ai4_assigned)

    if ai4_errors:
        raise RuntimeError(
            "AI4 split validation failed:\n"
            + "\n".join(f"  - {x}" for x in ai4_errors[:50])
        )

    print(f"  AI4 tiles: {len(ai4_assigned)}")

    # --------------------------------------------------------
    # 4. Combined manifest
    # --------------------------------------------------------
    print("\n[4/5] Writing split manifests...")

    ensure_dir(OUTPUT_ROOT)

    combined = {
        "project": "AI-Powered Underwater Marine Debris & Anomaly Detection System",
        "phase": "Phase 1",
        "scope": "SSS-only",
        "created_at": utc_now(),
        "training_performed": False,
        "source_datasets_modified": False,
        "strategy": {
            "subpipe": {
                "method": "acquisition_window_split",
                "train_group": TRAIN_GROUP,
                "test_group": TEST_GROUP,
                "validation_split": None,
                "random_image_split": False,
                "representation_leakage_prevented": True,
            },
            "ai4shipwrecks": {
                "method": "preserve_original_dataset_train_test",
                "random_image_split": False,
            },
        },
        "subpipe": subpipe_assigned,
        "subpipe_excluded_unannotated": unannotated,
        "ai4shipwrecks": ai4_assigned,
    }

    save_json(
        combined,
        OUTPUT_ROOT / "split_manifest.json",
    )

    # Separate manifests make downstream loading easier.
    save_json(
        subpipe_assigned,
        OUTPUT_ROOT / "subpipe_manifest.json",
    )

    save_json(
        ai4_assigned,
        OUTPUT_ROOT / "ai4shipwrecks_manifest.json",
    )

    # --------------------------------------------------------
    # 5. Summary and final checks
    # --------------------------------------------------------
    print("\n[5/5] Generating split summary...")

    subpipe_summary = summarize_subpipe(subpipe_assigned)
    ai4_summary = summarize_ai4(ai4_assigned)

    # Explicit leakage check at acquisition-group level.
    train_groups = {
        r["acquisition_group"]
        for r in subpipe_assigned
        if r["split"] == "train"
    }

    test_groups = {
        r["acquisition_group"]
        for r in subpipe_assigned
        if r["split"] == "test"
    }

    group_overlap = train_groups & test_groups

    final_errors = []

    if group_overlap:
        final_errors.append(
            f"Acquisition-group leakage detected: {sorted(group_overlap)}"
        )

    if train_groups != {TRAIN_GROUP}:
        final_errors.append(
            f"Unexpected SubPipe train groups: {sorted(train_groups)}"
        )

    if test_groups != {TEST_GROUP}:
        final_errors.append(
            f"Unexpected SubPipe test groups: {sorted(test_groups)}"
        )

    # Check representation consistency.
    source_rep_split = defaultdict(set)

    for r in subpipe_assigned:
        key = r["source_image"]
        source_rep_split[key].add(r["split"])

    leaked_sources = [
        source
        for source, splits in source_rep_split.items()
        if len(splits) > 1
    ]

    if leaked_sources:
        final_errors.append(
            "Source-image leakage across splits detected: "
            + str(leaked_sources[:10])
        )

    # Require that the test acquisition actually contains data.
    if not any(r["split"] == "test" for r in subpipe_assigned):
        final_errors.append("SubPipe test split is empty.")

    summary = {
        "created_at": utc_now(),
        "validation_passed": len(final_errors) == 0,
        "errors": final_errors,
        "subpipe": subpipe_summary,
        "ai4shipwrecks": ai4_summary,
        "subpipe_excluded_unannotated_source_images": len(unannotated),
        "acquisition_windows": windows,
        "policy": {
            "unannotated_subpipe_used_as_training_negative": False,
            "hf_lf_same_acquisition_window_same_split": True,
            "random_image_level_split": False,
            "ai4_original_train_test_preserved": True,
        },
    }

    save_json(
        summary,
        OUTPUT_ROOT / "split_summary.json",
    )

    # Optional materialized copies.
    if CREATE_SPLIT_FILE_COPIES:
        print("\nCreating split file copies/symlinks...")
        materialize_subpipe_split_files(subpipe_assigned)

    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("SPLIT MANIFEST BUILD COMPLETE")
    print("=" * 72)

    print("\nSubPipe")
    print(f"  Train tiles:       {subpipe_summary['by_split'].get('train', 0)}")
    print(f"  Test tiles:        {subpipe_summary['by_split'].get('test', 0)}")
    print(
        "  Train source imgs: "
        f"{subpipe_summary['unique_source_images_by_split'].get('train', 0)}"
    )
    print(
        "  Test source imgs:  "
        f"{subpipe_summary['unique_source_images_by_split'].get('test', 0)}"
    )

    print("\nSubPipe representation distribution")
    for key, value in sorted(
        subpipe_summary["by_split_and_representation"].items()
    ):
        print(f"  {key}: {value}")

    print("\nSubPipe positive tiles")
    for split, count in sorted(
        subpipe_summary["positive_tile_records_by_split"].items()
    ):
        print(f"  {split}: {count}")

    print("\nAI4Shipwrecks")
    for split, count in sorted(ai4_summary["by_split"].items()):
        print(f"  {split}: {count} tiles")

    print("\nLeakage checks")
    print(f"  Acquisition-group overlap: {len(group_overlap)}")
    print(f"  Source-image split overlap: {len(leaked_sources)}")

    print("\nValidation")
    print(f"  Passed: {len(final_errors) == 0}")
    print(f"  Errors: {len(final_errors)}")

    print(f"\nOutput: {OUTPUT_ROOT}")
    print(f"Manifest: {OUTPUT_ROOT / 'split_manifest.json'}")
    print(f"Summary:  {OUTPUT_ROOT / 'split_summary.json'}")

    if final_errors:
        print("\nFINAL ERRORS:")
        for e in final_errors:
            print(f"  - {e}")
        raise RuntimeError(
            f"Split manifest failed with {len(final_errors)} error(s)."
        )

    print("\nNO TRAINING WAS PERFORMED.")
    print("SOURCE DATASETS WERE NOT MODIFIED.")
    print("NO RANDOM IMAGE-LEVEL SPLIT WAS USED.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFATAL ERROR")
        traceback.print_exc()
        raise
