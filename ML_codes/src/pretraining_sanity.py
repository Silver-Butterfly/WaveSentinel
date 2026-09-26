"""
Phase-1 Final Pre-Training Dataset Sanity Check
Project: AI-Powered Underwater Marine Debris & Anomaly Detection System

Purpose:
    Final gate before any model training. This script validates the already-built
    preprocessing output and leakage-safe split without modifying source data,
    preprocessing data, manifests, or labels.

Checks:
    1. Manifest integrity and expected split structure
    2. SubPipe train/test source separation
    3. SubPipe HF/LF representation distribution
    4. SubPipe positive/background-candidate distribution
    5. YOLO label syntax, class IDs, normalized geometry and bounds
    6. Partial/truncated bounding-box statistics
    7. Duplicate tile IDs / duplicate source references
    8. Exact image hashes within and across splits
    9. AI4 image/mask existence, dimensions, binary masks
   10. AI4 empty/non-empty mask distribution by split
   11. AI4 source separation
   12. Final class mapping and training-readiness gate

IMPORTANT:
    - This is an AUDIT ONLY.
    - No training is performed.
    - No files are changed.
    - SubPipe background_candidate_tile records are NOT treated as confirmed
      negatives because the preprocessing policy explicitly marks them as
      candidates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PREPROCESSED_ROOT = Path(r"K:\Debris model\preprocessing_output")
SPLIT_ROOT = PREPROCESSED_ROOT / "splits"

SPLIT_MANIFEST = SPLIT_ROOT / "split_manifest.json"
SPLIT_SUMMARY = SPLIT_ROOT / "split_summary.json"

# Strict current Phase-1 taxonomy.
EXPECTED_SUBPIPE_CLASS = 0
EXPECTED_SUBPIPE_CLASS_NAME = "Pipeline"

# AI4 is retained as a segmentation dataset in this audit.
AI4_ALLOWED_SPLITS = {"train", "test"}

# Hashing every PNG is intentionally enabled for a strong duplicate check.
# SHA-256 is used for exact byte-level equality.
HASH_CHUNK = 1024 * 1024


# ============================================================
# UTILITIES
# ============================================================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def add_error(errors, msg):
    errors.append(msg)
    print(f"  ERROR: {msg}")


def add_warning(warnings, msg):
    warnings.append(msg)
    print(f"  WARNING: {msg}")


def fmt_counter(counter):
    return ", ".join(f"{k}={v}" for k, v in sorted(counter.items(), key=lambda x: str(x[0])))


# ============================================================
# SUBPIPE LABEL CHECK
# ============================================================

def inspect_subpipe_label(label_path: Path):
    """
    Returns:
        object_count, invalid_count, classes, out_of_bounds_count
    """
    objects = 0
    invalid = 0
    classes = Counter()
    out_of_bounds = 0

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0, 0, classes, 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            invalid += 1
            continue

        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            invalid += 1
            continue

        cx, cy, w, h = vals

        if cls != EXPECTED_SUBPIPE_CLASS:
            invalid += 1
            continue

        if not np.all(np.isfinite(vals)):
            invalid += 1
            continue

        if w <= 0 or h <= 0:
            invalid += 1
            continue

        # YOLO normalized geometry should lie in [0,1].
        # Small floating tolerance is allowed for serialization noise.
        if any(v < -1e-6 or v > 1.000001 for v in vals):
            out_of_bounds += 1
            invalid += 1
            continue

        # Center must be inside the image and box must fit inside canvas.
        if cx - w / 2 < -1e-6 or cx + w / 2 > 1.000001:
            out_of_bounds += 1
            invalid += 1
            continue
        if cy - h / 2 < -1e-6 or cy + h / 2 > 1.000001:
            out_of_bounds += 1
            invalid += 1
            continue

        objects += 1
        classes[cls] += 1

    return objects, invalid, classes, out_of_bounds


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    print("=" * 72)
    print("PHASE-1 FINAL PRE-TRAINING DATASET SANITY CHECK")
    print("=" * 72)
    print(f"Preprocessed root: {PREPROCESSED_ROOT}")
    print(f"Split root:        {SPLIT_ROOT}")

    errors = []
    warnings = []

    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(f"Missing split manifest: {SPLIT_MANIFEST}")

    manifest = load_json(SPLIT_MANIFEST)

    # --------------------------------------------------------
    # 1. Manifest integrity
    # --------------------------------------------------------
    print("\n[1/10] Manifest integrity...")

    subpipe = manifest.get("subpipe", [])
    ai4 = manifest.get("ai4shipwrecks", [])

    if not subpipe:
        add_error(errors, "SubPipe manifest is empty.")
    if not ai4:
        add_error(errors, "AI4Shipwrecks manifest is empty.")

    subpipe_split = Counter(r.get("split") for r in subpipe)
    ai4_split = Counter(r.get("split") for r in ai4)

    print(f"  SubPipe records: {len(subpipe)}")
    print(f"  SubPipe splits:  {fmt_counter(subpipe_split)}")
    print(f"  AI4 records:     {len(ai4)}")
    print(f"  AI4 splits:      {fmt_counter(ai4_split)}")

    # --------------------------------------------------------
    # 2. SubPipe source/split/representation checks
    # --------------------------------------------------------
    print("\n[2/10] SubPipe split and representation checks...")

    source_by_split = defaultdict(set)
    rep_by_split = Counter()
    type_by_split = Counter()
    acquisition_by_split = defaultdict(set)

    for r in subpipe:
        split = r.get("split")
        source = r.get("source_image")
        rep = r.get("representation")
        record_type = r.get("record_type")
        group = r.get("acquisition_group")

        if split not in {"train", "test"}:
            add_error(errors, f"Invalid SubPipe split: {split}")
        if not source:
            add_error(errors, f"Missing SubPipe source_image for {r.get('tile_id')}")
        else:
            source_by_split[split].add(source)

        rep_by_split[(split, rep)] += 1
        type_by_split[(split, record_type)] += 1

        if group is not None:
            acquisition_by_split[split].add(group)

        out_img = Path(r["output_image"])
        out_lbl = Path(r["output_label"])

        if not out_img.exists():
            add_error(errors, f"Missing preprocessed image: {out_img}")
        if not out_lbl.exists():
            add_error(errors, f"Missing preprocessed label: {out_lbl}")

        if r.get("model_canvas_width") != 1024 or r.get("model_canvas_height") != 1024:
            add_error(errors, f"Unexpected model canvas for {r.get('tile_id')}")
        if r.get("tile_height") != 500:
            add_error(errors, f"Unexpected tile height for {r.get('tile_id')}")

    overlap = source_by_split["train"] & source_by_split["test"]
    if overlap:
        add_error(errors, f"SubPipe source leakage: {len(overlap)} source images in both splits.")

    group_overlap = acquisition_by_split["train"] & acquisition_by_split["test"]
    if group_overlap:
        add_error(errors, f"SubPipe acquisition-group leakage: {sorted(group_overlap)}")

    print(f"  Source images: train={len(source_by_split['train'])}, test={len(source_by_split['test'])}")
    print(f"  Acquisition groups: train={sorted(acquisition_by_split['train'])}, test={sorted(acquisition_by_split['test'])}")
    print("  Representation distribution:")
    for k, v in sorted(rep_by_split.items()):
        print(f"    {k[0]}/{k[1]}: {v}")
    print("  Record distribution:")
    for k, v in sorted(type_by_split.items()):
        print(f"    {k[0]}/{k[1]}: {v}")

    # --------------------------------------------------------
    # 3. SubPipe annotation / bbox checks
    # --------------------------------------------------------
    print("\n[3/10] SubPipe YOLO geometry and annotation checks...")

    tile_ids = set()
    label_hashes = defaultdict(list)

    total_objects = Counter()
    invalid_labels = Counter()
    out_of_bounds = Counter()
    positive_tiles = Counter()
    background_tiles = Counter()
    partial_boxes = Counter()
    partial_fraction_values = []

    for r in subpipe:
        split = r["split"]
        tile_id = r.get("tile_id")

        if tile_id in tile_ids:
            add_error(errors, f"Duplicate SubPipe tile_id: {tile_id}")
        tile_ids.add(tile_id)

        record_type = r.get("record_type")
        if record_type == "positive_tile":
            positive_tiles[split] += 1
        elif record_type == "background_candidate_tile":
            background_tiles[split] += 1
        else:
            add_error(errors, f"Unexpected SubPipe record_type: {record_type}")

        annotations = r.get("annotations", [])
        for ann in annotations:
            if ann.get("category_name") != EXPECTED_SUBPIPE_CLASS_NAME:
                add_error(
                    errors,
                    f"Unexpected SubPipe category {ann.get('category_name')} "
                    f"in {tile_id}"
                )

            if ann.get("partially_truncated", False):
                partial_boxes[split] += 1

            frac = ann.get("retained_bbox_fraction")
            if frac is not None:
                try:
                    partial_fraction_values.append(float(frac))
                except (TypeError, ValueError):
                    add_error(errors, f"Invalid retained_bbox_fraction in {tile_id}")

        label_path = Path(r["output_label"])

        # The preprocessing output may contain a dataset-level classes.txt file
        # alongside YOLO labels. It is metadata, not an annotation record.
        # Only manifest-referenced per-tile .txt files are inspected here.
        if label_path.name.lower() == "classes.txt":
            add_warning(
                warnings,
                f"Ignoring dataset metadata file as an annotation label: {label_path}"
            )
            continue

        objects, invalid, classes, oob = inspect_subpipe_label(label_path)

        total_objects[split] += objects
        invalid_labels[split] += invalid
        out_of_bounds[split] += oob

        for cls, count in classes.items():
            if cls != EXPECTED_SUBPIPE_CLASS:
                add_error(errors, f"Unexpected class {cls} in {label_path}")

        label_hashes[split].append((sha256_file(label_path), label_path))

    print(f"  Positive tiles: train={positive_tiles['train']}, test={positive_tiles['test']}")
    print(f"  Background candidates: train={background_tiles['train']}, test={background_tiles['test']}")
    print(f"  Valid YOLO objects: train={total_objects['train']}, test={total_objects['test']}")
    print(f"  Invalid label records: train={invalid_labels['train']}, test={invalid_labels['test']}")
    print(f"  Out-of-bounds boxes: train={out_of_bounds['train']}, test={out_of_bounds['test']}")
    print(f"  Partially truncated boxes: train={partial_boxes['train']}, test={partial_boxes['test']}")

    if invalid_labels["train"] or invalid_labels["test"]:
        add_error(errors, "At least one SubPipe YOLO label contains invalid geometry/syntax.")
    if out_of_bounds["train"] or out_of_bounds["test"]:
        add_error(errors, "At least one SubPipe YOLO box is out of bounds.")

    if partial_boxes["train"] or partial_boxes["test"]:
        add_warning(
            warnings,
            "Partial/truncated object boxes exist. They are explicitly marked in the manifest; "
            "do not silently discard or treat them as full-object annotations."
        )

    # Label duplicate hashes are not automatically errors because identical
    # labels can legitimately occur for different tiles. They are reported only.
    duplicate_label_hashes = sum(
        1 for split, pairs in label_hashes.items()
        if any(len(v) > 1 for _, v in defaultdict(list).items())
    )

    # --------------------------------------------------------
    # 4. SubPipe image exact duplicates
    # --------------------------------------------------------
    print("\n[4/10] SubPipe exact image duplicate checks...")

    image_hashes = defaultdict(list)

    for r in subpipe:
        split = r["split"]
        path = Path(r["output_image"])
        if path.exists():
            digest = sha256_file(path)
            image_hashes[digest].append((split, r.get("tile_id"), r.get("source_image")))

    within_duplicates = 0
    cross_duplicates = 0

    for digest, items in image_hashes.items():
        splits = {x[0] for x in items}
        if len(items) > 1:
            if len(splits) == 1:
                within_duplicates += len(items) - 1
            else:
                cross_duplicates += 1
                add_error(
                    errors,
                    f"Exact preprocessed image duplicate across splits: {digest[:16]}"
                )

    print(f"  Unique image hashes: {len(image_hashes)}")
    print(f"  Within-split duplicate instances: {within_duplicates}")
    print(f"  Cross-split duplicate hash groups: {cross_duplicates}")

    # --------------------------------------------------------
    # 5. AI4 structural checks
    # --------------------------------------------------------
    print("\n[5/10] AI4 image/mask structural checks...")

    ai4_sources = defaultdict(set)
    ai4_empty = Counter()
    ai4_nonempty = Counter()
    ai4_mask_values = defaultdict(set)
    ai4_hashes = defaultdict(list)

    for r in ai4:
        split = r.get("split")
        if split not in AI4_ALLOWED_SPLITS:
            add_error(errors, f"Invalid AI4 split: {split}")

        source = r.get("source_image")
        if not source:
            add_error(errors, f"Missing AI4 source image for {r.get('tile_id')}")
        else:
            ai4_sources[split].add(source)

        image_path = Path(r["output_image"])
        mask_path = Path(r["output_mask"])

        if not image_path.exists():
            add_error(errors, f"Missing AI4 image: {image_path}")
            continue
        if not mask_path.exists():
            add_error(errors, f"Missing AI4 mask: {mask_path}")
            continue

        with Image.open(image_path) as im:
            image = np.asarray(im.convert("L"))

        with Image.open(mask_path) as mm:
            mask = np.asarray(mm.convert("L"))

        if image.shape != mask.shape:
            add_error(errors, f"AI4 image/mask dimension mismatch: {r.get('tile_id')}")

        values = set(np.unique(mask).tolist())
        ai4_mask_values[split].update(values)

        bad_values = values - {0, 255}
        if bad_values:
            add_error(
                errors,
                f"AI4 mask has non-binary stored values {sorted(bad_values)} "
                f"in {r.get('tile_id')}"
            )

        if np.any(mask > 0):
            ai4_nonempty[split] += 1
        else:
            ai4_empty[split] += 1

        ai4_hashes[split].append((sha256_file(image_path), r.get("tile_id"), source))

    ai4_overlap = ai4_sources["train"] & ai4_sources["test"]
    if ai4_overlap:
        add_error(errors, f"AI4 source leakage: {len(ai4_overlap)} source images in both splits.")

    print(f"  Source images: train={len(ai4_sources['train'])}, test={len(ai4_sources['test'])}")
    print(f"  Empty-mask tiles: train={ai4_empty['train']}, test={ai4_empty['test']}")
    print(f"  Non-empty-mask tiles: train={ai4_nonempty['train']}, test={ai4_nonempty['test']}")
    print(f"  Mask values: train={sorted(ai4_mask_values['train'])}, test={sorted(ai4_mask_values['test'])}")

    # --------------------------------------------------------
    # 6. AI4 exact duplicate checks
    # --------------------------------------------------------
    print("\n[6/10] AI4 exact image duplicate checks...")

    ai4_image_hashes = defaultdict(list)
    for split, items in ai4_hashes.items():
        for digest, tile_id, source in items:
            ai4_image_hashes[digest].append((split, tile_id, source))

    ai4_cross_duplicates = 0
    ai4_within_duplicates = 0

    for digest, items in ai4_image_hashes.items():
        if len(items) > 1:
            splits = {x[0] for x in items}
            if len(splits) > 1:
                ai4_cross_duplicates += 1
                add_error(errors, f"Exact AI4 image duplicate across train/test: {digest[:16]}")
            else:
                ai4_within_duplicates += len(items) - 1

    print(f"  Unique AI4 image hashes: {len(ai4_image_hashes)}")
    print(f"  Within-split duplicate instances: {ai4_within_duplicates}")
    print(f"  Cross-split duplicate hash groups: {ai4_cross_duplicates}")

    # --------------------------------------------------------
    # 7. Dataset balance / usable sample counts
    # --------------------------------------------------------
    print("\n[7/10] Dataset balance and usable-sample summary...")

    subpipe_total = len(subpipe)
    subpipe_positive = sum(positive_tiles.values())
    subpipe_background = sum(background_tiles.values())

    print("  SubPipe:")
    print(f"    Total manifest tiles: {subpipe_total}")
    print(f"    Positive tiles:       {subpipe_positive} ({pct(subpipe_positive, subpipe_total):.2f}%)")
    print(f"    Background candidates:{subpipe_background} ({pct(subpipe_background, subpipe_total):.2f}%)")
    print("    NOTE: background candidates are NOT confirmed negatives.")

    ai4_total = len(ai4)
    ai4_nonempty_total = sum(ai4_nonempty.values())
    ai4_empty_total = sum(ai4_empty.values())

    print("  AI4Shipwrecks:")
    print(f"    Total tiles:          {ai4_total}")
    print(f"    Non-empty mask tiles: {ai4_nonempty_total} ({pct(ai4_nonempty_total, ai4_total):.2f}%)")
    print(f"    Empty-mask tiles:     {ai4_empty_total} ({pct(ai4_empty_total, ai4_total):.2f}%)")

    if len(source_by_split["test"]) < 30:
        add_warning(
            warnings,
            "SubPipe test contains fewer than 30 source images. "
            "Treat test performance as a held-out acquisition-window result, "
            "not as a statistically strong estimate of deployment performance."
        )

    if rep_by_split[("test", "PBM_HF")] == 0 or rep_by_split[("test", "PBM_LF")] == 0:
        add_warning(
            warnings,
            "SubPipe test does not contain both PBM_HF and PBM_LF representations."
        )

    # --------------------------------------------------------
    # 8. Cross-dataset taxonomy check
    # --------------------------------------------------------
    print("\n[8/10] Taxonomy / annotation semantics check...")

    print("  SubPipe:")
    print("    class 0 = Pipeline")
    print("    annotation type = YOLO bounding boxes derived from canonical COCO")
    print("  AI4Shipwrecks:")
    print("    annotation type = binary segmentation mask")
    print("    semantic content = shipwreck / other")
    print("  Unified current detector taxonomy:")
    print("    class 0 = Pipeline")
    print("    AI4 shipwreck masks are retained separately and are NOT fabricated into pipeline boxes.")

    # --------------------------------------------------------
    # 9. Training gate
    # --------------------------------------------------------
    print("\n[9/10] Training-readiness gate...")

    # Deliberately conservative: warnings do not automatically block training,
    # but errors do.
    gate_pass = len(errors) == 0

    if gate_pass:
        print("  PASS: No blocking integrity/leakage/geometry errors detected.")
    else:
        print(f"  BLOCKED: {len(errors)} error(s) detected.")

    # --------------------------------------------------------
    # 10. Save audit report
    # --------------------------------------------------------
    print("\n[10/10] Writing final sanity-check report...")

    report = {
        "project": "AI-Powered Underwater Marine Debris & Anomaly Detection System",
        "phase": "Phase 1",
        "scope": "SSS-only",
        "training_performed": False,
        "source_data_modified": False,
        "training_gate": "PASS" if gate_pass else "BLOCKED",
        "errors": errors,
        "warnings": warnings,
        "subpipe": {
            "manifest_tiles": subpipe_total,
            "source_images": {
                "train": len(source_by_split["train"]),
                "test": len(source_by_split["test"]),
            },
            "representations": {
                "train_PBM_HF": rep_by_split[("train", "PBM_HF")],
                "train_PBM_LF": rep_by_split[("train", "PBM_LF")],
                "test_PBM_HF": rep_by_split[("test", "PBM_HF")],
                "test_PBM_LF": rep_by_split[("test", "PBM_LF")],
            },
            "positive_tiles": dict(positive_tiles),
            "background_candidate_tiles": dict(background_tiles),
            "valid_yolo_objects": dict(total_objects),
            "invalid_label_records": dict(invalid_labels),
            "out_of_bounds_boxes": dict(out_of_bounds),
            "partially_truncated_boxes": dict(partial_boxes),
            "class_mapping": {"0": "Pipeline"},
            "background_candidates_are_confirmed_negatives": False,
        },
        "ai4shipwrecks": {
            "manifest_tiles": ai4_total,
            "source_images": {
                "train": len(ai4_sources["train"]),
                "test": len(ai4_sources["test"]),
            },
            "empty_mask_tiles": dict(ai4_empty),
            "nonempty_mask_tiles": dict(ai4_nonempty),
            "stored_mask_values": {
                "train": sorted(ai4_mask_values["train"]),
                "test": sorted(ai4_mask_values["test"]),
            },
            "annotation_type": "binary_segmentation_mask",
        },
        "duplicate_checks": {
            "subpipe_cross_split_exact_image_hash_groups": cross_duplicates,
            "subpipe_within_split_duplicate_instances": within_duplicates,
            "ai4_cross_split_exact_image_hash_groups": ai4_cross_duplicates,
            "ai4_within_split_duplicate_instances": ai4_within_duplicates,
        },
        "split_policy": {
            "subpipe": "long timestamp-derived acquisition window = train; short timestamp-derived acquisition window = test",
            "ai4": "preserve original train/test split",
            "random_image_level_split": False,
            "validation_split": None,
        },
    }

    out = SPLIT_ROOT / "final_pretraining_sanity_check.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  Report: {out}")

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    if gate_pass:
        print("TRAINING GATE: PASS")
        print("The Phase-1 dataset pipeline is ready for the next training-configuration step.")
        print("This does NOT mean model training was performed.")
    else:
        print("TRAINING GATE: BLOCKED")
        print("Fix the reported errors before configuring or running training.")

    if warnings:
        print(f"\nNon-blocking warnings: {len(warnings)}")
        for w in warnings:
            print(f"  - {w}")

    print("\nNO TRAINING WAS PERFORMED.")
    print("NO SOURCE DATASETS WERE MODIFIED.")
    print("=" * 72)


if __name__ == "__main__":
    main()
