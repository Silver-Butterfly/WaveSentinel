"""
Phase-1 SSS Preprocessing Pipeline
Project: AI-Powered Underwater Marine Debris & Anomaly Detection System

Scope:
    - Side-Scan Sonar (SSS) only
    - SubPipeMini2: PBM_HF + PBM_LF, COCO is canonical annotation
    - AI4Shipwrecks: image + binary mask, terrain extras excluded
    - No training
    - Source datasets are NEVER modified

Operational preprocessing configuration:
    - SubPipe representations: PBM_HF + PBM_LF
    - Canonical SubPipe annotations: COCO
    - Tile size before model padding: 1024 x 500
    - Horizontal overlap: 20%
    - Model input canvas: 1024 x 1024
    - Preserve aspect ratio; pad vertically, do NOT stretch 500 px sonar height to 1024 px
    - Normalization: per-image percentile clipping (1st/99th percentile) -> [0, 1]
    - Output image format: PNG, 8-bit
    - SubPipe source images without COCO annotation are NOT used as training negatives.
      They are recorded separately as unannotated/quarantine records.
    - AI4 masks remain masks. Empty masks are retained and explicitly marked.

Run:
    python preprocessing\\preprocess_phase1.py
"""

from __future__ import annotations

import json
import math
import hashlib
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

SUBPIPE_ROOT = Path(r"K:\Debris model\SubPipeMiniSSS")
AI4_ROOT = Path(r"K:\Debris model\AI4Shipwrecks\AI4Shipwrecks")

OUTPUT_ROOT = Path(r"K:\Debris model\preprocessing_output")

# SubPipe
SUBPIPE_COCO_HF = (
    SUBPIPE_ROOT
    / "DATA"
    / "SSS_HF_images"
    / "COCO_Annotation"
    / "coco_format.json"
)

SUBPIPE_COCO_LF = (
    SUBPIPE_ROOT
    / "DATA"
    / "SSS_LF_images"
    / "COCO_Annotation"
    / "coco_format.json"
)

SUBPIPE_HF_DIR = SUBPIPE_ROOT / "DATA" / "SSS_HF_images"
SUBPIPE_LF_DIR = SUBPIPE_ROOT / "DATA" / "SSS_LF_images"

# AI4
AI4_TRAIN_IMAGES = AI4_ROOT / "train" / "images"
AI4_TRAIN_LABELS = AI4_ROOT / "train" / "labels"
AI4_TEST_IMAGES = AI4_ROOT / "test" / "images"
AI4_TEST_LABELS = AI4_ROOT / "test" / "labels"

# Preprocessing
TILE_WIDTH = 1024
TILE_HEIGHT = 500
OVERLAP = 0.20
STRIDE = int(round(TILE_WIDTH * (1.0 - OVERLAP)))

MODEL_SIZE = 1024

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

PNG_COMPRESSION = 3

# For SubPipe:
# Only images that have at least one canonical COCO annotation are eligible
# for generated training tiles. Images with no COCO annotation are quarantined
# in metadata and are NOT automatically converted to negatives.
USE_UNANNOTATED_SUBPIPE_AS_TRAINING_NEGATIVES = False


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Percentile-clipping normalization.
    Input may be uint8/uint16/etc.
    Output is uint8 [0,255].
    """
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got shape {arr.shape}")

    arr = arr.astype(np.float32)

    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixel values.")

    values = arr[finite]

    lo = float(np.percentile(values, LOW_PERCENTILE))
    hi = float(np.percentile(values, HIGH_PERCENTILE))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Invalid normalization percentiles.")

    if hi <= lo:
        # Constant/near-constant image.
        mn = float(values.min())
        mx = float(values.max())
        if mx <= mn:
            return np.zeros(arr.shape, dtype=np.uint8)
        lo, hi = mn, mx

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr * 255.0, 0, 255)

    return np.rint(arr).astype(np.uint8)


def pad_to_model_canvas(image: np.ndarray, target=MODEL_SIZE) -> np.ndarray:
    """
    Preserve original geometry.
    1024x500 tile becomes 1024x1024 with vertical padding.
    Padding is placed symmetrically.
    """
    h, w = image.shape

    if h > target or w > target:
        raise ValueError(
            f"Input {w}x{h} exceeds model canvas {target}x{target}."
        )

    canvas = np.zeros((target, target), dtype=np.uint8)

    top = (target - h) // 2
    left = (target - w) // 2

    canvas[top:top + h, left:left + w] = image

    return canvas


def tile_starts(full_width: int):
    """
    Return horizontal tile starts.

    Guarantees that the right edge of the original strip is covered.
    """
    if full_width <= TILE_WIDTH:
        return [0]

    starts = []
    x = 0

    while x + TILE_WIDTH < full_width:
        starts.append(x)
        x += STRIDE

    final_start = full_width - TILE_WIDTH

    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return sorted(set(starts))


def box_intersection(box, x0, y0, x1, y1):
    """
    box = [x_min, y_min, x_max, y_max]
    Returns clipped box in tile coordinates or None.
    """
    bx0, by0, bx1, by1 = box

    ix0 = max(bx0, x0)
    iy0 = max(by0, y0)
    ix1 = min(bx1, x1)
    iy1 = min(by1, y1)

    if ix1 <= ix0 or iy1 <= iy0:
        return None

    return [ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0]


def intersection_over_box_area(box, tile_box):
    """
    Fraction of original bbox area retained inside tile.
    """
    bx0, by0, bx1, by1 = box
    tx0, ty0, tx1, ty1 = tile_box

    bw = max(0.0, bx1 - bx0)
    bh = max(0.0, by1 - by0)

    if bw <= 0 or bh <= 0:
        return 0.0

    ix0 = max(bx0, tx0)
    iy0 = max(by0, ty0)
    ix1 = min(bx1, tx1)
    iy1 = min(by1, ty1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)

    return (iw * ih) / (bw * bh)


def tile_box_to_yolo(clipped_box, image_width=TILE_WIDTH, image_height=TILE_HEIGHT):
    """
    Convert tile-local xyxy bbox into normalized YOLO cx cy w h.
    """
    x0, y0, x1, y1 = clipped_box

    bw = x1 - x0
    bh = y1 - y0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    return [
        cx / image_width,
        cy / image_height,
        bw / image_width,
        bh / image_height,
    ]


def safe_json_dump(obj, path: Path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ============================================================
# COCO LOADING
# ============================================================

def load_coco(coco_path: Path):
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {int(x["id"]): x for x in coco.get("images", [])}
    categories = {
        int(x["id"]): x.get("name", f"class_{x['id']}")
        for x in coco.get("categories", [])
    }

    anns_by_image = defaultdict(list)

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        bbox = ann.get("bbox")

        if bbox is None or len(bbox) != 4:
            raise ValueError(
                f"Invalid COCO bbox in {coco_path}: annotation {ann.get('id')}"
            )

        x, y, w, h = map(float, bbox)

        if w <= 0 or h <= 0:
            raise ValueError(
                f"Non-positive COCO bbox in {coco_path}: annotation {ann.get('id')}"
            )

        category_id = int(ann["category_id"])

        anns_by_image[image_id].append(
            {
                "annotation_id": int(ann.get("id", -1)),
                "category_id": category_id,
                "category_name": categories.get(category_id, "UNKNOWN"),
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": [x, y, x + w, y + h],
                "area": float(ann.get("area", w * h)),
            }
        )

    return images, categories, anns_by_image


def locate_image_for_coco_record(coco_file: Path, file_name: str, representation: str):
    """
    Resolve COCO file_name robustly.

    First try paths relative to the directory containing the COCO file,
    then recursively search the corresponding representation directory
    by basename/stem.
    """
    candidates = []

    p = Path(file_name)

    if p.is_absolute():
        candidates.append(p)

    candidates.append(coco_file.parent / p)

    if representation == "PBM_HF":
        candidates.append(SUBPIPE_HF_DIR / p.name)
    elif representation == "PBM_LF":
        candidates.append(SUBPIPE_LF_DIR / p.name)

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    base_dir = SUBPIPE_HF_DIR if representation == "PBM_HF" else SUBPIPE_LF_DIR

    matches = list(base_dir.rglob(p.name))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple candidate images found for COCO filename {file_name}: "
            f"{matches[:10]}"
        )

    return None


# ============================================================
# SUBPIPE PREPROCESSING
# ============================================================

def preprocess_subpipe_representation(
    representation: str,
    coco_path: Path,
    output_dir: Path,
):
    print(f"\n[SubPipe] Processing {representation}")

    images, categories, anns_by_image = load_coco(coco_path)

    if len(categories) != 1:
        raise ValueError(
            f"{representation}: expected exactly one canonical category, "
            f"found {categories}"
        )

    category_names = set(categories.values())

    if category_names != {"Pipeline"}:
        raise ValueError(
            f"{representation}: expected canonical category Pipeline, "
            f"found {category_names}"
        )

    image_out = output_dir / "images"
    label_out = output_dir / "labels"
    ensure_dir(image_out)
    ensure_dir(label_out)

    records = []
    counters = defaultdict(int)

    for image_id, info in sorted(images.items()):
        file_name = info["file_name"]
        src = locate_image_for_coco_record(
            coco_path,
            file_name,
            representation,
        )

        if src is None:
            raise FileNotFoundError(
                f"Could not resolve COCO image {file_name} for {representation}"
            )

        try:
            with Image.open(src) as im:
                arr = np.asarray(im.convert("L"))

            h, w = arr.shape

            expected_h = 500
            expected_w = 5000 if representation == "PBM_HF" else 2500

            if h != expected_h or w != expected_w:
                raise ValueError(
                    f"Unexpected {representation} dimensions for {src}: "
                    f"{w}x{h}; expected {expected_w}x{expected_h}"
                )

            normalized = normalize_to_uint8(arr)

        except Exception as exc:
            raise RuntimeError(
                f"Failed reading/normalizing {src}: {exc}"
            ) from exc

        anns = anns_by_image.get(image_id, [])

        # Explicitly quarantine images without canonical annotations.
        if not anns:
            counters["unannotated_source_images"] += 1

            records.append(
                {
                    "record_type": "unannotated_source_image",
                    "dataset": "SubPipeMini2",
                    "representation": representation,
                    "source_image": str(src),
                    "source_image_id": image_id,
                    "source_file_name": file_name,
                    "source_width": w,
                    "source_height": h,
                    "annotation_count": 0,
                    "training_negative": False,
                    "reason": (
                        "No canonical COCO annotation supplied; "
                        "not automatically treated as confirmed negative."
                    ),
                }
            )

            if not USE_UNANNOTATED_SUBPIPE_AS_TRAINING_NEGATIVES:
                continue

        starts = tile_starts(w)

        for tile_index, x0 in enumerate(starts):
            x1 = min(x0 + TILE_WIDTH, w)

            # Rightmost tile can be shorter than 1024 if the source itself
            # is shorter, but HF/LF are both >= 1024.
            tile = normalized[:, x0:x1]

            # For standard SubPipe geometry, tile width is always 1024.
            if tile.shape[1] != TILE_WIDTH:
                # Defensive right-padding.
                padded = np.zeros((TILE_HEIGHT, TILE_WIDTH), dtype=np.uint8)
                padded[:, :tile.shape[1]] = tile
                tile = padded

            tile_box = [float(x0), 0.0, float(x1), float(h)]

            tile_annotations = []

            for ann in anns:
                original_box = ann["bbox_xyxy"]

                clipped = box_intersection(
                    original_box,
                    x0,
                    0,
                    x1,
                    h,
                )

                if clipped is None:
                    continue

                # Guard against floating-point boundary contacts that produce
                # an infinitesimally positive intersection. Such a box can
                # serialize to a zero-width/zero-height YOLO annotation.
                clipped_width = clipped[2] - clipped[0]
                clipped_height = clipped[3] - clipped[1]
                if clipped_width <= 1e-6 or clipped_height <= 1e-6:
                    continue

                retained = intersection_over_box_area(
                    original_box,
                    tile_box,
                )

                # Keep any geometrically intersecting object.
                # Partial boxes are explicitly marked so later analysis
                # can filter them if desired.
                yolo = tile_box_to_yolo(
                    clipped,
                    image_width=TILE_WIDTH,
                    image_height=TILE_HEIGHT,
                )

                tile_annotations.append(
                    {
                        "source_annotation_id": ann["annotation_id"],
                        "category_id": ann["category_id"],
                        "category_name": ann["category_name"],
                        "original_bbox_xyxy": original_box,
                        "tile_bbox_xyxy": clipped,
                        "retained_bbox_fraction": retained,
                        "partially_truncated": retained < 0.999999,
                        "yolo": yolo,
                    }
                )

            # We create all tiles from annotated source images.
            # Tiles without a supplied bbox are marked as background candidates,
            # not silently promoted to confirmed negatives.
            if tile_annotations:
                record_type = "positive_tile"
                counters["positive_tiles"] += 1
            else:
                record_type = "background_candidate_tile"
                counters["background_candidate_tiles"] += 1

            padded = pad_to_model_canvas(tile, MODEL_SIZE)

            tile_id = (
                f"SubPipeMini2_{representation}_"
                f"{Path(file_name).stem}_tile_{tile_index:04d}"
            )

            image_name = f"{tile_id}.png"
            label_name = f"{tile_id}.txt"

            out_image = image_out / image_name
            out_label = label_out / label_name

            Image.fromarray(padded, mode="L").save(
                out_image,
                format="PNG",
                compress_level=PNG_COMPRESSION,
            )

            # YOLO labels refer to the 1024x500 content region, which occupies
            # the center of the 1024x1024 padded canvas.
            # Therefore, shift normalized coordinates into the 1024x1024 canvas.
            label_lines = []

            vertical_pad = (MODEL_SIZE - TILE_HEIGHT) // 2

            for ta in tile_annotations:
                cx, cy, bw, bh = ta["yolo"]

                # Convert from 1024x500 coordinates to 1024x1024 canvas.
                cy_px = cy * TILE_HEIGHT + vertical_pad
                bh_px = bh * TILE_HEIGHT

                cy_canvas = cy_px / MODEL_SIZE
                bh_canvas = bh_px / MODEL_SIZE

                # Final serialization guard. Never write a degenerate YOLO box.
                if bw <= 1e-8 or bh_canvas <= 1e-8:
                    continue

                # Class 0 = Pipeline in the prepared unified detector format.
                label_lines.append(
                    f"0 {cx:.8f} {cy_canvas:.8f} "
                    f"{bw:.8f} {bh_canvas:.8f}"
                )

            with out_label.open("w", encoding="utf-8") as f:
                if label_lines:
                    f.write("\n".join(label_lines) + "\n")

            records.append(
                {
                    "record_type": record_type,
                    "dataset": "SubPipeMini2",
                    "representation": representation,
                    "source_image": str(src),
                    "source_image_id": image_id,
                    "source_file_name": file_name,
                    "source_timestamp": Path(file_name).stem,
                    "source_width": w,
                    "source_height": h,
                    "tile_id": tile_id,
                    "tile_index": tile_index,
                    "tile_x0": x0,
                    "tile_x1": x1,
                    "tile_width_before_padding": int(tile.shape[1]),
                    "tile_height": TILE_HEIGHT,
                    "model_canvas_width": MODEL_SIZE,
                    "model_canvas_height": MODEL_SIZE,
                    "vertical_padding_top": vertical_pad,
                    "normalization": {
                        "method": "percentile_clip",
                        "low_percentile": LOW_PERCENTILE,
                        "high_percentile": HIGH_PERCENTILE,
                    },
                    "annotation_count": len(tile_annotations),
                    "annotations": tile_annotations,
                    "training_negative": False,
                    "output_image": str(out_image),
                    "output_label": str(out_label),
                }
            )

    return {
        "representation": representation,
        "source_coco": str(coco_path),
        "source_image_count": len(images),
        "source_annotation_count": sum(len(v) for v in anns_by_image.values()),
        "counters": dict(counters),
        "records": records,
    }


# ============================================================
# AI4 PREPROCESSING
# ============================================================

def discover_ai4_pairs():
    pairs = []

    for split, image_dir, mask_dir in [
        ("train", AI4_TRAIN_IMAGES, AI4_TRAIN_LABELS),
        ("test", AI4_TEST_IMAGES, AI4_TEST_LABELS),
    ]:
        if not image_dir.exists():
            raise FileNotFoundError(f"AI4 image directory not found: {image_dir}")

        if not mask_dir.exists():
            raise FileNotFoundError(f"AI4 mask directory not found: {mask_dir}")

        images = {
            p.stem: p
            for p in image_dir.glob("*.png")
            if p.is_file()
        }

        masks = {
            p.stem: p
            for p in mask_dir.glob("*.png")
            if p.is_file()
        }

        for stem in sorted(set(images) | set(masks)):
            pairs.append(
                {
                    "split": split,
                    "stem": stem,
                    "image": images.get(stem),
                    "mask": masks.get(stem),
                }
            )

    return pairs


def preprocess_ai4(output_dir: Path):
    print("\n[AI4Shipwrecks] Processing")

    pairs = discover_ai4_pairs()

    image_out = output_dir / "images"
    mask_out = output_dir / "masks"

    ensure_dir(image_out)
    ensure_dir(mask_out)

    records = []
    counters = defaultdict(int)

    for pair in pairs:
        split = pair["split"]
        stem = pair["stem"]
        src_image = pair["image"]
        src_mask = pair["mask"]

        if src_image is None or src_mask is None:
            raise ValueError(
                f"AI4 incomplete pair: split={split}, stem={stem}, "
                f"image={src_image}, mask={src_mask}"
            )

        with Image.open(src_image) as im:
            image = np.asarray(im.convert("L"))

        with Image.open(src_mask) as mm:
            mask = np.asarray(mm.convert("L"))

        if image.shape != mask.shape:
            raise ValueError(
                f"AI4 dimension mismatch: {stem}: "
                f"image={image.shape}, mask={mask.shape}"
            )

        unique_mask = set(np.unique(mask).tolist())

        if not unique_mask.issubset({0, 1}):
            raise ValueError(
                f"AI4 mask {stem} contains values outside {{0,1}}: "
                f"{sorted(unique_mask)}"
            )

        image_h, image_w = image.shape

        normalized = normalize_to_uint8(image)

        # Convert mask to uint8 {0,255} for storage.
        mask_binary = (mask > 0).astype(np.uint8) * 255

        starts = tile_starts(image_w)

        for tile_index, x0 in enumerate(starts):
            x1 = min(x0 + TILE_WIDTH, image_w)

            image_tile = normalized[:, x0:x1]
            mask_tile = mask_binary[:, x0:x1]

            # For AI4, images may be taller than 500 px.
            # We therefore tile horizontally but preserve the complete vertical
            # dimension at this stage. Then split vertically into 500 px windows.
            vertical_starts = []

            if image_h <= TILE_HEIGHT:
                vertical_starts = [0]
            else:
                y = 0
                while y + TILE_HEIGHT < image_h:
                    vertical_starts.append(y)
                    y += STRIDE
                final_y = image_h - TILE_HEIGHT
                if not vertical_starts or vertical_starts[-1] != final_y:
                    vertical_starts.append(final_y)

            for y0 in sorted(set(vertical_starts)):
                y1 = min(y0 + TILE_HEIGHT, image_h)

                img_crop = image_tile[y0:y1, :]
                mask_crop = mask_tile[y0:y1, :]

                if img_crop.shape[0] < TILE_HEIGHT:
                    padded_img = np.zeros(
                        (TILE_HEIGHT, TILE_WIDTH),
                        dtype=np.uint8,
                    )
                    padded_mask = np.zeros(
                        (TILE_HEIGHT, TILE_WIDTH),
                        dtype=np.uint8,
                    )

                    padded_img[:img_crop.shape[0], :img_crop.shape[1]] = img_crop
                    padded_mask[:mask_crop.shape[0], :mask_crop.shape[1]] = mask_crop

                    img_crop = padded_img
                    mask_crop = padded_mask

                if img_crop.shape[1] < TILE_WIDTH:
                    padded_img = np.zeros(
                        (TILE_HEIGHT, TILE_WIDTH),
                        dtype=np.uint8,
                    )
                    padded_mask = np.zeros(
                        (TILE_HEIGHT, TILE_WIDTH),
                        dtype=np.uint8,
                    )

                    padded_img[:, :img_crop.shape[1]] = img_crop
                    padded_mask[:, :mask_crop.shape[1]] = mask_crop

                    img_crop = padded_img
                    mask_crop = padded_mask

                canvas_image = pad_to_model_canvas(img_crop, MODEL_SIZE)
                canvas_mask = pad_to_model_canvas(mask_crop, MODEL_SIZE)

                tile_id = (
                    f"AI4Shipwrecks_{split}_{stem}_"
                    f"x{tile_index:04d}_y{y0:04d}"
                )

                out_image = image_out / split / f"{tile_id}.png"
                out_mask = mask_out / split / f"{tile_id}.png"

                ensure_dir(out_image.parent)
                ensure_dir(out_mask.parent)

                Image.fromarray(canvas_image, mode="L").save(
                    out_image,
                    format="PNG",
                    compress_level=PNG_COMPRESSION,
                )

                Image.fromarray(canvas_mask, mode="L").save(
                    out_mask,
                    format="PNG",
                    compress_level=PNG_COMPRESSION,
                )

                foreground_pixels = int(np.count_nonzero(mask_crop))

                if foreground_pixels > 0:
                    counters["nonempty_tiles"] += 1
                    record_type = "shipwreck_tile"
                else:
                    counters["empty_tiles"] += 1
                    record_type = "empty_mask_tile"

                records.append(
                    {
                        "record_type": record_type,
                        "dataset": "AI4Shipwrecks",
                        "split": split,
                        "source_image": str(src_image),
                        "source_mask": str(src_mask),
                        "source_stem": stem,
                        "source_width": image_w,
                        "source_height": image_h,
                        "tile_x0": x0,
                        "tile_x1": x1,
                        "tile_y0": y0,
                        "tile_y1": y1,
                        "tile_width": TILE_WIDTH,
                        "tile_height": TILE_HEIGHT,
                        "model_canvas_width": MODEL_SIZE,
                        "model_canvas_height": MODEL_SIZE,
                        "foreground_pixels_in_tile": foreground_pixels,
                        "empty_mask": foreground_pixels == 0,
                        "normalization": {
                            "method": "percentile_clip",
                            "low_percentile": LOW_PERCENTILE,
                            "high_percentile": HIGH_PERCENTILE,
                        },
                        "output_image": str(out_image),
                        "output_mask": str(out_mask),
                    }
                )

    return {
        "source_pair_count": len(pairs),
        "counters": dict(counters),
        "records": records,
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_subpipe_records(records):
    errors = []

    for r in records:
        if r["record_type"] not in {
            "positive_tile",
            "background_candidate_tile",
            "unannotated_source_image",
        }:
            errors.append(f"Unknown SubPipe record type: {r['record_type']}")

        if r["record_type"] == "unannotated_source_image":
            continue

        if not Path(r["output_image"]).exists():
            errors.append(f"Missing output image: {r['output_image']}")

        if not Path(r["output_label"]).exists():
            errors.append(f"Missing output label: {r['output_label']}")

        for ann in r["annotations"]:
            yolo = ann["yolo"]
            if not all(0.0 <= v <= 1.0 for v in yolo):
                errors.append(
                    f"YOLO coordinate outside [0,1] in {r['tile_id']}: {yolo}"
                )

    return errors


def validate_ai4_records(records):
    errors = []

    for r in records:
        if not Path(r["output_image"]).exists():
            errors.append(f"Missing AI4 image: {r['output_image']}")

        if not Path(r["output_mask"]).exists():
            errors.append(f"Missing AI4 mask: {r['output_mask']}")

    return errors


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("PHASE-1 SSS PREPROCESSING PIPELINE")
    print("=" * 72)
    print(f"Started: {utc_now()}")
    print(f"SubPipe root: {SUBPIPE_ROOT}")
    print(f"AI4 root:     {AI4_ROOT}")
    print(f"Output root:  {OUTPUT_ROOT}")
    print()
    print("Configuration:")
    print(f"  Tile:              {TILE_WIDTH} x {TILE_HEIGHT}")
    print(f"  Overlap:           {OVERLAP * 100:.1f}%")
    print(f"  Horizontal stride: {STRIDE}px")
    print(f"  Model canvas:      {MODEL_SIZE} x {MODEL_SIZE}")
    print(f"  Normalization:     P{LOW_PERCENTILE}/P{HIGH_PERCENTILE} -> [0,255]")
    print("  SubPipe:            PBM_HF + PBM_LF")
    print("  SubPipe canonical:  COCO")
    print("  AI4 terrain:        EXCLUDED")
    print("  Training:           NOT PERFORMED")
    print()

    if not SUBPIPE_ROOT.exists():
        raise FileNotFoundError(f"SubPipe root does not exist: {SUBPIPE_ROOT}")

    if not AI4_ROOT.exists():
        raise FileNotFoundError(f"AI4 root does not exist: {AI4_ROOT}")

    ensure_dir(OUTPUT_ROOT)

    # Never put outputs inside either source dataset.
    if OUTPUT_ROOT.resolve().is_relative_to(SUBPIPE_ROOT.resolve()):
        raise RuntimeError("OUTPUT_ROOT must not be inside SubPipeMini2.")

    if OUTPUT_ROOT.resolve().is_relative_to(AI4_ROOT.resolve()):
        raise RuntimeError("OUTPUT_ROOT must not be inside AI4Shipwrecks.")

    report = {
        "project": "AI-Powered Underwater Marine Debris & Anomaly Detection System",
        "phase": "Phase 1",
        "scope": "SSS-only",
        "training_performed": False,
        "source_modification": False,
        "started_at": utc_now(),
        "configuration": {
            "subpipe_representations": ["PBM_HF", "PBM_LF"],
            "subpipe_canonical_annotation": "COCO",
            "tile_width": TILE_WIDTH,
            "tile_height": TILE_HEIGHT,
            "overlap": OVERLAP,
            "stride": STRIDE,
            "model_canvas": [MODEL_SIZE, MODEL_SIZE],
            "normalization": {
                "method": "percentile_clip",
                "low_percentile": LOW_PERCENTILE,
                "high_percentile": HIGH_PERCENTILE,
            },
            "use_unannotated_subpipe_as_training_negatives":
                USE_UNANNOTATED_SUBPIPE_AS_TRAINING_NEGATIVES,
            "ai4_masks_preserved": True,
            "ai4_extras_terrain_included": False,
        },
        "subpipe": {},
        "ai4shipwrecks": {},
        "validation": {},
    }

    # --------------------------------------------------------
    # SubPipe HF
    # --------------------------------------------------------
    print("[1/4] Preprocessing SubPipe PBM_HF...")
    hf = preprocess_subpipe_representation(
        "PBM_HF",
        SUBPIPE_COCO_HF,
        OUTPUT_ROOT / "subpipe" / "PBM_HF",
    )
    report["subpipe"]["PBM_HF"] = {
        k: v for k, v in hf.items() if k != "records"
    }

    safe_json_dump(
        hf["records"],
        OUTPUT_ROOT / "subpipe" / "PBM_HF" / "manifest.json",
    )

    # --------------------------------------------------------
    # SubPipe LF
    # --------------------------------------------------------
    print("[2/4] Preprocessing SubPipe PBM_LF...")
    lf = preprocess_subpipe_representation(
        "PBM_LF",
        SUBPIPE_COCO_LF,
        OUTPUT_ROOT / "subpipe" / "PBM_LF",
    )
    report["subpipe"]["PBM_LF"] = {
        k: v for k, v in lf.items() if k != "records"
    }

    safe_json_dump(
        lf["records"],
        OUTPUT_ROOT / "subpipe" / "PBM_LF" / "manifest.json",
    )

    # --------------------------------------------------------
    # AI4
    # --------------------------------------------------------
    print("[3/4] Preprocessing AI4Shipwrecks...")
    ai4 = preprocess_ai4(
        OUTPUT_ROOT / "ai4shipwrecks"
    )
    report["ai4shipwrecks"] = {
        k: v for k, v in ai4.items() if k != "records"
    }

    safe_json_dump(
        ai4["records"],
        OUTPUT_ROOT / "ai4shipwrecks" / "manifest.json",
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    print("[4/4] Validating generated outputs...")

    hf_errors = validate_subpipe_records(hf["records"])
    lf_errors = validate_subpipe_records(lf["records"])
    ai4_errors = validate_ai4_records(ai4["records"])

    errors = hf_errors + lf_errors + ai4_errors

    report["validation"] = {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors[:200],
    }

    report["finished_at"] = utc_now()

    safe_json_dump(
        report,
        OUTPUT_ROOT / "preprocessing_report.json",
    )

    # Combined manifest
    combined_records = (
        hf["records"]
        + lf["records"]
    )

    safe_json_dump(
        combined_records,
        OUTPUT_ROOT / "subpipe" / "combined_manifest.json",
    )

    print()
    print("=" * 72)
    print("PREPROCESSING COMPLETE")
    print("=" * 72)

    print("\nSubPipe PBM_HF")
    print(f"  Positive tiles:          {hf['counters'].get('positive_tiles', 0)}")
    print(f"  Background candidates:   {hf['counters'].get('background_candidate_tiles', 0)}")
    print(f"  Unannotated source imgs: {hf['counters'].get('unannotated_source_images', 0)}")

    print("\nSubPipe PBM_LF")
    print(f"  Positive tiles:          {lf['counters'].get('positive_tiles', 0)}")
    print(f"  Background candidates:   {lf['counters'].get('background_candidate_tiles', 0)}")
    print(f"  Unannotated source imgs: {lf['counters'].get('unannotated_source_images', 0)}")

    print("\nAI4Shipwrecks")
    print(f"  Non-empty tiles:         {ai4['counters'].get('nonempty_tiles', 0)}")
    print(f"  Empty-mask tiles:        {ai4['counters'].get('empty_tiles', 0)}")

    print("\nValidation")
    print(f"  Valid:                   {len(errors) == 0}")
    print(f"  Errors:                  {len(errors)}")

    print(f"\nOutput: {OUTPUT_ROOT}")
    print(f"Report: {OUTPUT_ROOT / 'preprocessing_report.json'}")

    if errors:
        print("\nFIRST VALIDATION ERRORS:")
        for e in errors[:20]:
            print(f"  - {e}")

        raise RuntimeError(
            f"Preprocessing completed with {len(errors)} validation errors."
        )

    print("\nNO TRAINING WAS PERFORMED.")
    print("SOURCE DATASETS WERE NOT MODIFIED.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nFATAL ERROR")
        traceback.print_exc()
        raise
