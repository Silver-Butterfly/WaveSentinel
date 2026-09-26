from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASSES = {0: "pipe", 1: "shipwreck", 2: "mine", 3: "ghost_net"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_candidates(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_yolo(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    rows = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 5:
            raise ValueError(f"Malformed YOLO label: {label_path}:{line_no}")
        try:
            c, x, y, w, h = map(float, p)
        except ValueError as e:
            raise ValueError(f"Non-numeric YOLO label: {label_path}:{line_no}") from e
        if c != int(c) or int(c) not in CLASSES:
            raise ValueError(f"Invalid class in {label_path}:{line_no}: {c}")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid normalized box in {label_path}:{line_no}")
        rows.append([float(int(c)), x, y, w, h])
    return rows


def save_yolo(label_path: Path, rows: list[list[float]]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for c, x, y, w, h in rows:
            f.write(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(row, W, H):
    c, xc, yc, w, h = row
    return int(c), np.array(
        [
            (xc - w / 2) * W,
            (yc - h / 2) * H,
            (xc + w / 2) * W,
            (yc + h / 2) * H,
        ],
        dtype=np.float32,
    )


def xyxy_to_yolo(c, box, W, H):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(W - 1), x1))
    y1 = max(0.0, min(float(H - 1), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [c, ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H]


def transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    pts = np.concatenate([points, ones], axis=1)
    return pts @ M.T


def apply_motion_transform(img, rows, rng):
    H, W = img.shape[:2]
    roll_deg = float(rng.uniform(-3.0, 3.0))
    pitch_shear = float(rng.uniform(-0.035, 0.035))
    heave_scale = float(rng.uniform(0.92, 1.08))
    lateral_shift = float(rng.uniform(-0.025, 0.025)) * W
    vertical_shift = float(rng.uniform(-0.02, 0.02)) * H
    gain = float(rng.uniform(0.90, 1.10))

    center = (W / 2.0, H / 2.0)
    base = cv2.getRotationMatrix2D(center, roll_deg, 1.0).astype(np.float32)
    base[0, 2] += lateral_shift
    base[1, 2] += vertical_shift

    shear = np.array(
        [
            [1.0, pitch_shear, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, heave_scale, H * (1.0 - heave_scale) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    full_M = (scale @ shear @ np.vstack([base, [0, 0, 1]])).astype(np.float32)

    warped = cv2.warpAffine(
        img,
        full_M[:2],
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    out_rows = []
    for row in rows:
        c, box = yolo_to_xyxy(row, W, H)
        x1, y1, x2, y2 = box
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        t = transform_points(pts, full_M[:2])
        converted = xyxy_to_yolo(c, [t[:, 0].min(), t[:, 1].min(), t[:, 0].max(), t[:, 1].max()], W, H)
        if converted is not None:
            out_rows.append(converted)

    params = {
        "roll_deg": roll_deg,
        "pitch_shear": pitch_shear,
        "heave_scale_proxy": heave_scale,
        "lateral_shift_px": lateral_shift,
        "vertical_shift_px": vertical_shift,
        "gain": gain,
    }
    return warped, out_rows, params


def copy_split(src_root: Path, dst_root: Path, split: str):
    s_img = src_root / split / "images"
    s_lbl = src_root / split / "labels"
    d_img = dst_root / split / "images"
    d_lbl = dst_root / split / "labels"
    d_img.mkdir(parents=True, exist_ok=True)
    d_lbl.mkdir(parents=True, exist_ok=True)

    records = []
    for p in image_candidates(s_img):
        shutil.copy2(p, d_img / p.name)
        lp = s_lbl / f"{p.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, d_lbl / lp.name)
        else:
            (d_lbl / f"{p.stem}.txt").write_text("", encoding="utf-8")
        records.append({"name": p.name, "sha256": sha256(p)})
    return records


def split_hashes(root: Path, split: str, include_labels=False):
    result = {}
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    for p in image_candidates(img_dir):
        item = {"image_sha256": sha256(p)}
        if include_labels:
            lp = lbl_dir / f"{p.stem}.txt"
            item["label_sha256"] = sha256(lp) if lp.exists() else hashlib.sha256(b"").hexdigest()
        result[p.name] = item
    return result


def validate_dataset(root: Path):
    bad = []
    counts = {}
    hashes = {}
    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        counts[split] = len(image_candidates(img_dir))
        hashes[split] = {}
        for p in image_candidates(img_dir):
            lp = lbl_dir / f"{p.stem}.txt"
            try:
                load_yolo(lp)
            except ValueError as e:
                bad.append(str(e))
                continue
            hashes[split][p.name] = sha256(p)
            if not lp.exists():
                bad.append(f"Missing label file: {lp}")

    split_names = ("train", "val", "test")
    overlap = {}
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            common = set(hashes[a].values()) & set(hashes[b].values())
            overlap[f"{a}_vs_{b}"] = len(common)
            if common:
                bad.append(f"Cross-split image hash overlap {a} vs {b}: {len(common)}")

    return {"counts": counts, "cross_split_image_hash_overlap": overlap, "errors": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--clean_source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hard_negative_manifest", required=True)
    ap.add_argument("--motion_variants", type=int, default=188)
    ap.add_argument("--hard_negative_extra_copies", type=int, default=1)
    ap.add_argument("--seed", type=int, default=26057)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    source = Path(args.source)
    clean_source = Path(args.clean_source)
    out = Path(args.output)
    manifest_path = Path(args.hard_negative_manifest)

    if out.exists():
        raise SystemExit(f"Output already exists: {out}. Refusing to overwrite.")
    for p in (source, clean_source, manifest_path):
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hard_names = manifest.get("frozen_hard_negative_pool")
    if not isinstance(hard_names, list) or len(hard_names) != 15 or len(set(hard_names)) != 15:
        raise SystemExit("Phase-C manifest must contain exactly 15 unique hard-negative source names.")
    if manifest.get("status") != "COMPLETE" or manifest.get("summary", {}).get("false_positives") != 15:
        raise SystemExit("Phase-C manifest is not the expected completed 15-FP freeze.")

    out.mkdir(parents=True)
    copy_records = {split: copy_split(source, out, split) for split in ("train", "val", "test")}

    clean_train_img = clean_source / "train" / "images"
    clean_train_lbl = clean_source / "train" / "labels"
    base_train_img = out / "train" / "images"
    base_train_lbl = out / "train" / "labels"

    for source_name in hard_names:
        clean_img = clean_train_img / source_name
        base_img = base_train_img / source_name
        clean_lbl = clean_train_lbl / f"{Path(source_name).stem}.txt"
        base_lbl = base_train_lbl / f"{Path(source_name).stem}.txt"
        for p in (clean_img, base_img, clean_lbl, base_lbl):
            if not p.exists():
                raise SystemExit(f"Hard-negative integrity failure: missing {p}")
        if load_yolo(clean_lbl) or load_yolo(base_lbl):
            raise SystemExit(f"Hard-negative is not empty-label: {source_name}")

    records = []
    for source_name in hard_names:
        src = clean_train_img / source_name
        for k in range(1, args.hard_negative_extra_copies + 1):
            dst_name = f"{Path(source_name).stem}__HN{k}{src.suffix.lower()}"
            shutil.copy2(src, base_train_img / dst_name)
            (base_train_lbl / f"{Path(dst_name).stem}.txt").write_text("", encoding="utf-8")
            records.append(
                {
                    "variant": "hard_negative",
                    "source": source_name,
                    "output": dst_name,
                    "reason": "verified clean-source false positive",
                }
            )

    all_train = image_candidates(clean_train_img)
    hard_set = set(hard_names)
    motion_pool = [p for p in all_train if p.name not in hard_set]
    if args.motion_variants > len(motion_pool):
        raise SystemExit("Not enough clean train images for requested motion variants.")
    selected = rng.sample(motion_pool, args.motion_variants)

    for idx, src in enumerate(selected, 1):
        lbl = clean_train_lbl / f"{src.stem}.txt"
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"Unreadable image: {src}")
        if img.ndim not in (2, 3):
            raise SystemExit(f"Unsupported image shape: {src} -> {img.shape}")
        rows = load_yolo(lbl)
        warped, out_rows, params = apply_motion_transform(img, rows, rng)
        out_name = f"{src.stem}__MOTION{idx:04d}.png"
        if not cv2.imwrite(str(base_train_img / out_name), warped):
            raise SystemExit(f"Failed to write motion variant: {out_name}")
        save_yolo(base_train_lbl / f"{Path(out_name).stem}.txt", out_rows)
        records.append(
            {
                "variant": "motion_pose_robustness",
                "source": src.name,
                "output": out_name,
                "parameters": params,
            }
        )

    root_text = str(out).replace("\\", "/")
    yaml_text = (
        f"path: {root_text}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: pipe\n"
        "  1: shipwreck\n"
        "  2: mine\n"
        "  3: ghost_net\n"
    )
    (out / "drishti.yaml").write_text(yaml_text, encoding="utf-8")

    integrity = {}
    for split in ("val", "test"):
        src_hashes = split_hashes(source, split, include_labels=True)
        dst_hashes = split_hashes(out, split, include_labels=True)
        integrity[split] = {
            "identical_images_and_labels": src_hashes == dst_hashes,
            "source_count": len(src_hashes),
            "output_count": len(dst_hashes),
        }
        if src_hashes != dst_hashes:
            raise SystemExit(f"{split} integrity check failed.")

    validation = validate_dataset(out)
    if validation["errors"]:
        raise SystemExit("Dataset integrity validation failed:\n" + "\n".join(validation["errors"][:50]))

    hard_copy_checks = []
    for r in records:
        if r["variant"] != "hard_negative":
            continue
        p = out / "train" / "images" / r["output"]
        lp = out / "train" / "labels" / f"{p.stem}.txt"
        hard_copy_checks.append(
            {"output": r["output"], "sha256": sha256(p), "empty_label": not load_yolo(lp)}
        )

    counts = validation["counts"]
    summary = {
        "phase": "D",
        "dataset": str(out),
        "base_dataset": str(source),
        "clean_source": str(clean_source),
        "hard_negative_manifest": str(manifest_path),
        "hard_negative_pool_size": len(hard_names),
        "hard_negative_extra_copies_per_image": args.hard_negative_extra_copies,
        "motion_variants": args.motion_variants,
        "seed": args.seed,
        "train_count": counts["train"],
        "val_count": counts["val"],
        "test_count": counts["test"],
        "integrity": integrity,
        "validation_gates": validation,
        "hard_negative_copy_checks": hard_copy_checks,
        "model_relevant_features": {
            "speckle": "inherited from Robust-V2",
            "gain_contrast": "inherited from Robust-V2",
            "dropout": "inherited from Robust-V2",
            "resolution_variation": "inherited from Robust-V2",
            "acoustic_shadow_variation": "inherited from Robust-V2",
            "hard_negative_mining": "15 verified false positives oversampled 2x total presence",
            "heave_pitch_roll_training_robustness": "explicit offline motion/pose robustness variants",
            "real_motion_correction": "not implemented here; reserved for telemetry-driven system layer",
        },
        "records": records,
    }
    out_summary = out / "phase_d_build_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(out),
                "hard_negative_pool_size": len(hard_names),
                "motion_variants": args.motion_variants,
                "train_count": counts["train"],
                "val_count": counts["val"],
                "test_count": counts["test"],
                "val_test_identical": {k: v["identical_images_and_labels"] for k, v in integrity.items()},
                "cross_split_overlap": validation["cross_split_image_hash_overlap"],
                "errors": validation["errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
