import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def read_labels(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) != 5:
            continue
        try:
            cls = int(p[0])
            vals = [float(x) for x in p[1:]]
        except ValueError:
            continue
        rows.append((cls, *vals))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Create a ghost-net oversampled YOLO training set without touching val/test."
    )
    ap.add_argument("--source", required=True, help="DRISHTI-SSS-TRAIN root")
    ap.add_argument("--output", required=True, help="Output training root")
    ap.add_argument("--ghost-class", type=int, default=3)
    ap.add_argument("--factor", type=float, default=2.0,
                    help="Target ghost-net exposure multiplier. 2.0 = original + one duplicate exposure.")
    ap.add_argument("--seed", type=int, default=26057)
    args = ap.parse_args()

    src = Path(args.source)
    dst = Path(args.output)

    if dst.exists():
        raise SystemExit(f"Output already exists: {dst}\nChoose a new output directory.")

    random.seed(args.seed)

    for split in ("train", "val", "test"):
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy original dataset exactly first.
    for split in ("train", "val", "test"):
        for p in (src / split / "images").iterdir():
            if p.is_file():
                shutil.copy2(p, dst / split / "images" / p.name)
        for p in (src / split / "labels").iterdir():
            if p.is_file():
                shutil.copy2(p, dst / split / "labels" / p.name)

    train_images = []
    ghost_images = []

    for img in sorted((src / "train" / "images").iterdir()):
        if not img.is_file():
            continue
        label = src / "train" / "labels" / f"{img.stem}.txt"
        rows = read_labels(label)
        if rows:
            train_images.append(img)
            if any(r[0] == args.ghost_class for r in rows):
                ghost_images.append(img)

    # Factor 2 means add one extra exposure for every ghost image.
    extra_n = round(len(ghost_images) * max(0.0, args.factor - 1.0))
    selected = random.choices(ghost_images, k=extra_n) if extra_n else []

    # Duplicate images under unique names. Labels remain identical.
    created = []
    for i, img in enumerate(selected, 1):
        label = src / "train" / "labels" / f"{img.stem}.txt"

        new_stem = f"{img.stem}__ghostos{i:05d}"
        new_img = dst / "train" / "images" / f"{new_stem}{img.suffix}"
        new_lbl = dst / "train" / "labels" / f"{new_stem}.txt"

        shutil.copy2(img, new_img)
        shutil.copy2(label, new_lbl)

        created.append({
            "original_image": str(img),
            "original_label": str(label),
            "new_image": str(new_img),
            "new_label": str(new_lbl),
        })

    # Manifest makes the oversampling fully auditable.
    manifest = {
        "method": "image_level_oversampling",
        "source": str(src),
        "output": str(dst),
        "seed": args.seed,
        "ghost_class": args.ghost_class,
        "factor": args.factor,
        "original_train_images": len(train_images),
        "original_ghost_images": len(ghost_images),
        "extra_ghost_exposures": extra_n,
        "final_train_images": len(train_images) + extra_n,
        "val_images_unchanged": len(list((dst / "val" / "images").iterdir())),
        "test_images_unchanged": len(list((dst / "test" / "images").iterdir())),
        "duplicates": created,
    }

    (dst / "ghost_oversampling_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "status": "ok",
        "source": str(src),
        "output": str(dst),
        "original_train_images": len(train_images),
        "original_ghost_images": len(ghost_images),
        "extra_ghost_exposures": extra_n,
        "final_train_images": len(train_images) + extra_n,
        "val_images": len(list((dst / "val" / "images").iterdir())),
        "test_images": len(list((dst / "test" / "images").iterdir())),
        "manifest": str(dst / "ghost_oversampling_manifest.json")
    }, indent=2))


if __name__ == "__main__":
    main()
