"""
DRISHTI-SSS V4 source-stratified, leakage-safe splitter.

Design:
- Split within every provenance/source family so all image types appear in train/val/test.
- Preserve meaningful pipe acquisition groups.
- Deduplicate exact image content before splitting.
- Quarantine images with invalid YOLO labels.
- Exclude configured classes, default: class 0 (crab_pot).
- Keep the original dataset untouched.
- Produce train/val/test folders plus a JSON/TXT split report.

Expected source layout:
DATASET/
  train/images, train/labels
  val/images,   val/labels
  test/images,  test/labels

The script also accepts a dataset root containing images/labels recursively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_ORDER = ["bg", "mine", "other", "pipe", "synth", "wreckA", "wreckR"]

SOURCE_RULES = [
    ("pipe", re.compile(r"^pipe(?:_|$)", re.I)),
    ("mine", re.compile(r"^(?:mine|mine_cylinder)(?:_|$)", re.I)),
    ("synth", re.compile(r"^(?:synth|ghost_net)(?:_|$)", re.I)),
    ("bg", re.compile(r"^(?:bg|background)(?:_|$)", re.I)),
    ("wreckA", re.compile(r"^wrecka(?:_|$)", re.I)),
    ("wreckR", re.compile(r"^wreckr(?:_|$)", re.I)),
]


@dataclass
class Sample:
    image: Path
    label: Path | None
    source: str
    group: str
    sha256: str
    classes: tuple[int, ...]
    objects: int
    empty: bool


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def source_family(stem: str) -> str:
    for name, pattern in SOURCE_RULES:
        if pattern.search(stem):
            return name
    return "other"


def source_group(stem: str, source: str) -> str:
    if source == "pipe":
        m = re.match(r"^(pipe_[^_]+)", stem, re.I)
        if m:
            return m.group(1)
    return stem


def parse_label(label: Path | None, excluded: set[int]):
    if label is None or not label.exists():
        return [], True, False

    classes = []
    invalid = False
    objects = 0

    for line_no, raw in enumerate(label.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            invalid = True
            continue

        try:
            cls = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            invalid = True
            continue

        if cls < 0:
            invalid = True
            continue

        if any(not math.isfinite(x) for x in vals):
            invalid = True
            continue

        x, y, w, h = vals
        if w <= 0 or h <= 0:
            invalid = True
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            invalid = True

        if cls not in excluded:
            classes.append(cls)
            objects += 1

    return classes, objects == 0, invalid


def find_pairs(root: Path):
    images = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    pairs = []
    for image in sorted(images):
        label = None
        parts = list(image.parts)
        lower_parts = [x.lower() for x in parts]

        if "images" in lower_parts:
            idx = len(lower_parts) - 1 - lower_parts[::-1].index("images")
            label_parts = parts[:idx] + ["labels"] + parts[idx + 1:]
            candidate = Path(*label_parts).with_suffix(".txt")
            if candidate.exists():
                label = candidate

        if label is None:
            candidates = [
                image.with_suffix(".txt"),
                image.parent.parent / "labels" / (image.stem + ".txt"),
                root / "labels" / (image.stem + ".txt"),
            ]
            for c in candidates:
                if c.exists():
                    label = c
                    break

        pairs.append((image, label))

    return pairs


def allocate_targets(counts: dict[str, int], fractions: tuple[float, float, float],
                     total_targets: tuple[int, int, int]):
    """Largest-remainder allocation, then exact correction to requested global totals."""
    splits = ["train", "val", "test"]
    targets = {s: {} for s in splits}

    for source, n in counts.items():
        raw = [n * f for f in fractions]
        base = [math.floor(x) for x in raw]
        rem = [raw[i] - base[i] for i in range(3)]

        for i in sorted(range(3), key=lambda j: rem[j], reverse=True)[:n - sum(base)]:
            base[i] += 1

        for i, s in enumerate(splits):
            targets[s][source] = base[i]

    for i, s in enumerate(splits):
        diff = total_targets[i] - sum(targets[s].values())
        if diff == 0:
            continue
        ranked = sorted(
            counts,
            key=lambda src: (
                counts[src] * fractions[i] - targets[s][src],
                counts[src],
            ),
            reverse=(diff > 0),
        )
        step = 1 if diff > 0 else -1
        for src in ranked:
            if diff == 0:
                break
            if step < 0 and targets[s][src] <= 1:
                continue
            targets[s][src] += step
            diff -= step

    return targets


def assign_groups(samples: list[Sample], source_targets: dict[str, int], seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[s.group].append(s)

    group_items = list(groups.items())
    rng.shuffle(group_items)

    group_items.sort(
        key=lambda kv: (
            -len(kv[1]),
            -sum(kv[1][0].classes.count(c) for c in kv[1][0].classes),
        )
    )

    assigned = {"train": [], "val": [], "test": []}
    counts = Counter()

    target = dict(source_targets)

    for group, items in group_items:
        size = len(items)

        def score(split):
            current = counts[split]
            desired = target[split]
            after = current + size
            overflow = max(0, after - desired)
            gap = abs(after - desired)
            return (overflow, gap, current / max(desired, 1), rng.random())

        split = min(("train", "val", "test"), key=score)
        assigned[split].extend(items)
        counts[split] += size

    return assigned, counts


def rebalance_singletons(assigned, targets):
    """Improve exact source counts when a source has singleton groups."""
    changed = True
    while changed:
        changed = False
        for src, target_by_split in targets.items():
            pass

        for src in SOURCE_ORDER:
            for dst in ("train", "val", "test"):
                excess = len([x for x in assigned[dst] if x.source == src]) - targets[dst].get(src, 0)
                if excess <= 0:
                    continue

                for src2 in ("train", "val", "test"):
                    if src2 == dst:
                        continue
                    need = targets[src2].get(src, 0) - len([x for x in assigned[src2] if x.source == src])
                    if need <= 0:
                        continue

                    candidates = [
                        x for x in assigned[dst]
                        if x.source == src and x.group == x.image.stem
                    ]
                    if not candidates:
                        continue

                    item = candidates[0]
                    assigned[dst].remove(item)
                    assigned[src2].append(item)
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break

    return assigned


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_report(samples, assigned, invalid, duplicates, output, target_totals, seed):
    report = {
        "version": "DRISHTI-SSS-V5",
        "output_naming": "stem__sha256prefix",
        "seed": seed,
        "targets": {
            "train": target_totals[0],
            "val": target_totals[1],
            "test": target_totals[2],
        },
        "input_clean_unique": len(samples),
        "invalid_images": len(invalid),
        "exact_duplicate_images_removed": len(duplicates),
        "splits": {},
        "source_distribution": {},
        "class_distribution": {},
        "empty_images": {},
        "overlap": {},
        "invalid_manifest": sorted(str(x) for x in invalid),
        "duplicate_manifest": sorted(str(x) for x in duplicates),
    }

    for split in ("train", "val", "test"):
        items = assigned[split]
        report["splits"][split] = {"images": len(items), "objects": sum(x.objects for x in items)}
        report["source_distribution"][split] = dict(Counter(x.source for x in items))
        report["class_distribution"][split] = dict(
            sorted(Counter(c for x in items for c in x.classes).items())
        )
        report["empty_images"][split] = sum(x.empty for x in items)

    split_hashes = {
        s: {x.sha256 for x in assigned[s]} for s in ("train", "val", "test")
    }
    split_stems = {
        s: {x.image.stem for x in assigned[s]} for s in ("train", "val", "test")
    }
    split_groups = {
        s: {f"{x.source}:{x.group}" for x in assigned[s]} for s in ("train", "val", "test")
    }

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        report["overlap"][f"{a}_vs_{b}"] = {
            "hashes": len(split_hashes[a] & split_hashes[b]),
            "stems": len(split_stems[a] & split_stems[b]),
            "groups": len(split_groups[a] & split_groups[b]),
        }

    output.mkdir(parents=True, exist_ok=True)
    (output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "DRISHTI-SSS V5 SPLIT SUMMARY",
        f"Input clean unique: {len(samples)}",
        f"Invalid images quarantined: {len(invalid)}",
        f"Exact duplicate images removed: {len(duplicates)}",
        "",
        "TARGETS",
        f"train: {target_totals[0]}",
        f"val:   {target_totals[1]}",
        f"test:  {target_totals[2]}",
        "",
        "SPLITS",
    ]

    for split in ("train", "val", "test"):
        lines.append(
            f"{split}: {report['splits'][split]['images']} images, "
            f"{report['splits'][split]['objects']} objects, "
            f"{report['empty_images'][split]} empty"
        )
        lines.append(f"  sources: {report['source_distribution'][split]}")
        lines.append(f"  classes: {report['class_distribution'][split]}")

    lines += ["", "OVERLAP"]
    for k, v in report["overlap"].items():
        lines.append(f"{k}: {v}")

    (output / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=26057)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--exclude-classes", default="0")
    ap.add_argument("--keep-invalid", action="store_true")
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-9:
        raise SystemExit("train + val + test must equal 1.0")

    excluded = {int(x.strip()) for x in args.exclude_classes.split(",") if x.strip()}

    pairs = find_pairs(args.source_root)
    if not pairs:
        raise SystemExit(f"No images found under {args.source_root}")

    invalid = []
    raw = []
    seen_hash = {}
    duplicates = []

    for image, label in pairs:
        classes, empty, bad = parse_label(label, excluded)

        if bad and not args.keep_invalid:
            invalid.append(image)
            continue

        digest = sha256_file(image)
        if digest in seen_hash:
            duplicates.append(image)
            continue
        seen_hash[digest] = image

        src = source_family(image.stem)
        group = source_group(image.stem, src)

        raw.append(
            Sample(
                image=image,
                label=label,
                source=src,
                group=group,
                sha256=digest,
                classes=tuple(sorted(set(classes))),
                objects=len(classes),
                empty=empty,
            )
        )

    if not raw:
        raise SystemExit("No usable samples remain after filtering.")

    counts = Counter(x.source for x in raw)
    n = len(raw)
    total_targets = (
        round(n * args.train),
        round(n * args.val),
        n - round(n * args.train) - round(n * args.val),
    )

    targets = allocate_targets(
        dict(counts),
        (args.train, args.val, args.test),
        total_targets,
    )

    assigned = {"train": [], "val": [], "test": []}

    for src in sorted(counts):
        subset = [x for x in raw if x.source == src]
        src_targets = {s: targets[s].get(src, 0) for s in ("train", "val", "test")}

        groups = defaultdict(list)
        for x in subset:
            groups[x.group].append(x)

        if all(len(v) == 1 for v in groups.values()):
            stable_src_seed = int(hashlib.sha256(src.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(args.seed + stable_src_seed)
            rng.shuffle(subset)
            a = src_targets["train"]
            b = a + src_targets["val"]
            assigned["train"].extend(subset[:a])
            assigned["val"].extend(subset[a:b])
            assigned["test"].extend(subset[b:])
        else:
            local, local_counts = assign_groups(subset, src_targets, args.seed)
            for s in assigned:
                assigned[s].extend(local[s])

    for s in assigned:
        random.Random(args.seed + {"train": 1, "val": 2, "test": 3}[s]).shuffle(assigned[s])

    # Use a content-derived suffix so distinct source files with the same
    # basename can never overwrite each other in the flat output folders.
    # The original source path remains recorded in the manifest/report.
    output_manifest = []
    output_names_seen = set()

    for split in ("train", "val", "test"):
        for item in assigned[split]:
            suffix = item.image.suffix.lower()
            output_stem = f"{item.image.stem}__{item.sha256[:12]}"
            output_image_name = output_stem + suffix
            output_label_name = output_stem + ".txt"

            if output_image_name in output_names_seen:
                raise RuntimeError(
                    f"Output filename collision after hashing: {output_image_name}"
                )
            output_names_seen.add(output_image_name)

            out_img = args.output / split / "images" / output_image_name
            out_lbl = args.output / split / "labels" / output_label_name

            safe_copy(item.image, out_img)

            if item.label and item.label.exists():
                safe_copy(item.label, out_lbl)
            else:
                out_lbl.parent.mkdir(parents=True, exist_ok=True)
                out_lbl.write_text("", encoding="utf-8")

            output_manifest.append({
                "split": split,
                "source_image": str(item.image),
                "source_label": str(item.label) if item.label else None,
                "output_image": output_image_name,
                "output_label": output_label_name,
                "sha256": item.sha256,
                "source": item.source,
                "group": item.group,
            })

    (args.output / "output_manifest.json").write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )

    quarantine = args.output / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    (quarantine / "invalid_images.txt").write_text(
        "\n".join(str(x) for x in invalid) + ("\n" if invalid else ""),
        encoding="utf-8",
    )
    (quarantine / "duplicate_images.txt").write_text(
        "\n".join(str(x) for x in duplicates) + ("\n" if duplicates else ""),
        encoding="utf-8",
    )

    report = build_report(
        raw,
        assigned,
        invalid,
        duplicates,
        args.output,
        total_targets,
        args.seed,
    )

    print(json.dumps({
        "status": "ok",
        "input_images": len(pairs),
        "clean_unique": len(raw),
        "invalid": len(invalid),
        "duplicates": len(duplicates),
        "targets": report["targets"],
        "actual": {s: report["splits"][s]["images"] for s in ("train", "val", "test")},
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
