from pathlib import Path
import argparse

MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(source)

    total_files = 0
    total_objects = 0
    empty_files = 0

    for split in SPLITS:
        src_labels = source / split / "labels"
        dst_labels = output / split / "labels"

        if not src_labels.exists():
            raise FileNotFoundError(src_labels)

        dst_labels.mkdir(parents=True, exist_ok=True)

        for src in src_labels.glob("*.txt"):
            dst = dst_labels / src.name
            text = src.read_text(encoding="utf-8").strip()

            if not text:
                dst.write_text("", encoding="utf-8")
                empty_files += 1
                total_files += 1
                continue

            output_lines = []

            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.split()

                if len(parts) != 5:
                    raise ValueError(
                        f"Invalid YOLO row: {src}:{line_no}"
                    )

                old_class = int(parts[0])

                if old_class not in MAPPING:
                    raise ValueError(
                        f"Unexpected class {old_class} in {src}:{line_no}. "
                        f"Expected only {sorted(MAPPING)}."
                    )

                parts[0] = str(MAPPING[old_class])
                output_lines.append(" ".join(parts))
                total_objects += 1

            dst.write_text(
                "\n".join(output_lines) + "\n",
                encoding="utf-8"
            )

            total_files += 1

    print("DRISHTI-SSS V5 LABEL REMAP")
    print("=" * 60)
    print(f"Source:      {source}")
    print(f"Output:      {output}")
    print(f"Label files: {total_files}")
    print(f"Empty:       {empty_files}")
    print(f"Objects:     {total_objects}")
    print()
    print("Mapping:")
    print("  1 -> 0  pipe")
    print("  2 -> 1  shipwreck")
    print("  3 -> 2  mine")
    print("  4 -> 3  ghost_net")


if __name__ == "__main__":
    main()