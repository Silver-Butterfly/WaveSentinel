import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path, w, h):
    gt = []
    if not label_path.exists():
        return gt
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        p = line.split()
        if len(p) != 5:
            continue
        try:
            cls = int(p[0])
            cx, cy, bw, bh = map(float, p[1:])
        except ValueError:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        gt.append({"cls": cls, "box": (x1, y1, x2, y2), "line": line_no, "area": bw * bh})
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", required=True, help="DRISHTI-SSS-TRAIN root")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou-match", type=float, default=0.50)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.data_root)
    images_dir = root / args.split / "images"
    labels_dir = root / args.split / "labels"
    out = Path(args.output) if args.output else Path("K:/Debris model/runs") / f"confidence_analysis_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    names = {0: "pipe", 1: "shipwreck", 2: "mine", 3: "ghost_net"}
    records = []
    class_stats = defaultdict(lambda: {
        "gt": 0, "matched": 0, "unmatched": 0,
        "matched_conf": [], "unmatched_best_conf": [],
        "matched_iou": [], "all_pred_conf": [], "fp_conf": []
    })

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    )

    for idx, image_path in enumerate(image_paths, 1):
        label_path = labels_dir / (image_path.stem + ".txt")

        try:
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.7,
                device=0,
                verbose=False,
                max_det=300,
            )[0]
        except Exception as e:
            print(f"ERROR {image_path.name}: {e}")
            continue

        orig_h, orig_w = result.orig_shape
        gt = load_gt(label_path, orig_w, orig_h)

        preds = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confs, classes):
                preds.append({
                    "cls": int(cls),
                    "conf": float(conf),
                    "box": tuple(float(x) for x in box),
                })
                class_stats[int(cls)]["all_pred_conf"].append(float(conf))

        used = set()

        # Match each GT to its highest-IoU same-class prediction.
        for g in gt:
            c = g["cls"]
            cs = class_stats[c]
            cs["gt"] += 1

            candidates = [
                (j, iou_xyxy(g["box"], p["box"]))
                for j, p in enumerate(preds)
                if j not in used and p["cls"] == c
            ]

            if candidates:
                best_j, best_iou = max(candidates, key=lambda x: x[1])
                if best_iou >= args.iou_match:
                    used.add(best_j)
                    cs["matched"] += 1
                    cs["matched_conf"].append(preds[best_j]["conf"])
                    cs["matched_iou"].append(best_iou)
                    records.append({
                        "image": image_path.name,
                        "class": names[c],
                        "class_id": c,
                        "status": "TP",
                        "confidence": preds[best_j]["conf"],
                        "iou": best_iou,
                        "gt_area": g["area"],
                    })
                    continue

                cs["unmatched_best_conf"].append(preds[best_j]["conf"])
                records.append({
                    "image": image_path.name,
                    "class": names[c],
                    "class_id": c,
                    "status": "LOCALIZATION_FAIL",
                    "confidence": preds[best_j]["conf"],
                    "iou": best_iou,
                    "gt_area": g["area"],
                })
                continue

            cs["unmatched"] += 1
            records.append({
                "image": image_path.name,
                "class": names[c],
                "class_id": c,
                "status": "FN",
                "confidence": 0.0,
                "iou": 0.0,
                "gt_area": g["area"],
            })

        for j, p in enumerate(preds):
            if j not in used:
                class_stats[p["cls"]]["fp_conf"].append(p["conf"])

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)}")

    def mean(xs):
        return round(float(np.mean(xs)), 5) if xs else None

    def pct(xs, threshold):
        return round(100.0 * sum(x < threshold for x in xs) / len(xs), 2) if xs else None

    summary = {}
    for c, name in names.items():
        s = class_stats[c]
        tp = s["matched"]
        gt = s["gt"]
        locfail = len(s["unmatched_best_conf"])
        fn = s["unmatched"]
        summary[name] = {
            "gt": gt,
            "matched_iou_ge_0.50": tp,
            "recall_at_iou_0.50": round(tp / gt, 4) if gt else None,
            "localization_failures": locfail,
            "true_fns_without_same_class_prediction": fn,
            "mean_tp_conf": mean(s["matched_conf"]),
            "median_tp_conf": mean([float(np.median(s["matched_conf"]))]) if s["matched_conf"] else None,
            "tp_conf_lt_0.25_pct": pct(s["matched_conf"], 0.25),
            "tp_conf_lt_0.50_pct": pct(s["matched_conf"], 0.50),
            "tp_conf_lt_0.75_pct": pct(s["matched_conf"], 0.75),
            "mean_localization_fail_conf": mean(s["unmatched_best_conf"]),
            "median_localization_fail_conf": mean([float(np.median(s["unmatched_best_conf"]))]) if s["unmatched_best_conf"] else None,
            "locfail_conf_lt_0.25_pct": pct(s["unmatched_best_conf"], 0.25),
            "locfail_conf_lt_0.50_pct": pct(s["unmatched_best_conf"], 0.50),
            "locfail_conf_lt_0.75_pct": pct(s["unmatched_best_conf"], 0.75),
            "false_positive_count": len(s["fp_conf"]),
            "mean_fp_conf": mean(s["fp_conf"]),
            "median_fp_conf": mean([float(np.median(s["fp_conf"]))]) if s["fp_conf"] else None,
        }

    report = {
        "model": args.model,
        "split": args.split,
        "images": len(image_paths),
        "imgsz": args.imgsz,
        "prediction_conf": args.conf,
        "match_iou": args.iou_match,
        "summary": summary,
    }

    (out / "confidence_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (out / "prediction_records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "class", "class_id", "status", "confidence", "iou", "gt_area"
        ])
        writer.writeheader()
        writer.writerows(records)

    lines = [
        f"CONFIDENCE / LOCALIZATION ANALYSIS ({args.split})",
        "=" * 62,
        f"Images: {len(image_paths)}",
        f"Prediction conf floor: {args.conf}",
        f"GT match IoU: {args.iou_match}",
        "",
    ]

    for name in names.values():
        s = summary[name]
        lines.extend([
            f"[{name}]",
            f"GT: {s['gt']}",
            f"Matched @ IoU>=0.50: {s['matched_iou_ge_0.50']}",
            f"Recall @ IoU>=0.50: {s['recall_at_iou_0.50']}",
            f"Localization failures: {s['localization_failures']}",
            f"True FNs without same-class prediction: {s['true_fns_without_same_class_prediction']}",
            f"Mean TP confidence: {s['mean_tp_conf']}",
            f"Median TP confidence: {s['median_tp_conf']}",
            f"TP confidence <0.25: {s['tp_conf_lt_0.25_pct']}%",
            f"TP confidence <0.50: {s['tp_conf_lt_0.50_pct']}%",
            f"TP confidence <0.75: {s['tp_conf_lt_0.75_pct']}%",
            f"Mean localization-fail confidence: {s['mean_localization_fail_conf']}",
            f"Median localization-fail confidence: {s['median_localization_fail_conf']}",
            f"Loc-fail confidence <0.25: {s['locfail_conf_lt_0.25_pct']}%",
            f"Loc-fail confidence <0.50: {s['locfail_conf_lt_0.50_pct']}%",
            f"Loc-fail confidence <0.75: {s['locfail_conf_lt_0.75_pct']}%",
            f"False positives: {s['false_positive_count']}",
            f"Mean FP confidence: {s['mean_fp_conf']}",
            f"Median FP confidence: {s['median_fp_conf']}",
            "",
        ])

    (out / "confidence_analysis.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"Reports written to: {out}")


if __name__ == "__main__":
    main()
