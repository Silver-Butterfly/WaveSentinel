# SubPipeMini2 Level-2 Data Audit

**Dataset root:** `K:\Debris model\SubPipeMiniSSS`  
**Audit status:** `BLOCKED_UNMATCHED_COCO`

## Executive verdict

Some COCO image references do not resolve to local SSS images. Training is blocked until the path/name mapping is resolved.

## 1. Modality separation

- **CAM0:** 16200 images; dimensions/channels={'2704x1520x3': 16200}
- **CAM1:** 430 images; dimensions/channels={'1936x1216x3': 430}


## 2. COCO annotation audit

- `DATA/SSS_HF_images/COCO_Annotation/coco_format.json`: 903 images, 593 annotations, 566 annotated images, 337 unannotated images, 0 invalid bboxes, 0 matched local SSS images.
- `DATA/SSS_LF_images/COCO_Annotation/coco_format.json`: 1055 images, 726 annotations, 696 annotated images, 359 unannotated images, 0 invalid bboxes, 0 matched local SSS images.


## 3. YOLO annotation audit

- Discovered YOLO annotation files: 1368
- Matched to local SSS images: 0
- By modality: {}
- Invalid YOLO rows: 0

## 4. Temporal redundancy

- SSS images: None
- Timestamp gap statistics: None
- Possible >0.5 s sequence breaks: None
- Consecutive similarity: None

## 5. Effective-sample planning

{
  "status": "not_computed"
}


## 6. Telemetry integrity

- `DATA/Acceleration.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/Altitude.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/AngularVelocity.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/Depth.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/EstimatedState.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/ForwardDistance.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/Pressure.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/Rpm.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/Temperature.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.
- `DATA/WaterVelocity.csv`: rows=16200, timestamp_present=True, blank_cells=0, duplicate_rows=0, negative_timestamp_steps=0.


## 7. Required actions before model implementation

1. Keep LF/HF SSS images as separate audited populations until preprocessing and frequency harmonization are justified.
1. Use survey/site/pass/date/sequence groups for train/validation/test partitioning. Do not perform a random image-level split.
1. Resolve exact per-image COCO↔YOLO equivalence before choosing one annotation format as the canonical training source.
1. Quantify annotated-frame clustering and avoid placing adjacent frames from the same continuous run into different splits.
1. Map telemetry timestamps to SSS image timestamps only after confirming timestamp semantics and camera/SSS synchronization.
1. Inspect telemetry extreme values semantically. Do not label negative or large values invalid solely from magnitude.
1. Document the final leakage-safe grouping rule in /audit/data_audit_report.md before model implementation.


## 8. Evidence classification

- **Verified from local dataset:** counts, filenames, annotation structure, timestamps, dimensions, and telemetry fields actually observed by this script.
- **Computed by audit:** distributions, matching rates, similarity statistics, gap statistics, and outlier counts.
- **Inference:** sequence-break candidates and planning-bound effective sample estimate.
- **Unknown until survey metadata is mapped:** true survey/site/pass boundaries and leakage-safe group IDs.
- **Out of scope:** FLS, MBES, synthetic sonar generation, edge deployment.
