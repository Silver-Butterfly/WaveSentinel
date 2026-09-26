from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from motion_telemetry import MotionTelemetry


SCHEMA_VERSION = "1.0"


class CorrectionNotReady(RuntimeError):
    pass


def parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc


def timestamp_delta_s(a: str, b: str) -> float:
    da = parse_timestamp(a)
    db = parse_timestamp(b)
    if da.tzinfo is None or db.tzinfo is None:
        raise ValueError("timestamps must include timezone information")
    return abs((da - db).total_seconds())


@dataclass(frozen=True)
class MotionReference:
    heave_m: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    reference_id: str = "zero"

    def validate(self) -> None:
        for name, value in {
            "heave_m": self.heave_m,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
        }.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not self.reference_id.strip():
            raise ValueError("reference_id must not be empty")


@dataclass(frozen=True)
class AlignmentConfig:
    max_error_s: float = 0.050
    require_valid_telemetry: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.max_error_s) or self.max_error_s < 0:
            raise ValueError("max_error_s must be finite and >= 0")


@dataclass(frozen=True)
class SonarGeometryConfig:
    geometry_id: str
    coordinate_convention: str
    pixel_to_range_model: str
    validated: bool = False

    def validate(self) -> None:
        if not self.geometry_id.strip():
            raise ValueError("geometry_id must not be empty")
        if not self.coordinate_convention.strip():
            raise ValueError("coordinate_convention must not be empty")
        if not self.pixel_to_range_model.strip():
            raise ValueError("pixel_to_range_model must not be empty")

    def ready(self) -> bool:
        self.validate()
        return self.validated


@dataclass(frozen=True)
class TelemetryAlignment:
    frame_timestamp: str
    telemetry_timestamp: str
    telemetry_sequence: int
    telemetry_source: str
    error_s: float
    within_tolerance: bool
    valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MotionCorrectionPlan:
    schema_version: str
    frame_timestamp: str
    telemetry_timestamp: str
    telemetry_sequence: int
    telemetry_source: str
    telemetry_valid: bool
    alignment_error_s: float
    reference: MotionReference
    relative_heave_m: float
    relative_pitch_deg: float
    relative_roll_deg: float
    correction_mode: str
    heave_action: str
    pitch_action: str
    roll_action: str
    applied_to_pixels: bool
    geometry_ready: bool
    limitation: str

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        for value in (
            self.alignment_error_s,
            self.relative_heave_m,
            self.relative_pitch_deg,
            self.relative_roll_deg,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("plan contains non-finite numeric value")
        if self.applied_to_pixels and self.correction_mode != "geometric":
            raise ValueError("pixel application requires geometric mode")

    def to_dict(self) -> dict:
        self.reference.validate()
        self.validate()
        data = asdict(self)
        data["reference"] = asdict(self.reference)
        return data


class MotionCorrectionEngine:
    """Build motion-correction plans without pretending image geometry is known."""

    def __init__(
        self,
        reference: MotionReference | None = None,
        alignment: AlignmentConfig | None = None,
    ) -> None:
        self.reference = reference or MotionReference()
        self.alignment = alignment or AlignmentConfig()
        self.reference.validate()
        self.alignment.validate()

    def nearest_alignment(
        self,
        frame_timestamp: str,
        records: Iterable[MotionTelemetry],
    ) -> TelemetryAlignment:
        frame_ts = parse_timestamp(frame_timestamp)
        candidates = []
        for record in records:
            record.validate()
            if self.alignment.require_valid_telemetry and not record.valid:
                continue
            ts = parse_timestamp(record.timestamp)
            error = abs((ts - frame_ts).total_seconds())
            candidates.append((error, record))

        if not candidates:
            raise ValueError("No usable telemetry records available")

        error, record = min(candidates, key=lambda item: (item[0], item[1].sequence))
        return TelemetryAlignment(
            frame_timestamp=frame_timestamp,
            telemetry_timestamp=record.timestamp,
            telemetry_sequence=record.sequence,
            telemetry_source=record.telemetry_source,
            error_s=error,
            within_tolerance=error <= self.alignment.max_error_s,
            valid=record.valid,
        )

    def plan_from_telemetry(
        self,
        frame_timestamp: str,
        telemetry: MotionTelemetry,
        geometry: SonarGeometryConfig | None = None,
        alignment_error_s: float = 0.0,
    ) -> MotionCorrectionPlan:
        telemetry.validate()
        parse_timestamp(frame_timestamp)
        if alignment_error_s < 0 or not math.isfinite(alignment_error_s):
            raise ValueError("alignment_error_s must be finite and >= 0")

        geometry_ready = bool(geometry and geometry.ready())
        within_tolerance = alignment_error_s <= self.alignment.max_error_s

        relative_heave = telemetry.heave_m - self.reference.heave_m
        relative_pitch = telemetry.pitch_deg - self.reference.pitch_deg
        relative_roll = telemetry.roll_deg - self.reference.roll_deg

        if not telemetry.valid:
            mode = "blocked_invalid_telemetry"
            actions = ("BLOCKED_INVALID_TELEMETRY",) * 3
            applied = False
        elif not within_tolerance:
            mode = "blocked_alignment"
            actions = ("BLOCKED_ALIGNMENT_ERROR",) * 3
            applied = False
        elif geometry_ready:
            mode = "geometric_ready"
            actions = (
                "READY_FOR_GEOMETRIC_HEAVE",
                "READY_FOR_GEOMETRIC_PITCH",
                "READY_FOR_GEOMETRIC_ROLL",
            )
            applied = False
        else:
            mode = "metadata_only"
            actions = (
                "DEFERRED_NO_SONAR_GEOMETRY",
                "DEFERRED_NO_SONAR_GEOMETRY",
                "DEFERRED_NO_SONAR_GEOMETRY",
            )
            applied = False

        plan = MotionCorrectionPlan(
            schema_version=SCHEMA_VERSION,
            frame_timestamp=frame_timestamp,
            telemetry_timestamp=telemetry.timestamp,
            telemetry_sequence=telemetry.sequence,
            telemetry_source=telemetry.telemetry_source,
            telemetry_valid=telemetry.valid,
            alignment_error_s=alignment_error_s,
            reference=self.reference,
            relative_heave_m=relative_heave,
            relative_pitch_deg=relative_pitch,
            relative_roll_deg=relative_roll,
            correction_mode=mode,
            heave_action=actions[0],
            pitch_action=actions[1],
            roll_action=actions[2],
            applied_to_pixels=applied,
            geometry_ready=geometry_ready,
            limitation=(
                "Pixel correction is intentionally not performed until validated sonar "
                "geometry, coordinate convention, and pixel-to-range model are supplied."
            ),
        )
        plan.validate()
        return plan

    def require_geometric_ready(self, geometry: SonarGeometryConfig) -> None:
        if not geometry.ready():
            raise CorrectionNotReady(
                "Validated sonar geometry is required before pixel-level motion correction."
            )


def read_jsonl(path: Path) -> list[MotionTelemetry]:
    records: list[MotionTelemetry] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                records.append(MotionTelemetry.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid telemetry at line {line_no}: {exc}") from exc
    return records


def write_plans(plans: Iterable[MotionCorrectionPlan], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for plan in plans:
            f.write(json.dumps(plan.to_dict(), separators=(",", ":")) + "\n")


def build_plans(records: list[MotionTelemetry], reference: MotionReference) -> list[MotionCorrectionPlan]:
    if not records:
        raise ValueError("Telemetry input contains no records")
    engine = MotionCorrectionEngine(reference=reference)
    plans = []
    for record in records:
        plans.append(
            engine.plan_from_telemetry(
                frame_timestamp=record.timestamp,
                telemetry=record,
                alignment_error_s=0.0,
            )
        )
    return plans


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--input", required=True)
    plan_parser.add_argument("--output", required=True)
    plan_parser.add_argument("--reference-heave", type=float, default=0.0)
    plan_parser.add_argument("--reference-pitch", type=float, default=0.0)
    plan_parser.add_argument("--reference-roll", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "plan":
        input_path = Path(args.input).expanduser().resolve()
        output_dir = Path(args.output).expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        reference = MotionReference(
            heave_m=args.reference_heave,
            pitch_deg=args.reference_pitch,
            roll_deg=args.reference_roll,
            reference_id="manual",
        )
        records = read_jsonl(input_path)
        plans = build_plans(records, reference)
        write_plans(plans, output_dir / "motion_correction_plan.jsonl")

        summary = {
            "schema_version": SCHEMA_VERSION,
            "records": len(plans),
            "telemetry_sources": sorted({p.telemetry_source for p in plans}),
            "correction_mode_counts": {
                mode: sum(p.correction_mode == mode for p in plans)
                for mode in sorted({p.correction_mode for p in plans})
            },
            "pixels_modified": any(p.applied_to_pixels for p in plans),
            "reference": asdict(reference),
            "status": "PASS",
            "note": "Architecture-only plan generation. No image pixels were modified.",
        }
        (output_dir / "motion_correction_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (output_dir / "motion_correction_config.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "alignment": asdict(AlignmentConfig()),
                    "geometry": {
                        "required": True,
                        "validated": False,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
