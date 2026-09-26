from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable


class TelemetrySource(str, Enum):
    SIMULATION = "simulation"
    REAL_IMU = "real_imu"


@dataclass(frozen=True)
class MotionTelemetry:
    timestamp: str
    sequence: int
    telemetry_source: str
    valid: bool
    heave_m: float
    pitch_deg: float
    roll_deg: float
    schema_version: str = "1.0"

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("Unsupported schema_version")
        if self.telemetry_source not in {"simulation", "real_imu"}:
            raise ValueError("telemetry_source must be simulation or real_imu")
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        for name, value in {
            "heave_m": self.heave_m,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
        }.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not -90.0 <= self.pitch_deg <= 90.0:
            raise ValueError("pitch_deg out of range")
        if not -180.0 <= self.roll_deg <= 180.0:
            raise ValueError("roll_deg out of range")
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO-8601") from exc

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MotionTelemetry":
        obj = cls(
            timestamp=str(data["timestamp"]),
            sequence=int(data["sequence"]),
            telemetry_source=str(data["telemetry_source"]),
            valid=bool(data["valid"]),
            heave_m=float(data["heave_m"]),
            pitch_deg=float(data["pitch_deg"]),
            roll_deg=float(data["roll_deg"]),
            schema_version=str(data.get("schema_version", "1.0")),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class MotionTelemetryConfig:
    duration_s: float = 30.0
    rate_hz: float = 10.0
    seed: int = 26057
    heave_amplitude_m: float = 0.45
    heave_period_s: float = 5.0
    pitch_amplitude_deg: float = 4.0
    pitch_period_s: float = 7.0
    roll_amplitude_deg: float = 5.0
    roll_period_s: float = 9.0
    noise_heave_m: float = 0.03
    noise_pitch_deg: float = 0.35
    noise_roll_deg: float = 0.35
    invalid_fraction: float = 0.0

    def validate(self) -> None:
        for name, value in {
            "duration_s": self.duration_s,
            "rate_hz": self.rate_hz,
            "heave_period_s": self.heave_period_s,
            "pitch_period_s": self.pitch_period_s,
            "roll_period_s": self.roll_period_s,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")
        for name, value in {
            "heave_amplitude_m": self.heave_amplitude_m,
            "pitch_amplitude_deg": self.pitch_amplitude_deg,
            "roll_amplitude_deg": self.roll_amplitude_deg,
            "noise_heave_m": self.noise_heave_m,
            "noise_pitch_deg": self.noise_pitch_deg,
            "noise_roll_deg": self.noise_roll_deg,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.invalid_fraction <= 1.0:
            raise ValueError("invalid_fraction must be in [0,1]")


class MotionTelemetrySimulator:
    def __init__(self, config: MotionTelemetryConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(self, start_time: datetime | None = None) -> list[MotionTelemetry]:
        c = self.config
        start = start_time or datetime.now(timezone.utc)
        n = max(1, int(round(c.duration_s * c.rate_hz)))
        dt = 1.0 / c.rate_hz
        out = []

        for i in range(n):
            t = i * dt
            record = MotionTelemetry(
                timestamp=(start + timedelta(seconds=t)).astimezone(
                    timezone.utc
                ).isoformat(),
                sequence=i,
                telemetry_source=TelemetrySource.SIMULATION.value,
                valid=self.rng.random() >= c.invalid_fraction,
                heave_m=(
                    c.heave_amplitude_m
                    * math.sin(2 * math.pi * t / c.heave_period_s)
                    + self.rng.gauss(0, c.noise_heave_m)
                ),
                pitch_deg=(
                    c.pitch_amplitude_deg
                    * math.sin(2 * math.pi * t / c.pitch_period_s)
                    + self.rng.gauss(0, c.noise_pitch_deg)
                ),
                roll_deg=(
                    c.roll_amplitude_deg
                    * math.sin(
                        2 * math.pi * t / c.roll_period_s + math.pi / 4
                    )
                    + self.rng.gauss(0, c.noise_roll_deg)
                ),
            )
            record.validate()
            out.append(record)

        return out


def write_jsonl(records: Iterable[MotionTelemetry], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), separators=(",", ":")) + "\n")


def write_json(records: Iterable[MotionTelemetry], path: Path) -> None:
    path.write_text(
        json.dumps([r.to_dict() for r in records], indent=2),
        encoding="utf-8",
    )


def write_csv(records: Iterable[MotionTelemetry], path: Path) -> None:
    fields = [
        "schema_version", "timestamp", "sequence", "telemetry_source",
        "valid", "heave_m", "pitch_deg", "roll_deg"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())


def validate_jsonl(path: Path) -> dict:
    total = valid = invalid_state = 0
    errors = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            total += 1
            try:
                r = MotionTelemetry.from_dict(json.loads(raw))
                if r.valid:
                    valid += 1
                else:
                    invalid_state += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append({"line": line_no, "error": str(exc)})

    return {
        "total_records": total,
        "valid_records": valid,
        "invalid_state_records": invalid_state,
        "schema_errors": len(errors),
        "errors": errors[:20],
        "status": "PASS" if not errors else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--output", required=True)
    sim.add_argument("--duration", type=float, default=30.0)
    sim.add_argument("--rate", type=float, default=10.0)
    sim.add_argument("--seed", type=int, default=26057)
    sim.add_argument("--invalid-fraction", type=float, default=0.0)

    val = sub.add_parser("validate")
    val.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.command == "simulate":
        out = Path(args.output).expanduser().resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
        out.mkdir(parents=True, exist_ok=True)

        cfg = MotionTelemetryConfig(
            duration_s=args.duration,
            rate_hz=args.rate,
            seed=args.seed,
            invalid_fraction=args.invalid_fraction,
        )
        records = MotionTelemetrySimulator(cfg).generate()

        write_jsonl(records, out / "motion_telemetry.jsonl")
        write_json(records, out / "motion_telemetry.json")
        write_csv(records, out / "motion_telemetry.csv")
        (out / "simulation_config.json").write_text(
            json.dumps({
                "schema_version": "1.0",
                "telemetry_source": "simulation",
                "config": asdict(cfg),
                "note": "Synthetic software-integration telemetry, not real IMU data."
            }, indent=2),
            encoding="utf-8",
        )

        print("=" * 78)
        print("PHASE F2 | MOTION TELEMETRY SIMULATION COMPLETE")
        print("=" * 78)
        print(f"Records: {len(records)}")
        print(f"Rate: {cfg.rate_hz:.2f} Hz")
        print(f"Duration: {cfg.duration_s:.2f} s")
        print(f"Output: {out}")
        print("Source is explicitly marked as simulation.")
        return 0

    path = Path(args.input).expanduser().resolve()
    report = validate_jsonl(path)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
