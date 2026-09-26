from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "1.0"
EARTH_RADIUS_M = 6371008.8
KNOT_TO_MPS = 0.514444


class GeotagNotReady(RuntimeError):
    pass


def parse_timestamp(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone information")
    return dt


def timestamp_error_s(a: str, b: str) -> float:
    return abs((parse_timestamp(a) - parse_timestamp(b)).total_seconds())


@dataclass(frozen=True)
class NavigationFix:
    timestamp: str
    sequence: int
    navigation_source: str
    valid: bool
    latitude_deg: float
    longitude_deg: float
    depth_m: float | None = None
    heading_deg: float | None = None
    speed_mps: float | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        if self.navigation_source not in {"simulation", "real_gnss"}:
            raise ValueError("navigation_source must be simulation or real_gnss")
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        parse_timestamp(self.timestamp)
        for name, value in {
            "latitude_deg": self.latitude_deg,
            "longitude_deg": self.longitude_deg,
        }.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg out of range")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg out of range")
        if self.depth_m is not None:
            if not math.isfinite(float(self.depth_m)) or self.depth_m < 0:
                raise ValueError("depth_m must be finite and >= 0")
        if self.heading_deg is not None:
            if not math.isfinite(float(self.heading_deg)) or not 0 <= self.heading_deg < 360:
                raise ValueError("heading_deg must be in [0, 360)")
        if self.speed_mps is not None:
            if not math.isfinite(float(self.speed_mps)) or self.speed_mps < 0:
                raise ValueError("speed_mps must be finite and >= 0")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NavigationFix":
        obj = cls(
            timestamp=str(data["timestamp"]),
            sequence=int(data["sequence"]),
            navigation_source=str(data["navigation_source"]),
            valid=bool(data["valid"]),
            latitude_deg=float(data["latitude_deg"]),
            longitude_deg=float(data["longitude_deg"]),
            depth_m=None if data.get("depth_m") is None else float(data["depth_m"]),
            heading_deg=None if data.get("heading_deg") is None else float(data["heading_deg"]),
            speed_mps=None if data.get("speed_mps") is None else float(data["speed_mps"]),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )
        obj.validate()
        return obj


@dataclass(frozen=True)
class GeotagConfig:
    max_alignment_error_s: float = 0.050
    require_valid_navigation: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.max_alignment_error_s) or self.max_alignment_error_s < 0:
            raise ValueError("max_alignment_error_s must be finite and >= 0")


@dataclass(frozen=True)
class SonarGeoreferenceConfig:
    geometry_id: str
    coordinate_convention: str
    range_model: str
    validated: bool = False

    def validate(self) -> None:
        if not self.geometry_id.strip():
            raise ValueError("geometry_id must not be empty")
        if not self.coordinate_convention.strip():
            raise ValueError("coordinate_convention must not be empty")
        if not self.range_model.strip():
            raise ValueError("range_model must not be empty")

    def ready(self) -> bool:
        self.validate()
        return self.validated


@dataclass(frozen=True)
class FrameEvent:
    mission_id: str
    frame_id: str
    frame_timestamp: str
    detection_id: str | None = None
    object_class: str | None = None
    confidence: float | None = None
    depth_m: float | None = None

    def validate(self) -> None:
        if not self.mission_id.strip():
            raise ValueError("mission_id must not be empty")
        if not self.frame_id.strip():
            raise ValueError("frame_id must not be empty")
        parse_timestamp(self.frame_timestamp)
        if self.confidence is not None:
            if not math.isfinite(float(self.confidence)) or not 0 <= self.confidence <= 1:
                raise ValueError("confidence must be in [0, 1]")
        if self.depth_m is not None:
            if not math.isfinite(float(self.depth_m)) or self.depth_m < 0:
                raise ValueError("depth_m must be finite and >= 0")


@dataclass(frozen=True)
class GeotagRecord:
    schema_version: str
    mission_id: str
    frame_id: str
    detection_id: str | None
    frame_timestamp: str
    navigation_timestamp: str | None
    navigation_sequence: int | None
    navigation_source: str | None
    alignment_error_s: float | None
    latitude_deg: float | None
    longitude_deg: float | None
    depth_m: float | None
    heading_deg: float | None
    speed_mps: float | None
    geotag_mode: str
    target_georeferenced: bool
    geometry_ready: bool
    status: str
    limitation: str

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        parse_timestamp(self.frame_timestamp)
        if self.navigation_timestamp is not None:
            parse_timestamp(self.navigation_timestamp)
        if self.alignment_error_s is not None:
            if not math.isfinite(float(self.alignment_error_s)) or self.alignment_error_s < 0:
                raise ValueError("alignment_error_s must be finite and >= 0")
        if self.latitude_deg is not None and not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg out of range")
        if self.longitude_deg is not None and not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude_deg out of range")
        if self.geotag_mode not in {
            "simulated_frame_anchor",
            "real_gnss_frame_anchor",
            "target_georeference_ready",
            "target_georeferenced",
            "blocked_no_navigation",
            "blocked_alignment",
            "blocked_invalid_navigation",
        }:
            raise ValueError("Unsupported geotag_mode")
        if self.status not in {"PASS", "BLOCKED"}:
            raise ValueError("status must be PASS or BLOCKED")
        if self.target_georeferenced and self.geotag_mode != "target_georeferenced":
            raise ValueError("target_georeferenced requires target_georeferenced mode")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


class GeotaggingEngine:
    """Align frames with navigation and produce provenance-safe geotags."""

    def __init__(self, config: GeotagConfig | None = None) -> None:
        self.config = config or GeotagConfig()
        self.config.validate()

    def nearest_fix(self, frame_timestamp: str, fixes: Iterable[NavigationFix]) -> tuple[NavigationFix | None, float | None]:
        frame_ts = parse_timestamp(frame_timestamp)
        candidates: list[tuple[float, NavigationFix]] = []
        for fix in fixes:
            fix.validate()
            if self.config.require_valid_navigation and not fix.valid:
                continue
            error = abs((parse_timestamp(fix.timestamp) - frame_ts).total_seconds())
            candidates.append((error, fix))
        if not candidates:
            return None, None
        error, fix = min(candidates, key=lambda item: (item[0], item[1].sequence))
        return fix, error

    def geotag_frame(
        self,
        event: FrameEvent,
        fixes: Iterable[NavigationFix],
        sonar_geometry: SonarGeoreferenceConfig | None = None,
        require_target_georeference: bool = False,
    ) -> GeotagRecord:
        event.validate()
        fix, error = self.nearest_fix(event.frame_timestamp, fixes)

        if fix is None:
            mode = "blocked_no_navigation"
            return GeotagRecord(
                schema_version=SCHEMA_VERSION,
                mission_id=event.mission_id,
                frame_id=event.frame_id,
                detection_id=event.detection_id,
                frame_timestamp=event.frame_timestamp,
                navigation_timestamp=None,
                navigation_sequence=None,
                navigation_source=None,
                alignment_error_s=None,
                latitude_deg=None,
                longitude_deg=None,
                depth_m=event.depth_m,
                heading_deg=None,
                speed_mps=None,
                geotag_mode=mode,
                target_georeferenced=False,
                geometry_ready=False,
                status="BLOCKED",
                limitation="No valid navigation fix is available within the configured alignment policy.",
            )

        if error is None or error > self.config.max_alignment_error_s:
            return GeotagRecord(
                schema_version=SCHEMA_VERSION,
                mission_id=event.mission_id,
                frame_id=event.frame_id,
                detection_id=event.detection_id,
                frame_timestamp=event.frame_timestamp,
                navigation_timestamp=fix.timestamp,
                navigation_sequence=fix.sequence,
                navigation_source=fix.navigation_source,
                alignment_error_s=error,
                latitude_deg=fix.latitude_deg,
                longitude_deg=fix.longitude_deg,
                depth_m=event.depth_m if event.depth_m is not None else fix.depth_m,
                heading_deg=fix.heading_deg,
                speed_mps=fix.speed_mps,
                geotag_mode="blocked_alignment",
                target_georeferenced=False,
                geometry_ready=False,
                status="BLOCKED",
                limitation="Navigation fix exists but frame-to-navigation time error exceeds the configured tolerance.",
            )

        if not fix.valid:
            mode = "blocked_invalid_navigation"
            status = "BLOCKED"
        elif require_target_georeference:
            geometry_ready = bool(sonar_geometry and sonar_geometry.ready())
            if not geometry_ready:
                raise GeotagNotReady(
                    "Validated sonar georeference geometry is required before converting a detection inside the sonar swath into a target coordinate."
                )
            mode = "target_georeference_ready"
            status = "PASS"
        else:
            mode = (
                "simulated_frame_anchor"
                if fix.navigation_source == "simulation"
                else "real_gnss_frame_anchor"
            )
            status = "PASS"

        return GeotagRecord(
            schema_version=SCHEMA_VERSION,
            mission_id=event.mission_id,
            frame_id=event.frame_id,
            detection_id=event.detection_id,
            frame_timestamp=event.frame_timestamp,
            navigation_timestamp=fix.timestamp,
            navigation_sequence=fix.sequence,
            navigation_source=fix.navigation_source,
            alignment_error_s=error,
            latitude_deg=fix.latitude_deg,
            longitude_deg=fix.longitude_deg,
            depth_m=event.depth_m if event.depth_m is not None else fix.depth_m,
            heading_deg=fix.heading_deg,
            speed_mps=fix.speed_mps,
            geotag_mode=mode,
            target_georeferenced=False,
            geometry_ready=bool(sonar_geometry and sonar_geometry.ready()),
            status=status,
            limitation=(
                "Coordinate is a frame-position anchor in the navigation reference; target position inside the sonar swath is not georeferenced."
                if mode in {"simulated_frame_anchor", "real_gnss_frame_anchor"}
                else "Architecture gate passed; target georeferencing transform is still not implemented."
            ),
        )

    def require_target_georeferencing_ready(self, geometry: SonarGeoreferenceConfig) -> None:
        if not geometry.ready():
            raise GeotagNotReady(
                "Validated sonar georeference geometry is required before target-level geotagging."
            )


@dataclass(frozen=True)
class NavigationSimulationConfig:
    duration_s: float = 60.0
    rate_hz: float = 10.0
    seed: int = 26057
    start_latitude_deg: float = 20.4521
    start_longitude_deg: float = 85.1294
    start_heading_deg: float = 142.0
    speed_kts: float = 4.2
    depth_m: float = 42.3
    invalid_fraction: float = 0.0

    def validate(self) -> None:
        if self.duration_s <= 0 or self.rate_hz <= 0:
            raise ValueError("duration_s and rate_hz must be > 0")
        if not -90 <= self.start_latitude_deg <= 90:
            raise ValueError("start_latitude_deg out of range")
        if not -180 <= self.start_longitude_deg <= 180:
            raise ValueError("start_longitude_deg out of range")
        if not 0 <= self.start_heading_deg < 360:
            raise ValueError("start_heading_deg must be in [0, 360)")
        if self.speed_kts < 0 or self.depth_m < 0:
            raise ValueError("speed_kts and depth_m must be >= 0")
        if not 0 <= self.invalid_fraction <= 1:
            raise ValueError("invalid_fraction must be in [0,1]")


class NavigationSimulator:
    def __init__(self, config: NavigationSimulationConfig) -> None:
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(self, start_time: datetime | None = None) -> list[NavigationFix]:
        c = self.config
        start = start_time or datetime.now(timezone.utc)
        count = max(1, int(round(c.duration_s * c.rate_hz)))
        dt = 1.0 / c.rate_hz
        speed_mps = c.speed_kts * KNOT_TO_MPS
        out: list[NavigationFix] = []

        for i in range(count):
            t = i * dt
            distance_m = speed_mps * t
            heading = (c.start_heading_deg + 1.5 * math.sin(2 * math.pi * t / 35.0)) % 360.0
            north = distance_m * math.cos(math.radians(heading))
            east = distance_m * math.sin(math.radians(heading))
            lat = c.start_latitude_deg + math.degrees(north / EARTH_RADIUS_M)
            lon_scale = max(math.cos(math.radians(c.start_latitude_deg)), 1e-6)
            lon = c.start_longitude_deg + math.degrees(east / (EARTH_RADIUS_M * lon_scale))
            fix = NavigationFix(
                timestamp=(start + timedelta(seconds=t)).astimezone(timezone.utc).isoformat(),
                sequence=i,
                navigation_source="simulation",
                valid=self.rng.random() >= c.invalid_fraction,
                latitude_deg=lat,
                longitude_deg=lon,
                depth_m=c.depth_m + 0.25 * math.sin(2 * math.pi * t / 23.0),
                heading_deg=heading,
                speed_mps=speed_mps,
            )
            fix.validate()
            out.append(fix)
        return out


def write_jsonl(items: Iterable[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")


def write_navigation_jsonl(fixes: Iterable[NavigationFix], path: Path) -> None:
    write_jsonl((f.to_dict() for f in fixes), path)


def write_csv_rows(rows: Iterable[dict], path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_navigation_jsonl(path: Path) -> list[NavigationFix]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                out.append(NavigationFix.from_dict(json.loads(raw)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid navigation record at line {line_no}: {exc}") from exc
    return out


def read_frame_events_jsonl(path: Path) -> list[FrameEvent]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                data = json.loads(raw)
                event = FrameEvent(
                    mission_id=str(data["mission_id"]),
                    frame_id=str(data["frame_id"]),
                    frame_timestamp=str(data["frame_timestamp"]),
                    detection_id=None if data.get("detection_id") is None else str(data["detection_id"]),
                    object_class=None if data.get("object_class") is None else str(data["object_class"]),
                    confidence=None if data.get("confidence") is None else float(data["confidence"]),
                    depth_m=None if data.get("depth_m") is None else float(data["depth_m"]),
                )
                event.validate()
                out.append(event)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid frame event at line {line_no}: {exc}") from exc
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="WaveSentinel geotagging architecture")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--output", required=True)
    sim.add_argument("--duration", type=float, default=60.0)
    sim.add_argument("--rate", type=float, default=10.0)
    sim.add_argument("--seed", type=int, default=26057)
    sim.add_argument("--lat", type=float, default=20.4521)
    sim.add_argument("--lon", type=float, default=85.1294)
    sim.add_argument("--heading", type=float, default=142.0)
    sim.add_argument("--speed-kts", type=float, default=4.2)
    sim.add_argument("--depth", type=float, default=42.3)
    sim.add_argument("--invalid-fraction", type=float, default=0.0)

    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)

    demo = sub.add_parser("demo")
    demo.add_argument("--navigation", required=True)
    demo.add_argument("--output", required=True)
    demo.add_argument("--mission-id", default="WS-2026-001")
    demo.add_argument("--frame-step", type=int, default=60)

    args = parser.parse_args()

    if args.command == "simulate":
        output = Path(args.output).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        cfg = NavigationSimulationConfig(
            duration_s=args.duration,
            rate_hz=args.rate,
            seed=args.seed,
            start_latitude_deg=args.lat,
            start_longitude_deg=args.lon,
            start_heading_deg=args.heading,
            speed_kts=args.speed_kts,
            depth_m=args.depth,
            invalid_fraction=args.invalid_fraction,
        )
        fixes = NavigationSimulator(cfg).generate(datetime.now(timezone.utc))
        write_navigation_jsonl(fixes, output / "navigation.jsonl")
        write_csv_rows(
            (fix.to_dict() for fix in fixes),
            output / "navigation.csv",
            ["schema_version", "timestamp", "sequence", "navigation_source", "valid", "latitude_deg", "longitude_deg", "depth_m", "heading_deg", "speed_mps"],
        )
        (output / "navigation_simulation_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        print(json.dumps({"records": len(fixes), "rate_hz": cfg.rate_hz, "duration_s": cfg.duration_s, "navigation_source": "simulation", "status": "PASS"}, indent=2))
        return 0

    if args.command == "validate":
        fixes = read_navigation_jsonl(Path(args.input).expanduser().resolve())
        print(json.dumps({"records": len(fixes), "schema_errors": 0, "status": "PASS"}, indent=2))
        return 0

    if args.command == "demo":
        nav_path = Path(args.navigation).expanduser().resolve()
        output = Path(args.output).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        fixes = read_navigation_jsonl(nav_path)
        if args.frame_step <= 0:
            raise ValueError("frame-step must be > 0")
        events = []
        for idx, fix in enumerate(fixes[::args.frame_step]):
            events.append({
                "mission_id": args.mission_id,
                "frame_id": f"WS_{idx + 1:06d}",
                "frame_timestamp": fix.timestamp,
                "detection_id": f"DET_{idx + 1:05d}",
                "object_class": ["shipwreck", "pipe", "mine", "ghost_net"][idx % 4],
                "confidence": [0.87, 0.94, 0.98, 0.69][idx % 4],
                "depth_m": fix.depth_m,
            })
        events_path = output / "frame_events.jsonl"
        write_jsonl(events, events_path)
        engine = GeotaggingEngine()
        event_objects = read_frame_events_jsonl(events_path)
        records = [engine.geotag_frame(event, fixes) for event in event_objects]
        write_jsonl((record.to_dict() for record in records), output / "geotags.jsonl")
        write_csv_rows(
            (record.to_dict() for record in records),
            output / "geotags.csv",
            list(records[0].to_dict().keys()) if records else ["schema_version"],
        )
        summary = {
            "records": len(records),
            "navigation_sources": sorted({r.navigation_source for r in records if r.navigation_source}),
            "geotag_mode_counts": {mode: sum(r.geotag_mode == mode for r in records) for mode in sorted({r.geotag_mode for r in records})},
            "target_georeferenced": any(r.target_georeferenced for r in records),
            "status": "PASS" if all(r.status == "PASS" for r in records) else "BLOCKED",
            "note": "Simulation frame-anchor geotags only. Target position inside the sonar swath is not physically georeferenced.",
        }
        (output / "geotagging_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
