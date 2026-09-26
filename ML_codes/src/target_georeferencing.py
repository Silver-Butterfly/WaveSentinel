from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

EARTH_RADIUS_M = 6_371_008.8
SCHEMA_VERSION = "1.0"


class GeoreferenceNotReady(RuntimeError):
    pass


def parse_timestamp(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if dt.tzinfo is None:
        raise ValueError("timestamp must include timezone information")
    return dt


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class NavigationPose:
    timestamp: str
    sequence: int
    navigation_source: str
    latitude_deg: float
    longitude_deg: float
    depth_m: float
    heading_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        parse_timestamp(self.timestamp)
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        if self.navigation_source not in {"simulation", "real_gnss"}:
            raise ValueError("navigation_source must be simulation or real_gnss")
        _finite("latitude_deg", self.latitude_deg)
        _finite("longitude_deg", self.longitude_deg)
        _finite("depth_m", self.depth_m)
        _finite("heading_deg", self.heading_deg)
        _finite("pitch_deg", self.pitch_deg)
        _finite("roll_deg", self.roll_deg)
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg out of range")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg out of range")
        if self.depth_m < 0:
            raise ValueError("depth_m must be >= 0")
        if not 0.0 <= self.heading_deg < 360.0:
            raise ValueError("heading_deg must be in [0, 360)")
        if not -90.0 <= self.pitch_deg <= 90.0:
            raise ValueError("pitch_deg out of range")
        if not -180.0 <= self.roll_deg <= 180.0:
            raise ValueError("roll_deg out of range")


@dataclass(frozen=True)
class SonarGeometrySpec:
    geometry_id: str
    validated: bool
    body_frame: str = "FRD"
    navigation_frame: str = "NED"
    boresight_yaw_deg: float = 0.0
    lever_arm_forward_m: float = 0.0
    lever_arm_starboard_m: float = 0.0
    lever_arm_down_m: float = 0.0
    pixel_model: str = "external_metric_offsets"

    def validate(self) -> None:
        if not self.geometry_id.strip():
            raise ValueError("geometry_id must not be empty")
        if self.body_frame != "FRD":
            raise ValueError("body_frame must be FRD")
        if self.navigation_frame != "NED":
            raise ValueError("navigation_frame must be NED")
        if not math.isfinite(self.boresight_yaw_deg):
            raise ValueError("boresight_yaw_deg must be finite")
        for name in (
            "lever_arm_forward_m",
            "lever_arm_starboard_m",
            "lever_arm_down_m",
        ):
            _finite(name, getattr(self, name))
        if self.pixel_model not in {"external_metric_offsets", "linear_swath_simulation"}:
            raise ValueError("unsupported pixel_model")

    def ready(self) -> bool:
        self.validate()
        return self.validated


@dataclass(frozen=True)
class TargetOffset:
    forward_m: float
    starboard_m: float
    down_m: float

    def validate(self) -> None:
        _finite("forward_m", self.forward_m)
        _finite("starboard_m", self.starboard_m)
        _finite("down_m", self.down_m)


@dataclass(frozen=True)
class DetectionPixel:
    x_px: float
    y_px: float
    image_width_px: int
    image_height_px: int

    def validate(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be > 0")
        _finite("x_px", self.x_px)
        _finite("y_px", self.y_px)
        if not 0 <= self.x_px <= self.image_width_px:
            raise ValueError("x_px out of image bounds")
        if not 0 <= self.y_px <= self.image_height_px:
            raise ValueError("y_px out of image bounds")


@dataclass(frozen=True)
class LinearSwathModel:
    swath_width_m: float
    along_track_length_m: float
    center_x_px: float | None = None
    center_y_px: float | None = None
    vertical_origin_at_frame_center: bool = True

    def validate(self, pixel: DetectionPixel) -> None:
        if self.swath_width_m <= 0 or self.along_track_length_m <= 0:
            raise ValueError("swath_width_m and along_track_length_m must be > 0")
        if self.center_x_px is not None and not 0 <= self.center_x_px <= pixel.image_width_px:
            raise ValueError("center_x_px out of bounds")
        if self.center_y_px is not None and not 0 <= self.center_y_px <= pixel.image_height_px:
            raise ValueError("center_y_px out of bounds")

    def pixel_to_offset(self, pixel: DetectionPixel) -> TargetOffset:
        self.validate(pixel)
        cx = self.center_x_px if self.center_x_px is not None else pixel.image_width_px / 2.0
        cy = self.center_y_px if self.center_y_px is not None else pixel.image_height_px / 2.0
        starboard = ((pixel.x_px - cx) / pixel.image_width_px) * self.swath_width_m
        forward = ((pixel.y_px - cy) / pixel.image_height_px) * self.along_track_length_m
        return TargetOffset(forward_m=forward, starboard_m=starboard, down_m=0.0)


def _matmul(a: tuple[tuple[float, float, float], ...], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        sum(a[i][j] * v[j] for j in range(3)) for i in range(3)
    )


def body_frd_to_ned(forward_m: float, starboard_m: float, down_m: float, heading_deg: float, pitch_deg: float, roll_deg: float, boresight_yaw_deg: float = 0.0) -> tuple[float, float, float]:
    yaw = math.radians((heading_deg + boresight_yaw_deg) % 360.0)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0))
    ry = ((cp, 0.0, -sp), (0.0, 1.0, 0.0), (sp, 0.0, cp))
    rx = ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr))

    body = (forward_m, starboard_m, down_m)
    after_roll = _matmul(rx, body)
    after_pitch = _matmul(ry, after_roll)
    return _matmul(rz, after_pitch)


def ned_to_latlon(latitude_deg: float, longitude_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    lat_rad = math.radians(latitude_deg)
    dlat = north_m / EARTH_RADIUS_M
    cos_lat = max(1e-9, math.cos(lat_rad))
    dlon = east_m / (EARTH_RADIUS_M * cos_lat)
    return latitude_deg + math.degrees(dlat), longitude_deg + math.degrees(dlon)


@dataclass(frozen=True)
class TargetGeoreferenceRequest:
    mission_id: str
    frame_id: str
    detection_id: str
    frame_timestamp: str
    object_class: str
    confidence: float
    navigation: NavigationPose
    target_offset: TargetOffset | None = None
    detection_pixel: DetectionPixel | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        if not self.mission_id.strip() or not self.frame_id.strip() or not self.detection_id.strip():
            raise ValueError("identifiers must not be empty")
        if not self.object_class.strip():
            raise ValueError("object_class must not be empty")
        parse_timestamp(self.frame_timestamp)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.navigation.validate()
        if self.target_offset is not None:
            self.target_offset.validate()
        if self.detection_pixel is not None:
            self.detection_pixel.validate()
        if self.target_offset is None and self.detection_pixel is None:
            raise ValueError("provide target_offset or detection_pixel")


@dataclass(frozen=True)
class TargetGeoreferenceRecord:
    schema_version: str
    mission_id: str
    frame_id: str
    detection_id: str
    frame_timestamp: str
    object_class: str
    confidence: float
    navigation_source: str
    geometry_id: str
    geometry_validated: bool
    heading_deg: float
    pitch_deg: float
    roll_deg: float
    forward_m: float
    starboard_m: float
    down_m: float
    north_offset_m: float
    east_offset_m: float
    latitude_deg: float
    longitude_deg: float
    depth_m: float
    georeference_mode: str
    status: str
    limitation: str

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported schema_version")
        if self.status not in {"PASS", "BLOCKED"}:
            raise ValueError("status must be PASS or BLOCKED")
        if self.georeference_mode not in {"simulated_target_georeferenced", "real_target_georeferenced"}:
            raise ValueError("unsupported georeference_mode")
        if not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg out of range")
        if not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude_deg out of range")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


class TargetGeoreferencingEngine:
    def __init__(self, geometry: SonarGeometrySpec) -> None:
        geometry.validate()
        self.geometry = geometry

    def _offset_from_request(self, request: TargetGeoreferenceRequest) -> TargetOffset:
        if request.target_offset is not None:
            return request.target_offset
        assert request.detection_pixel is not None
        if self.geometry.pixel_model != "linear_swath_simulation":
            raise GeoreferenceNotReady(
                "Pixel-to-range geometry is not validated; provide a validated metric target offset instead."
            )
        raise GeoreferenceNotReady(
            "Linear pixel-to-swath conversion is simulation-only and must be supplied through an explicit simulation model."
        )

    def georeference(self, request: TargetGeoreferenceRequest) -> TargetGeoreferenceRecord:
        request.validate()
        if not self.geometry.ready():
            raise GeoreferenceNotReady("Validated sonar geometry is required before target-level georeferencing.")

        offset = self._offset_from_request(request)
        lever = TargetOffset(
            self.geometry.lever_arm_forward_m,
            self.geometry.lever_arm_starboard_m,
            self.geometry.lever_arm_down_m,
        )
        total_f = offset.forward_m + lever.forward_m
        total_s = offset.starboard_m + lever.starboard_m
        total_d = offset.down_m + lever.down_m
        north, east, down = body_frd_to_ned(
            total_f,
            total_s,
            total_d,
            request.navigation.heading_deg,
            request.navigation.pitch_deg,
            request.navigation.roll_deg,
            self.geometry.boresight_yaw_deg,
        )
        lat, lon = ned_to_latlon(
            request.navigation.latitude_deg,
            request.navigation.longitude_deg,
            north,
            east,
        )
        depth = request.navigation.depth_m + down
        mode = (
            "simulated_target_georeferenced"
            if request.navigation.navigation_source == "simulation"
            else "real_target_georeferenced"
        )
        record = TargetGeoreferenceRecord(
            schema_version=SCHEMA_VERSION,
            mission_id=request.mission_id,
            frame_id=request.frame_id,
            detection_id=request.detection_id,
            frame_timestamp=request.frame_timestamp,
            object_class=request.object_class,
            confidence=request.confidence,
            navigation_source=request.navigation.navigation_source,
            geometry_id=self.geometry.geometry_id,
            geometry_validated=self.geometry.validated,
            heading_deg=request.navigation.heading_deg,
            pitch_deg=request.navigation.pitch_deg,
            roll_deg=request.navigation.roll_deg,
            forward_m=offset.forward_m,
            starboard_m=offset.starboard_m,
            down_m=offset.down_m,
            north_offset_m=north,
            east_offset_m=east,
            latitude_deg=lat,
            longitude_deg=lon,
            depth_m=depth,
            georeference_mode=mode,
            status="PASS",
            limitation=(
                "Target coordinate derived from an explicitly validated geometry contract."
                if request.navigation.navigation_source == "real_gnss"
                else "Simulated target coordinate using validated simulation geometry; not a field measurement."
            ),
        )
        record.validate()
        return record


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 26057
    latitude_deg: float = 20.4521
    longitude_deg: float = 85.1294
    heading_deg: float = 142.0
    pitch_deg: float = 2.4
    roll_deg: float = -1.1
    depth_m: float = 42.3
    target_count: int = 8


TARGET_CLASSES = ("shipwreck", "pipe", "mine", "ghost_net")


def simulate(config: SimulationConfig) -> list[TargetGeoreferenceRecord]:
    rng = random.Random(config.seed)
    nav = NavigationPose(
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        sequence=0,
        navigation_source="simulation",
        latitude_deg=config.latitude_deg,
        longitude_deg=config.longitude_deg,
        depth_m=config.depth_m,
        heading_deg=config.heading_deg,
        pitch_deg=config.pitch_deg,
        roll_deg=config.roll_deg,
    )
    geometry = SonarGeometrySpec(
        geometry_id="SIM-SSS-FRD-NED-001",
        validated=True,
        lever_arm_forward_m=1.2,
        lever_arm_starboard_m=0.0,
        lever_arm_down_m=0.8,
        boresight_yaw_deg=0.0,
        pixel_model="external_metric_offsets",
    )
    engine = TargetGeoreferencingEngine(geometry)
    out: list[TargetGeoreferenceRecord] = []
    timestamp = nav.timestamp
    for i in range(config.target_count):
        request = TargetGeoreferenceRequest(
            mission_id="WS-2026-001",
            frame_id=f"WS_{i + 1:06d}",
            detection_id=f"DET_{i + 1:05d}",
            frame_timestamp=timestamp,
            object_class=TARGET_CLASSES[i % len(TARGET_CLASSES)],
            confidence=[0.87, 0.94, 0.98, 0.69][i % 4],
            navigation=nav,
            target_offset=TargetOffset(
                forward_m=rng.uniform(-8.0, 12.0),
                starboard_m=rng.uniform(-25.0, 25.0),
                down_m=rng.uniform(-0.5, 0.5),
            ),
        )
        out.append(engine.georeference(request))
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Target-level georeferencing architecture and simulation")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--output", required=True)
    sim.add_argument("--count", type=int, default=8)
    sim.add_argument("--seed", type=int, default=26057)

    args = parser.parse_args()
    if args.command == "simulate":
        if args.count <= 0:
            raise SystemExit("--count must be > 0")
        records = simulate(SimulationConfig(seed=args.seed, target_count=args.count))
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in records]
        (out / "target_georeferences.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        write_csv(rows, out / "target_georeferences.csv")
        summary = {
            "records": len(records),
            "navigation_sources": sorted({r.navigation_source for r in records}),
            "georeference_modes": sorted({r.georeference_mode for r in records}),
            "geometry_ids": sorted({r.geometry_id for r in records}),
            "target_georeferenced": all(r.status == "PASS" for r in records),
            "status": "PASS" if records and all(r.status == "PASS" for r in records) else "BLOCKED",
            "note": "Simulation only. Target coordinates are generated from an explicitly validated synthetic geometry contract.",
        }
        (out / "target_georeferencing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
