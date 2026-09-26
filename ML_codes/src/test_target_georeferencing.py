from __future__ import annotations

import pytest

from target_georeferencing import (
    GeoreferenceNotReady,
    NavigationPose,
    SonarGeometrySpec,
    TargetGeoreferenceRequest,
    TargetGeoreferencingEngine,
    TargetOffset,
    body_frd_to_ned,
    ned_to_latlon,
)


def make_nav(source: str = "simulation") -> NavigationPose:
    return NavigationPose(
        timestamp="2026-09-18T12:00:00Z",
        sequence=0,
        navigation_source=source,
        latitude_deg=20.4521,
        longitude_deg=85.1294,
        depth_m=42.3,
        heading_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )


def make_request(nav: NavigationPose | None = None) -> TargetGeoreferenceRequest:
    nav = nav or make_nav()
    return TargetGeoreferenceRequest(
        mission_id="WS-2026-001",
        frame_id="WS_000001",
        detection_id="DET_00001",
        frame_timestamp=nav.timestamp,
        object_class="pipe",
        confidence=0.94,
        navigation=nav,
        target_offset=TargetOffset(forward_m=10.0, starboard_m=5.0, down_m=1.0),
    )


def test_zero_heading_maps_forward_to_north_and_starboard_to_east() -> None:
    north, east, down = body_frd_to_ned(10.0, 5.0, 2.0, 0.0, 0.0, 0.0)
    assert north == pytest.approx(10.0)
    assert east == pytest.approx(5.0)
    assert down == pytest.approx(2.0)


def test_heading_90_maps_forward_to_east() -> None:
    north, east, down = body_frd_to_ned(10.0, 0.0, 0.0, 90.0, 0.0, 0.0)
    assert north == pytest.approx(0.0, abs=1e-9)
    assert east == pytest.approx(10.0)
    assert down == pytest.approx(0.0)


def test_ned_offsets_change_latitude_and_longitude() -> None:
    lat, lon = ned_to_latlon(20.4521, 85.1294, 100.0, 100.0)
    assert lat > 20.4521
    assert lon > 85.1294


def test_unvalidated_geometry_is_blocked() -> None:
    geometry = SonarGeometrySpec(geometry_id="SIM-001", validated=False)
    engine = TargetGeoreferencingEngine(geometry)
    with pytest.raises(GeoreferenceNotReady):
        engine.georeference(make_request())


def test_validated_geometry_produces_simulated_target_coordinate() -> None:
    geometry = SonarGeometrySpec(geometry_id="SIM-001", validated=True)
    engine = TargetGeoreferencingEngine(geometry)
    record = engine.georeference(make_request())
    assert record.status == "PASS"
    assert record.georeference_mode == "simulated_target_georeferenced"
    assert record.geometry_validated is True
    assert record.latitude_deg > 20.4521
    assert record.longitude_deg > 85.1294
    assert record.depth_m == pytest.approx(43.3)


def test_real_source_preserved_without_being_claimed_from_simulation() -> None:
    geometry = SonarGeometrySpec(geometry_id="REAL-CONTRACT-01", validated=True)
    engine = TargetGeoreferencingEngine(geometry)
    record = engine.georeference(make_request(make_nav("real_gnss")))
    assert record.navigation_source == "real_gnss"
    assert record.georeference_mode == "real_target_georeferenced"


def test_boresight_yaw_changes_result() -> None:
    nav = make_nav()
    base = TargetGeoreferencingEngine(SonarGeometrySpec(geometry_id="A", validated=True)).georeference(make_request(nav))
    rotated = TargetGeoreferencingEngine(
        SonarGeometrySpec(geometry_id="B", validated=True, boresight_yaw_deg=90.0)
    ).georeference(make_request(nav))
    assert abs(base.latitude_deg - rotated.latitude_deg) > 1e-9
    assert abs(base.longitude_deg - rotated.longitude_deg) > 1e-9
