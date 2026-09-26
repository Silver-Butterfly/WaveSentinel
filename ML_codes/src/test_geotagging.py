from datetime import datetime, timezone, timedelta

import pytest

from geotagging import (
    FrameEvent,
    GeotagConfig,
    GeotagNotReady,
    GeotaggingEngine,
    NavigationFix,
    SonarGeoreferenceConfig,
)


def ts(offset=0):
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset)).isoformat()


def fix(seq=0, offset=0, valid=True, source="simulation", lat=20.0, lon=85.0):
    return NavigationFix(
        timestamp=ts(offset),
        sequence=seq,
        navigation_source=source,
        valid=valid,
        latitude_deg=lat,
        longitude_deg=lon,
        depth_m=42.3,
        heading_deg=142.0,
        speed_mps=2.16,
    )


def event(offset=0):
    return FrameEvent(
        mission_id="WS-2026-001",
        frame_id="WS_000001",
        frame_timestamp=ts(offset),
        detection_id="DET_00001",
        object_class="shipwreck",
        confidence=0.87,
    )


def test_nearest_navigation_fix_is_selected():
    engine = GeotaggingEngine()
    records = [fix(seq=0, offset=0), fix(seq=1, offset=0.10, lat=20.1, lon=85.1)]
    out = engine.geotag_frame(event(offset=0.02), records)
    assert out.status == "PASS"
    assert out.latitude_deg == 20.0
    assert out.longitude_deg == 85.0
    assert out.alignment_error_s == pytest.approx(0.02)


def test_alignment_error_blocks_geotag():
    engine = GeotaggingEngine(GeotagConfig(max_alignment_error_s=0.05))
    out = engine.geotag_frame(event(offset=1), [fix(offset=0)])
    assert out.status == "BLOCKED"
    assert out.geotag_mode == "blocked_alignment"


def test_invalid_navigation_is_skipped():
    engine = GeotaggingEngine()
    records = [fix(seq=0, offset=0, valid=False), fix(seq=1, offset=0.02, valid=True, lat=20.2)]
    out = engine.geotag_frame(event(offset=0), records)
    assert out.status == "PASS"
    assert out.latitude_deg == 20.2
    assert out.navigation_sequence == 1


def test_simulation_and_real_sources_are_preserved():
    engine = GeotaggingEngine()
    sim = engine.geotag_frame(event(), [fix(source="simulation")])
    real = engine.geotag_frame(event(), [fix(source="real_gnss")])
    assert sim.navigation_source == "simulation"
    assert sim.geotag_mode == "simulated_frame_anchor"
    assert real.navigation_source == "real_gnss"
    assert real.geotag_mode == "real_gnss_frame_anchor"


def test_no_navigation_blocks_cleanly():
    engine = GeotaggingEngine()
    out = engine.geotag_frame(event(), [])
    assert out.status == "BLOCKED"
    assert out.geotag_mode == "blocked_no_navigation"
    assert out.latitude_deg is None
    assert out.longitude_deg is None


def test_target_georeferencing_requires_validated_geometry():
    engine = GeotaggingEngine()
    geometry = SonarGeoreferenceConfig(
        geometry_id="demo",
        coordinate_convention="undefined",
        range_model="undefined",
        validated=False,
    )
    with pytest.raises(GeotagNotReady):
        engine.geotag_frame(event(), [fix()], sonar_geometry=geometry, require_target_georeference=True)


def test_validated_geometry_changes_gate_state_without_fabricating_transform():
    engine = GeotaggingEngine()
    geometry = SonarGeoreferenceConfig(
        geometry_id="validated-demo",
        coordinate_convention="verified-convention",
        range_model="verified-range-model",
        validated=True,
    )
    out = engine.geotag_frame(event(), [fix()], sonar_geometry=geometry, require_target_georeference=True)
    assert out.status == "PASS"
    assert out.geotag_mode == "target_georeference_ready"
    assert out.target_georeferenced is False
    assert out.geometry_ready is True
    assert "not implemented" in out.limitation
