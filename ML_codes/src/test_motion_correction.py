from datetime import datetime, timedelta, timezone

from motion_correction import (
    AlignmentConfig,
    CorrectionNotReady,
    MotionCorrectionEngine,
    MotionReference,
    SonarGeometryConfig,
)
from motion_telemetry import MotionTelemetry


def make_record(
    timestamp="2026-09-18T00:00:00+00:00",
    sequence=0,
    source="simulation",
    valid=True,
    heave=0.25,
    pitch=2.0,
    roll=-3.0,
):
    return MotionTelemetry(
        timestamp=timestamp,
        sequence=sequence,
        telemetry_source=source,
        valid=valid,
        heave_m=heave,
        pitch_deg=pitch,
        roll_deg=roll,
    )


def test_relative_motion_is_computed_from_reference():
    engine = MotionCorrectionEngine(MotionReference(0.1, 1.0, -1.0, "ref"))
    record = make_record()
    plan = engine.plan_from_telemetry(record.timestamp, record)
    assert plan.relative_heave_m == 0.15
    assert plan.relative_pitch_deg == 1.0
    assert plan.relative_roll_deg == -2.0
    assert plan.correction_mode == "metadata_only"
    assert not plan.applied_to_pixels


def test_nearest_alignment_selects_closest_valid_record():
    t0 = datetime(2026, 9, 18, tzinfo=timezone.utc)
    records = [
        make_record((t0 + timedelta(seconds=0.00)).isoformat(), 0, valid=False),
        make_record((t0 + timedelta(seconds=0.04)).isoformat(), 1, valid=True),
        make_record((t0 + timedelta(seconds=0.09)).isoformat(), 2, valid=True),
    ]
    engine = MotionCorrectionEngine(alignment=AlignmentConfig(max_error_s=0.05))
    alignment = engine.nearest_alignment((t0 + timedelta(seconds=0.06)).isoformat(), records)
    assert alignment.telemetry_sequence == 1
    assert alignment.within_tolerance
    assert alignment.valid


def test_alignment_error_blocks_correction():
    record = make_record()
    engine = MotionCorrectionEngine(alignment=AlignmentConfig(max_error_s=0.01))
    plan = engine.plan_from_telemetry(record.timestamp, record, alignment_error_s=0.02)
    assert plan.correction_mode == "blocked_alignment"
    assert not plan.applied_to_pixels


def test_invalid_telemetry_blocks_correction():
    record = make_record(valid=False)
    engine = MotionCorrectionEngine()
    plan = engine.plan_from_telemetry(record.timestamp, record)
    assert plan.correction_mode == "blocked_invalid_telemetry"
    assert not plan.applied_to_pixels


def test_real_and_simulation_source_are_preserved():
    record = make_record(source="real_imu")
    engine = MotionCorrectionEngine()
    plan = engine.plan_from_telemetry(record.timestamp, record)
    assert plan.telemetry_source == "real_imu"


def test_validated_geometry_changes_plan_state_but_not_pixels():
    geometry = SonarGeometryConfig(
        geometry_id="future-sonar-01",
        coordinate_convention="TO_BE_CONFIRMED",
        pixel_to_range_model="TO_BE_CONFIRMED",
        validated=True,
    )
    record = make_record()
    engine = MotionCorrectionEngine()
    plan = engine.plan_from_telemetry(record.timestamp, record, geometry=geometry)
    assert plan.correction_mode == "geometric_ready"
    assert plan.geometry_ready
    assert not plan.applied_to_pixels


def test_require_geometric_ready_rejects_unvalidated_geometry():
    geometry = SonarGeometryConfig(
        geometry_id="future-sonar-01",
        coordinate_convention="TO_BE_CONFIRMED",
        pixel_to_range_model="TO_BE_CONFIRMED",
        validated=False,
    )
    try:
        MotionCorrectionEngine().require_geometric_ready(geometry)
    except CorrectionNotReady as exc:
        assert "geometry" in str(exc).lower()
    else:
        raise AssertionError("Expected CorrectionNotReady")
