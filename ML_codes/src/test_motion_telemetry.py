from datetime import datetime, timezone
from pathlib import Path
import tempfile

from motion_telemetry import (
    MotionTelemetry,
    MotionTelemetryConfig,
    MotionTelemetrySimulator,
    validate_jsonl,
    write_jsonl,
)


def test_round_trip():
    r = MotionTelemetry(
        "2026-09-18T00:00:00+00:00", 0, "simulation", True,
        0.2, -1.5, 2.0
    )
    assert MotionTelemetry.from_dict(r.to_dict()) == r


def test_simulator_count_and_source():
    cfg = MotionTelemetryConfig(duration_s=2, rate_hz=10, seed=26057)
    records = MotionTelemetrySimulator(cfg).generate(
        datetime(2026, 9, 18, tzinfo=timezone.utc)
    )
    assert len(records) == 20
    assert all(r.telemetry_source == "simulation" for r in records)
    assert [r.sequence for r in records] == list(range(20))


def test_deterministic():
    cfg = MotionTelemetryConfig(duration_s=3, rate_hz=5, seed=26057)
    start = datetime(2026, 9, 18, tzinfo=timezone.utc)
    assert MotionTelemetrySimulator(cfg).generate(start) == MotionTelemetrySimulator(cfg).generate(start)


def test_invalid_records_are_supported():
    cfg = MotionTelemetryConfig(duration_s=1, rate_hz=10, invalid_fraction=1.0)
    records = MotionTelemetrySimulator(cfg).generate()
    assert records and all(not r.valid for r in records)


def test_jsonl_validation():
    cfg = MotionTelemetryConfig(duration_s=1, rate_hz=5, seed=26057)
    records = MotionTelemetrySimulator(cfg).generate()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "telemetry.jsonl"
        write_jsonl(records, path)
        report = validate_jsonl(path)
    assert report["status"] == "PASS"
    assert report["total_records"] == 5
    assert report["schema_errors"] == 0


def test_bad_pitch_rejected():
    r = MotionTelemetry(
        "2026-09-18T00:00:00+00:00", 0, "simulation", True,
        0.0, 100.0, 0.0
    )
    try:
        r.validate()
    except ValueError as exc:
        assert "pitch" in str(exc)
    else:
        raise AssertionError("Expected invalid pitch to be rejected")
