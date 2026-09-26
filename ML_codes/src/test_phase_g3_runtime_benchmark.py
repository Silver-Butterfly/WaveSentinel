from __future__ import annotations

import json
from pathlib import Path

import pytest

import phase_g3_runtime_benchmark as g3


def test_percentile_linear_known_values() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert g3.percentile_linear(values, 50) == 3.0
    assert g3.percentile_linear(values, 95) == pytest.approx(4.8)
    assert g3.percentile_linear(values, 99) == pytest.approx(4.96)


def test_summarize_latency_contains_throughput() -> None:
    result = g3.summarize_latencies([10.0, 20.0, 30.0, 40.0])
    assert result["count"] == 4.0
    assert result["p50_ms"] == pytest.approx(25.0)
    assert result["throughput_fps"] == pytest.approx(40.0)
    assert result["p50_fps"] == pytest.approx(40.0)


def test_select_reference_images_is_deterministic(tmp_path: Path) -> None:
    for name in ["c.jpg", "a.jpg", "b.png", "d.jpeg"]:
        (tmp_path / name).write_bytes(b"x")
    one = g3.select_reference_images(tmp_path, 3, 26057)
    two = g3.select_reference_images(tmp_path, 3, 26057)
    assert one == two
    assert len(one) == 3


def test_select_reference_images_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        g3.select_reference_images(tmp_path, 1, 26057)


def test_schema_result_can_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "g3.json"
    payload = {"status": "PASS", "schema_version": g3.SCHEMA_VERSION, "phase": "G3"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["phase"] == "G3"
    assert loaded["schema_version"] == "1.0"
