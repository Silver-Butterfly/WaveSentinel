from pathlib import Path

from phase_g1_onnx import ExportConfig, sha256_file


def test_export_config_defaults_are_conservative():
    cfg = ExportConfig()
    cfg.validate()
    assert cfg.imgsz == 640
    assert cfg.batch == 1
    assert cfg.half is False
    assert cfg.dynamic is False
    assert cfg.simplify is False


def test_export_config_rejects_invalid_image_size():
    cfg = ExportConfig(imgsz=0)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "imgsz" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_export_config_rejects_invalid_batch():
    cfg = ExportConfig(batch=0)
    try:
        cfg.validate()
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_sha256_file_is_deterministic(tmp_path: Path):
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"wavesentinel-phase-g1")
    first = sha256_file(p)
    second = sha256_file(p)
    assert first == second
    assert len(first) == 64


def test_export_config_accepts_explicit_opset():
    cfg = ExportConfig(opset=17)
    cfg.validate()
    assert cfg.opset == 17
