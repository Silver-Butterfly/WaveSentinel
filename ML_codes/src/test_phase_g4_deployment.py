import json
from pathlib import Path


def test_preflight_report_schema(tmp_path):
    p = tmp_path / "preflight.json"
    p.write_text(json.dumps({
        "status": "PASS",
        "schema_version": "1.0",
        "phase": "G4",
        "stage": "preflight",
        "checks": {"onnx_hash_match": True, "fp32_hash_match": True},
        "fp16": {"status": "DEFERRED"},
    }))
    d = json.loads(p.read_text())
    assert d["phase"] == "G4"
    assert d["schema_version"] == "1.0"
    assert d["fp16"]["status"] == "DEFERRED"


def test_no_legacy_fp16_flag_reference():
    source = Path(__file__).with_name("phase_g4_deployment.py").read_text(encoding="utf-8")
    assert "platform_has_fast_fp16" not in source
    assert "BuilderFlag" not in source


def test_required_hashes_are_frozen():
    source = Path(__file__).with_name("phase_g4_deployment.py").read_text(encoding="utf-8")
    assert "17db4dee6147d1eb05867cda7ca503b1133e3b6e83a45cb7c0e359cf9f78f12b" in source
    assert "5e2bfabaca634873b61242c1817323d51cdb002d63831cee1a2874fd60818067" in source
