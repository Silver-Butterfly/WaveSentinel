from phase_g2_tensorrt import (
    EXPECTED_PHASE_D_SHA256,
    EXPECTED_PHASE_G1_ONNX_SHA256,
    SCHEMA_VERSION,
    pair_predictions,
)


def test_expected_checkpoint_hash_is_pinned():
    assert len(EXPECTED_PHASE_D_SHA256) == 64
    int(EXPECTED_PHASE_D_SHA256, 16)


def test_expected_onnx_hash_is_pinned():
    assert len(EXPECTED_PHASE_G1_ONNX_SHA256) == 64
    int(EXPECTED_PHASE_G1_ONNX_SHA256, 16)


def test_hashes_are_distinct():
    assert EXPECTED_PHASE_D_SHA256 != EXPECTED_PHASE_G1_ONNX_SHA256


def test_schema_version():
    assert SCHEMA_VERSION == "1.2"


def test_pair_predictions_identical():
    sample = {
        "boxes": [[0.0, 0.0, 10.0, 10.0]],
        "confidence": [0.9],
        "classes": [0],
    }
    result = pair_predictions(sample, sample)
    assert result["paired_count"] == 1
    assert result["unmatched_a"] == 0
    assert result["unmatched_b"] == 0
    assert result["numerically_stable"] is True
