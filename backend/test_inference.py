from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference import (
    CLASS_THRESHOLDS,
    decode_yolo_output,
    make_prepared_tensor,
    make_raw_tiles,
    nms_classwise,
)


def test_thresholds_frozen():
    assert CLASS_THRESHOLDS == {
        0: 0.56,
        1: 0.45,
        2: 0.89,
        3: 0.36,
    }


def test_prepared_tensor_contract():
    import numpy as np
    gray = np.zeros((640, 640), dtype=np.uint8)
    x = make_prepared_tensor(gray)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32


def test_raw_tiling_contract():
    import numpy as np
    gray = np.zeros((500, 2500), dtype=np.uint8)
    tiles = make_raw_tiles(gray)
    assert len(tiles) == 3
    assert all(t.shape == (1, 3, 640, 640) for t, _ in tiles)


def test_decode_filters_class_thresholds():
    import numpy as np
    out = np.zeros((1, 8, 8400), dtype=np.float32)
    out[0, 0, 0] = 320
    out[0, 1, 0] = 320
    out[0, 2, 0] = 100
    out[0, 3, 0] = 100
    out[0, 4, 0] = 0.90
    assert len(decode_yolo_output(out, (640, 640))) == 1

    out[0, 4, 0] = 0.50
    assert len(decode_yolo_output(out, (640, 640))) == 0


def test_classwise_nms():
    import numpy as np
    boxes = np.array([[0,0,10,10],[1,1,11,11],[100,100,110,110]], dtype=np.float32)
    scores = np.array([0.9,0.8,0.7], dtype=np.float32)
    classes = np.array([0,0,0], dtype=np.int32)
    keep = nms_classwise(boxes, scores, classes, 0.5, 300)
    assert keep.tolist() == [0,2]
