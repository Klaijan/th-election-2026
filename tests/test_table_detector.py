import numpy as np
import cv2

from vote69_form_ocr.table_detector import TableDetector
from vote69_form_ocr.types import Box


def _make_grid(w=600, h=400, cols=5, rows=6, margin=20, thickness=3):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness)

    # verticals
    xs = np.linspace(x0, x1, cols + 1).astype(int)
    ys = np.linspace(y0, y1, rows + 1).astype(int)
    for x in xs[1:-1]:
        cv2.line(img, (x, y0), (x, y1), (0, 0, 0), thickness)
    for y in ys[1:-1]:
        cv2.line(img, (x0, y), (x1, y), (0, 0, 0), thickness)
    return img, xs.tolist(), ys.tolist()


def test_detect_grid_lines_and_column3():
    img, xs, ys = _make_grid(cols=5, rows=6)
    det = TableDetector()
    bbox = Box(x=10, y=10, w=img.shape[1] - 20, h=img.shape[0] - 20)
    struct = det.parse_table_structure(img, bbox)

    assert struct.cols >= 4  # can merge lines; but must have multiple columns
    x_last = struct.target_column
    assert x_last[0] < x_last[1]
    assert 0 <= x_last[0] <= img.shape[1]
    assert 0 <= x_last[1] <= img.shape[1]


