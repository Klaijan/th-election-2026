import numpy as np
import cv2

from extract_handwritten_numbers.dot_detector import DotDetector


def _make_dotted_line_image(w=600, h=300, y=120, x0=60, x1=520, step=12, r=2, missing_every=0):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    xs = list(range(x0, x1 + 1, step))
    for i, x in enumerate(xs):
        if missing_every and (i % missing_every == 0):
            continue
        cv2.circle(img, (x, y), r, (0, 0, 0), -1)
    return img


def test_detects_single_dotted_line():
    img = _make_dotted_line_image(missing_every=0)
    det = DotDetector()
    lines = det.detect_dotted_lines(img, (0, img.shape[0]))
    assert len(lines) >= 1
    best = max(lines, key=lambda d: d.dot_count)
    assert best.dot_count >= 10
    assert best.length >= 200


def test_allows_small_gaps():
    img = _make_dotted_line_image(missing_every=7)  # missing dots
    det = DotDetector()
    lines = det.detect_dotted_lines(img, (0, img.shape[0]))
    assert len(lines) >= 1


