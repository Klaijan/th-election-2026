from __future__ import annotations

import cv2
import numpy as np

from extract_handwritten_numbers.logo_detector import LogoDetector


def _make_template() -> np.ndarray:
    # Simple synthetic "logo": a circle with a cross
    img = np.full((60, 60), 255, dtype=np.uint8)
    cv2.circle(img, (30, 30), 20, 0, 3)
    cv2.line(img, (15, 30), (45, 30), 0, 3)
    cv2.line(img, (30, 15), (30, 45), 0, 3)
    return img


def test_logo_detector_detects_logo_in_top_band(tmp_path):
    tmpl = _make_template()
    tmpl_path = tmp_path / "logo.png"
    assert cv2.imwrite(str(tmpl_path), tmpl)

    # Create a page and stamp the logo near the top
    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    x0, y0 = 200, 40
    h, w = tmpl.shape[:2]
    stamp = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
    page[y0 : y0 + h, x0 : x0 + w] = stamp

    det = LogoDetector(template_path=tmpl_path, threshold=0.70, search_y_frac=0.20, scales=(0.8, 1.0, 1.2))
    hit = det.detect_logo(page)

    assert hit["has_logo"] is True
    assert float(hit["confidence"]) >= 0.70
    bx, by, bw, bh = hit["bbox"]
    assert abs(int(bx) - x0) <= 3
    assert abs(int(by) - y0) <= 3
    assert abs(int(bw) - w) <= 3
    assert abs(int(bh) - h) <= 3


def test_logo_detector_rejects_when_only_below_threshold(tmp_path):
    tmpl = _make_template()
    tmpl_path = tmp_path / "logo.png"
    assert cv2.imwrite(str(tmpl_path), tmpl)

    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    # Put something else in the top band (a rectangle), should not match well with our template.
    # Use filled shape to avoid accidental edge correlations.
    cv2.rectangle(page, (50, 50), (250, 120), (0, 0, 0), -1)

    det = LogoDetector(template_path=tmpl_path, threshold=0.95, search_y_frac=0.20, scales=(1.0,))
    hit = det.detect_logo(page)
    assert hit["has_logo"] is False
    assert float(hit["confidence"]) < 0.95


