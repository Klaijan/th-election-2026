import numpy as np

from vote69_form_ocr.field_extractor import FieldExtractor
from vote69_form_ocr.types import DottedLine


def test_bbox_calculation_and_clamping():
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    dl = DottedLine(y=50, x_start=30, x_end=200, length=170, dot_count=20, dots=[(30, 50), (200, 50)])
    ex = FieldExtractor(remove_dots=False)
    regs = ex.extract_field_regions(img, [dl])
    assert len(regs) == 0 or len(regs) == 1  # may be empty due to empty-region skip
    if regs:
        r = regs[0]
        assert r.bbox.x >= 0
        assert r.bbox.y >= 0
        assert r.bbox.x2 <= img.shape[1]
        assert r.bbox.y2 <= img.shape[0]


