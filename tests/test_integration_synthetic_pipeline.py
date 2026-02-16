import numpy as np
import cv2

from extract_handwritten_numbers import config
from extract_handwritten_numbers.dot_detector import DotDetector
from extract_handwritten_numbers.field_extractor import FieldExtractor
from extract_handwritten_numbers.table_detector import TableDetector
from extract_handwritten_numbers.table_extractor import TableExtractor
from extract_handwritten_numbers.types import Box, TableStructure


def _draw_dotted_line(img, y, x0, x1, step=12, r=2):
    for x in range(x0, x1 + 1, step):
        cv2.circle(img, (x, y), r, (0, 0, 0), -1)


def _make_page(w=800, h=1200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_fields(page):
    # 12 dotted lines in the configured fields zone
    h = int(page.shape[0])
    y0 = int(h * float(config.FIELD_ZONE[0]))
    y1 = int(h * float(config.FIELD_ZONE[1]))
    y_base = y0 + 30
    step_y = max(28, int((y1 - y0 - 60) / 12))
    for i in range(12):
        y = y_base + i * step_y
        _draw_dotted_line(page, y, 120, 600, step=12, r=2)
        cv2.putText(page, str(100 + i), (130, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


def _draw_table(page, bbox: Box, cols=5, header_plus_rows=17):
    x0, y0, x1, y1 = bbox.x, bbox.y, bbox.x2, bbox.y2
    cv2.rectangle(page, (x0, y0), (x1, y1), (0, 0, 0), 3)

    xs = np.linspace(x0, x1, cols + 1).astype(int).tolist()
    ys = np.linspace(y0, y1, header_plus_rows + 1).astype(int).tolist()
    for x in xs[1:-1]:
        cv2.line(page, (x, y0), (x, y1), (0, 0, 0), 3)
    for y in ys[1:-1]:
        cv2.line(page, (x0, y), (x1, y), (0, 0, 0), 3)

    # Draw dots + numbers only in the LAST column
    c = cols - 1
    cx0, cx1 = xs[c], xs[c + 1]
    for r in range(1, header_plus_rows):  # skip header row
        ry0, ry1 = ys[r], ys[r + 1]
        y_dot = int(ry0 + (ry1 - ry0) * 0.7)
        _draw_dotted_line(page, y_dot, cx0 + 10, cx1 - 10, step=12, r=2)
        cv2.putText(page, str(r), (cx0 + 15, ry0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)


def test_synthetic_pipeline_extraction_counts():
    pages = [_make_page() for _ in range(3)]
    _draw_fields(pages[0])

    table_bbox = Box(x=80, y=650, w=640, h=480)
    for p in pages:
        _draw_table(p, table_bbox, cols=5, header_plus_rows=17)

    # Fields (use dot detector directly to avoid bullet detector dependency in this test)
    dot = DotDetector()
    h0 = int(pages[0].shape[0])
    field_zone = (int(h0 * float(config.FIELD_ZONE[0])), int(h0 * float(config.FIELD_ZONE[1])))
    dotted = dot.detect_dotted_lines(pages[0], field_zone)
    fields = FieldExtractor(remove_dots=True).extract_field_regions(pages[0], dotted)

    # Tables (reuse target column across pages)
    det = TableDetector()
    ext = TableExtractor(remove_grid=True)
    col_range = None
    col_idx0 = None
    table_regions = []
    row_offset = 0
    for i, pg in enumerate(pages):
        struct = det.parse_table_structure(pg, table_bbox)
        if i == 0:
            col_range = struct.target_column
            col_idx0 = int(struct.target_column_index)
        else:
            struct = TableStructure(
                bbox=struct.bbox,
                grid_horizontal=struct.grid_horizontal,
                grid_vertical=struct.grid_vertical,
                target_column=col_range,
                target_column_index=int(col_idx0),
                rows=struct.rows,
                cols=struct.cols,
            )
        cells = ext.extract_target_column_cells(pg, struct, i, row_offset=row_offset, skip_header_rows=1)
        table_regions.extend(cells)
        row_offset += len(cells)

    assert len(fields) == 12
    assert len(table_regions) == 48
    assert len(fields) + len(table_regions) == 60