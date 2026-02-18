"""
Central configuration for the OCR pipeline.

All "magic numbers" live here so tuning for new scan batches is easy.
"""

from __future__ import annotations

# -------------------------
# PDF conversion
# -------------------------
PDF_DPI: int = 400

# -------------------------
# Zone definitions (fractions of page height)
# -------------------------
# Note: header/fields percent zones are no longer used (zone-1 is inferred via templates 4/5).
# Keep only table zone for page-1 table guidance.
TABLE_ZONE = (0.50, 1.0)  # table start and continuation
# Backward-compatible synthetic test constant.
FIELD_ZONE = (0.15, 0.50)

# -------------------------
# Zone 1 (fields) y-range templates (page 1)
# -------------------------
ZONE1_TEMPLATE_FILES: tuple[str, str] = ("template_4.png", "template_5.png")
ZONE1_TEMPLATE_THRESHOLD: float = 0.60
ZONE1_TEMPLATE_SCALES: tuple[float, float, float] = (0.90, 1.00, 1.10)
ZONE1_TEMPLATE_PAD_PX: int = 80
ZONE1_TEMPLATE_SEARCH_X_FRAC: float = 0.5  # search left 50% of the page (templates 4/5 appear on left half)
ZONE1_TEMPLATE_SEARCH_Y_FRAC: float = 0.65  # search top 65% of the page

# -------------------------
# Region templates (fields zone for first pages with logo)
# -------------------------
# These anchors define the top/bottom of the "fields" region on first pages.
REGION_TEMPLATE_FILES_1: tuple[str, str] = ("region_begin.png", "region_end.png")
REGION_TEMPLATE_FILES_2: tuple[str, str] = ("region_begin_2.png", "region_end_2.png")
REGION_TEMPLATE_THRESHOLD: float = 0.70
REGION_TEMPLATE_SCALES: tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20)
REGION_TEMPLATE_PAD_PX: int = 60
REGION_TEMPLATE_SEARCH_Y_FRAC: float = 0.65  # search top 65% of the page
REGION_TEMPLATE_SEARCH_X_FRAC: float = 1.0   # search full width by default

# -------------------------
# Dot detection
# -------------------------
DOT_MIN_DIAMETER: int = 3
DOT_MAX_DIAMETER: int = 8
DOT_SPACING_MIN: int = 5
DOT_SPACING_MAX: int = 15
DOT_SEQUENCE_MIN: int = 5
DOT_LINE_MIN_LENGTH: int = 100

DOT_Y_CLUSTER_TOLERANCE_PX: int = 3  # rotation/scan tolerance for horizontal alignment
DOT_CIRCULARITY_MIN: float = 0.70
DOT_ALLOW_GAP_MULTIPLIER: float = 2.2  # allow occasional missing dots
DOT_MAX_GAPS_PER_LINE: int = 2

# -------------------------
# Extraction box around dots
# -------------------------
EXTRACTION_PAD_LEFT_RIGHT: int = 10
EXTRACTION_PAD_TOP: int = 60   # capture more handwriting above dots
EXTRACTION_PAD_BOTTOM: int = 60  # include overlap below/at dots if needed
# Kept for backwards-compat; actual crop height is PAD_TOP + PAD_BOTTOM in FieldExtractor.
EXTRACTION_TOTAL_HEIGHT: int = 120

# -------------------------
# Table extraction (target column only)
# -------------------------
TABLE_TARGET_COLUMN: str = "last"  # supported: "last" or "index"
TABLE_TARGET_COLUMN_INDEX: int = 2  # used only when TABLE_TARGET_COLUMN == "index" (0-indexed)
TABLE_COLUMN_ALIGNMENT_TOLERANCE: int = 20  # px tolerance across pages
TABLE_GRID_MERGE_TOLERANCE: int = 10  # merge nearby grid lines within this many px

# Morphology kernels (scaled dynamically; these are minimums)
TABLE_MIN_HORIZ_KERNEL: int = 40
TABLE_MIN_VERT_KERNEL: int = 40

# -------------------------
# OCR
# -------------------------
OCR_PROVIDER: str = "google"
OCR_LANGUAGES: list[str] = ["th", "en"]
OCR_MODEL: str = "gemini-3-flash"
OCR_RETRIES: int = 3
OCR_TIMEOUT_S: float = 90.0
OCR_CONFIDENCE_THRESHOLD: float = 0.70

# -------------------------
# Validation rules
# -------------------------
FIELD_VALUE_RANGE = (0, 10_000)
TABLE_VALUE_RANGE = (0, 1_000)

# -------------------------
# Debugging
# -------------------------
DEBUG_SAVE_INTERMEDIATES: bool = False

# -------------------------
# Logo detection (multi-document PDFs)
# -------------------------
# Place a grayscale PNG template under extract_handwritten_numbers/templates/
LOGO_TEMPLATE_FILE: str = "logo.png"
# Detect logo only in top band of the page (fraction of height)
LOGO_SEARCH_Y_FRAC: float = 0.20
# Template match threshold (TM_CCOEFF_NORMED)
LOGO_DETECTION_THRESHOLD: float = 0.70
# Multi-scale matching (±20% size variation)
# NOTE: In practice, template DPI/cropping can differ from rendered page DPI.
# Keep a wider scale sweep so we can still detect the crest logo reliably.
LOGO_SCALES: tuple[float, ...] = (0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.75, 1.90, 2.00)

# Only enable logo-based multi-document split for PDFs with at least this many pages.
MIN_PAGES_FOR_LOGO_DETECTION: int = 3
# Always treat page 0 as a first page (safety).
ALWAYS_TREAT_PAGE0_AS_FIRST: bool = True
# If two detected logos are too close, treat the later one as suspicious (helps false positives).
LOGO_MIN_PAGE_GAP: int = 1


