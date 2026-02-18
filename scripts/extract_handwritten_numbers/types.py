from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in absolute pixel coordinates (x,y,w,h)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return int(self.x + self.w)

    @property
    def y2(self) -> int:
        return int(self.y + self.h)

    def clamp(self, *, width: int, height: int) -> "Box":
        x0 = max(0, min(int(self.x), int(width)))
        y0 = max(0, min(int(self.y), int(height)))
        x1 = max(0, min(int(self.x2), int(width)))
        y1 = max(0, min(int(self.y2), int(height)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return Box(x=x0, y=y0, w=max(0, x1 - x0), h=max(0, y1 - y0))

    def pad(self, *, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> "Box":
        return Box(x=int(self.x - left), y=int(self.y - top), w=int(self.w + left + right), h=int(self.h + top + bottom))

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.x), int(self.y), int(self.x2), int(self.y2))


@dataclass(frozen=True)
class DottedLine:
    y: int
    x_start: int
    x_end: int
    length: int
    dot_count: int
    dots: list[tuple[int, int]]


@dataclass(frozen=True)
class ExtractedRegion:
    region_id: str
    bbox: Box
    image: np.ndarray  # preprocessed
    raw_image: np.ndarray  # raw crop for debugging
    meta: dict[str, Any]


@dataclass(frozen=True)
class TableStructure:
    bbox: Box  # table bbox in page coords
    grid_horizontal: list[int]  # absolute y positions
    grid_vertical: list[int]  # absolute x positions
    target_column: tuple[int, int]  # absolute (x_start, x_end)
    target_column_index: int  # 0-indexed
    rows: int
    cols: int


@dataclass(frozen=True)
class OCRItem:
    image_id: str
    text: str
    raw_text: str
    confidence: float
    provider: Literal["google", "gemini", "tesseract", "unknown"] = "unknown"


@dataclass(frozen=True)
class ValidationOutcome:
    value: str
    confidence: float
    status: Literal["auto_accepted", "review_queue", "manual_entry"]
    validation: dict[str, str]
    bbox: Optional[Box] = None
    source: Optional[str] = None


