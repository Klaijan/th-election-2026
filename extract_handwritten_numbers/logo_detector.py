from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

log = logging.getLogger("extract_handwritten_numbers")


@dataclass(frozen=True)
class LogoHit:
    has_logo: bool
    confidence: float
    position: tuple[int, int]  # (cx, cy) absolute in page coords
    bbox: tuple[int, int, int, int]  # (x, y, w, h) absolute in page coords
    scale: float


class LogoDetector:
    """
    Detect the government logo/header on a page to identify "first pages" of forms.
    Uses template matching in the top header band for speed and robustness.
    """

    def __init__(
        self,
        *,
        template_path: Optional[str | Path] = None,
        threshold: Optional[float] = None,
        search_y_frac: Optional[float] = None,
        scales: Optional[tuple[float, ...]] = None,
    ) -> None:
        templates_dir = Path(__file__).resolve().parent / "templates"
        default_template = templates_dir / str(getattr(config, "LOGO_TEMPLATE_FILE", "logo.png"))

        def _rewrite_legacy_template_path(p: Path) -> Path:
            """
            Backwards-compat: older revisions used templates under:
              scripts/extract_handwritten_numbers/templates/
            The library now lives at:
              extract_handwritten_numbers/templates/

            If a caller/environment still passes the legacy path, rewrite it so the
            logo detector keeps working after the refactor.
            """
            parts = p.parts
            # If the path ends with .../scripts/extract_handwritten_numbers/templates/<file>
            if len(parts) >= 4 and parts[-4:-1] == ("scripts", "extract_handwritten_numbers", "templates"):
                return templates_dir / parts[-1]

            # Otherwise, map the first occurrence of scripts/extract_handwritten_numbers/templates/<file>
            # to this package's templates directory.
            for i in range(len(parts) - 3):
                if parts[i : i + 3] == ("scripts", "extract_handwritten_numbers", "templates") and (i + 3) < len(parts):
                    return templates_dir / parts[i + 3]

            return p

        if template_path is not None:
            self.template_path = _rewrite_legacy_template_path(Path(template_path))
        else:
            self.template_path = _rewrite_legacy_template_path(default_template)
        self.threshold = float(threshold if threshold is not None else getattr(config, "LOGO_DETECTION_THRESHOLD", 0.70))
        self.search_y_frac = float(search_y_frac if search_y_frac is not None else getattr(config, "LOGO_SEARCH_Y_FRAC", 0.20))
        if scales is not None:
            self.scales = tuple(float(s) for s in scales)
        else:
            # Prefer a range sweep (min,max,step) to match debug_template_match.py behavior.
            rng = getattr(config, "LOGO_SCALE_RANGE", None)
            parsed: list[float] = []
            if isinstance(rng, (tuple, list)) and len(rng) == 3:
                try:
                    a, b, step = float(rng[0]), float(rng[1]), float(rng[2])
                    if step > 0 and a > 0 and b >= a:
                        x = a
                        for _ in range(2000):
                            if x > b + 1e-9:
                                break
                            parsed.append(float(x))
                            x += step
                except Exception:
                    parsed = []
            if not parsed:
                parsed = [float(s) for s in getattr(config, "LOGO_SCALES", (0.80, 0.90, 1.00, 1.10, 1.20))]
            # Keep scale list bounded to avoid pathological perf if misconfigured.
            self.scales = tuple(parsed[:200])

        self._tmpl_gray: Optional[np.ndarray] = None
        self._tmpl_ds_cache: dict[float, np.ndarray] = {}

    def _load_template(self) -> Optional[np.ndarray]:
        if self._tmpl_gray is not None:
            return self._tmpl_gray
        if not self.template_path.exists():
            log.info("Logo template not found: %s (logo detection disabled)", str(self.template_path))
            self._tmpl_gray = None
            return None
        t = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
        if t is None or t.size == 0:
            log.warning("Failed to load logo template: %s (logo detection disabled)", str(self.template_path))
            self._tmpl_gray = None
            return None
        self._tmpl_gray = t
        return t

    def _load_template_downscaled(self, downscale: float) -> Optional[np.ndarray]:
        tmpl = self._load_template()
        if tmpl is None:
            return None
        ds = float(downscale)
        ds = max(0.05, min(ds, 1.0))
        if ds in self._tmpl_ds_cache:
            return self._tmpl_ds_cache[ds]
        if ds == 1.0:
            self._tmpl_ds_cache[ds] = tmpl
            return tmpl
        tw = max(8, int(round(tmpl.shape[1] * ds)))
        th = max(8, int(round(tmpl.shape[0] * ds)))
        t2 = cv2.resize(tmpl, (int(tw), int(th)), interpolation=cv2.INTER_AREA)
        self._tmpl_ds_cache[ds] = t2
        return t2

    def detect_logo(self, page_bgr: np.ndarray) -> dict:
        """
        Returns a JSON-friendly dict:
          {
            'has_logo': bool,
            'confidence': float,
            'position': (x, y),
            'bbox': (x, y, w, h),
            'scale': float,
          }
        """
        ds = float(getattr(config, "LOGO_MATCH_DOWNSCALE", 0.25))
        tmpl = self._load_template_downscaled(ds)
        if tmpl is None:
            return {"has_logo": False, "confidence": 0.0, "position": (0, 0), "bbox": (0, 0, 0, 0), "scale": 1.0}

        h, w = page_bgr.shape[:2]
        y_end = int(round(float(h) * max(0.05, min(float(self.search_y_frac), 0.50))))
        y_end = max(1, min(y_end, h))
        # Optional X-range restriction for speed/stability
        x0, x1 = 0, int(w)
        xr = getattr(config, "LOGO_SEARCH_X_RANGE_FRAC", None)
        if isinstance(xr, (tuple, list)) and len(xr) == 2:
            try:
                a, b = float(xr[0]), float(xr[1])
                a = max(0.0, min(a, 1.0))
                b = max(0.0, min(b, 1.0))
                if b < a:
                    a, b = b, a
                x0 = int(round(a * w))
                x1 = int(round(b * w))
            except Exception:
                x0, x1 = 0, int(w)
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))

        roi = page_bgr[0:y_end, x0:x1]
        if roi.size == 0:
            return {"has_logo": False, "confidence": 0.0, "position": (0, 0), "bbox": (0, 0, 0, 0), "scale": 1.0}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ds = max(0.05, min(float(ds), 1.0))
        if ds != 1.0:
            gray = cv2.resize(
                gray,
                (max(32, int(round(gray.shape[1] * ds))), max(32, int(round(gray.shape[0] * ds)))),
                interpolation=cv2.INTER_AREA,
            )

        best_score = float("-inf")
        best_xy = (0, 0)
        best_wh = (0, 0)
        best_scale = 1.0

        for s in self.scales:
            s = float(s)
            if s <= 0:
                continue
            tw = max(8, int(round(tmpl.shape[1] * s)))
            th = max(8, int(round(tmpl.shape[0] * s)))
            if tw >= gray.shape[1] or th >= gray.shape[0]:
                continue
            t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
            if float(maxv) > float(best_score):
                best_score = float(maxv)
                best_xy = (int(maxl[0]), int(maxl[1]))
                best_wh = (int(tw), int(th))
                best_scale = float(s)

        x, y = best_xy
        tw, th = best_wh
        conf = float(best_score) if best_score != float("-inf") else 0.0
        has_logo = bool(conf >= float(self.threshold)) and (tw > 0 and th > 0)
        # Convert coords back to original page scale + offset
        if ds != 1.0:
            x = int(round(x / ds))
            y = int(round(y / ds))
            tw = int(round(tw / ds))
            th = int(round(th / ds))
        cx = int(x + tw // 2) + int(x0)
        cy = int(y + th // 2)

        return {
            "has_logo": bool(has_logo),
            "confidence": float(conf if has_logo else conf),
            "position": (int(cx), int(cy)),  # ROI==page coords for y, since ROI starts at 0
            "bbox": (int(x + int(x0)), int(y), int(tw), int(th)),
            "scale": float(best_scale),
        }


