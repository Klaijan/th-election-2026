from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


log = logging.getLogger("extract_handwritten_numbers")


def setup_logging(*, level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = time.perf_counter()
        self.dt: Optional[float] = None

    def __enter__(self) -> "Timer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dt = time.perf_counter() - self.t0


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_image(path: str | Path, img: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(p), img)
    if not ok:
        raise IOError(f"Failed to write image: {p}")


def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return {k: to_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, np.ndarray):
        return {"_type": "ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def save_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=indent), encoding="utf-8")


def env_flag(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "yes", "y", "on"}


