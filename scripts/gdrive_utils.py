"""Shared utilities for Google Drive survey scripts.

Provides province-to-region mapping, CSV parsing, human-readable byte
formatting, tqdm with fallback, folder-ID regex, and atomic file writes.
"""

from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

__all__ = [
    "PROVINCE_TO_REGION",
    "REGION_NAMES",
    "FOLDER_ID_RE",
    "parse_province_links",
    "human_bytes",
    "get_tqdm",
    "fallback_progress",
    "atomic_write_text",
]

# ---------------------------------------------------------------------------
# Region mapping (6 regions, 77 provinces)
# ---------------------------------------------------------------------------

REGION_NAMES: list[str] = [
    "ภาคกลาง",
    "ภาคตะวันออก",
    "ภาคอีสาน",
    "ภาคเหนือ",
    "ภาคตะวันตก",
    "ภาคใต้",
]

_REGION_PROVINCES: dict[str, list[str]] = {
    "ภาคกลาง": [
        "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี",
        "พระนครศรีอยุธยา", "อ่างทอง", "ลพบุรี", "สิงห์บุรี",
        "ชัยนาท", "สระบุรี", "นครปฐม", "สมุทรสาคร", "สมุทรสงคราม",
    ],
    "ภาคตะวันออก": [
        "ชลบุรี", "ระยอง", "จันทบุรี", "ตราด",
        "ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", "สระแก้ว",
    ],
    "ภาคอีสาน": [
        "นครราชสีมา", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ",
        "อุบลราชธานี", "ยโสธร", "ชัยภูมิ", "อำนาจเจริญ",
        "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี",
        "เลย", "หนองคาย", "มหาสารคาม", "ร้อยเอ็ด",
        "กาฬสินธุ์", "สกลนคร", "นครพนม", "มุกดาหาร",
    ],
    "ภาคเหนือ": [
        "เชียงใหม่", "ลำพูน", "ลำปาง", "อุตรดิตถ์", "แพร่",
        "น่าน", "พะเยา", "เชียงราย", "แม่ฮ่องสอน",
        "นครสวรรค์", "อุทัยธานี", "กำแพงเพชร", "ตาก",
        "สุโขทัย", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์",
    ],
    "ภาคตะวันตก": [
        "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "เพชรบุรี",
        "ประจวบคีรีขันธ์",
    ],
    "ภาคใต้": [
        "นครศรีธรรมราช", "กระบี่", "พังงา", "ภูเก็ต",
        "สุราษฎร์ธานี", "ระนอง", "ชุมพร", "สงขลา",
        "สตูล", "ตรัง", "พัทลุง", "ปัตตานี", "ยะลา", "นราธิวาส",
    ],
}

PROVINCE_TO_REGION: dict[str, str] = {}
for _region, _provs in _REGION_PROVINCES.items():
    for _p in _provs:
        PROVINCE_TO_REGION[_p] = _region

# ---------------------------------------------------------------------------
# Folder-ID regex
# ---------------------------------------------------------------------------

FOLDER_ID_RE = re.compile(r"folders/([a-zA-Z0-9_-]+)")

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_province_links(path: Path) -> list[dict]:
    """Read province_links.csv (utf-8-sig BOM) and extract folder IDs."""
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            province = row["province"].strip()
            url = (row.get("folder_url") or "").strip()
            m = FOLDER_ID_RE.search(url)
            rows.append({
                "province": province,
                "folder_url": url,
                "folder_id": m.group(1) if m else "",
            })
    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def human_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    exp = min(int(math.log(abs(n), 1024)), len(units) - 1)
    val = n / (1024 ** exp)
    if exp == 0:
        return f"{int(val)} B"
    return f"{val:.2f} {units[exp]}"


# ---------------------------------------------------------------------------
# tqdm with fallback
# ---------------------------------------------------------------------------


def get_tqdm():
    """Return ``tqdm.tqdm`` if available, else ``None``."""
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def fallback_progress(done: int, total: int, width: int = 28) -> str:
    """Simple ASCII progress bar for when tqdm is not installed."""
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * (done / total)))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via tmp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
