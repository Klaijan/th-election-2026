#!/usr/bin/env python3
"""Survey Google Drive folder sizes for all provinces via rclone.

Scans province folders on Google Drive using ``rclone size --json`` and
produces a terminal report, with optional CSV and JSON outputs.

Usage::

    python scripts/survey_gdrive.py [--workers 10] [--output-dir data/] \
        [--csv] [--json] [--timeout 120]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# tqdm with fallback
# ---------------------------------------------------------------------------

def _get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def _fallback_progress(done: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    filled = int(round(width * (done / total)))
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


# ---------------------------------------------------------------------------
# Region mapping (6 regions, 77 provinces)
# ---------------------------------------------------------------------------

PROVINCE_TO_REGION: dict[str, str] = {}

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

for _region, _provs in _REGION_PROVINCES.items():
    for _p in _provs:
        PROVINCE_TO_REGION[_p] = _region


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ProvinceSize:
    province: str
    folder_id: str
    folder_url: str
    file_count: int = 0
    total_bytes: int = 0
    constituencies: int = 0
    region: str = ""
    error: str = ""
    anomalies: list[str] = field(default_factory=list)


@dataclass
class SurveyReport:
    metadata: dict = field(default_factory=dict)
    provinces: list[ProvinceSize] = field(default_factory=list)
    regions: dict[str, dict] = field(default_factory=dict)
    anomalies: list[dict] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    size_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FOLDER_ID_RE = re.compile(r"folders/([a-zA-Z0-9_-]+)")


def parse_province_links(path: Path) -> list[dict]:
    """Read province_links.csv (utf-8-sig BOM) and extract folder IDs."""
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            province = row["province"].strip()
            url = (row.get("folder_url") or "").strip()
            folder_id = ""
            if url:
                m = _FOLDER_ID_RE.search(url)
                if m:
                    folder_id = m.group(1)
            rows.append({
                "province": province,
                "folder_url": url,
                "folder_id": folder_id,
            })
    return rows


def load_constituency_counts(path: Path) -> dict[str, int]:
    """Parse cand_clean.csv and count unique constituencies per province."""
    const_ids: dict[str, set[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["const_id"].strip()
            province = cid.rsplit("-", 1)[0]
            const_ids.setdefault(province, set()).add(cid)
    return {prov: len(ids) for prov, ids in const_ids.items()}


# ---------------------------------------------------------------------------
# rclone
# ---------------------------------------------------------------------------

def rclone_size(folder_id: str, timeout: int = 120, remote: str = "gdrive:") -> dict:
    """Run ``rclone size --json`` on a Google Drive folder.

    Returns dict with ``count`` and ``bytes`` on success, or
    ``error`` key on failure.
    """
    try:
        result = subprocess.run(
            [
                "rclone", "size", "--json",
                remote,
                "--drive-root-folder-id", folder_id,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"error": "rclone not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}

    if result.returncode != 0:
        err = result.stderr.strip()[:200] or f"exit code {result.returncode}"
        return {"error": err}

    try:
        data = json.loads(result.stdout)
        return {"count": data.get("count", 0), "bytes": data.get("bytes", 0)}
    except (json.JSONDecodeError, KeyError) as exc:
        return {"error": f"bad JSON: {exc}"}


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(provinces: list[ProvinceSize]) -> list[dict]:
    """Flag anomalies: missing_link, duplicate_folder_id, empty_folder, size_outlier."""
    all_anomalies: list[dict] = []

    # Duplicate folder IDs
    id_to_provs: dict[str, list[str]] = {}
    for p in provinces:
        if p.folder_id:
            id_to_provs.setdefault(p.folder_id, []).append(p.province)
    for fid, provs in id_to_provs.items():
        if len(provs) > 1:
            for prov in provs:
                tag = "duplicate_folder_id"
                msg = f"Folder ID shared with: {', '.join(n for n in provs if n != prov)}"
                all_anomalies.append({"province": prov, "type": tag, "detail": msg})
                for ps in provinces:
                    if ps.province == prov and tag not in ps.anomalies:
                        ps.anomalies.append(tag)

    # Per-province checks
    for p in provinces:
        if not p.folder_id:
            tag = "missing_link"
            all_anomalies.append({"province": p.province, "type": tag, "detail": "No folder URL in CSV"})
            if tag not in p.anomalies:
                p.anomalies.append(tag)
        elif p.file_count == 0 and not p.error:
            tag = "empty_folder"
            all_anomalies.append({"province": p.province, "type": tag, "detail": "0 files returned"})
            if tag not in p.anomalies:
                p.anomalies.append(tag)

    # Size outlier (>2σ per-constituency bytes)
    per_const: list[float] = []
    for p in provinces:
        if p.constituencies > 0 and p.total_bytes > 0:
            per_const.append(p.total_bytes / p.constituencies)
    if len(per_const) >= 3:
        mean = sum(per_const) / len(per_const)
        variance = sum((x - mean) ** 2 for x in per_const) / len(per_const)
        std = math.sqrt(variance)
        if std > 0:
            for p in provinces:
                if p.constituencies > 0 and p.total_bytes > 0:
                    val = p.total_bytes / p.constituencies
                    if abs(val - mean) > 2 * std:
                        tag = "size_outlier"
                        direction = "high" if val > mean else "low"
                        sigma = (val - mean) / std
                        detail = f"{direction} outlier: {sigma:+.1f}σ per-constituency"
                        all_anomalies.append({"province": p.province, "type": tag, "detail": detail})
                        if tag not in p.anomalies:
                            p.anomalies.append(tag)

    return all_anomalies


# ---------------------------------------------------------------------------
# Formatting helpers
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


_SIZE_BUCKETS = [
    ("0 B", 0),
    ("< 500 MB", 500 * 1024 ** 2),
    ("< 1 GB", 1 * 1024 ** 3),
    ("< 5 GB", 5 * 1024 ** 3),
    ("< 10 GB", 10 * 1024 ** 3),
    ("< 20 GB", 20 * 1024 ** 3),
    ("20 GB+", float("inf")),
]


def _bucket_label(total_bytes: int) -> str:
    if total_bytes == 0:
        return "0 B"
    for label, upper in _SIZE_BUCKETS[1:]:
        if total_bytes < upper:
            return label
        if upper == float("inf"):
            return label
    return "20 GB+"


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def build_report(
    provinces: list[ProvinceSize],
    anomalies: list[dict],
) -> SurveyReport:
    """Build the full survey report from province data."""
    report = SurveyReport()
    report.provinces = provinces
    report.anomalies = anomalies

    # Metadata
    report.metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_provinces": len(provinces),
        "scanned": sum(1 for p in provinces if p.folder_id and not p.error),
        "errors": sum(1 for p in provinces if p.error),
        "skipped": sum(1 for p in provinces if not p.folder_id),
    }

    # Grand total — deduplicated by folder_id
    seen_ids: set[str] = set()
    dedup_files = 0
    dedup_bytes = 0
    for p in sorted(provinces, key=lambda x: x.total_bytes, reverse=True):
        if p.folder_id and p.folder_id not in seen_ids:
            dedup_files += p.file_count
            dedup_bytes += p.total_bytes
            seen_ids.add(p.folder_id)
    report.summary = {
        "total_files": dedup_files,
        "total_bytes": dedup_bytes,
        "total_human": human_bytes(dedup_bytes),
        "unique_folders": len(seen_ids),
    }

    # Per-region aggregation (also deduplicated)
    region_data: dict[str, dict] = {}
    region_seen_ids: dict[str, set[str]] = {}
    for p in provinces:
        r = p.region or "ไม่ระบุ"
        if r not in region_data:
            region_data[r] = {"files": 0, "bytes": 0, "provinces": 0}
            region_seen_ids[r] = set()
        region_data[r]["provinces"] += 1
        if p.folder_id and p.folder_id not in region_seen_ids[r]:
            region_data[r]["files"] += p.file_count
            region_data[r]["bytes"] += p.total_bytes
            region_seen_ids[r].add(p.folder_id)
    for r in region_data:
        region_data[r]["human"] = human_bytes(region_data[r]["bytes"])
    report.regions = region_data

    # Size distribution
    dist: dict[str, int] = {label: 0 for label, _ in _SIZE_BUCKETS}
    for p in provinces:
        dist[_bucket_label(p.total_bytes)] += 1
    report.size_distribution = dist

    return report


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def format_terminal_report(report: SurveyReport) -> str:
    """Build a multi-section ASCII report for the terminal."""
    lines: list[str] = []

    lines.append("")
    lines.append("=" * 80)
    lines.append("  Google Drive Survey Report")
    lines.append(f"  {report.metadata.get('timestamp', '')}")
    lines.append("=" * 80)

    # Grand total
    s = report.summary
    lines.append("")
    lines.append(f"  Grand Total (deduplicated): {s['total_human']}")
    lines.append(f"  Files: {s['total_files']:,}  |  Unique folders: {s['unique_folders']}")
    m = report.metadata
    lines.append(f"  Provinces: {m['total_provinces']}  |  Scanned: {m['scanned']}  |  "
                 f"Errors: {m['errors']}  |  Skipped: {m['skipped']}")

    # Per-province table
    lines.append("")
    lines.append("-" * 80)
    header = f"  {'Province':<25} {'Region':<14} {'Const':>5} {'Files':>7} {'Size':>10} {'Per-Const':>10} {'Anomaly'}"
    lines.append(header)
    lines.append("-" * 80)

    sorted_provs = sorted(report.provinces, key=lambda p: p.total_bytes, reverse=True)
    for p in sorted_provs:
        per_c = human_bytes(p.total_bytes // p.constituencies) if p.constituencies > 0 else "-"
        anom = ",".join(p.anomalies) if p.anomalies else ""
        if p.error:
            anom = f"ERR:{p.error[:20]}" if not anom else f"{anom},ERR"
        line = (
            f"  {p.province:<25} {p.region:<14} {p.constituencies:>5} "
            f"{p.file_count:>7,} {human_bytes(p.total_bytes):>10} {per_c:>10} {anom}"
        )
        lines.append(line)

    # Per-region aggregation
    lines.append("")
    lines.append("-" * 60)
    lines.append(f"  {'Region':<20} {'Provinces':>9} {'Files':>8} {'Size':>12}")
    lines.append("-" * 60)
    for region in ["ภาคกลาง", "ภาคตะวันออก", "ภาคอีสาน", "ภาคเหนือ", "ภาคตะวันตก", "ภาคใต้"]:
        rd = report.regions.get(region, {"provinces": 0, "files": 0, "human": "0 B"})
        lines.append(f"  {region:<20} {rd['provinces']:>9} {rd['files']:>8,} {rd['human']:>12}")

    # Anomalies
    if report.anomalies:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  Anomalies ({len(report.anomalies)}):")
        lines.append("-" * 60)
        for a in report.anomalies:
            lines.append(f"  [{a['type']}] {a['province']}: {a['detail']}")

    # Size distribution histogram
    lines.append("")
    lines.append("-" * 60)
    lines.append("  Size Distribution:")
    lines.append("-" * 60)
    max_count = max(report.size_distribution.values()) if report.size_distribution else 1
    for label, count in report.size_distribution.items():
        bar_len = int(30 * count / max(max_count, 1))
        bar = "#" * bar_len
        lines.append(f"  {label:<10} {bar:<30} {count}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File outputs
# ---------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via tmp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_csv_report(provinces: list[ProvinceSize], path: Path) -> None:
    """Write per-province CSV report with atomic write."""
    rows: list[dict] = []
    for p in sorted(provinces, key=lambda x: x.total_bytes, reverse=True):
        rows.append({
            "province": p.province,
            "region": p.region,
            "constituencies": p.constituencies,
            "folder_id": p.folder_id,
            "file_count": p.file_count,
            "total_bytes": p.total_bytes,
            "total_human": human_bytes(p.total_bytes),
            "per_constituency_bytes": p.total_bytes // p.constituencies if p.constituencies > 0 else 0,
            "anomalies": ";".join(p.anomalies),
            "error": p.error,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_tmp" + path.suffix)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_json_report(report: SurveyReport, path: Path) -> None:
    """Write full JSON report with atomic write."""
    data = {
        "metadata": report.metadata,
        "summary": report.summary,
        "regions": report.regions,
        "size_distribution": report.size_distribution,
        "anomalies": report.anomalies,
        "provinces": [asdict(p) for p in report.provinces],
    }
    text = json.dumps(data, ensure_ascii=False, indent=2)
    _atomic_write_text(path, text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Survey Google Drive folder sizes for all provinces via rclone.",
    )
    ap.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel rclone workers. Default: 10",
    )
    ap.add_argument(
        "--output-dir", default="data",
        help="Output directory for CSV/JSON reports. Default: data/",
    )
    ap.add_argument(
        "--csv", action="store_true",
        help="Write per-province CSV report.",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Write full JSON report.",
    )
    ap.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per rclone call in seconds. Default: 120",
    )
    ap.add_argument(
        "--remote", default="gdrive:",
        help="rclone remote name for Google Drive. Default: gdrive:",
    )
    ap.add_argument(
        "--province-links", default="configs/province_links.csv",
        help="Path to province_links.csv. Default: configs/province_links.csv",
    )
    ap.add_argument(
        "--cand-csv", default="data/cand_clean.csv",
        help="Path to cand_clean.csv. Default: data/cand_clean.csv",
    )
    args = ap.parse_args(argv)

    links_path = Path(args.province_links)
    cand_path = Path(args.cand_csv)
    output_dir = Path(args.output_dir)

    # --- Parse inputs ---
    if not links_path.exists():
        print(f"Error: province links not found: {links_path}", file=sys.stderr)
        return 1

    rows = parse_province_links(links_path)
    if not rows:
        print("Error: no provinces found in CSV", file=sys.stderr)
        return 1

    const_counts: dict[str, int] = {}
    if cand_path.exists():
        const_counts = load_constituency_counts(cand_path)
    else:
        print(f"Warning: cand_clean.csv not found: {cand_path}", file=sys.stderr)

    # --- Build initial ProvinceSize objects ---
    provinces: list[ProvinceSize] = []
    to_scan: list[ProvinceSize] = []
    for row in rows:
        ps = ProvinceSize(
            province=row["province"],
            folder_id=row["folder_id"],
            folder_url=row["folder_url"],
            constituencies=const_counts.get(row["province"], 0),
            region=PROVINCE_TO_REGION.get(row["province"], ""),
        )
        provinces.append(ps)
        if ps.folder_id:
            to_scan.append(ps)

    skipped = len(provinces) - len(to_scan)
    if skipped:
        print(f"Skipping {skipped} province(s) with no folder link.", file=sys.stderr)

    # --- Pre-scan anomaly: duplicate folder IDs ---
    id_counts: dict[str, int] = {}
    for ps in provinces:
        if ps.folder_id:
            id_counts[ps.folder_id] = id_counts.get(ps.folder_id, 0) + 1
    dups = {fid for fid, c in id_counts.items() if c > 1}
    if dups:
        dup_provs = [ps.province for ps in provinces if ps.folder_id in dups]
        print(f"Note: {len(dups)} shared folder ID(s) across: {', '.join(dup_provs)}",
              file=sys.stderr)

    # --- Parallel rclone scanning ---
    print(f"Scanning {len(to_scan)} folders with {args.workers} workers...",
          file=sys.stderr)

    tqdm_cls = _get_tqdm()
    pbar = tqdm_cls(total=len(to_scan), desc="rclone size", unit="prov") if tqdm_cls else None
    processed = 0
    last_fb = 0.0

    # Deduplicate: only scan each folder_id once
    unique_ids: dict[str, ProvinceSize] = {}
    dup_mapping: dict[str, str] = {}  # folder_id -> first province name
    for ps in to_scan:
        if ps.folder_id not in unique_ids:
            unique_ids[ps.folder_id] = ps
            dup_mapping[ps.folder_id] = ps.province
    scan_targets = list(unique_ids.values())
    dup_targets = [ps for ps in to_scan if ps.folder_id in unique_ids and
                   unique_ids[ps.folder_id] is not ps]

    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(rclone_size, ps.folder_id, args.timeout, args.remote): ps
            for ps in scan_targets
        }

        for fut in as_completed(futs):
            ps = futs[fut]
            result = fut.result()

            if "error" in result:
                ps.error = result["error"]
                failures.append((ps.province, ps.error))
            else:
                ps.file_count = result.get("count", 0)
                ps.total_bytes = result.get("bytes", 0)

            processed += 1
            if pbar:
                pbar.update(1)
            else:
                now = time.time()
                if processed == len(scan_targets) or (now - last_fb) >= 0.25:
                    print("\r" + _fallback_progress(processed, len(scan_targets)),
                          end="", flush=True, file=sys.stderr)
                    last_fb = now

    if pbar:
        pbar.close()
    else:
        if scan_targets:
            print(file=sys.stderr)  # newline after progress bar

    # Copy results to duplicate-mapped provinces
    for ps in dup_targets:
        source = unique_ids[ps.folder_id]
        ps.file_count = source.file_count
        ps.total_bytes = source.total_bytes
        ps.error = source.error

    # Also update the progress bar count for skipped dups
    if pbar and dup_targets:
        pass  # already closed

    # --- Anomaly detection ---
    anomalies = detect_anomalies(provinces)

    # --- Build report ---
    report = build_report(provinces, anomalies)

    # --- Terminal output ---
    print(format_terminal_report(report))

    # --- File outputs ---
    if args.csv:
        csv_path = output_dir / "survey_gdrive.csv"
        write_csv_report(provinces, csv_path)
        print(f"CSV written: {csv_path}", file=sys.stderr)

    if args.json:
        json_path = output_dir / "survey_gdrive.json"
        write_json_report(report, json_path)
        print(f"JSON written: {json_path}", file=sys.stderr)

    # --- Summary ---
    if failures:
        print(f"\n{len(failures)} error(s):", file=sys.stderr)
        for prov, msg in failures[:20]:
            print(f"  - {prov}: {msg}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
