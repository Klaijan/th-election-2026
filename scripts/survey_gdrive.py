#!/usr/bin/env python3
"""Survey Google Drive province folders via rclone.

Unified CLI with three subcommands:

* **size**   — measure folder sizes (``rclone size --json``)
* **files**  — list every file (``rclone lsjson -R``)
* **owners** — identify folder owners (``rclone lsjson -M``)

Usage::

    python scripts/survey_gdrive.py size  [--csv] [--json] [--cand-csv ...]
    python scripts/survey_gdrive.py files [--workers 5] [--timeout 600]
    python scripts/survey_gdrive.py owners
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from gdrive_utils import (
    PROVINCE_TO_REGION,
    REGION_NAMES,
    atomic_write_text,
    fallback_progress,
    get_tqdm,
    human_bytes,
    parse_province_links,
)


# ---------------------------------------------------------------------------
# Data models (size command)
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


# ===================================================================
# Size command helpers
# ===================================================================

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
# rclone: size
# ---------------------------------------------------------------------------

def rclone_size(folder_id: str, timeout: int = 120, remote: str = "gdrive:") -> dict:
    """Run ``rclone size --json`` on a Google Drive folder."""
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
# Anomaly detection (size command)
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
# Size formatting helpers
# ---------------------------------------------------------------------------

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
# Size report building
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
# Size terminal report
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
    for region in REGION_NAMES:
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
# Size file outputs
# ---------------------------------------------------------------------------

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
    atomic_write_text(path, text)


# ===================================================================
# Files command helpers
# ===================================================================

def rclone_list_files(
    province: str, folder_id: str, timeout: int, remote: str,
) -> tuple[str, list[dict], str]:
    """List all files recursively. Returns (province, file_list, error)."""
    if not folder_id:
        return province, [], "no_link"
    try:
        result = subprocess.run(
            [
                "rclone", "lsjson", "-R", "-M", "--files-only",
                remote, "--drive-root-folder-id", folder_id,
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return province, [], "rclone not found"
    except subprocess.TimeoutExpired:
        return province, [], f"timeout after {timeout}s"

    if result.returncode != 0:
        return province, [], result.stderr.strip()[:200]

    try:
        items = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return province, [], f"bad JSON: {e}"

    return province, items, ""


# ===================================================================
# Owners command helpers
# ===================================================================

def rclone_folder_owner(
    folder_id: str,
    remote: str = "gdrive:",
    timeout: int = 60,
) -> tuple[str, str]:
    """Return (owner_email, error) for a Google Drive folder."""
    if not folder_id:
        return "", "no_link"
    for flag in ["--dirs-only", "--files-only"]:
        try:
            result = subprocess.run(
                [
                    "rclone", "lsjson", "-M", "--max-depth", "1", flag,
                    remote, "--drive-root-folder-id", folder_id,
                ],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                items = json.loads(result.stdout)
                if items:
                    owner = items[0].get("Metadata", {}).get("owner", "unknown")
                    return owner, ""
        except FileNotFoundError:
            return "", "rclone not found"
        except subprocess.TimeoutExpired:
            return "", f"timeout after {timeout}s"
        except Exception as e:
            return "", str(e)[:100]
    return "", "empty_folder"


def classify_email(email: str) -> str:
    """Classify an email into account type."""
    if not email:
        return "error"
    if email.endswith("@ect.go.th"):
        return "official_ect"
    if "ect" in email.lower():
        return "ect_gmail"
    if email.endswith(("@gmail.com", "@hotmail.com", "@yahoo.com")):
        return "personal"
    return "institutional"


# ===================================================================
# Subcommand: size
# ===================================================================

def cmd_size(args: argparse.Namespace) -> int:
    """Run folder-size survey."""
    links_path = Path(args.province_links)
    cand_path = Path(args.cand_csv)
    output_dir = Path(args.output_dir)

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

    tqdm_cls = get_tqdm()
    pbar = tqdm_cls(total=len(to_scan), desc="rclone size", unit="prov") if tqdm_cls else None
    processed = 0
    last_fb = 0.0

    # Deduplicate: only scan each folder_id once
    unique_ids: dict[str, ProvinceSize] = {}
    for ps in to_scan:
        if ps.folder_id not in unique_ids:
            unique_ids[ps.folder_id] = ps
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
                    print("\r" + fallback_progress(processed, len(scan_targets)),
                          end="", flush=True, file=sys.stderr)
                    last_fb = now

    if pbar:
        pbar.close()
    else:
        if scan_targets:
            print(file=sys.stderr)

    # Copy results to duplicate-mapped provinces
    for ps in dup_targets:
        source = unique_ids[ps.folder_id]
        ps.file_count = source.file_count
        ps.total_bytes = source.total_bytes
        ps.error = source.error

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


# ===================================================================
# Subcommand: files
# ===================================================================

def cmd_files(args: argparse.Namespace) -> int:
    """Run full file listing for all provinces."""
    links_path = Path(args.province_links)
    output_dir = Path(args.output_dir)

    if not links_path.exists():
        print(f"Error: {links_path} not found", file=sys.stderr)
        return 1

    rows = parse_province_links(links_path)
    if not rows:
        print("Error: no provinces in CSV", file=sys.stderr)
        return 1

    # Deduplicate by folder_id
    unique: dict[str, dict] = {}
    dup_map: dict[str, list[str]] = {}  # folder_id -> list of provinces
    for r in rows:
        fid = r["folder_id"]
        if not fid:
            unique[r["province"]] = r
        elif fid not in dup_map:
            dup_map[fid] = [r["province"]]
            unique[r["province"]] = r
        else:
            dup_map[fid].append(r["province"])

    scan_list = list(unique.values())
    print(f"Scanning {len(scan_list)} folders ({len(rows)} provinces) "
          f"with {args.workers} workers, timeout {args.timeout}s ...",
          file=sys.stderr)

    # --- Parallel scan ---
    tqdm_cls = get_tqdm()
    pbar = (tqdm_cls(total=len(scan_list), desc="listing", unit="prov")
            if tqdm_cls else None)
    processed = 0
    last_fb = 0.0

    results: dict[str, tuple[list[dict], str]] = {}
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                rclone_list_files,
                r["province"], r["folder_id"], args.timeout, args.remote,
            ): r
            for r in scan_list
        }
        for fut in as_completed(futs):
            province, files, error = fut.result()
            results[province] = (files, error)
            if error:
                failures.append((province, error))

            processed += 1
            if pbar:
                pbar.update(1)
            else:
                now = time.time()
                if processed == len(scan_list) or (now - last_fb) >= 0.25:
                    print("\r" + fallback_progress(processed, len(scan_list)),
                          end="", flush=True, file=sys.stderr)
                    last_fb = now

    if pbar:
        pbar.close()
    else:
        if scan_list:
            print(file=sys.stderr)

    # --- Write master CSV ---
    csv_path = output_dir / "gdrive_files.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_name(csv_path.stem + "_tmp" + csv_path.suffix)

    fieldnames = [
        "province", "region", "owner", "path", "name",
        "size_bytes", "size_human", "mime_type", "modified", "created",
        "file_id", "folder_id",
    ]

    total_files = 0
    total_bytes = 0
    owner_set: set[str] = set()
    province_stats: list[dict] = []

    try:
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in rows:
                province = r["province"]
                fid = r["folder_id"]
                region = PROVINCE_TO_REGION.get(province, "")

                # Get results — may come from deduplicated scan
                if province in results:
                    files, error = results[province]
                else:
                    # Shared folder — copy from first province
                    source = next(
                        (p for p, rr in results.items()
                         if any(row["folder_id"] == fid and row["province"] == p
                                for row in scan_list)),
                        None,
                    )
                    if source:
                        files, error = results[source]
                    else:
                        files, error = [], "shared_folder_not_found"

                prov_files = 0
                prov_bytes = 0

                for item in files:
                    meta = item.get("Metadata", {})
                    size = item.get("Size", 0)
                    owner = meta.get("owner", "")
                    owner_set.add(owner)

                    writer.writerow({
                        "province": province,
                        "region": region,
                        "owner": owner,
                        "path": item.get("Path", ""),
                        "name": item.get("Name", ""),
                        "size_bytes": size,
                        "size_human": human_bytes(size),
                        "mime_type": meta.get("content-type", item.get("MimeType", "")),
                        "modified": meta.get("mtime", item.get("ModTime", "")),
                        "created": meta.get("btime", ""),
                        "file_id": item.get("ID", ""),
                        "folder_id": fid,
                    })
                    prov_files += 1
                    prov_bytes += size

                total_files += prov_files
                total_bytes += prov_bytes
                province_stats.append({
                    "province": province,
                    "region": region,
                    "folder_id": fid,
                    "owner": files[0].get("Metadata", {}).get("owner", "") if files else "",
                    "file_count": prov_files,
                    "total_bytes": prov_bytes,
                    "total_human": human_bytes(prov_bytes),
                    "error": error,
                })

        os.replace(str(tmp_csv), str(csv_path))
    except Exception:
        tmp_csv.unlink(missing_ok=True)
        raise

    # --- Write summary JSON ---
    json_path = output_dir / "gdrive_files_summary.json"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_provinces": len(rows),
        "scanned": len(scan_list) - len(failures),
        "errors": len(failures),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_human": human_bytes(total_bytes),
        "unique_owners": sorted(owner_set - {""}),
        "provinces": province_stats,
        "failures": [{"province": p, "error": e} for p, e in failures],
    }
    tmp_json = json_path.with_name(json_path.stem + "_tmp" + json_path.suffix)
    try:
        tmp_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp_json), str(json_path))
    except Exception:
        tmp_json.unlink(missing_ok=True)
        raise

    # --- Terminal report ---
    print(f"\n{'='*70}")
    print(f"  Full File Listing Report")
    print(f"  {summary['timestamp']}")
    print(f"{'='*70}")
    print(f"\n  Total: {total_files:,} files, {human_bytes(total_bytes)}")
    print(f"  Scanned: {summary['scanned']}/{len(rows)} provinces")
    print(f"  Unique owners: {len(summary['unique_owners'])}")
    print(f"\n{'-'*70}")
    print(f"  {'Province':<25} {'Owner':<35} {'Files':>7} {'Size':>10}")
    print(f"{'-'*70}")
    for ps in sorted(province_stats, key=lambda x: x["total_bytes"], reverse=True):
        owner = ps["owner"][:33] if ps["owner"] else ps["error"][:33] if ps["error"] else ""
        print(f"  {ps['province']:<25} {owner:<35} {ps['file_count']:>7,} {ps['total_human']:>10}")

    print(f"\n  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"{'='*70}")

    if failures:
        print(f"\n{len(failures)} error(s):", file=sys.stderr)
        for p, e in failures:
            print(f"  - {p}: {e}", file=sys.stderr)

    return 1 if failures else 0


# ===================================================================
# Subcommand: owners
# ===================================================================

def cmd_owners(args: argparse.Namespace) -> int:
    """Run folder ownership forensics."""
    links_path = Path(args.province_links)
    output_dir = Path(args.output_dir)

    if not links_path.exists():
        print(f"Error: {links_path} not found", file=sys.stderr)
        return 1

    rows = parse_province_links(links_path)
    if not rows:
        print("Error: no provinces in CSV", file=sys.stderr)
        return 1

    # Deduplicate by folder_id
    seen_ids: dict[str, dict] = {}
    scan_tasks: list[dict] = []
    for r in rows:
        fid = r["folder_id"]
        if not fid:
            scan_tasks.append(r)
        elif fid not in seen_ids:
            seen_ids[fid] = r
            scan_tasks.append(r)

    print(f"Scanning {len(scan_tasks)} folders with {args.workers} workers...",
          file=sys.stderr)

    # --- Parallel scan ---
    tqdm_cls = get_tqdm()
    pbar = (tqdm_cls(total=len(scan_tasks), desc="owners", unit="prov")
            if tqdm_cls else None)
    processed = 0

    results: dict[str, tuple[str, str]] = {}  # key -> (owner, error)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                rclone_folder_owner, r["folder_id"], args.remote, args.timeout,
            ): r
            for r in scan_tasks
        }
        for fut in as_completed(futs):
            r = futs[fut]
            owner, err = fut.result()
            key = r["folder_id"] or r["province"]
            results[key] = (owner, err)

            processed += 1
            if pbar:
                pbar.update(1)
            else:
                print("\r" + fallback_progress(processed, len(scan_tasks)),
                      end="", flush=True, file=sys.stderr)

    if pbar:
        pbar.close()
    elif scan_tasks:
        print(file=sys.stderr)

    # --- Build full report ---
    report: list[dict] = []
    for r in rows:
        fid = r["folder_id"]
        key = fid if fid and fid in results else r["province"]
        owner, err = results.get(key, ("", "not_scanned"))
        report.append({**r, "owner": owner, "error": err})

    # --- Terminal report ---
    sep = "=" * 75
    dash = "-" * 75

    print(f"\n{sep}")
    print("  FOLDER OWNERSHIP FORENSICS REPORT")
    print(f"  {links_path} — {len(report)} provinces")
    print(sep)

    # Type breakdown
    type_counts: dict[str, int] = {}
    for r in report:
        t = classify_email(r["owner"])
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n  Account type breakdown:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {c:>3}")

    # Anomalies
    print(f"\n{dash}")
    print("  ANOMALIES")
    print(dash)

    # Shared folders
    fid_provs: dict[str, list[str]] = {}
    for r in report:
        if r["folder_id"]:
            fid_provs.setdefault(r["folder_id"], []).append(r["province"])
    for fid, provs in fid_provs.items():
        if len(provs) > 1:
            print(f"  [SHARED_FOLDER] {' + '.join(provs)}")
            print(f"                  folder_id: {fid}")

    for r in report:
        t = classify_email(r["owner"])
        if t == "personal":
            print(f"  [PERSONAL]      {r['province']}: {r['owner']}")
        elif t == "official_ect":
            print(f"  [OFFICIAL_GOV]  {r['province']}: {r['owner']}")
        elif t == "institutional":
            print(f"  [INSTITUTIONAL] {r['province']}: {r['owner']}")

    for r in report:
        if r["error"]:
            print(f"  [ERROR]         {r['province']}: {r['error']}")

    # Full table
    print(f"\n{dash}")
    print(f"  {'#':<3} {'Province':<25} {'Owner':<40} {'Type'}")
    print(dash)
    for i, r in enumerate(report, 1):
        etype = classify_email(r["owner"])
        owner = r["owner"][:38] if r["owner"] else f"[{r['error']}]"
        flag = " *" if etype == "personal" else ""
        print(f"  {i:<3} {r['province']:<25} {owner:<40} {etype}{flag}")
    print(sep)

    # --- CSV output ---
    csv_path = output_dir / "folder_owners.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "province", "folder_owner", "email_type",
            "folder_id", "folder_url", "error",
        ])
        writer.writeheader()
        for r in report:
            writer.writerow({
                "province": r["province"],
                "folder_owner": r["owner"],
                "email_type": classify_email(r["owner"]),
                "folder_id": r["folder_id"],
                "folder_url": r["folder_url"],
                "error": r["error"],
            })
    print(f"\n  CSV saved: {csv_path}")

    return 0


# ===================================================================
# CLI: argparse with subparsers
# ===================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Survey Google Drive province folders via rclone.",
    )

    # Shared arguments on parent parser
    ap.add_argument(
        "--workers", type=int, default=10,
        help="Number of parallel rclone workers (default: 10)",
    )
    ap.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per rclone call in seconds (default varies by subcommand)",
    )
    ap.add_argument(
        "--remote", default="gdrive:",
        help="rclone remote name (default: gdrive:)",
    )
    ap.add_argument(
        "--province-links", default="configs/province_links.csv",
        help="Path to province_links.csv (default: configs/province_links.csv)",
    )
    ap.add_argument(
        "--output-dir", default="data",
        help="Output directory (default: data/)",
    )

    sub = ap.add_subparsers(dest="command", required=True)

    # --- size ---
    sp_size = sub.add_parser("size", help="Measure folder sizes (rclone size --json)")
    sp_size.add_argument("--csv", action="store_true", help="Write per-province CSV report")
    sp_size.add_argument("--json", action="store_true", help="Write full JSON report")
    sp_size.add_argument(
        "--cand-csv", default="data/cand_clean.csv",
        help="Path to cand_clean.csv (default: data/cand_clean.csv)",
    )

    # --- files ---
    sp_files = sub.add_parser("files", help="List every file (rclone lsjson -R)")

    # --- owners ---
    sp_owners = sub.add_parser("owners", help="Identify folder owners (rclone lsjson -M)")

    args = ap.parse_args(argv)

    # Apply per-subcommand defaults when the user hasn't explicitly set them
    if args.command == "files":
        # Re-parse to check if user explicitly passed --timeout / --workers
        if "--timeout" not in (argv or sys.argv[1:]):
            args.timeout = 600
        if "--workers" not in (argv or sys.argv[1:]):
            args.workers = 5
    elif args.command == "owners":
        if "--timeout" not in (argv or sys.argv[1:]):
            args.timeout = 60

    dispatch = {
        "size": cmd_size,
        "files": cmd_files,
        "owners": cmd_owners,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
