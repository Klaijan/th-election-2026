#!/usr/bin/env python3
"""List every file on Google Drive for all provinces via rclone.

Produces a master CSV with one row per file and a summary JSON.

Usage::

    python scripts/survey_gdrive_files.py [--workers 5] [--timeout 300]
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
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# tqdm fallback
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
# Region mapping
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
# Parsing
# ---------------------------------------------------------------------------

_FOLDER_ID_RE = re.compile(r"folders/([a-zA-Z0-9_-]+)")


def parse_province_links(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("folder_url") or "").strip()
            m = _FOLDER_ID_RE.search(url)
            rows.append({
                "province": row["province"].strip(),
                "folder_url": url,
                "folder_id": m.group(1) if m else "",
            })
    return rows


# ---------------------------------------------------------------------------
# rclone listing
# ---------------------------------------------------------------------------

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


def human_bytes(n: int) -> str:
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    exp = min(int(math.log(abs(n), 1024)), len(units) - 1)
    val = n / (1024 ** exp)
    if exp == 0:
        return f"{int(val)} B"
    return f"{val:.2f} {units[exp]}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="List every file on Google Drive for all provinces.",
    )
    ap.add_argument("--workers", type=int, default=5,
                    help="Parallel workers. Default: 5")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Timeout per province in seconds. Default: 600")
    ap.add_argument("--remote", default="gdrive:",
                    help="rclone remote. Default: gdrive:")
    ap.add_argument("--output-dir", default="data",
                    help="Output directory. Default: data/")
    ap.add_argument("--province-links", default="configs/province_links.csv")
    args = ap.parse_args(argv)

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
    tqdm_cls = _get_tqdm()
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
                    print("\r" + _fallback_progress(processed, len(scan_list)),
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


if __name__ == "__main__":
    raise SystemExit(main())
