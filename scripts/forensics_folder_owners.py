#!/usr/bin/env python3
"""Forensics: identify the owner of every Google Drive folder in province_links.csv."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

FOLDER_ID_RE = re.compile(r"folders/([a-zA-Z0-9_-]+)")


def parse_links(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            url = (r.get("folder_url") or "").strip()
            m = FOLDER_ID_RE.search(url)
            rows.append({
                "province": r["province"].strip(),
                "folder_id": m.group(1) if m else "",
                "url": url,
            })
    return rows


def get_owner(province: str, folder_id: str) -> tuple[str, str, str, str]:
    if not folder_id:
        return province, folder_id, "", "no_link"
    for flag in ["--dirs-only", "--files-only"]:
        try:
            result = subprocess.run(
                ["rclone", "lsjson", "-M", "--max-depth", "1", flag,
                 "gdrive:", "--drive-root-folder-id", folder_id],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                items = json.loads(result.stdout)
                if items:
                    owner = items[0].get("Metadata", {}).get("owner", "unknown")
                    return province, folder_id, owner, ""
        except Exception as e:
            return province, folder_id, "", str(e)[:100]
    return province, folder_id, "", "empty_folder"


def classify(email: str) -> str:
    if not email:
        return "error"
    if email.endswith("@ect.go.th"):
        return "official_ect"
    if "ect" in email.lower():
        return "ect_gmail"
    if email.endswith(("@gmail.com", "@hotmail.com", "@yahoo.com")):
        return "personal"
    return "institutional"


def main() -> int:
    rows = parse_links("configs/province_links.csv")

    # Dedup by folder_id
    seen_ids: dict[str, dict] = {}
    scan_tasks: list[dict] = []
    for r in rows:
        fid = r["folder_id"]
        if not fid:
            scan_tasks.append(r)
        elif fid not in seen_ids:
            seen_ids[fid] = r
            scan_tasks.append(r)

    print(f"Scanning {len(scan_tasks)} folders...", file=sys.stderr)

    results: dict[str, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(get_owner, r["province"], r["folder_id"]): r for r in scan_tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            prov, fid, owner, err = fut.result()
            results[fid or prov] = (owner, err)
            print(f"\r[{done}/{len(scan_tasks)}]", end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)

    # Build full report
    report = []
    for r in rows:
        fid = r["folder_id"]
        key = fid if fid and fid in results else r["province"]
        owner, err = results.get(key, ("", "not_scanned"))
        report.append({**r, "owner": owner, "error": err})

    # ===== TERMINAL REPORT =====
    sep = "=" * 75
    dash = "-" * 75

    print(f"\n{sep}")
    print("  FOLDER OWNERSHIP FORENSICS REPORT")
    print(f"  configs/province_links.csv — {len(report)} provinces")
    print(sep)

    # Type breakdown
    type_counts: dict[str, int] = {}
    for r in report:
        t = classify(r["owner"])
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

    # Personal emails (non-ECT)
    for r in report:
        if classify(r["owner"]) == "personal":
            print(f"  [PERSONAL]      {r['province']}: {r['owner']}")

    # Official ect.go.th
    for r in report:
        if classify(r["owner"]) == "official_ect":
            print(f"  [OFFICIAL_GOV]  {r['province']}: {r['owner']}")

    # Institutional (non-gmail, non-ect)
    for r in report:
        if classify(r["owner"]) == "institutional":
            print(f"  [INSTITUTIONAL] {r['province']}: {r['owner']}")

    # Errors
    for r in report:
        if r["error"]:
            print(f"  [ERROR]         {r['province']}: {r['error']}")

    # Full table
    print(f"\n{dash}")
    print(f"  {'#':<3} {'Province':<25} {'Owner':<40} {'Type'}")
    print(dash)
    for i, r in enumerate(report, 1):
        etype = classify(r["owner"])
        owner = r["owner"][:38] if r["owner"] else f"[{r['error']}]"
        flag = " *" if etype == "personal" else ""
        print(f"  {i:<3} {r['province']:<25} {owner:<40} {etype}{flag}")
    print(sep)

    # CSV output
    csv_path = "data/folder_owners.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "province", "folder_owner", "email_type", "folder_id", "folder_url", "error",
        ])
        writer.writeheader()
        for r in report:
            writer.writerow({
                "province": r["province"],
                "folder_owner": r["owner"],
                "email_type": classify(r["owner"]),
                "folder_id": r["folder_id"],
                "folder_url": r["url"],
                "error": r["error"],
            })
    print(f"\n  CSV saved: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
