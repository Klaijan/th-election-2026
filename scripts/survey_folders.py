#!/usr/bin/env python3
"""
Survey folder structures under a data root (e.g. data/raw/).

For each province subdirectory, prints:
- Tree structure with file counts per folder
- Depth pattern summary (how many levels deep the PDFs sit)
- Naming patterns (numeric filenames like 1.pdf vs Thai names)

Usage:
  python scripts/survey_folders.py --root data/raw
  python scripts/survey_folders.py --root data/raw --province ลำปาง
  python scripts/survey_folders.py --root data/raw --csv survey_output.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PDF_EXTS = {".pdf"}
ALL_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp",
            ".csv", ".xlsx", ".xls"}


@dataclass
class FolderStats:
    """Stats for a single folder (non-recursive)."""
    path: Path
    rel_path: str
    depth: int
    pdf_count: int = 0
    other_file_count: int = 0
    subfolder_count: int = 0
    filenames: list[str] = field(default_factory=list)


@dataclass
class ProvinceProfile:
    """Aggregated profile for one province directory."""
    name: str
    root: Path
    total_pdfs: int = 0
    total_files: int = 0
    total_folders: int = 0
    max_depth: int = 0
    depth_distribution: Counter = field(default_factory=Counter)
    leaf_folders: list[FolderStats] = field(default_factory=list)
    folder_tree: list[FolderStats] = field(default_factory=list)
    naming_patterns: Counter = field(default_factory=Counter)
    depth_pattern: str = ""
    structure_signature: str = ""


def classify_filename(name: str) -> str:
    """Classify a filename into a pattern category."""
    stem = Path(name).stem
    if re.fullmatch(r"\d+", stem):
        return "numeric"
    if re.fullmatch(r"\d+-\d+", stem):
        return "numeric-range"
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", stem):
        return "ascii"
    return "thai-or-mixed"


def scan_folder(folder: Path, province_root: Path, depth: int) -> FolderStats:
    """Gather stats for a single directory."""
    rel = str(folder.relative_to(province_root)) if folder != province_root else "."
    stats = FolderStats(path=folder, rel_path=rel, depth=depth)

    for child in sorted(folder.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            stats.subfolder_count += 1
        elif child.is_file():
            if child.suffix.lower() in PDF_EXTS:
                stats.pdf_count += 1
                stats.filenames.append(child.name)
            elif child.suffix.lower() in ALL_EXTS:
                stats.other_file_count += 1
                stats.filenames.append(child.name)

    return stats


def survey_province(province_dir: Path) -> ProvinceProfile:
    """Walk a province directory and build its profile."""
    name = province_dir.name
    profile = ProvinceProfile(name=name, root=province_dir)

    stack: list[tuple[Path, int]] = [(province_dir, 0)]
    while stack:
        folder, depth = stack.pop()
        stats = scan_folder(folder, province_dir, depth)
        profile.folder_tree.append(stats)
        profile.total_folders += 1
        profile.total_pdfs += stats.pdf_count
        profile.total_files += stats.pdf_count + stats.other_file_count

        if depth > profile.max_depth:
            profile.max_depth = depth

        if stats.pdf_count > 0:
            profile.depth_distribution[depth] += stats.pdf_count
            for fn in stats.filenames:
                profile.naming_patterns[classify_filename(fn)] += 1

        if stats.subfolder_count == 0 and (stats.pdf_count + stats.other_file_count) > 0:
            profile.leaf_folders.append(stats)

        # Queue subfolders
        for child in sorted(folder.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                stack.append((child, depth + 1))

    # Sort tree by path for display
    profile.folder_tree.sort(key=lambda s: s.rel_path)
    profile.leaf_folders.sort(key=lambda s: s.rel_path)

    # Summarize depth pattern
    if profile.depth_distribution:
        dominant = profile.depth_distribution.most_common(1)[0]
        profile.depth_pattern = (
            f"depth={dominant[0]} ({dominant[1]} PDFs, "
            f"{len(profile.depth_distribution)} distinct depth(s))"
        )

    # Structure signature: folder-name patterns at each level
    level_names: dict[int, list[str]] = {}
    for fs in profile.folder_tree:
        if fs.depth > 0:
            level_names.setdefault(fs.depth, []).append(
                Path(fs.rel_path).name
            )
    sig_parts = []
    for d in sorted(level_names.keys()):
        names = level_names[d]
        sample = names[:3]
        tag = f"L{d}({len(names)}): {', '.join(sample)}"
        if len(names) > 3:
            tag += ", ..."
        sig_parts.append(tag)
    profile.structure_signature = " / ".join(sig_parts)

    return profile


def print_tree(profile: ProvinceProfile, *, max_depth: int = 0) -> None:
    """Print a tree view of the province folder structure."""
    print(f"\n{'=' * 70}")
    print(f"  {profile.name}")
    print(f"  PDFs: {profile.total_pdfs}  |  Files: {profile.total_files}"
          f"  |  Folders: {profile.total_folders}  |  Max depth: {profile.max_depth}")
    print(f"  Depth pattern: {profile.depth_pattern}")
    print(f"  Naming: {dict(profile.naming_patterns)}")
    print(f"  Signature: {profile.structure_signature}")
    print(f"{'=' * 70}")

    for fs in profile.folder_tree:
        if max_depth and fs.depth > max_depth:
            continue
        indent = "  " * fs.depth
        count_parts = []
        if fs.pdf_count:
            count_parts.append(f"{fs.pdf_count} pdf")
        if fs.other_file_count:
            count_parts.append(f"{fs.other_file_count} other")
        if fs.subfolder_count:
            count_parts.append(f"{fs.subfolder_count} dirs")
        count_str = f"  ({', '.join(count_parts)})" if count_parts else ""

        # Show sample filenames for leaf folders
        sample = ""
        if fs.pdf_count > 0 and fs.subfolder_count == 0:
            names = fs.filenames[:5]
            if len(fs.filenames) > 5:
                names.append(f"...+{len(fs.filenames) - 5}")
            sample = f"  [{', '.join(names)}]"

        name = Path(fs.rel_path).name if fs.rel_path != "." else "."
        print(f"{indent}{name}/{count_str}{sample}")


def write_csv(profiles: list[ProvinceProfile], out_path: Path) -> None:
    """Write a flat CSV with one row per leaf folder for analysis."""
    rows = []
    for p in profiles:
        for fs in p.leaf_folders:
            parts = Path(fs.rel_path).parts
            rows.append({
                "province": p.name,
                "rel_path": fs.rel_path,
                "depth": fs.depth,
                "pdf_count": fs.pdf_count,
                "other_count": fs.other_file_count,
                "level_1": parts[0] if len(parts) > 0 else "",
                "level_2": parts[1] if len(parts) > 1 else "",
                "level_3": parts[2] if len(parts) > 2 else "",
                "level_4": parts[3] if len(parts) > 3 else "",
                "sample_files": "; ".join(fs.filenames[:10]),
                "naming_dominant": (
                    Counter(classify_filename(f) for f in fs.filenames)
                    .most_common(1)[0][0] if fs.filenames else ""
                ),
            })

    if not rows:
        print(f"No leaf folders to write to {out_path}")
        return

    fieldnames = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote CSV: {out_path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Survey folder structures under a data root.",
    )
    ap.add_argument(
        "--root", default="data/raw",
        help="Root directory containing province subdirectories. Default: data/raw",
    )
    ap.add_argument(
        "--province", default=None,
        help="Only survey this province (exact folder name match).",
    )
    ap.add_argument(
        "--csv", default=None,
        help="Write a CSV summary of leaf folders (one row per leaf).",
    )
    ap.add_argument(
        "--max-depth", type=int, default=0,
        help="Max tree depth to display (0 = unlimited). Default: 0.",
    )
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root}", file=sys.stderr)
        print("Have you synced any provinces yet? See: bash scripts/sync_selected_from_csv.sh", file=sys.stderr)
        return 1

    # Find province directories (immediate children of root)
    if args.province:
        target = root / args.province
        if not target.is_dir():
            print(f"Province directory not found: {target}", file=sys.stderr)
            return 1
        province_dirs = [target]
    else:
        province_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )

    if not province_dirs:
        print(f"No province subdirectories found under: {root}", file=sys.stderr)
        return 1

    profiles: list[ProvinceProfile] = []
    for d in province_dirs:
        p = survey_province(d)
        profiles.append(p)
        print_tree(p, max_depth=args.max_depth)

    # Summary across provinces
    if len(profiles) > 1:
        print(f"\n{'=' * 70}")
        print("  CROSS-PROVINCE SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Provinces surveyed: {len(profiles)}")
        print(f"  Total PDFs: {sum(p.total_pdfs for p in profiles)}")

        depth_groups: dict[str, list[str]] = {}
        for p in profiles:
            key = str(dict(p.depth_distribution.most_common()))
            depth_groups.setdefault(key, []).append(p.name)

        print(f"\n  Depth pattern groups ({len(depth_groups)}):")
        for pattern, provinces in sorted(depth_groups.items()):
            print(f"    {pattern}: {', '.join(provinces[:5])}"
                  + (f" ...+{len(provinces)-5}" if len(provinces) > 5 else ""))

    if args.csv:
        write_csv(profiles, Path(args.csv))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
