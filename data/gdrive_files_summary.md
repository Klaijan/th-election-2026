# Google Drive Files Summary

Snapshot from `survey_gdrive.py` scans on 2026-02-18.

## Overview

- **138,437 files** across **74 unique folders** (77 provinces, 71 scanned)
- **417.15 GB** total size (deduplicated)
- 4 provinces timed out, 2 missing links, 2 empty folders

## Tracked reports

| File | Description |
|------|-------------|
| `survey_gdrive.csv` | Folder size per province (file count, bytes, per-constituency) |
| `folder_owners.csv` | Folder owner email and account type per province |

## Regenerating reports

All commands require `rclone` configured with a `gdrive:` remote.

```bash
# Folder sizes — writes survey_gdrive.csv
python scripts/survey_gdrive.py size --csv

# Folder ownership — writes folder_owners.csv
python scripts/survey_gdrive.py owners

# Full file listing — writes gdrive_files.csv (~80 MB) + gdrive_files_summary.json
python scripts/survey_gdrive.py files
```
