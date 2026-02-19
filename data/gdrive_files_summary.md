# Google Drive Files Summary

Snapshot from `survey_gdrive.py` scans on 2026-02-19.

Source: <https://www.ect.go.th/ect_th/th/election-2026/>

## Overview

- **176,127 files** across **77 unique folders** (77 provinces, all scanned)
- **511.80 GB** total size (deduplicated)

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
