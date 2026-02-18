# Google Drive Files Summary

Snapshot from `survey_gdrive.py` scans on 2026-02-19.

## Overview

- **170,033 files** across **76 unique folders** (77 provinces, all scanned)
- **486.44 GB** total size (deduplicated)
- 2 empty folders (แม่ฮ่องสอน, ระนอง)

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
