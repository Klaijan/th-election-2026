## Vote69 — Mirror ECT drives and OCR Pipeline

This repo currently does four things:

1. Locally download official ECT google drive for selected provinces (via `.configs/province.txt`)
2. Crop election-form PDFs into a consistent band / region (PyMuPDF)
3. OCR cropped outputs via **Typhoon OCR** (remote API)
4. Multi-page Thai form pipeline to extract handwritten numbers on/above dotted lines and **table column 3 only** (Google Cloud Vision)

### Setup

Recommended: use **pyenv** to pin a Python version for this project.

This repo includes `.python-version` (used by pyenv).

If you want `pyenv activate` to work in **zsh**, ensure you have `pyenv` + `pyenv-virtualenv` installed and your `~/.zshrc` contains:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

One-time setup (pyenv + venv):

```bash
bash scripts/pyenv_setup.sh
```

Activate the venv later (switchable):

```bash
source scripts/activate_venv.sh .venv311
python -m pip install -r requirements.txt
```

If you don't use pyenv, you can still create a venv with any `python3` on PATH.

Install Python deps:

```bash
python -m pip install -r requirements.txt
```

### A) Mirror ECT google drive (download PDFs locally)

Before running OCR, mirror the official ECT Google Drive folders to your local `data/` directory

1. Install rclone prerequisite and configure a Google Drive remote (one-time setup):

```bash
brew install rclone
rclone config
```

Create a remote called `ect_drive`.

2. Select province for mirroring

For the provinces you want to mirror, enter one province per line in `configs/province.txt`. `configs/province_links.csv` provides the corresponding drive folder URL.

3. Run the mirroring script

```bash
bash scripts/sync_selected_from_csv.sh ect_drive
```

The script will write files under `data/`, and can be re-run as it will only copy missing/changed files depending on flags.

### B) Crop PDFs

Default crop (full width, 30%..60% of page height) into a new 1-page PDF:

```bash
python scripts/crop_pdf_page.py --pdf data/sample/district/1.pdf --out data/sample/cropped/district/1.pdf
```

Batch crop and keep filenames under `cropped/{district,partylist}/`:

```bash
python scripts/batch_crop_pdfs.py --input-dir data/sample --out-root data/sample --crop-script crop_pdf_page.py
```

### C) Typhoon OCR (remote API)

This mirrors the structure / robustness of the reference pipeline:
[`mjenmana/thai-election-2026`](https://github.com/mjenmana/thai-election-2026/tree/master)

Set env vars (do not commit keys):

```bash
cp env.example env.local
# edit env.local
```

Run OCR (writes Markdown outputs + JSONL manifest, and supports resume/skips):

```bash
python scripts/run_typhoon_ocr.py \
  --raw-root data/sample/cropped \
  --out-root data/sample/typhoon_md \
  --manifest-jsonl data/sample/typhoon_manifest.jsonl \
  --workers 3 \
  --max-seconds 7200
```

Notes:

- Outputs are `*.md` mirroring `--raw-root` under `--out-root`.
- Manifest is append-only JSONL (source of truth for resuming).
- `run_typhoon_ocr.py` will automatically load `env.local` by default (see `--env-file`).
- Progress bar is enabled by default; disable with `--no-progress`.
- At the end of each run, the script appends a `row_type=run_summary` record into a separate stats JSONL file (default: `*.stats.jsonl` next to your manifest).
- Preflight without API calls:

```bash
python scripts/run_typhoon_ocr.py \
  --raw-root data/sample/cropped \
  --out-root data/sample/typhoon_md \
  --manifest-jsonl data/sample/typhoon_manifest.jsonl \
  --dry-run
```

### C) Extract the numbers you care about (post-processing)

This scans Typhoon OCR Markdown and extracts every occurrence of:

- `จำนวน <n> คน`
- `จำนวน <n> บัตร`

It keeps the **order** (so you can rely on position) and also captures an optional **schema** prefix (e.g. `2.2.1`) + the **label text** before the number (for foolproofing).

Example (partylist):

```bash
python scripts/extract_typhoon_counts.py \
  --md-root data/sample/typhoon_md \
  --out-jsonl data/sample/typhoon_counts_partylist.jsonl \
  --kind partylist
```

### D) Multi-page Thai form OCR (Google Cloud Vision) — dotted lines + table column 3 only

This is the **production-style pipeline** described in the prompt (`extract_handwritten_numbers/` package). It:

- Converts a multi-page PDF to images (default **400 DPI**)
- Detects dotted lines in the fields zone (page 1)
- Detects zone-1 y-range on page 1 using `template_4.png` (top anchor) and `template_5.png` (bottom anchor), then searches for dotted lines inside that band.
- Detects table grids and extracts **only the last column** across continuation pages
- Batches all crops into **one** OCR call (Google Cloud Vision)
- Validates outputs and produces a review queue

#### System dependencies

- **Poppler** (needed by `pdf2image`)
  - macOS: `brew install poppler`
- (Optional) **Tesseract** (fallback OCR)
  - macOS: `brew install tesseract`

#### Google credentials

Set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` in your environment.

#### Run

```bash
python main.py --input data/sample/district --out output --debug
```

Outputs:

- `output/result.json`
- `output/debug_output/` (zones, dotted-line overlays, sample crops, OCR+timing JSON)

#### Tests

```bash
pytest -q
```
