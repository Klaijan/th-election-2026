## Vote69 — Mirror ECT drives and OCR Pipeline

Extract structured data from Thai election form PDFs (สส.5/18, สส.5/11) published by the Election Commission of Thailand (กกต.) and make it available as open data.

### Data source

Province-level Google Drive folders are published by ECT at:
<https://www.ect.go.th/ect_th/th/election-2026/>

### Which path should I use?

| Path | Script | Output | When to use |
|------|--------|--------|-------------|
| A — Mirror | `sync_selected_from_csv.sh` | PDF files | Download ECT Google Drive locally |
| B — Crop | `crop_pdf_page.py` | Cropped PDFs | Normalize PDF bands before OCR |
| C — Typhoon OCR | `run_typhoon_ocr.py` | Markdown + JSONL | Fast line-level OCR (remote API) |
| D — CV Pipeline | `main.py` | result.json | Precise cropped-region OCR (Google Vision) |
| E — Vision LLM | `run_gemini_extract.py` | Structured JSON | Full-form extraction (Gemini / Ollama) |

**Recommended for production**: Path **E** (Vision LLM) extracts the complete form as structured JSON in one shot — constituency, candidates, vote counts, voter stats — without the multi-step CV pipeline.

### Architecture

```
scripts/
  run_typhoon_ocr.py ─── Typhoon remote API ──> Markdown + JSONL
  run_gemini_extract.py ─ Vision LLM (Gemini/Ollama) ──> Structured JSON
  crop_pdf_page.py ───── PDF cropping
  survey_folders.py ──── Data folder recon

extract_handwritten_numbers/     (CV Pipeline package)
  main.py ─── Orchestrator
  pdf_loader.py ─── PDF -> images (400 DPI)
  zone_detector.py ─── Template matching (logo, region anchors)
  dot_detector.py ─── Dotted-line detection (fields)
  table_detector.py ─── Grid detection (tables)
  field_extractor.py ─── Crop field regions
  table_extractor.py ─── Crop table columns
  ocr_processor.py ─── Google Vision / Tesseract OCR
  validator.py ─── Cross-field validation
  config.py ─── Configuration
```

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
- Preflight without API calls:

```bash
python scripts/run_typhoon_ocr.py \
  --raw-root data/sample/cropped \
  --out-root data/sample/typhoon_md \
  --manifest-jsonl data/sample/typhoon_manifest.jsonl \
  --dry-run
```

#### Extract counts (post-processing)

Scans Typhoon OCR Markdown and extracts `จำนวน <n> คน/บัตร` occurrences:

```bash
python scripts/extract_typhoon_counts.py \
  --md-root data/sample/typhoon_md \
  --out-jsonl data/sample/typhoon_counts_partylist.jsonl \
  --kind partylist
```

### D) CV Pipeline — Multi-page Thai form OCR (Google Cloud Vision)

The `extract_handwritten_numbers/` package. It:

- Converts a multi-page PDF to images (default **400 DPI**)
- Detects dotted lines in the fields zone (page 1)
- Detects zone-1 y-range on page 1 using `template_4.png` (top anchor) and `template_5.png` (bottom anchor), then searches for dotted lines inside that band
- Detects table grids and extracts **only the last column** across continuation pages
- Batches all crops into **one** OCR call (Google Cloud Vision)
- Validates outputs (cross-field checks: valid + invalid + no_vote = ballots_used)

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

Override OCR provider (default: google):

```bash
python main.py --input data/sample/district --out output --ocr-provider tesseract
```

Outputs:

- `output/result.json`
- `output/debug_output/` (zones, dotted-line overlays, sample crops, OCR+timing JSON)

### E) Vision LLM — Gemini / Ollama structured extraction

Unlike OCR-only paths (C, D) which read cropped regions, this path sends a **whole page image** to a Vision LLM and extracts the complete form as structured JSON in one shot.

Extraction output per page:

```json
{
  "form_type": "สส.5/18 แบ่งเขต",
  "province": "ลำปาง",
  "constituency": 4,
  "amphoe": "แม่พริก",
  "tambon": "แม่พริก",
  "unit_number": 1,
  "voter_stats": {
    "eligible_voters": 445,
    "ballots_used": 300,
    "valid_ballots": 290,
    "invalid_ballots": 8,
    "no_vote_ballots": 2
  },
  "candidates": [
    {"number": 1, "name": "...", "party": "...", "votes": 120},
    {"number": 2, "name": "...", "party": "...", "votes": 170}
  ]
}
```

#### Backend options

| Backend | Flag | Default model | Use case |
|---------|------|---------------|----------|
| Gemini (Google API) | `--backend gemini` | `gemini-2.0-flash` | Cloud; requires `GEMINI_API_KEY` in `env.local` |
| Ollama (local) | `--backend ollama` | `qwen3-vl:8b` | Air-gapped; no API key needed |

#### Gemini (cloud)

```bash
python scripts/run_gemini_extract.py \
  --input "data/raw/ลำปาง/เขตเลือกตั้งที่ 4" \
  --out-root data/gemini_output \
  --manifest-jsonl data/gemini_output/manifest.jsonl \
  --backend gemini \
  --model gemini-2.0-flash \
  --workers 5
```

#### Ollama (local vision model)

```bash
ollama pull qwen3-vl:8b
python scripts/run_gemini_extract.py \
  --input data/sample/district \
  --out-root data/ollama_output \
  --manifest-jsonl data/ollama_output/manifest.jsonl \
  --backend ollama \
  --model qwen3-vl:8b \
  --workers 1
```

Notes:

- Only **odd pages** are processed (page 1, 3, 5... = vote tables; even pages = signatures)
- Resume/skip is automatic via manifest JSONL
- Rate limiting is built-in for Gemini API
- Dry-run mode: add `--dry-run`

### Tests

```bash
pytest -q
```

### Status / WIP

- **Template detector tuning**: zone/template matching thresholds + robustness across scan variants is still in progress.
- **OCR review/tuning**: OCR accuracy and post-processing rules need review on target scans.
- **Gemini extraction validation**: compare structured JSON output against manual counts for accuracy measurement.
