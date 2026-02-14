## Vote69 — Crop PDFs + Typhoon OCR

This repo contains two things:

- Crop election-form PDFs into a consistent band / region (PyMuPDF)
- OCR cropped outputs via **Typhoon OCR** (remote API)

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

### A) Crop PDFs

Default crop (full width, 30%..60% of page height) into a new 1-page PDF:

```bash
python crop_pdf_page.py --pdf data/sample/district/1.pdf --out data/sample/cropped/district/1.pdf
```

Batch crop and keep filenames under `cropped/{district,partylist}/`:

```bash
python batch_crop_pdfs.py --input-dir data/sample --out-root data/sample --crop-script crop_pdf_page.py
```

### B) Typhoon OCR (remote API)

This mirrors the structure / robustness of the reference pipeline:
[`mjenmana/thai-election-2026`](https://github.com/mjenmana/thai-election-2026/tree/master)

Set env vars (do not commit keys):

```bash
cp env.example env.local
# edit env.local
```

Run OCR (writes Markdown outputs + JSONL manifest, and supports resume/skips):

```bash
python run_typhoon_ocr.py \
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
python run_typhoon_ocr.py \
  --raw-root data/sample/cropped \
  --out-root data/sample/typhoon_md \
  --manifest-jsonl data/sample/typhoon_manifest.jsonl \
  --dry-run
```
