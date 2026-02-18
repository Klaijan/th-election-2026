#!/usr/bin/env python3
import time
from pathlib import Path

import camelot
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

PDF = "data/หน่วยเลือกตั้ง210169.pdf"
OUT = "data/ballotlist.csv"
FAIL = "data/ballotlist_failures.csv"

FLAVOR = "stream"
UPDATE_EVERY = 25  # write partial output every N pages (safety)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize whitespace/newlines in cells
    df = df.replace({"\n": " "}, regex=True)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None"]), c] = ""
    return df


def main():
    pdf_path = Path(PDF)
    if not pdf_path.exists():
        raise FileNotFoundError(PDF)

    n_pages = len(PdfReader(PDF).pages)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    failures = []
    empty_pages = 0

    # optional: streaming safety snapshot
    last_flush = 0
    t0 = time.time()

    bar = tqdm(total=n_pages, unit="page", dynamic_ncols=True, smoothing=0.05)
    for page in range(1, n_pages + 1):
        try:
            tables = camelot.read_pdf(PDF, pages=str(page), flavor=FLAVOR)
            if tables.n == 0:
                empty_pages += 1
            else:
                for t in tables:
                    df = clean_df(t.df)
                    df.insert(0, "page", page)
                    chunks.append(df)

        except Exception as e:
            failures.append({"page": page, "error": repr(e)})

        # update bar stats
        elapsed = time.time() - t0
        rate = page / elapsed if elapsed > 0 else 0.0
        bar.set_postfix(
            tables=len(chunks),
            empty=empty_pages,
            fail=len(failures),
            rate=f"{rate:.2f}/s",
        )
        bar.update(1)

        # optional periodic flush to disk (prevents total loss if interrupted)
        if page - last_flush >= UPDATE_EVERY and chunks:
            tmp_out = Path(OUT).with_suffix(".partial.csv")
            pd.concat(chunks, ignore_index=True).to_csv(tmp_out, index=False)
            last_flush = page

    bar.close()

    if not chunks:
        raise RuntimeError("No tables extracted (chunks is empty).")

    big = pd.concat(chunks, ignore_index=True)
    big.to_csv(OUT, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(FAIL, index=False)

    print(f"\nWrote: {OUT} (rows={len(big)})")
    if failures:
        print(f"Wrote: {FAIL} (n={len(failures)})")
    print(f"Empty pages with no tables: {empty_pages}")


if __name__ == "__main__":
    main()
