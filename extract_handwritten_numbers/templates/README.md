# Zone-1 anchor templates

This pipeline uses **template matching** on page 1 to infer the **zone-1 (fields) y-range**.

Place these grayscale PNGs in this directory:

- `template_4.png`: **top** anchor
- `template_5.png`: **bottom** anchor

Recommended:
- crop size ~30–50px square
- clean background (just the printed numeral)
- high-contrast sample from a clean scan

If templates are missing, the pipeline falls back to a broad "above table" band on page 1 (less precise, but keeps the pipeline working).


