import json
from pathlib import Path

import numpy as np

from extract_handwritten_numbers.utils import save_json, save_image


def test_save_json_writes_correctly(tmp_path):
    out = tmp_path / "result.json"
    save_json(out, {"key": "value", "n": 42})
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["key"] == "value"
    assert data["n"] == 42


def test_save_json_no_tmp_leftover(tmp_path):
    out = tmp_path / "sub" / "result.json"
    save_json(out, {"x": 1})
    assert out.exists()
    tmp_sibling = out.with_name("result_tmp.json")
    assert not tmp_sibling.exists()


def test_save_json_overwrites_existing(tmp_path):
    out = tmp_path / "result.json"
    save_json(out, {"v": 1})
    save_json(out, {"v": 2})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["v"] == 2


def test_save_image_writes_png(tmp_path):
    out = tmp_path / "img.png"
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    save_image(out, img)
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_image_no_tmp_leftover(tmp_path):
    out = tmp_path / "img.jpg"
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    save_image(out, img)
    tmp_sibling = out.with_name("img_tmp.jpg")
    assert not tmp_sibling.exists()
