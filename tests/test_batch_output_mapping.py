from pathlib import Path

from extract_handwritten_numbers.main import _out_dir_for_input


def test_out_dir_mirrors_input_structure():
    in_root = Path("/tmp/in")
    out_root = Path("/tmp/out")
    pdf = Path("/tmp/in/a/b/c.pdf")
    out_dir = _out_dir_for_input(in_root, pdf, out_root)
    assert str(out_dir).endswith("/tmp/out/a/b/c")


def test_out_dir_single_file_uses_stem():
    in_root = Path("/tmp/in_one.pdf")
    out_root = Path("/tmp/out")
    pdf = Path("/tmp/in_one.pdf")
    out_dir = _out_dir_for_input(in_root, pdf, out_root)
    assert str(out_dir).endswith("/tmp/out/in_one")


