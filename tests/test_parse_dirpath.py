"""Tests for scripts/parse_dirpath.py directory path parser."""
import sys
from pathlib import Path

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from parse_dirpath import parse_election_path


def test_full_path_with_all_fields():
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 3/อำเภอแม่ทะ/1.ตำบลบ้านบอม/1.แบบแบ่งเขต 5-18/หน่วย3 ม.3/บอม 5-18 น3ม3.pdf"
    )
    assert result["province"] == "ลำปาง"
    assert result["constituency"] == 3
    assert result["amphoe"] == "แม่ทะ"
    assert result["tambon"] == "บ้านบอม"
    assert result["form_type"] == "district"
    assert result["unit_number"] == 3
    assert result["is_advance_voting"] is False


def test_short_amphoe_format():
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 4/อ.แม่พริก/ส.ส5-18/สส5-18แบ่งเขต ต.แม่พริก.pdf"
    )
    assert result["amphoe"] == "แม่พริก"
    assert result["form_type"] == "district"
    assert result["tambon"] == "แม่พริก"


def test_partylist_with_bch():
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 4/อ.แม่พริก/ส.ส.5-18(บช)/สส5-18(บช) ต.แม่พริก.pdf"
    )
    assert result["form_type"] == "partylist"
    assert result["constituency"] == 4


def test_partylist_with_full_name():
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 3/อำเภอเมือง/2.ทม.พิชัย/สส.แบบบัญชีรายชื่อ 5-18 (บช)/หน่วยที่ 5 ทม.พิชัย.pdf"
    )
    assert result["form_type"] == "partylist"
    assert result["tambon"] == "พิชัย"
    assert result["unit_number"] == 5


def test_municipality_tambon_formats():
    # ทม. (เทศบาลเมือง)
    result = parse_election_path("ลำปาง/เขตเลือกตั้งที่ 3/อำเภอเมือง/3.ทม.เขลางค์นคร/สส.แบบแบ่งเขต 5-18")
    assert result["tambon"] == "เขลางค์นคร"

    # อบต.
    result = parse_election_path("ลำปาง/เขตเลือกตั้งที่ 3/อำเภอเมือง/1.อบต.พิชัย/สส.แบ่งเขต 5-18")
    assert result["tambon"] == "พิชัย"


def test_advance_voting():
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 4/นอกเขต/ส.ส.5-17/ชุดที่1.pdf"
    )
    assert result["is_advance_voting"] is True
    assert result["constituency"] == 4


def test_data_raw_prefix_stripped():
    result = parse_election_path(
        "data/raw/ลำปาง/เขตเลือกตั้งที่ 1/อำเภอเมือง/ตำบลหัวเวียง/สส5-18/test.pdf"
    )
    assert result["province"] == "ลำปาง"
    assert result["constituency"] == 1


def test_no_unit_number():
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 4/อ.เถิน/สส5-18/สส5-18แบ่งเขต ต.แม่ปะ.pdf"
    )
    assert result["unit_number"] is None
    assert result["tambon"] == "แม่ปะ"


def test_empty_path():
    result = parse_election_path("")
    assert result["province"] is None
    assert result["constituency"] is None


def test_pdf_extension_stripped_from_tambon():
    """Regression: tambon should not include .pdf extension."""
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 4/อ.แม่พริก/ส.ส.5-18(บช)/สส5-18(บช) ต.แม่พริก.pdf"
    )
    assert ".pdf" not in result["tambon"]


def test_mixed_district_and_partylist_dir():
    """A directory containing both forms: "สส.5-18 5-18(บช)" → partylist wins because บช present."""
    result = parse_election_path(
        "ลำปาง/เขตเลือกตั้งที่ 4/อ.เกาะคา/สส.5-18 5-18(บช)/test.pdf"
    )
    assert result["form_type"] == "partylist"
