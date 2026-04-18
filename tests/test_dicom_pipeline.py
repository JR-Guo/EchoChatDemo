import pytest

from tests.fixtures.make_dicom import make_still_dicom, make_non_dicom
from app.services.dicom_pipeline import looks_like_dicom


def test_recognizes_dicom_by_magic_bytes(tmp_path):
    p = make_still_dicom(tmp_path / "renamed_no_ext")
    assert looks_like_dicom(p) is True


def test_renamed_to_png_still_recognized(tmp_path):
    p = make_still_dicom(tmp_path / "looks_like.png")
    assert looks_like_dicom(p) is True


def test_non_dicom_rejected(tmp_path):
    p = make_non_dicom(tmp_path / "real.png")
    assert looks_like_dicom(p) is False


def test_zero_byte_file_rejected(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    assert looks_like_dicom(p) is False
