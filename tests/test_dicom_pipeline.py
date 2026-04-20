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


from app.services.dicom_pipeline import convert_dicom, ConvertResult
from tests.fixtures.make_dicom import make_cine_dicom


def test_convert_still(tmp_path):
    src = make_still_dicom(tmp_path / "s.dcm")
    out = tmp_path / "out"
    out.mkdir()
    result = convert_dicom(src, out / "target")
    assert isinstance(result, ConvertResult)
    assert result.is_video is False
    assert result.output_path.suffix == ".png"
    assert result.output_path.exists()
    assert result.thumbnail_path.exists()


def test_convert_cine(tmp_path):
    src = make_cine_dicom(tmp_path / "c.dcm", frames=5)
    out = tmp_path / "out"
    out.mkdir()
    result = convert_dicom(src, out / "target")
    assert result.is_video is True
    assert result.output_path.suffix == ".mp4"
    assert result.output_path.exists()
    assert result.frame_count == 5
    assert result.thumbnail_path.exists()


def test_convert_rejects_non_dicom(tmp_path):
    src = make_non_dicom(tmp_path / "bad")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="not a DICOM"):
        convert_dicom(src, out / "target")
