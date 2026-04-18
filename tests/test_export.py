import io
from pathlib import Path

from app.services.export import render_report_pdf, render_report_docx


def _sample_data():
    return {
        "sections": [
            {"name": "Aortic Valve", "content": "Normal.", "edited": False},
            {"name": "Left Ventricle", "content": "EF 55%.", "edited": True},
            {"name": "Summary", "content": "Unremarkable.", "edited": False},
        ]
    }


def test_render_pdf_writes_non_empty_pdf(tmp_path):
    out = tmp_path / "r.pdf"
    render_report_pdf(_sample_data(), out)
    data = out.read_bytes()
    assert data[:4] == b"%PDF"
    assert len(data) > 1000


def test_render_docx_writes_non_empty_docx(tmp_path):
    out = tmp_path / "r.docx"
    render_report_docx(_sample_data(), out)
    assert out.read_bytes()[:2] == b"PK"
