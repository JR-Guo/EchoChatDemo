"""Tests for DICOM-zip extraction + upload GC lifespan task."""
from __future__ import annotations

import base64
import io
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services import mobile_media
from app.services.mobile_uploads import ORPHAN_AGE_SECONDS, UploadStore


# ---- detect_kind on zip ---------------------------------------------------

def _make_dicom_bytes() -> bytes:
    """Minimum bytes to pass the DICM magic check — 128 NUL preamble + 'DICM'."""
    return b"\x00" * mobile_media.DICOM_MAGIC_OFFSET + b"DICM" + b"\x00" * 32


def _make_zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_detect_kind_zip_with_dicom_by_extension(tmp_path):
    path = tmp_path / "study.zip"
    path.write_bytes(_make_zip_bytes({"a.dcm": _make_dicom_bytes()}))
    assert mobile_media.detect_kind(path) == "dicom"


def test_detect_kind_zip_with_dicom_by_magic(tmp_path):
    # Member has no .dcm extension; relies on DICM magic scan
    path = tmp_path / "study.zip"
    path.write_bytes(_make_zip_bytes({"frame_001.bin": _make_dicom_bytes()}))
    assert mobile_media.detect_kind(path) == "dicom"


def test_detect_kind_zip_rejects_pure_text(tmp_path):
    path = tmp_path / "random.zip"
    path.write_bytes(_make_zip_bytes({
        "a.txt": b"hello",
        "b.json": b'{"x":1}',
    }))
    assert mobile_media.detect_kind(path) is None


def test_extract_dicoms_from_zip_happy_path(tmp_path):
    zpath = tmp_path / "study.zip"
    zpath.write_bytes(_make_zip_bytes({
        "subdir/a.dcm": _make_dicom_bytes(),
        "b.dcm": _make_dicom_bytes(),
        "readme.txt": b"not dicom",
    }))
    dest = tmp_path / "raw"
    written = mobile_media.extract_dicoms_from_zip(zpath, dest)
    assert len(written) == 2
    assert all(p.suffix == ".dcm" for p in written)
    # Each file has unique 16-hex id as stem
    stems = {p.stem for p in written}
    assert len(stems) == 2
    assert all(len(s) == 16 for s in stems)


def test_extract_dicoms_raises_when_no_dicom(tmp_path):
    zpath = tmp_path / "empty.zip"
    zpath.write_bytes(_make_zip_bytes({"a.txt": b"hello"}))
    with pytest.raises(ValueError):
        mobile_media.extract_dicoms_from_zip(zpath, tmp_path / "out")


# ---- POST /studies with zip upload ---------------------------------------

def _tus(extra=None):
    h = {"Tus-Resumable": "1.0.0"}
    if extra:
        h.update(extra)
    return h


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _make_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    monkeypatch.setenv("ECHOCHAT_SKIP_MODEL", "1")
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    return TestClient(create_app())


def _auth(c: TestClient) -> dict:
    tokens = c.post("/api/mobile/v1/auth/login", json={"password": "pw"}).json()
    return {"Authorization": f"Bearer {tokens['access']}"}


def _upload(c, h, payload, filename, filetype):
    hdrs = {**h, **_tus({
        "Upload-Length": str(len(payload)),
        "Upload-Metadata": f"filename {_b64(filename)},filetype {_b64(filetype)}",
    })}
    create = c.post("/api/mobile/v1/uploads", headers=hdrs)
    uid = create.headers["location"].rsplit("/", 1)[-1]
    patch_h = {**h, **_tus({"Upload-Offset": "0", "Content-Type": "application/offset+octet-stream"})}
    r = c.patch(f"/api/mobile/v1/uploads/{uid}", headers=patch_h, content=payload)
    assert r.status_code == 204
    return uid


def test_zip_with_no_dicom_rejected(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    zbytes = _make_zip_bytes({"a.txt": b"hello"})
    uid = _upload(c, h, zbytes, "empty.zip", "application/zip")
    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid})
    assert r.status_code == 422


def test_zip_routes_to_dicom_install(monkeypatch, tmp_path):
    """When the zip has DICOMs, the kind is 'dicom' and _install_dicom runs.
    run_study_preprocess will fail in this test env (no view classifier model),
    so we expect a 500 — but the important thing is we got past type detection."""
    c = _make_client(monkeypatch, tmp_path)
    h = _auth(c)
    zbytes = _make_zip_bytes({
        "a.dcm": _make_dicom_bytes(),
        "b.dcm": _make_dicom_bytes(),
    })
    uid = _upload(c, h, zbytes, "study.zip", "application/zip")
    r = c.post("/api/mobile/v1/studies", headers=h, json={"uploadId": uid})
    # 500 (preprocess fails without real model) OR 201 (if env happens to have
    # working preprocess). Either way NOT 422 — that would mean type detection
    # rejected the zip, which is the bug this test catches.
    assert r.status_code != 422, f"zip should not be rejected at detection: {r.text}"


# ---- Upload GC ------------------------------------------------------------

def test_gc_orphans_removes_old_dir(tmp_path):
    store = UploadStore(base_dir=tmp_path)
    meta = store.create(length=10, filename="a.bin", filetype="application/octet-stream")
    meta_path = tmp_path / meta.upload_id / "meta.json"
    # Backdate the meta.json mtime past the orphan horizon
    old = time.time() - (ORPHAN_AGE_SECONDS + 60)
    import os
    os.utime(meta_path, (old, old))

    removed = store.gc_orphans()
    assert removed == 1
    assert not (tmp_path / meta.upload_id).exists()


def test_gc_orphans_keeps_recent(tmp_path):
    store = UploadStore(base_dir=tmp_path)
    meta = store.create(length=10, filename="a.bin", filetype="application/octet-stream")
    removed = store.gc_orphans()
    assert removed == 0
    assert (tmp_path / meta.upload_id / "meta.json").exists()
