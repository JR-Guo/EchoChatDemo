"""tus 1.0.0 server state management.

Storage layout under `<data_dir>/uploads/<upload-id>/`:
    data            raw bytes, appended to on each PATCH
    meta.json       {filename, filetype, length, created_at, offset, owner?}

`data_dir` is `settings.data_dir` — same NFS volume as `sessions/`, so a
completed upload can be atomically `os.rename`'d into a study directory
later without copying bytes.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


MAX_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB
ORPHAN_AGE_SECONDS = 24 * 3600


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class UploadMeta:
    upload_id: str
    filename: str
    filetype: str
    length: int
    offset: int
    created_at: str
    root: Path

    def to_dict(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "filename": self.filename,
            "filetype": self.filetype,
            "length": self.length,
            "offset": self.offset,
            "created_at": self.created_at,
        }


class UploadStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _upload_dir(self, upload_id: str) -> Path:
        return self.base_dir / upload_id

    def _meta_path(self, upload_id: str) -> Path:
        return self._upload_dir(upload_id) / "meta.json"

    def data_path(self, upload_id: str) -> Path:
        return self._upload_dir(upload_id) / "data"

    def exists(self, upload_id: str) -> bool:
        return self._meta_path(upload_id).exists()

    def create(self, *, length: int, filename: str, filetype: str) -> UploadMeta:
        upload_id = uuid.uuid4().hex
        root = self._upload_dir(upload_id)
        root.mkdir(parents=True, exist_ok=False)
        meta = UploadMeta(
            upload_id=upload_id,
            filename=_sanitize_name(filename) or "upload.bin",
            filetype=filetype or "application/octet-stream",
            length=length,
            offset=0,
            created_at=_now_iso(),
            root=root,
        )
        self._meta_path(upload_id).write_text(json.dumps(meta.to_dict(), indent=2))
        self.data_path(upload_id).touch()
        return meta

    def load(self, upload_id: str) -> Optional[UploadMeta]:
        p = self._meta_path(upload_id)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        return UploadMeta(
            upload_id=d["upload_id"],
            filename=d["filename"],
            filetype=d["filetype"],
            length=int(d["length"]),
            offset=int(d["offset"]),
            created_at=d["created_at"],
            root=self._upload_dir(upload_id),
        )

    def append_chunk(self, upload_id: str, offset: int, chunk: bytes) -> int:
        """Write `chunk` at byte `offset` (must match current). Returns new offset."""
        meta = self.load(upload_id)
        if meta is None:
            raise LookupError("upload not found")
        if offset != meta.offset:
            raise ValueError(f"offset mismatch: have {meta.offset}, got {offset}")
        if meta.offset + len(chunk) > meta.length:
            raise ValueError("chunk exceeds declared Upload-Length")
        with self.data_path(upload_id).open("ab") as fh:
            fh.write(chunk)
        new_offset = meta.offset + len(chunk)
        meta.offset = new_offset
        self._meta_path(upload_id).write_text(json.dumps(meta.to_dict(), indent=2))
        return new_offset

    def delete(self, upload_id: str) -> None:
        root = self._upload_dir(upload_id)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)

    def consume(self, upload_id: str) -> Optional[UploadMeta]:
        """Return meta for a completed upload and leave the data file intact
        under `<root>/data`. Caller is responsible for moving it + calling
        `delete()` once done. Returns None if upload does not exist or is
        incomplete."""
        meta = self.load(upload_id)
        if meta is None:
            return None
        if meta.offset != meta.length or meta.length == 0:
            return None
        return meta

    def gc_orphans(self, now: Optional[float] = None) -> int:
        """Delete upload dirs whose meta.json is older than ORPHAN_AGE_SECONDS
        AND whose offset < length (never completed OR already consumed)."""
        now = now if now is not None else time.time()
        removed = 0
        if not self.base_dir.exists():
            return 0
        for entry in self.base_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "meta.json"
            if not meta_path.exists():
                continue
            try:
                mtime = meta_path.stat().st_mtime
            except OSError:
                continue
            if (now - mtime) < ORPHAN_AGE_SECONDS:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed


_METADATA_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./")


def _sanitize_name(name: str) -> str:
    """Strip path separators & suspicious chars so the filename can't escape upload dir."""
    if not name:
        return ""
    base = os.path.basename(name)  # drop any path prefix
    cleaned = "".join(c if c in _METADATA_ALLOWED_CHARS else "_" for c in base)
    return cleaned[:200]  # cap length


def parse_upload_metadata(header: str) -> dict[str, str]:
    """Parse tus Upload-Metadata: 'key b64val,key b64val,...' → {key: value}."""
    out: dict[str, str] = {}
    if not header:
        return out
    for pair in header.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(" ", 1)
        key = parts[0].strip()
        if not key:
            continue
        if len(parts) == 1:
            out[key] = ""
            continue
        try:
            value = base64.b64decode(parts[1].strip(), validate=True).decode("utf-8", errors="replace")
        except Exception:
            value = ""
        out[key] = value
    return out
