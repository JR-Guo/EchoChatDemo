from __future__ import annotations

from pathlib import Path


DICM_OFFSET = 128
DICM_MAGIC = b"DICM"


def looks_like_dicom(path: Path) -> bool:
    """Check DICOM preamble magic bytes (offset 128, 'DICM').

    Tolerates any filename/extension. Zero-byte files are rejected.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return False
    if size < DICM_OFFSET + len(DICM_MAGIC):
        return False
    with p.open("rb") as f:
        f.seek(DICM_OFFSET)
        return f.read(len(DICM_MAGIC)) == DICM_MAGIC
