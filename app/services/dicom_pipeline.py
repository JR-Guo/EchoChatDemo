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


from dataclasses import dataclass
from typing import Optional

import numpy as np
import pydicom
import cv2


@dataclass(frozen=True)
class ConvertResult:
    output_path: Path
    thumbnail_path: Path
    is_video: bool
    frame_count: int
    width: int
    height: int


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Return an 8-bit BGR image from an arbitrary DICOM frame."""
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        if hi > lo:
            f = (f - lo) / (hi - lo) * 255.0
        frame = f.astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _make_thumbnail(bgr: np.ndarray, out: Path, max_side: int = 240) -> Path:
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), bgr)
    return out

def validate_US(dcm_path):
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    modal = ds.Modality if hasattr(ds, 'Modality') else 'US'
    if modal != 'US':
        return False
    return True

def convert_dicom(src: Path, target_stem: Path, *, fps: int = 20) -> ConvertResult:
    """Convert a DICOM at `src` into mp4 (cine) or png (still).

    `target_stem` is a path *without* extension; we add .mp4 / .png.
    A sibling `<stem>.thumb.png` is also produced.

    Raises ValueError if the file is not a DICOM.
    """
    src = Path(src)
    target_stem = Path(target_stem)
    if not looks_like_dicom(src):
        raise ValueError(f"not a Echocardiography DICOM: {src}")
    
    if not validate_US(src):
        raise ValueError(f"not a Echocardiography DICOM: {src}")

    ds = pydicom.dcmread(str(src))
    pixels = ds.pixel_array

    thumb_path = target_stem.with_suffix("").with_name(target_stem.name + ".thumb.png")

    if pixels.ndim == 2 or (pixels.ndim == 3 and pixels.shape[-1] in (3, 4)):
        frame = _normalize_frame(pixels)
        out = target_stem.with_suffix(".png")
        cv2.imwrite(str(out), frame)
        _make_thumbnail(frame, thumb_path)
        h, w = frame.shape[:2]
        return ConvertResult(out, thumb_path, is_video=False, frame_count=1, width=w, height=h)

    if pixels.ndim == 3:
        frames = pixels
    elif pixels.ndim == 4:
        frames = pixels
    else:
        raise ValueError(f"unexpected pixel shape {pixels.shape}")

    h, w = frames.shape[1], frames.shape[2]
    out = target_stem.with_suffix(".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    try:
        for i in range(frames.shape[0]):
            writer.write(_normalize_frame(frames[i]))
    finally:
        writer.release()
    _make_thumbnail(_normalize_frame(frames[0]), thumb_path)
    return ConvertResult(out, thumb_path, is_video=True, frame_count=int(frames.shape[0]),
                         width=w, height=h)
