"""Non-DICOM media preprocessing: detect file type, copy into study, generate thumbnails.

DICOM uploads still go through the existing `preprocess_adapter.run_study_preprocess`
pipeline. This module handles the two new branches mobile adds in M3:

- **Video** (.mp4/.mov): copy/rename into `converted/`, extract a middle frame
  via OpenCV as the thumbnail at `converted/<file_id>.thumb.png`.
- **Image** (.png/.jpg/.jpeg): copy/rename into `converted/`, re-encode as
  PNG thumbnail at `converted/<file_id>.thumb.png`.

No view classification here — that's best-effort / future work. `Clip.view`
is left None; mobile studies will show `status="processing"` (or "ready" if
at least one clip was DICOM-preprocessed into a known view).
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Literal, Optional


MediaKind = Literal["dicom", "video", "image"]


DICOM_MAGIC_OFFSET = 128
DICOM_MAGIC = b"DICM"

_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def detect_kind(raw_path: Path, filename_hint: str = "") -> MediaKind | None:
    """Return 'dicom' / 'video' / 'image' / None based on content then extension."""
    try:
        with raw_path.open("rb") as fh:
            head = fh.read(DICOM_MAGIC_OFFSET + 4)
    except OSError:
        return None
    if len(head) >= DICOM_MAGIC_OFFSET + 4 and head[DICOM_MAGIC_OFFSET:DICOM_MAGIC_OFFSET + 4] == DICOM_MAGIC:
        return "dicom"

    name_lower = (filename_hint or raw_path.name).lower()
    for ext in _VIDEO_EXTS:
        if name_lower.endswith(ext):
            return "video"
    for ext in _IMAGE_EXTS:
        if name_lower.endswith(ext):
            return "image"
    # File content sniff for mp4 (ftyp box)
    if b"ftyp" in head[:32]:
        return "video"
    # PNG / JPEG magic
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "image"
    return None


def new_file_id() -> str:
    return uuid.uuid4().hex[:16]


def install_video(
    *,
    upload_data: Path,
    original_filename: str,
    converted_dir: Path,
    file_id: Optional[str] = None,
) -> tuple[str, Path, Path]:
    """Move the uploaded video into the study's `converted/` dir and emit a
    thumbnail PNG next to it. Returns (file_id, converted_path, thumbnail_path).

    The upload bytes are *renamed* (not copied) when the dest is on the same
    filesystem — O(1) on NFS.
    """
    file_id = file_id or new_file_id()
    ext = _safe_ext(original_filename, default=".mp4", allowed=_VIDEO_EXTS)
    converted_dir.mkdir(parents=True, exist_ok=True)
    dest = converted_dir / f"{file_id}{ext}"
    _move_or_copy(upload_data, dest)

    thumb = converted_dir / f"{file_id}.thumb.png"
    try:
        _extract_video_middle_frame(dest, thumb)
    except Exception:
        # Non-fatal: leave thumbnail missing. Mobile UI handles null gracefully.
        pass
    return file_id, dest, thumb


def install_image(
    *,
    upload_data: Path,
    original_filename: str,
    converted_dir: Path,
    file_id: Optional[str] = None,
) -> tuple[str, Path, Path]:
    """Move the uploaded image into `converted/`. Thumbnail = a downscaled PNG copy."""
    file_id = file_id or new_file_id()
    ext = _safe_ext(original_filename, default=".png", allowed=_IMAGE_EXTS)
    converted_dir.mkdir(parents=True, exist_ok=True)
    dest = converted_dir / f"{file_id}{ext}"
    _move_or_copy(upload_data, dest)

    thumb = converted_dir / f"{file_id}.thumb.png"
    try:
        _shrink_image(dest, thumb, max_width=320)
    except Exception:
        try:
            shutil.copyfile(dest, thumb)
        except Exception:
            pass
    return file_id, dest, thumb


def _safe_ext(filename: str, default: str, allowed: set[str]) -> str:
    name = (filename or "").lower()
    for ext in allowed:
        if name.endswith(ext):
            return ext
    return default


def _move_or_copy(src: Path, dst: Path) -> None:
    try:
        src.rename(dst)
    except OSError:
        shutil.copyfile(src, dst)
        try:
            src.unlink()
        except OSError:
            pass


def _extract_video_middle_frame(video_path: Path, out_png: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cv2 could not open video")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = max(total // 2, 0)
        if target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("no frame decoded")
        # Resize to a reasonable thumbnail size (max 320 wide)
        h, w = frame.shape[:2]
        if w > 320:
            scale = 320 / w
            frame = cv2.resize(frame, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_png), frame)
    finally:
        cap.release()


def _shrink_image(src: Path, out_png: Path, max_width: int = 320) -> None:
    import cv2

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2 could not read image")
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_png), img)
