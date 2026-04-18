from __future__ import annotations

import re
from typing import Iterable


def collect_media(clips) -> tuple[list[str], list[str]]:
    """Split usable clips into (images, videos) based on the flag+path."""
    images: list[str] = []
    videos: list[str] = []
    for c in clips:
        if not getattr(c, "converted_path", None):
            continue
        if getattr(c, "is_video", False):
            videos.append(c.converted_path)
        else:
            images.append(c.converted_path)
    return images, videos


_NUMERIC = re.compile(r"(-?\d+(?:\.\d+)?)\s*([%A-Za-z/\u00b5°][\w\s/\u00b5\-\^]*)?")


def split_value_unit(s: str) -> tuple[str | None, str | None]:
    """Best-effort extractor: from 'EF is 55 %' pull ('55', '%').

    Returns (None, None) if no number is found. Keep simple — model outputs
    vary, and the raw response is shown beside the parsed value.
    """
    if not s:
        return None, None
    m = _NUMERIC.search(s)
    if not m:
        return None, None
    value = m.group(1)
    unit = (m.group(2) or "").strip() or None
    if unit:
        unit = unit.split()[0].rstrip(".,;:").strip() or None
    return value, unit
