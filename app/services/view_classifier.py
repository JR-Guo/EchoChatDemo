from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from constants.view_labels import VIEW_LABELS


@dataclass(frozen=True)
class ClassifyOutcome:
    view: Optional[str]
    confidence: Optional[float]
    raw_class: Optional[str]
    error: Optional[str] = None


class ViewClassifier:
    """Async client for the EchoView38 service (POST /classify)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def classify(self, abs_path: Path) -> ClassifyOutcome:
        url = f"{self.base_url}/classify"
        payload = {"path": str(Path(abs_path).resolve()), "topk": 1}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=payload)
        except httpx.TimeoutException:
            return ClassifyOutcome(view=None, confidence=None, raw_class=None, error="timeout")
        except httpx.HTTPError as e:
            return ClassifyOutcome(view=None, confidence=None, raw_class=None, error=str(e))

        if r.status_code != 200:
            return ClassifyOutcome(
                view=None, confidence=None, raw_class=None,
                error=f"http {r.status_code}",
            )
        body = r.json()
        raw = body.get("class_name") or body.get("detected_view_type") or body.get("top1")
        confidence = body.get("confidence")
        view = raw if raw in VIEW_LABELS else None
        return ClassifyOutcome(view=view, confidence=confidence, raw_class=raw)
