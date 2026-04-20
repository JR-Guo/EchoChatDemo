from __future__ import annotations

import json
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
    """Async client for the OpenAI-compatible view classification service.

    Sends `POST {base_url}/v1/chat/completions` with
    `{"model": "view-classifier-v1", "file_path": <abs_path>}`.
    Reads `classification.original_view_name` (38-class raw label) and
    `classification.confidence` from the response.

    Optional Bearer token via `api_key` — passed as
    `Authorization: Bearer <key>`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def classify(self, abs_path: Path) -> ClassifyOutcome:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": "view-classifier-v1",
            "file_path": str(Path(abs_path).resolve()),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            return ClassifyOutcome(view=None, confidence=None, raw_class=None,
                                   error="timeout")
        except httpx.HTTPError as e:
            return ClassifyOutcome(view=None, confidence=None, raw_class=None,
                                   error=str(e))

        if r.status_code != 200:
            return ClassifyOutcome(
                view=None, confidence=None, raw_class=None,
                error=f"http {r.status_code}",
            )
        body = r.json()

        # Preferred: top-level `classification` block (present in the
        # OpenAI-compatible service).
        classification = body.get("classification") or {}
        if not classification:
            # Fallback: parse JSON out of assistant message content.
            try:
                content = body["choices"][0]["message"]["content"]
                classification = json.loads(content)
            except Exception:
                classification = {}

        raw = (classification.get("original_view_name")
               or body.get("class_name")
               or body.get("detected_view_type"))
        confidence = classification.get("confidence") or body.get("confidence")
        view = raw if raw in VIEW_LABELS else None
        return ClassifyOutcome(view=view, confidence=confidence, raw_class=raw)
