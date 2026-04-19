from __future__ import annotations

import asyncio
import os
from typing import Protocol, Sequence


class EngineBackend(Protocol):
    def infer_sync(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ) -> str: ...


class EchoChatEngine:
    """Async wrapper with a global lock so the single GPU model runs one job at a time."""

    def __init__(self, backend: EngineBackend):
        self._backend = backend
        self._lock = asyncio.Lock()

    async def infer(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ) -> str:
        async with self._lock:
            return await asyncio.to_thread(
                self._backend.infer_sync, system, query, images, videos
            )


class SwiftPtBackend:
    """Real backend using ms_swift PtEngine. Imported lazily so unit tests never load torch.

    attn_impl defaults to torch's native `sdpa` (Scaled Dot Product Attention). Set
    ECHOCHAT_ATTN_IMPL=flash_attn to use flash-attention 2 when available.
    """

    def __init__(self, model_path: str):
        from swift.llm import PtEngine, get_template

        attn_impl = os.environ.get("ECHOCHAT_ATTN_IMPL", "sdpa")
        self.engine = PtEngine(
            model_path, torch_dtype="bfloat16", attn_impl=attn_impl, use_hf=True
        )
        self.template = get_template(
            self.engine.model_meta.template,
            self.engine.tokenizer,
            default_system=None,
        )

    def infer_sync(
        self,
        system: str,
        query: str,
        images,
        videos,
    ) -> str:
        from swift.llm import InferRequest, RequestConfig

        placeholders = " ".join(["<video>"] * len(videos) + ["<image>"] * len(images))
        prompt = (placeholders + " " + query).strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
        req = InferRequest(messages=messages, images=list(images), videos=list(videos))
        cfg = RequestConfig(max_tokens=1024, temperature=0, stream=False)
        gen = self.engine.infer([req], cfg)
        return gen[0].choices[0].message.content
