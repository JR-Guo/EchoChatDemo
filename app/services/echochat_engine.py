from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Protocol, Sequence


class EngineBackend(Protocol):
    def infer_sync(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ) -> str: ...

    def infer_stream_sync(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ): ...


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

    async def infer_stream(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ) -> AsyncIterator[str]:
        """Yield assistant text chunks as they are produced.

        Holds the engine lock for the duration of the stream, serializing
        requests across clients (the GPU model runs one job at a time).
        Each yielded string is a *delta* (new tokens since the previous
        yield), not a running total — the caller concatenates.
        """
        async with self._lock:
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _producer() -> None:
                try:
                    last = ""
                    for chunk in self._backend.infer_stream_sync(system, query, images, videos):
                        if chunk.startswith(last):
                            delta = chunk[len(last):]
                            last = chunk
                        else:
                            # Model reset or non-monotonic — send whole chunk
                            delta = chunk
                            last = chunk
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, delta)
                except Exception as exc:  # surface via sentinel + re-raise
                    loop.call_soon_threadsafe(queue.put_nowait, _ProducerError(exc))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = asyncio.to_thread(_producer)
            runner = asyncio.create_task(task)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, _ProducerError):
                        raise item.inner
                    yield item
            finally:
                if not runner.done():
                    runner.cancel()
                    try:
                        await runner
                    except (asyncio.CancelledError, Exception):
                        pass


class _ProducerError:
    __slots__ = ("inner",)
    def __init__(self, inner: BaseException) -> None:
        self.inner = inner


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

    def _build_request(self, system: str, query: str, images, videos):
        from swift.llm import InferRequest

        placeholders = " ".join(["<image>"] * len(images))+" ".join(["<video>"] * len(videos))
        prompt = (placeholders + "" + query).strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
        return InferRequest(messages=messages, images=list(images), videos=list(videos))

    def infer_sync(
        self,
        system: str,
        query: str,
        images,
        videos,
    ) -> str:
        from swift.llm import RequestConfig

        req = self._build_request(system, query, images, videos)
        cfg = RequestConfig(max_tokens=10000, temperature=0, stream=False)
        gen = self.engine.infer([req], cfg)
        return gen[0].choices[0].message.content

    def infer_stream_sync(self, system: str, query: str, images, videos):
        """Yield running-total strings as the model generates. Caller computes deltas."""
        from swift.llm import RequestConfig

        req = self._build_request(system, query, images, videos)
        cfg = RequestConfig(max_tokens=10000, temperature=0, stream=True)
        running = ""
        for batch in self.engine.infer([req], cfg):
            # PtEngine in stream mode yields a batch list on each tick
            item = batch[0] if isinstance(batch, list) else batch
            choice = item.choices[0]
            # delta schema: message.content holds incremental text in swift>=3
            delta = getattr(getattr(choice, "delta", None), "content", None)
            if delta is None:
                delta = getattr(getattr(choice, "message", None), "content", "") or ""
            if delta:
                running += delta
                yield running
