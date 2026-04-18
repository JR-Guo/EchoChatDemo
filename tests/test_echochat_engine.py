import asyncio

import pytest

from app.services.echochat_engine import EchoChatEngine


class FakeBackend:
    def __init__(self, response: str = "stub response"):
        self.response = response
        self.calls: list[dict] = []

    def infer_sync(self, system, query, images, videos):
        self.calls.append(
            dict(system=system, query=query, images=list(images), videos=list(videos))
        )
        return self.response


async def test_engine_delegates_to_backend():
    be = FakeBackend("hello")
    eng = EchoChatEngine(backend=be)
    out = await eng.infer(system="S", query="Q", images=["a.png"], videos=["b.mp4"])
    assert out == "hello"
    assert be.calls[0]["system"] == "S"
    assert be.calls[0]["query"] == "Q"


async def test_engine_serializes_concurrent_calls():
    order: list[str] = []

    class SeqBackend:
        def __init__(self):
            self.i = 0
        def infer_sync(self, system, query, images, videos):
            self.i += 1
            order.append(f"start-{self.i}")
            import time
            time.sleep(0.02)
            order.append(f"end-{self.i}")
            return str(self.i)

    eng = EchoChatEngine(backend=SeqBackend())
    results = await asyncio.gather(
        eng.infer(system="s", query="q", images=[], videos=[]),
        eng.infer(system="s", query="q", images=[], videos=[]),
        eng.infer(system="s", query="q", images=[], videos=[]),
    )
    assert order == ["start-1", "end-1", "start-2", "end-2", "start-3", "end-3"]
    assert results == ["1", "2", "3"]
