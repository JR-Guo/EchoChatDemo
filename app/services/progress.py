from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ProgressEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class _Channel:
    def __init__(self):
        self.queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self.buffer: list[ProgressEvent] = []
        self.done: bool = False


class ProgressHub:
    """Per-task in-memory pub/sub.

    Publishers push events; subscribers receive them in order. Late subscribers
    get a replay of everything already emitted plus live events until a 'done'
    or 'error' event arrives.

    Single-process only.
    """

    def __init__(self):
        self._channels: dict[str, _Channel] = {}

    def _chan(self, task_id: str) -> _Channel:
        if task_id not in self._channels:
            self._channels[task_id] = _Channel()
        return self._channels[task_id]

    async def publish(self, task_id: str, evt: ProgressEvent) -> None:
        ch = self._chan(task_id)
        ch.buffer.append(evt)
        await ch.queue.put(evt)
        if evt.kind in ("done", "error"):
            ch.done = True

    async def subscribe(self, task_id: str) -> AsyncIterator[ProgressEvent]:
        ch = self._chan(task_id)
        for evt in list(ch.buffer):
            yield evt
            if evt.kind in ("done", "error"):
                return
        if ch.done:
            return
        while True:
            evt = await ch.queue.get()
            yield evt
            if evt.kind in ("done", "error"):
                return


hub = ProgressHub()
