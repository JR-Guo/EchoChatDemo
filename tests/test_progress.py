import asyncio

import pytest

from app.services.progress import ProgressHub, ProgressEvent


async def test_publish_and_subscribe():
    hub = ProgressHub()
    events: list[ProgressEvent] = []

    async def reader():
        async for evt in hub.subscribe("task-1"):
            events.append(evt)
            if evt.kind == "done":
                break

    r = asyncio.create_task(reader())
    await asyncio.sleep(0)
    await hub.publish("task-1", ProgressEvent(kind="phase", data={"phase": "start"}))
    await hub.publish("task-1", ProgressEvent(kind="done", data={}))
    await asyncio.wait_for(r, timeout=1.0)
    assert [e.kind for e in events] == ["phase", "done"]


async def test_late_subscriber_still_gets_end():
    hub = ProgressHub()
    await hub.publish("t2", ProgressEvent(kind="phase", data={"phase": "x"}))
    await hub.publish("t2", ProgressEvent(kind="done", data={}))

    seen: list[str] = []
    async for evt in hub.subscribe("t2"):
        seen.append(evt.kind)
        if evt.kind == "done":
            break
    assert seen == ["phase", "done"]
