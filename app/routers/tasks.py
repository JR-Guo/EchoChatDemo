from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.models.task import TaskKind, TaskStatus
from app.services.progress import hub, ProgressEvent
from app.services.tasks.report import run_report
from app.services.tasks.measurement import run_measurement
from app.services.tasks.disease import run_disease
from app.services.tasks.vqa import run_vqa
from app.storage import Storage

router = APIRouter()

_engine_holder: dict = {"engine": None}


def set_engine(engine) -> None:
    _engine_holder["engine"] = engine


def get_engine():
    engine = _engine_holder["engine"]
    if engine is None:
        raise HTTPException(status_code=503,
                            detail="Model is still warming up. Try again shortly.")
    return engine


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


_tasks: dict[str, dict] = {}


def _register_task(tid: str, kind: TaskKind, sid: str) -> None:
    _tasks[tid] = {"kind": kind, "study_id": sid, "result": None,
                   "status": TaskStatus.QUEUED}


def _set_result(tid: str, result) -> None:
    rec = _tasks.get(tid)
    if not rec:
        return
    rec["result"] = result.model_dump() if hasattr(result, "model_dump") else result
    rec["status"] = (result.status if hasattr(result, "status") else TaskStatus.DONE)


async def _run_and_persist(tid: str, kind: TaskKind, sid: str, coro_factory):
    _tasks[tid]["status"] = TaskStatus.RUNNING
    try:
        result = await coro_factory()
    except Exception as e:
        await hub.publish(tid, ProgressEvent(kind="error", data={"reason": str(e)}))
        _tasks[tid]["status"] = TaskStatus.ERROR
        return
    _set_result(tid, result)
    s = _store()
    filename = {
        TaskKind.REPORT: "report.json",
        TaskKind.MEASUREMENT: "measurements.json",
        TaskKind.DISEASE: "diseases.json",
        TaskKind.VQA: "vqa.json",
    }[kind]
    s.save_result(sid, filename, result.model_dump())


@router.post("/api/study/{sid}/task/report")
async def start_report(sid: str, engine=Depends(get_engine)):
    s = _store()
    study = s.load_study(sid)
    if not study.clips:
        raise HTTPException(status_code=400, detail="Study has no clips.")
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.REPORT, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.REPORT, sid,
        lambda: run_report(task_id=tid, clips=study.clips, engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.post("/api/study/{sid}/task/measurement")
async def start_measurement(sid: str, body: dict = Body(...), engine=Depends(get_engine)):
    items = body.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items[] required")
    s = _store()
    study = s.load_study(sid)
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.MEASUREMENT, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.MEASUREMENT, sid,
        lambda: run_measurement(task_id=tid, clips=study.clips, items=items,
                                engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.post("/api/study/{sid}/task/disease")
async def start_disease(sid: str, body: dict = Body(...), engine=Depends(get_engine)):
    items = body.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items[] required")
    s = _store()
    study = s.load_study(sid)
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.DISEASE, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.DISEASE, sid,
        lambda: run_disease(task_id=tid, clips=study.clips, items=items,
                            engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.post("/api/study/{sid}/task/vqa")
async def start_vqa(sid: str, body: dict = Body(...), engine=Depends(get_engine)):
    q = (body.get("question") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question required")
    s = _store()
    study = s.load_study(sid)
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.VQA, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.VQA, sid,
        lambda: run_vqa(task_id=tid, clips=study.clips, question=q,
                        engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.get("/api/task/{tid}/stream")
async def stream_task(tid: str):
    if tid not in _tasks:
        raise HTTPException(status_code=404)

    async def gen():
        async for evt in hub.subscribe(tid):
            yield {"event": evt.kind, "data": json.dumps(evt.data)}

    return EventSourceResponse(gen())


@router.get("/api/task/{tid}")
def get_task(tid: str):
    rec = _tasks.get(tid)
    if not rec:
        raise HTTPException(status_code=404)
    if rec["result"] is None:
        return {"status": rec["status"].value if hasattr(rec["status"], "value") else rec["status"]}
    return rec["result"]
