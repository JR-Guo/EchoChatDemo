from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.config import get_settings
from app.mobile_auth import require_mobile_user, sign_path
from app.routers.tasks import get_engine
from app.models.task import TaskStatus
from app.services.progress import ProgressHub
from app.services.tasks.report import run_report
from app.storage import Storage


logger = logging.getLogger("echochat.mobile.report")

router = APIRouter(tags=["mobile-reports"], dependencies=[Depends(require_mobile_user)])


ReportStatus = Literal["pending", "ready", "failed"]


class Report(BaseModel):
    id: str
    studyId: str
    status: ReportStatus
    pdfUrl: Optional[str] = None
    docxUrl: Optional[str] = None
    generatedAt: Optional[str] = None


class ReportSectionOut(BaseModel):
    name: str
    content: str


class ReportFull(Report):
    sections: list[ReportSectionOut] = []


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


def _iso_utc(ts_epoch: float) -> str:
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _report_json_path(sid: str) -> Path:
    return _store().result_path(sid, "report.json")


def _pdf_path(sid: str) -> Path:
    return _store().export_path(sid, "report.pdf")


def _docx_path(sid: str) -> Path:
    return _store().export_path(sid, "report.docx")


def _render_exports(sid: str, data: dict) -> None:
    """Pre-render PDF + DOCX from the report dict so signed download URLs
    work as soon as the caller sees status=ready. Per-format failures are
    non-fatal — the caller can always re-POST to regenerate."""
    from app.services.export import render_report_docx, render_report_pdf

    try:
        pdf_path = _pdf_path(sid)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        render_report_pdf(data, pdf_path)
    except Exception as exc:
        logger.warning("report PDF render failed for %s: %s", sid, exc)
    try:
        docx_path = _docx_path(sid)
        docx_path.parent.mkdir(parents=True, exist_ok=True)
        render_report_docx(data, docx_path)
    except Exception as exc:
        logger.warning("report DOCX render failed for %s: %s", sid, exc)


def _load_report_json(sid: str) -> Optional[dict]:
    p = _report_json_path(sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _build_report(sid: str) -> Report:
    store = _store()
    root = store.study_root(sid)
    if not root.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")

    data = _load_report_json(sid)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No report generated yet")

    # Lazy-render PDF/DOCX if the JSON is newer than the exports. This handles
    # the case where report.json was written by a web-side task but mobile
    # exports haven't been rendered yet.
    json_mtime = _report_json_path(sid).stat().st_mtime
    pdf, docx = _pdf_path(sid), _docx_path(sid)
    needs_render = (
        not pdf.exists() or pdf.stat().st_mtime < json_mtime
        or not docx.exists() or docx.stat().st_mtime < json_mtime
    )
    if needs_render:
        _render_exports(sid, data)

    pdf_exists = pdf.exists()
    docx_exists = docx.exists()
    newest_ts = max(
        (pdf.stat().st_mtime if pdf_exists else 0),
        (docx.stat().st_mtime if docx_exists else 0),
        json_mtime,
    )

    return Report(
        id=f"{sid}-report",
        studyId=sid,
        status="ready",
        pdfUrl=sign_path(f"/api/mobile/v1/studies/{sid}/report/pdf") if pdf_exists else None,
        docxUrl=sign_path(f"/api/mobile/v1/studies/{sid}/report/docx") if docx_exists else None,
        generatedAt=_iso_utc(newest_ts),
    )


@router.get("/studies/{sid}/report", response_model=ReportFull)
def get_report(sid: str) -> ReportFull:
    base = _build_report(sid)
    data = _load_report_json(sid) or {}
    sections = [
        ReportSectionOut(name=s.get("name", ""), content=s.get("content", ""))
        for s in (data.get("sections") or [])
    ]
    return ReportFull(**base.model_dump(), sections=sections)


async def _generate_sync(sid: str, engine) -> dict:
    """Run report inference inline and persist report.json. Returns the dict
    that was written so the caller can render PDF/DOCX immediately.
    """
    store = _store()
    study = store.load_study(sid)
    if not study.clips:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Study has no clips; upload media before generating a report.",
        )

    # Pre-validate images so a broken PNG does not hang the ms-swift worker
    # pool. Same defense as the chat endpoint.
    from PIL import Image
    kept = []
    bad = 0
    for c in study.clips:
        if not c.converted_path:
            continue
        if c.is_video:
            kept.append(c)  # decord validates lazily
            continue
        try:
            with Image.open(c.converted_path) as im:
                im.verify()
            kept.append(c)
        except Exception:
            bad += 1
    if not kept:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No readable media in this study"
                + (f" ({bad} invalid file(s) skipped)." if bad else ".")
            ),
        )

    # ProgressHub is required by run_report but mobile doesn't surface progress
    # for M3 — the hub just buffers events no subscriber will drain.
    hub = ProgressHub()
    task_id = uuid.uuid4().hex[:16]

    result = await run_report(task_id=task_id, clips=kept, engine=engine, hub=hub)
    if result.status != TaskStatus.DONE:
        err = getattr(result, "error", None) or "inference error"
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {err}",
        )

    data = result.model_dump()
    out = _report_json_path(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    _render_exports(sid, data)
    return data


@router.post("/studies/{sid}/report", response_model=ReportFull)
async def generate_report(sid: str, engine=Depends(get_engine)) -> ReportFull:
    """Synchronously generate the report: model inference → report.json →
    PDF + DOCX rendered into exports/. Returns the full report once ready.

    Typical latency is 30-60s for a multi-clip study. Gunicorn --timeout=600
    gives ample headroom; the mobile client shows a spinner during this
    window via the existing ReportScreen flow.
    """
    store = _store()
    root = store.study_root(sid)
    if not root.exists() or (root / ".deleted").exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study not found")

    await _generate_sync(sid, engine)
    return get_report(sid)
