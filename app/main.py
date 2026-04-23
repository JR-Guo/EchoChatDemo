import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.auth import RequireAuthMiddleware, router as auth_router
from app.config import get_settings
from app.routers.meta import router as meta_router
from app.routers.upload import router as upload_router
from app.routers.study import router as study_router
from app.routers.tasks import router as tasks_router, set_engine
from app.routers.report_io import router as report_io_router
from app.routers.mobile import router as mobile_router

_start_time = time.monotonic()
_model_ready = False

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if os.environ.get("ECHOCHAT_SKIP_MODEL", "").lower() in ("1", "true", "yes"):
        yield
        return
    import asyncio
    from app.services.echochat_engine import EchoChatEngine, SwiftPtBackend

    settings_local = get_settings()

    def _build():
        be = SwiftPtBackend(str(settings_local.model_path))
        return EchoChatEngine(backend=be)

    eng = await asyncio.to_thread(_build)
    set_engine(eng)
    global _model_ready
    _model_ready = True
    try:
        yield
    finally:
        pass


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="EchoChat Demo", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(RequireAuthMiddleware)
    app.include_router(auth_router)
    app.include_router(meta_router)
    app.include_router(upload_router)
    app.include_router(study_router)
    app.include_router(tasks_router)
    app.include_router(report_io_router)
    app.include_router(mobile_router)

    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "uptime_s": int(time.monotonic() - _start_time),
            "model_loaded": _model_ready,
            "view_classifier_url": settings.view_classifier_url,
        }

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/home", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, error: str | None = None):
        return templates.TemplateResponse(
            request, "login.html", {"error": error}
        )

    @app.get("/home", response_class=HTMLResponse)
    def home_page(request: Request):
        cards = [
            {
                "title": "Report Generation",
                "desc": "Generate a sectioned echocardiography report.",
                "inputs": ["Full echo study are needed to generated the report."],
                "stat": "Sectioned",
                "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
            },
            {
                "title": "Measurement",
                "desc": "Select structured echo parameters to measure, with clinical units.",
                "inputs": ["Full echo study."],
                "stat": "26 measurements",
                "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 3H3v18h18z"/><path d="M7 7v10"/><path d="M11 10v7"/><path d="M15 13v4"/></svg>',
            },
            {
                "title": "Disease Diagnosis",
                "desc": "Evaluate presence of supported cardiac conditions from the uploaded study.",
                "inputs": ["Full echo study."],
                "stat": "28 conditions",
                "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-6 8-12a8 8 0 1 0-16 0c0 6 8 12 8 12z"/></svg>',
            },
            {
                "title": "Visual Question Answering",
                "desc": "Ask focused clinical questions about the uploaded study.",
                "inputs": ["Questions must be echo-related."],
                "stat": "Bounded input",
                "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>',
            },
        ]
        return templates.TemplateResponse(request, "home.html", {"cards": cards})

    @app.get("/upload", response_class=HTMLResponse)
    def upload_page(request: Request):
        sid = (request.query_params.get("study_id") or "").strip() or None
        return templates.TemplateResponse(
            request, "upload.html", {"study_id": sid}
        )

    @app.get("/workspace/{study_id}", response_class=HTMLResponse)
    def workspace_page(request: Request, study_id: str):
        return templates.TemplateResponse(
            request, "workspace.html", {"study_id": study_id},
        )

    return app


app = create_app()
