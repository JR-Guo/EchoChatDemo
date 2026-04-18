import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import RequireAuthMiddleware, router as auth_router
from app.config import get_settings
from app.routers.meta import router as meta_router
from app.routers.report_io import router as report_io_router
from app.routers.study import router as study_router
from app.routers.tasks import router as tasks_router, set_engine
from app.routers.upload import router as upload_router

_start_time = time.monotonic()
_model_ready = False


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="EchoChat Demo", version="0.1.0")
    app.add_middleware(RequireAuthMiddleware)
    app.include_router(auth_router)
    app.include_router(meta_router)
    app.include_router(upload_router)
    app.include_router(study_router)
    app.include_router(tasks_router)
    app.include_router(report_io_router)

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
    def login_page():
        return HTMLResponse(
            "<form method='post' action='/login'>"
            "<input type='password' name='password'/>"
            "<button>Login</button></form>"
        )

    @app.get("/home", response_class=HTMLResponse)
    def home_page():
        return HTMLResponse("<h1>Home (stub)</h1>")

    @app.on_event("startup")
    async def _load_engine():
        import os, asyncio
        if os.environ.get("ECHOCHAT_SKIP_MODEL", "").lower() in ("1", "true", "yes"):
            return
        from app.services.echochat_engine import EchoChatEngine, SwiftPtBackend

        settings_local = get_settings()
        def _build():
            be = SwiftPtBackend(str(settings_local.model_path))
            return EchoChatEngine(backend=be)

        eng = await asyncio.to_thread(_build)
        set_engine(eng)
        globals()["_model_ready"] = True

    return app


app = create_app()
