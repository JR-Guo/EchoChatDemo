import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth import RequireAuthMiddleware, router as auth_router
from app.config import get_settings

_start_time = time.monotonic()
_model_ready = False


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="EchoChat Demo", version="0.1.0")
    app.add_middleware(RequireAuthMiddleware)
    app.include_router(auth_router)

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

    return app


app = create_app()
