import time
from fastapi import FastAPI

from app.config import get_settings

_start_time = time.monotonic()
_model_ready = False


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="EchoChat Demo", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "uptime_s": int(time.monotonic() - _start_time),
            "model_loaded": _model_ready,
            "view_classifier_url": settings.view_classifier_url,
        }

    return app


app = create_app()
