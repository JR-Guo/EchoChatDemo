"""Mobile JSON + SSE surface, mounted at /api/mobile/v1/.

See EchoChatMobile/docs/API.md for the wire contract.
"""
from fastapi import APIRouter

from .auth import router as auth_router
from .chat import router as chat_router
from .downloads import router as downloads_router
from .reports import router as reports_router
from .studies import router as studies_router
from .uploads import router as uploads_router


router = APIRouter(prefix="/api/mobile/v1")
router.include_router(auth_router)
router.include_router(uploads_router)
router.include_router(studies_router)
router.include_router(chat_router)
router.include_router(reports_router)
router.include_router(downloads_router)
