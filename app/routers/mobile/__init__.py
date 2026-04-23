"""Mobile JSON + SSE surface, mounted at /api/mobile/v1/.

See EchoChatMobile/docs/API.md for the wire contract.
"""
from fastapi import APIRouter

from .auth import router as auth_router
from .downloads import router as downloads_router
from .reports import router as reports_router
from .studies import router as studies_router


router = APIRouter(prefix="/api/mobile/v1")
router.include_router(auth_router)
router.include_router(studies_router)
router.include_router(reports_router)
router.include_router(downloads_router)
