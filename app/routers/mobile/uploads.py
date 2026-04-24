"""tus 1.0.0 upload server for mobile, mounted at /api/mobile/v1/uploads.

Supports the `creation` and `termination` extensions. Validates Bearer token
on every request. State + storage lives under `<data_dir>/uploads/<id>/`
(see `app.services.mobile_uploads`).

Not using `fastapi-tus` — hand-rolled because we need to keep the dependency
set small (this repo runs on a shared conda env) and because the behavior we
need is well-scoped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.config import get_settings
from app.mobile_auth import require_mobile_user
from app.services.mobile_uploads import (
    MAX_UPLOAD_SIZE,
    UploadStore,
    parse_upload_metadata,
)


router = APIRouter(tags=["mobile-uploads"])


TUS_VERSION = "1.0.0"


def _store() -> UploadStore:
    return UploadStore(base_dir=Path(get_settings().data_dir) / "uploads")


def _tus_headers(extra: Optional[dict] = None) -> dict:
    h = {"Tus-Resumable": TUS_VERSION}
    if extra:
        h.update(extra)
    return h


def _require_tus_header(request: Request) -> None:
    got = request.headers.get("tus-resumable")
    if got != TUS_VERSION:
        raise HTTPException(
            status.HTTP_412_PRECONDITION_FAILED,
            detail=f"Tus-Resumable must be {TUS_VERSION}",
            headers=_tus_headers({"Tus-Version": TUS_VERSION}),
        )


@router.options("/uploads")
def options_uploads() -> Response:
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers=_tus_headers({
            "Tus-Version": TUS_VERSION,
            "Tus-Extension": "creation,termination",
            "Tus-Max-Size": str(MAX_UPLOAD_SIZE),
        }),
    )


@router.post("/uploads", dependencies=[Depends(require_mobile_user)])
async def create_upload(request: Request) -> Response:
    _require_tus_header(request)
    length_hdr = request.headers.get("upload-length")
    if not length_hdr:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Upload-Length required")
    try:
        length = int(length_hdr)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Upload-Length not an integer")
    if length <= 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Upload-Length must be positive")
    if length > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Upload-Length exceeds max {MAX_UPLOAD_SIZE} bytes",
            headers=_tus_headers(),
        )

    meta = parse_upload_metadata(request.headers.get("upload-metadata", ""))
    filename = meta.get("filename") or meta.get("clientFilename") or "upload.bin"
    filetype = meta.get("filetype") or "application/octet-stream"

    upload = _store().create(length=length, filename=filename, filetype=filetype)
    location = f"/api/mobile/v1/uploads/{upload.upload_id}"
    return Response(
        status_code=status.HTTP_201_CREATED,
        headers=_tus_headers({"Location": location, "Upload-Offset": "0"}),
    )


@router.head("/uploads/{upload_id}", dependencies=[Depends(require_mobile_user)])
def head_upload(upload_id: str, request: Request) -> Response:
    _require_tus_header(request)
    meta = _store().load(upload_id)
    if meta is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, headers=_tus_headers())
    return Response(
        status_code=status.HTTP_200_OK,
        headers=_tus_headers({
            "Upload-Offset": str(meta.offset),
            "Upload-Length": str(meta.length),
            "Cache-Control": "no-store",
        }),
    )


@router.patch("/uploads/{upload_id}", dependencies=[Depends(require_mobile_user)])
async def patch_upload(upload_id: str, request: Request) -> Response:
    _require_tus_header(request)
    ctype = request.headers.get("content-type", "")
    if ctype != "application/offset+octet-stream":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/offset+octet-stream",
            headers=_tus_headers(),
        )
    offset_hdr = request.headers.get("upload-offset")
    if offset_hdr is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Upload-Offset required")
    try:
        offset = int(offset_hdr)
    except ValueError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Upload-Offset not integer")

    store = _store()
    meta = store.load(upload_id)
    if meta is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, headers=_tus_headers())
    if offset != meta.offset:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Offset mismatch: have {meta.offset}, got {offset}",
            headers=_tus_headers({"Upload-Offset": str(meta.offset)}),
        )

    body = await request.body()
    try:
        new_offset = store.append_chunk(upload_id, offset, body)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc), headers=_tus_headers())

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers=_tus_headers({"Upload-Offset": str(new_offset)}),
    )


@router.delete("/uploads/{upload_id}", dependencies=[Depends(require_mobile_user)])
def delete_upload(upload_id: str, request: Request) -> Response:
    _require_tus_header(request)
    store = _store()
    if not store.exists(upload_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, headers=_tus_headers())
    store.delete(upload_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=_tus_headers())
