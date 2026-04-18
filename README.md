# EchoChat Demo

Web demo showcasing four echocardiography AI tasks — **Report generation**, **Measurement**, **Disease diagnosis**, **Visual question answering** — on uploaded DICOM studies.

- Product requirements: `docs/target.md`
- Design spec: `docs/superpowers/specs/2026-04-18-echochat-demo-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-18-echochat-demo.md`

## Prerequisites

- Python 3.10+ (tested with 3.12)
- GPU with sufficient VRAM for `echochatv1.5` (used on `cuda:2` by default)
- View classifier service reachable at `http://127.0.0.1:8995` (`EchoView38Classifier`)
- NFS-writable path for `ECHOCHAT_DATA_DIR`

## Local dev (hot reload, model loading skipped)

```bash
cp .env.example .env
# edit values as needed
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pytest -m "not slow"
bash scripts/run_dev.sh
```

Open `http://127.0.0.1:12345/login`. Password comes from `SHARED_PASSWORD` (default: `echochat`).

## Production (on eez195)

```bash
# From a local checkout (optional sync path):
bash scripts/deploy.sh
# Then, on the server:
ssh jguoaz@eez195 'cd /nfs/usrhome2/EchoChatDemo && bash scripts/run_prod.sh'
```

The service listens on `0.0.0.0:12345`. An Nginx reverse proxy maps `echochat.micro-heart.com` to `127.0.0.1:12345` (configured separately by the domain owner).

## Smoke test

```bash
bash scripts/smoke.sh /path/to/example.dcm
```

Exercises login → create study → upload → process (SSE) → run report → stream → export PDF.

## Repository layout

```
app/
  main.py             FastAPI app factory + page routes + startup engine loader
  config.py           pydantic-settings Settings from env
  auth.py             shared-password middleware + /login, /logout
  storage.py          filesystem accessor for sessions/<sid>/
  models/             Pydantic schemas (Clip, Study, TaskResult...)
  routers/
    meta.py           /api/constants, healthz baseline
    upload.py         /api/study, /api/study/<sid>/upload, /process SSE
    study.py          /api/study/<sid> GET/PATCH/DELETE, thumbnails
    tasks.py          /api/study/<sid>/task/* POST, /api/task/<tid>/stream SSE
    report_io.py      /api/study/<sid>/report/section PATCH, /export GET
  services/
    dicom_pipeline.py pydicom + opencv: DICOM → mp4 / png + thumbnails
    view_classifier.py httpx client for :8995
    echochat_engine.py async wrapper + asyncio.Lock + SwiftPtBackend
    export.py         report → PDF (weasyprint), DOCX (python-docx)
    progress.py       per-task in-memory pub/sub
    tasks/            report / measurement / disease / vqa orchestrators

constants/            authoritative lists (28 diseases, 22 measurements, 38 views, prompts, presets)
templates/            Jinja2 pages (base, login, home, upload, workspace) + 4 task-tab components
static/
  css/app.css         design tokens layered on Tailwind CDN
  js/                 upload.js, workspace.js, editor.js
  img/                logo.svg
scripts/              run_dev.sh, run_prod.sh, deploy.sh, smoke.sh
tests/                pytest suite (unit + integration, mocked engine + view classifier)
docs/                 target.md, spec, plan
```

## Extending capabilities

- **New measurement** — add the canonical name to `constants/measurements.py`; optionally add to a preset in `constants/presets.py`. `tests/test_constants.py` enforces the set-length invariant.
- **New disease** — add to `constants/diseases.py`; optionally add to `DISEASE_PRESETS`.
- **New view class** — add to `VIEW_LABELS` in `constants/view_labels.py` and update `_GROUP_PREFIXES` if the coarse group is new. Doppler / spectrum variants do not need separate handling; `is_doppler` already catches PW/CW/TDI/M-mode/Color/LVO/MCE tokens.
- **VQA example questions** — populate `VQA_EXAMPLES` in `constants/presets.py`. The UI dropdown is hidden while the list is empty.

No code change is required anywhere else; `/api/constants` re-serves the updated lists on each request and the workspace page picks them up on next load.

## License

Internal demo; not for clinical use.
