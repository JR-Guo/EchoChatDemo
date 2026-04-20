# EchoChat Demo — Design

**Date:** 2026-04-18
**Status:** Approved (brainstorming), pending implementation plan
**Audience:** Nature-paper reviewers will see this demo; visual polish and perceived stability are first-class requirements.

---

## 1. Context and Goals

We are building a clinician-facing web demo that showcases the EchoChat echocardiography AI platform in four tasks: **Report Generation**, **Measurement**, **Disease Diagnosis**, and **Visual Question Answering (VQA)**.

The full product requirements live in `docs/target.md`. The essence: doctors must see, at all times, *what they can upload, what they have uploaded, what the system can do with it, and what it cannot*. The flow is upload-all-DICOMs-first, then-run-tasks. The UI must be smooth, bounded, and clinically credible.

**Deployment target:** server `eez195.ece.ust.hk`, code at `/nfs/usrhome2/EchoChatDemo/`, exposed via `echochat.micro-heart.com`.

**Inputs we accept:** arbitrary DICOM files (various extensions and non-standard filenames) from a full echo study. Internally converted to mp4 (cine) or png (still) for the model.

**Supported capabilities (authoritative, from `docs/*.py`):**

- 28 diseases (see `docs/disease(1).py`)
- 22 measurements (see `docs/measurement(1).py`)
- 10-section report: Aortic Valve, Atria, Great Vessels, Left Ventricle, Mitral Valve, Pericardium Pleural, Pulmonic Valve, Right Ventricle, Tricuspid Valve, Summary
- VQA: preset examples (content TBD) + free-text input

**Non-goals (explicit):**

- No per-user account system. A single shared password (`echochat`) gates the UI.
- No persistent patient database. Per-study session directories are the storage model.
- No Docker packaging in this phase; a separate workstream owns that.
- No `stream=true` chat responses from the view classifier (8996 PAH API is not used anyway).
- No mobile-first layout. Desktop is the sole target.
- No real-time collaboration / multi-user edit of a single study.

---

## 2. Architecture

```
Browser (echochat.micro-heart.com)
    │
    │  HTTPS → Nginx → 127.0.0.1:12345
    ▼
┌─────────────────────────────────────────────────────────────┐
│   EchoChatDemo service (FastAPI + Jinja2, port 12345)       │
│                                                             │
│   ┌─── Routers ────────────────────────────────────────┐   │
│   │ pages | upload | study | tasks (+ SSE streams)      │   │
│   └─────────────────────────────────────────────────────┘   │
│   ┌─── Services ────────────────────────────────────────┐   │
│   │ dicom_pipeline   (pydicom + opencv)                 │   │
│   │ view_classifier  (HTTP client → :8995 on cuda:3)    │   │
│   │ echochat_engine  (ms_swift PtEngine, cuda:2)        │   │
│   │ tasks.report | tasks.measurement | tasks.disease |  │   │
│   │ tasks.vqa | export | progress (async pub/sub)       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Single gunicorn worker (model loaded exactly once).       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ reads/writes
                              ▼
/nfs/usrhome2/EchoChatDemo/data/sessions/<study_id>/
    raw/         uploaded .dcm (UUID-renamed)
    converted/   <uuid>.mp4 | <uuid>.png
    meta.json    per-clip view / confidence / user override
    results/     report.json | measurements.json | diseases.json | vqa.json
    exports/     report.pdf | report.docx
```

### 2.1 GPU and external-service allocation

| Service | GPU | Port | Owned by |
|---|---|---|---|
| EchoChat (our new inference engine, `echochatv1.5`) | `cuda:2` | 12345 (web) | this project |
| EchoView38 view classifier (reused as-is) | `cuda:3` | 8995 | `/home/jguoaz/EchoAgent/services/echoview38/` |

The existing `/nfs/usrhome2/echo/usdemo/` (old echochat on port 5000) and the PAH view classifier (port 8996, 4-class mapping) are **not** used. We reimplement from scratch to avoid the prompt-mixing concerns in the legacy `app.py`.

### 2.2 Process model

- One FastAPI app, one gunicorn worker with uvicorn worker class (`--workers 1`), because the EchoChat model is a ~16 GB PtEngine loaded once into GPU.
- In-process `asyncio` handles concurrency. DICOM parsing and file I/O use `run_in_executor` to avoid blocking the event loop.
- All inference calls go through a single `asyncio.Lock`. Concurrent task requests queue; the UI surfaces `Waiting for inference slot…` with queue position.
- Run via `tmux` (named session `echochat-demo`) in phase 1. A `systemd --user` unit is a later polish.

---

## 3. User Flow and Pages

```
/login ──shared password──► /home ──► /upload ──► /workspace/<study_id>
                                                      │
                                                      ├─► Tab: Report
                                                      ├─► Tab: Measurement
                                                      ├─► Tab: Disease
                                                      └─► Tab: VQA
```

### 3.1 Home

Four capability cards describing what each task does, what inputs it needs, and an inline "Start new study" CTA. Cards also call out supported diseases count (28), measurements count (22), and the 10-section report structure.

### 3.2 Upload

Drag-and-drop area accepting any file regardless of extension or filename (target.md 3.2 requires tolerance to non-standard names). Each file is validated client-side by magic byte (`DICM` at offset 128) rather than extension, so renamed DICOMs succeed and non-DICOM files are rejected before upload. Per-file progress bar. After upload, a single SSE stream walks through: `parsing_dicom` → `converting` → `classifying`. The view classifier returns one of 38 classes or `unknown`; unknown clips get a dropdown so the doctor can manually tag. Doppler variants are rolled up under a general "Doppler / Spectrum" bucket — doctors are told to upload all spectra they have; we do not sub-classify.

### 3.3 Workspace (the core page)

Two-pane layout:

- **Left (Study Panel, persistent):** list of detected views as pill badges, color-coded (green = present, amber = missing-but-not-required, gray = unknown). Shows task-availability summary (`Report ready`, `Measurement: all inputs present`, `Disease: can evaluate X of 28`, etc.). "Add more clips" button returns to upload flow, appending to the current study.
- **Right (Task Panel, tabbed):** one tab per task.
  - **Report:** single `Generate Report` button. Sections stream in one-by-one via SSE. Each section is contenteditable; a pencil icon reveals an edit state. `Export PDF` and `Export DOCX` buttons at the bottom. Regenerating after edits prompts a confirmation modal.
  - **Measurement:** left column is the 22-item checkbox list with preset buttons (`Basic 5`, `Valvular Doppler`, `All`). Right column is a running results table — name, value, unit, raw response (collapsible). Items are evaluated sequentially; each completed item renders immediately.
  - **Disease:** same pattern as Measurement — 28-item checkbox list, preset groupings (`Valvular`, `LV function`, `RV function`, `All`), results shown as name / Yes-No / expandable raw response.
  - **VQA:** preset-examples dropdown (placeholder content for now, wired via `/api/constants` so it can be populated later without code change) plus a free-text input. Chat-bubble transcript below.

### 3.4 Required-view policy (soft)

Missing views produce **amber warnings**, never hard blocks. Doctors can always press Run; the system warns what might be unreliable. The model itself is not expected to hallucinate missing evidence, so the warning is informational and lets the doctor judge.

---

## 4. Visual Design

### 4.1 Style direction

Clinical SaaS trust aesthetic. No generic Bootstrap defaults. The output must look intentional at every touchpoint — typography, spacing, empty states, error states, loading states.

### 4.2 Design tokens

| Token | Value | Usage |
|---|---|---|
| `--color-primary` | `#0E6AAD` | headers, primary buttons, active tabs |
| `--color-accent` | `#1F9D8F` | success badges, "ready" indicators |
| `--color-warning` | `#E09B3A` | missing-view warnings, soft alerts |
| `--color-danger` | `#C0392B` | errors, destructive confirmations |
| `--color-bg` | `#F7F9FB` | page background |
| `--color-surface` | `#FFFFFF` | cards, panels |
| `--color-text` | `#1F2937` | body |
| `--color-muted` | `#6B7280` | secondary text |
| shadow | `0 1px 3px rgba(16,24,40,0.04)` | default card |
| shadow-hover | `0 4px 12px rgba(16,24,40,0.08)` | interactive card hover |
| radius | `8px` default, `12px` for large panels | all surfaces |

### 4.3 Typography

- **UI text:** Inter (Google Fonts), weights 400/500/600.
- **Numeric tables:** `font-variant-numeric: tabular-nums` for measurement columns.
- **Monospace (technical fields only):** JetBrains Mono.
- **Display / headers:** Inter Display (fallback: Inter 600).

### 4.4 Framework choice

Tailwind CSS via CDN for utility classes + a small `app.css` defining the tokens above as custom properties and a handful of component classes (`.card`, `.badge-view`, `.tab-underline`). Icons: `lucide` SVG set (inline). No Bootstrap, no bootstrap-icons.

### 4.5 Interaction norms

- Transitions: 200–300 ms ease-out on hovers, tab switches, modal reveals.
- Drag-over on upload area: subtle pulse animation, not flashing borders.
- Tab switches: underline slides under active tab.
- Loading spinners: minimal SVG spinner in accent color; never a full-page blocking overlay.
- Empty states: illustrated SVG + one-line guidance + primary CTA.

### 4.6 Language

All user-facing copy is **English**. No Chinese strings anywhere the user can see. Code comments may be bilingual.

---

## 5. Code Layout

```
/root/sync_workspace/jiarong/eez195/echo_ui/              (local; git-tracked; JR-Guo/EchoChatDemo)
├── app/
│   ├── main.py                       FastAPI app + middleware + startup
│   ├── config.py                     settings (env-driven)
│   ├── auth.py                       shared-password session middleware
│   ├── routers/
│   │   ├── pages.py                  GET /login /home /upload /workspace/<id>
│   │   ├── upload.py                 POST /api/study, POST /api/study/<id>/upload,
│   │   │                             SSE /api/study/<id>/process
│   │   ├── study.py                  GET /api/study/<id>, PATCH clip, thumbnails
│   │   └── tasks.py                  POST task starts, SSE task streams
│   ├── services/
│   │   ├── dicom_pipeline.py         magic-byte sniff, pydicom+opencv → mp4/png, thumbnails
│   │   ├── view_classifier.py        HTTP client for :8995
│   │   ├── echochat_engine.py        PtEngine wrapper, single-use inference API
│   │   ├── tasks/
│   │   │   ├── report.py             10-section prompt orchestrator
│   │   │   ├── measurement.py        per-item prompt orchestrator
│   │   │   ├── disease.py            per-item prompt orchestrator
│   │   │   └── vqa.py                free-text + preset-hook
│   │   ├── export.py                 report → PDF (weasyprint) / DOCX (python-docx)
│   │   └── progress.py               in-memory asyncio.Queue pub/sub per task_id
│   ├── models/
│   │   ├── study.py                  Pydantic schemas
│   │   ├── task.py
│   │   └── view.py
│   └── storage.py                    filesystem accessor for sessions/<id>/
├── templates/
│   ├── base.html                     head: Tailwind CDN, Inter font, favicon
│   ├── components/                   study_panel.html, view_badge.html, ...
│   ├── login.html | home.html | upload.html | workspace.html
├── static/
│   ├── css/app.css                   design tokens + component overrides
│   ├── js/
│   │   ├── upload.js                 drag-drop, magic-byte sniff, SSE
│   │   ├── workspace.js              tabs, SSE consumers, form state
│   │   └── editor.js                 contenteditable report editor + autosave
│   └── img/                          logo, empty-state illustrations (SVG)
├── constants/
│   ├── diseases.py                   authoritative list (mirrors docs/disease(1).py)
│   ├── measurements.py               authoritative list
│   ├── report_sections.py            10-section order + labels
│   ├── view_labels.py                38-class names + grouping (a4c/plax/psax/...)
│   ├── presets.py                    Measurement / Disease quick-pick sets
│   └── prompts.py                    SYSTEM_PROMPT and QUERY templates per task
├── scripts/
│   ├── run_dev.sh                    uvicorn --reload for local
│   ├── run_prod.sh                   gunicorn w/ uvicorn worker under tmux
│   └── deploy.sh                     rsync to /nfs/usrhome2/EchoChatDemo/
├── tests/
│   ├── test_dicom_pipeline.py        synthetic DICOM fixtures, magic-byte cases
│   ├── test_view_classifier.py       mocked :8995, timeout fallback
│   ├── test_tasks.py                 mocked engine, SSE event-sequence checks
│   ├── test_auth.py                  password correct/wrong/forged cookie
│   └── fixtures/
├── docs/
│   ├── target.md                     product requirements (existing)
│   ├── disease(1).py | measurement(1).py | report(1).py
│   └── superpowers/specs/2026-04-18-echochat-demo-design.md   (this file)
├── requirements.txt                  pinned deps (see §8)
├── .env.example                      env var template
├── README.md                         how to run / deploy / extend
└── .gitignore
```

Runtime data (created on first run, not git-tracked):

```
/nfs/usrhome2/EchoChatDemo/data/sessions/<study_id>/
    raw/ converted/ meta.json results/ exports/
```

### 5.1 Boundaries

- `services/tasks/*` each own their own `SYSTEM_PROMPT` and `QUERY` template. They do not share a prompt builder — that is the explicit lesson from legacy `app.py`.
- `services/echochat_engine.py` exposes a single `async def infer(system, query, images, videos) -> str`. It does not know anything about diseases, measurements, or reports.
- `constants/` is pure data. Tests read these directly; runtime imports these.
- `routers/` is HTTP-transport only; it calls into `services/` for all work.

---

## 6. API Contract

### 6.1 Pages

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | redirect to `/home` (or `/login`) |
| GET | `/login` | login form |
| POST | `/login` | validates password, sets session cookie |
| POST | `/logout` | clears cookie |
| GET | `/home` | 4-task overview landing |
| GET | `/upload` | new-study upload page |
| GET | `/workspace/{study_id}` | workspace page (SSR shell; data via APIs) |

### 6.2 Study and upload

| Method | Path | Body / Params | Returns |
|---|---|---|---|
| POST | `/api/study` | — | `{study_id}` |
| POST | `/api/study/{sid}/upload` | multipart `file` | `{file_id, filename, kind, size, saved_as}` |
| GET | `/api/study/{sid}/process` | — | SSE: `phase`, `clip`, `error`, `done` |
| GET | `/api/study/{sid}` | — | study state incl. `clips[]` and `tasks_availability` |
| PATCH | `/api/study/{sid}/clip/{fid}` | `{user_view}` | updated clip |
| DELETE | `/api/study/{sid}/clip/{fid}` | — | 204 |
| GET | `/api/study/{sid}/clip/{fid}/thumbnail` | — | png |
| GET | `/api/study/{sid}/clip/{fid}/video` | — | mp4 |

### 6.3 Tasks

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/study/{sid}/task/report` | `{}` | `{task_id}` |
| POST | `/api/study/{sid}/task/measurement` | `{items: [str]}` | `{task_id}` |
| POST | `/api/study/{sid}/task/disease` | `{items: [str]}` | `{task_id}` |
| POST | `/api/study/{sid}/task/vqa` | `{question: str}` | `{task_id}` |
| GET | `/api/task/{tid}/stream` | — | SSE: `phase`, `partial`, `item`, `message`, `done`, `error` |
| GET | `/api/task/{tid}` | — | final result JSON |

### 6.4 Report edit / export

| Method | Path | Body / Params | Returns |
|---|---|---|---|
| PATCH | `/api/study/{sid}/report/section` | `{section, content}` | updated section |
| GET | `/api/study/{sid}/report/export` | `?format=pdf\|docx` | file stream |

### 6.5 Static catalog

| Method | Path | Returns |
|---|---|---|
| GET | `/api/constants` | `{diseases[], measurements[], report_sections[], views{}, presets{}}` |
| GET | `/healthz` | `{model_loaded: bool, view_classifier_reachable: bool, uptime_s: int}` |

### 6.6 SSE event payloads

```
event: phase
data: {"phase": "preparing_context" | "inference" | "done" | "queued", "detail": "..."}

event: partial                  # report only
data: {"section": "LV", "content": "..."}

event: item                     # measurement / disease
data: {"name": "LV ejection fraction", "value": "55", "unit": "%", "raw": "..."}
data: {"name": "Aortic stenosis", "answer": "no", "raw": "..."}

event: message                  # vqa
data: {"role": "assistant", "content": "..."}

event: done
data: {"task_id": "..."}

event: error
data: {"reason": "..."}
```

---

## 7. Progress Copy, Errors, Edge Cases

### 7.1 Progress copy

| Phase | Report | Measurement | Disease | VQA |
|---|---|---|---|---|
| queued | `Waiting for inference slot…` | same | same | same |
| start | `Preparing echocardiography context…` | `Preparing context for <item>…` | `Preparing context for <item>…` | `Analyzing your question…` |
| during | `Generating section: Left Ventricle` | `Measuring <item>` (i/N) | `Evaluating <item>` (i/N) | `Thinking…` |
| done | `Report generated. You can edit any section below.` | `<X> measurements complete.` | `<X> conditions evaluated.` | — (response inline) |
| error | `Generation failed: <short reason>. Please try again.` | same pattern | same pattern | same pattern |

Upload copy:

| Phase | Copy |
|---|---|
| uploading | `Uploading <filename>… (n of N)` |
| parsing | `Parsing DICOM headers…` |
| converting | `Converting video stream…` |
| classifying | `Identifying view for <filename>…` |
| done | `<N> clips ready. Review views and proceed to workspace.` |

### 7.2 Error classes

| Class | Trigger | UX |
|---|---|---|
| Fatal (500) | Model load fail / GPU OOM / NFS unavailable | Full-page "Service unavailable" banner + retry |
| Recoverable | A single clip fails to parse | Clip card turns red + `Skip` / `Re-upload`; other clips continue |
| Soft warning | Missing views / `unknown` views | Amber badge in Study Panel; never blocks |
| Client validation | Non-DICOM / zero-byte | Rejected in the upload JS; never reaches server |
| Rate / lock | Another task is running | `Waiting for inference slot…` with queue position |

### 7.3 Edge cases

1. **Workspace with no clips** → redirect to `/upload` with `at least 1 clip required` toast.
2. **All clips are `unknown`** → every task shows `manual view required`, except Report (Report does not require specific views).
3. **Page refresh mid-task** → reconnecting SSE is not supported; the client calls `GET /api/task/{tid}` to fetch the final result. In-progress tasks thus finish in background and are retrievable.
4. **User deletes a clip** → existing results are not auto-invalidated; they are flagged `stale: true` until the task is re-run.
5. **Regenerate Report when edits exist** → modal `This will discard your edits. Continue?`
6. **Export when report empty** → button disabled.
7. **Warm-up** → on startup, `/healthz` returns `model_loaded: false` until engine ready. Home page shows a top banner `Model is warming up…` that clears on first successful `/healthz`.
8. **User uploads 100+ DICOMs** → backend streams DICOM parsing to queue; UI shows overall progress. No hard cap, but we warn at 50+ that inference will be slower.

---

## 8. Dependencies

```
# pinned later during implementation
fastapi
uvicorn
gunicorn
sse-starlette
jinja2
python-multipart
pydantic
pydicom
opencv-python-headless
numpy
pillow
ms-swift               # for PtEngine
torch                  # matches existing echochat env
transformers
flash-attn             # optional, matches legacy run.sh
weasyprint             # PDF export
python-docx            # DOCX export
httpx                  # view classifier client
pytest
pytest-asyncio
```

Tailwind + Inter + lucide all load via CDN; no build toolchain required.

---

## 9. Deployment

```bash
# First-time (on server)
cd /nfs/usrhome2/EchoChatDemo
conda env create -f environment.yml       # or reuse an existing matching env
# then
bash scripts/run_prod.sh
```

`scripts/run_prod.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/<user>/anaconda3/etc/profile.d/conda.sh
conda activate echochat-demo
export CUDA_VISIBLE_DEVICES=2
export VIDEO_MAX_PIXELS=20971520
export VIDEO_MIN_PIXELS=13107
export FPS=1
export FPS_MAX_FRAMES=16
export ECHOCHAT_MODEL_PATH=/nfs/usrhome2/yqinar/HKUSData/script/echochatv1.5
export VIEW_CLASSIFIER_URL=http://127.0.0.1:8995
export SHARED_PASSWORD=echochat          # override via .env in prod
tmux new-session -dA -s echochat-demo \
  "exec gunicorn -k uvicorn.workers.UvicornWorker \
       -b 0.0.0.0:12345 --workers 1 --timeout 600 \
       app.main:app 2>&1 | tee -a logs/service.log"
```

Nginx reverse-proxy from `echochat.micro-heart.com` → `127.0.0.1:12345` is provisioned by the domain owner (not this project).

Developer workflow:

1. Write/modify code locally in `/root/sync_workspace/jiarong/eez195/echo_ui/`.
2. `git commit && git push` to `JR-Guo/EchoChatDemo`.
3. `bash scripts/deploy.sh` (runs rsync over SSH, excluding `.git`, `__pycache__`, `data/`).
4. On server, restart tmux session or `tmux send-keys -t echochat-demo C-c` + re-run.

---

## 10. Testing Strategy

| Layer | Scope | Tool |
|---|---|---|
| Unit | dicom pipeline (synthetic DICOM fixtures, magic-byte sniff, cine → mp4) | pytest |
| Unit | view classifier client (mocked httpx responses; timeout + `unknown` paths) | pytest + `respx` |
| Unit | each task orchestrator (mocked `echochat_engine.infer`, asserts prompt + SSE event sequence) | pytest-asyncio |
| Unit | auth middleware (correct, wrong, forged cookie) | pytest + TestClient |
| Integration (slow, opt-in) | real model, one report run end-to-end | `pytest -m slow` |
| Manual smoke | `scripts/smoke.sh` — upload 3 real DICOMs, observe views, run report, download PDF | shell |

A pre-merge check requires the fast unit suite to pass. Integration tests are expected to be run on the server before each demo session.

---

## 11. Open Items

The following are deferred and do not block implementation:

- **VQA preset content** — the list of example questions is surfaced through `/api/constants.presets.vqa` and rendered by the client. Empty list for launch; a domain expert fills it without code changes.
- **Additional view-to-task requirement mapping** — initial heuristic: Measurement items tagged as "requires Doppler" warn when no Doppler clips are present; Disease items tagged as "valvular" warn when no parasternal color clips are present. Heuristic table lives in `constants/view_labels.py`; refined by a clinician.
- **Authoring a proper empty-state illustration set** — launches with a minimal lucide-based placeholder; design refresh is later.
- **Report editor autosave cadence** — starts with blur-triggered save; consider debounced-on-type in a later iteration.
- **Systemd unit and log rotation** — tmux is the phase-1 answer; systemd later.

---

## 12. Traceability to `target.md`

| `target.md` requirement | Addressed in |
|---|---|
| 2.1 inputs clear | §3.1 Home; §3.2 Upload guidance; §4 visual design |
| 2.2 process smooth | §5.1 boundaries; §6.6 SSE streams; §7.1 progress copy |
| 2.3 capability boundaries | §1 non-goals; §3.3 Left Study Panel; §5 constants |
| 2.4 code clean split | §5 code layout (services/tasks isolation); §9 deployment |
| 3.2 strong upload guidance / format tolerance | §3.2 magic-byte sniff; §7.3 edge cases #8 |
| 3.4 staged progress feedback | §6.6 SSE event types; §7.1 progress copy |
| 3.5 results suit demo viewing | §3.3 task panel layouts; §4 typography |
| 5 required 4 capabilities | §1 goals; §3.3 tabs; §5 `services/tasks/*` |
| 6 items to finalize later | §11 open items |
