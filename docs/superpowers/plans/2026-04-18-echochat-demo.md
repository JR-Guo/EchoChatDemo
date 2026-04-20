# EchoChat Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + Jinja2 web demo for echocardiography AI (Report, Measurement, Disease, VQA) with DICOM upload, view classification, and polished clinical UI, deployed on eez195 under `echochat.micro-heart.com`.

**Architecture:** Single FastAPI process on port 12345 with one gunicorn worker. EchoChat model (`echochatv1.5`, ms_swift PtEngine) loaded once on `cuda:2`. View classification delegated to existing HTTP service at `http://127.0.0.1:8995` (cuda:3). Session data lives at `/nfs/usrhome2/EchoChatDemo/data/sessions/<study_id>/`. SSE streams progress; filesystem is the persistence layer.

**Tech Stack:** Python 3.10+, FastAPI, Jinja2, sse-starlette, pydantic, pydicom, opencv-python-headless, ms-swift, torch, weasyprint, python-docx, httpx, pytest + pytest-asyncio + respx. Frontend: Tailwind CSS (CDN), Inter font (Google Fonts), lucide SVG icons, vanilla JS (no build step).

**Design spec:** `docs/superpowers/specs/2026-04-18-echochat-demo-design.md`

---

## Phases (runnable milestones)

- **Phase 0 — Scaffold:** repo layout + env + lint + trivial FastAPI that boots.
- **Phase 1 — Data layer:** constants, prompts, Pydantic schemas, storage helpers. `pytest` green.
- **Phase 2 — Inference clients:** DICOM pipeline, view classifier client, EchoChat engine wrapper. Unit-testable with mocks; the engine has an opt-in integration test with `-m slow`.
- **Phase 3 — API + auth:** login/logout, upload, study process SSE, study state, clip patch/delete. `curl` flow end-to-end works without templates.
- **Phase 4 — Task orchestrators:** report, measurement, disease, VQA + task stream SSE. End-to-end via `curl` produces real results.
- **Phase 5 — Frontend:** base template + login/home/upload/workspace pages, CSS tokens, JS for upload/workspace/editor.
- **Phase 6 — Export:** report PDF via weasyprint + DOCX via python-docx.
- **Phase 7 — Deploy + smoke:** run scripts, tmux, README, smoke script.

---

## File Structure (locked in)

```
app/
  __init__.py
  main.py                FastAPI factory, static/templates mount, startup
  config.py              pydantic-settings; reads env
  auth.py                shared-password session middleware + /login + /logout
  storage.py             filesystem accessor for sessions/<sid>/
  routers/
    __init__.py
    pages.py             GET /login /home /upload /workspace/<sid>
    upload.py            POST /api/study, POST /api/study/<sid>/upload, SSE process
    study.py             GET /api/study/<sid>, PATCH/DELETE clip, thumbnails
    tasks.py             POST task starts, SSE task stream, GET final result
    report_io.py         PATCH section, GET export (PDF/DOCX)
    meta.py              GET /api/constants, GET /healthz
  services/
    __init__.py
    dicom_pipeline.py    magic-byte sniff, pydicom->mp4/png, thumbnails
    view_classifier.py   httpx client for :8995
    echochat_engine.py   ms_swift PtEngine wrapper + asyncio.Lock
    export.py            report -> PDF (weasyprint), DOCX (python-docx)
    progress.py          per-task asyncio.Queue pub/sub
    tasks/
      __init__.py
      report.py          10-section orchestrator
      measurement.py     per-item orchestrator
      disease.py         per-item orchestrator
      vqa.py             free-text orchestrator
  models/
    __init__.py
    study.py             Pydantic: Clip, Study, TasksAvailability
    task.py              Pydantic: TaskKind, TaskStatus, ReportResult,
                         MeasurementResult, DiseaseResult, VQAMessage
    view.py              ViewLabel enum-like + helpers

constants/
  __init__.py
  diseases.py            SUPPORT_DISEASES (28)
  measurements.py        SUPPORT_MEASUREMENTS (22)
  report_sections.py     REPORT_SECTIONS (10)
  view_labels.py         VIEW_LABELS (38) + view-group helpers
  presets.py             measurement/disease quick-picks + vqa preset stub
  prompts.py             SYSTEM_PROMPT and QUERY per task

templates/
  base.html
  login.html
  home.html
  upload.html
  workspace.html
  components/
    study_panel.html
    view_badge.html
    task_tab_report.html
    task_tab_measurement.html
    task_tab_disease.html
    task_tab_vqa.html
    progress_stream.html

static/
  css/app.css
  js/upload.js
  js/workspace.js
  js/editor.js
  img/logo.svg
  img/empty_upload.svg
  img/empty_results.svg

scripts/
  run_dev.sh
  run_prod.sh
  deploy.sh
  smoke.sh

tests/
  conftest.py
  fixtures/
    make_dicom.py        helper to synthesize DICOMs
    sample_still.dcm     (generated in fixtures setup)
    sample_cine.dcm      (generated in fixtures setup)
  test_config.py
  test_storage.py
  test_models.py
  test_constants.py
  test_dicom_pipeline.py
  test_view_classifier.py
  test_echochat_engine.py
  test_tasks_report.py
  test_tasks_measurement.py
  test_tasks_disease.py
  test_tasks_vqa.py
  test_auth.py
  test_upload_router.py
  test_study_router.py
  test_tasks_router.py
  test_export.py
  test_real_inference.py    # -m slow, opt-in

environment.yml          conda env 'echochat-demo'
requirements.txt         pip pins (alternative path)
.env.example
README.md
pytest.ini               markers config
```

The whole system reads config from env vars; `.env.example` documents the list. No secrets in git.

---

# Phase 0 — Scaffold

### Task 0.1: Create project skeleton directories and empty `__init__.py` files

**Files:**
- Create: `app/__init__.py`, `app/routers/__init__.py`, `app/services/__init__.py`, `app/services/tasks/__init__.py`, `app/models/__init__.py`, `constants/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create all package init files**

```bash
mkdir -p app/routers app/services/tasks app/models constants tests/fixtures \
        templates/components static/css static/js static/img scripts \
        docs/superpowers/specs docs/superpowers/plans
touch app/__init__.py app/routers/__init__.py app/services/__init__.py \
      app/services/tasks/__init__.py app/models/__init__.py \
      constants/__init__.py tests/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "Phase 0: scaffold empty package tree"
```

### Task 0.2: `pytest.ini` with markers

**Files:** Create `pytest.ini`

- [ ] **Step 1: Write file**

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
markers =
    slow: integration tests that need the real echochat model (opt-in with -m slow)
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 2: Commit**

```bash
git add pytest.ini
git commit -m "Phase 0: add pytest config with slow marker"
```

### Task 0.3: `requirements.txt`

**Files:** Create `requirements.txt`

- [ ] **Step 1: Write file**

```
fastapi>=0.110
uvicorn[standard]>=0.27
gunicorn>=21.2
sse-starlette>=2.1
jinja2>=3.1
python-multipart>=0.0.9
pydantic>=2.5
pydantic-settings>=2.2
pydicom>=2.4
opencv-python-headless>=4.9
numpy>=1.26
pillow>=10.0
httpx>=0.27
weasyprint>=60.2
python-docx>=1.1
itsdangerous>=2.2
# ML deps match legacy echochat env; align to /nfs/usrhome2/yqinar/HKUSData/script/echochatv1.5
ms-swift>=3.3
torch>=2.1
transformers>=4.51
# Dev
pytest>=8.0
pytest-asyncio>=0.23
respx>=0.21
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "Phase 0: pin dependency versions"
```

### Task 0.4: `environment.yml`

**Files:** Create `environment.yml`

- [ ] **Step 1: Write file**

```yaml
name: echochat-demo
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pip
  - pip:
      - -r requirements.txt
```

- [ ] **Step 2: Commit**

```bash
git add environment.yml
git commit -m "Phase 0: add conda environment spec"
```

### Task 0.5: `.env.example`

**Files:** Create `.env.example`

- [ ] **Step 1: Write file**

```
# Server
ECHOCHAT_HOST=0.0.0.0
ECHOCHAT_PORT=12345

# Paths
ECHOCHAT_MODEL_PATH=/nfs/usrhome2/yqinar/HKUSData/script/echochatv1.5
ECHOCHAT_DATA_DIR=/nfs/usrhome2/EchoChatDemo/data

# External services
VIEW_CLASSIFIER_URL=http://127.0.0.1:8995

# Auth
SHARED_PASSWORD=echochat
SESSION_SECRET=change-me-to-a-random-long-string

# GPU (set via CUDA_VISIBLE_DEVICES instead of device string)
# Example shell: CUDA_VISIBLE_DEVICES=2 python -m app.main
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "Phase 0: document env variables"
```

### Task 0.6: `app/config.py` with pydantic-settings

**Files:** Create `app/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write test first**

```python
# tests/test_config.py
import os
from pathlib import Path

import pytest


def test_settings_load_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VIEW_CLASSIFIER_URL", "http://x:1")
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "s")

    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    assert s.model_path == Path("/tmp/model")
    assert s.data_dir == tmp_path
    assert s.view_classifier_url == "http://x:1"
    assert s.shared_password == "pw"
    assert s.session_secret == "s"
    assert s.host == "0.0.0.0"
    assert s.port == 12345


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "s")
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))

    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()

    assert s.view_classifier_url == "http://127.0.0.1:8995"
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_config.py -v
# Expected: ImportError or ModuleNotFoundError for app.config
```

- [ ] **Step 3: Write `app/config.py`**

```python
# app/config.py
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="ECHOCHAT_HOST")
    port: int = Field(default=12345, alias="ECHOCHAT_PORT")

    model_path: Path = Field(alias="ECHOCHAT_MODEL_PATH")
    data_dir: Path = Field(alias="ECHOCHAT_DATA_DIR")

    view_classifier_url: str = Field(
        default="http://127.0.0.1:8995", alias="VIEW_CLASSIFIER_URL"
    )

    shared_password: str = Field(alias="SHARED_PASSWORD")
    session_secret: str = Field(alias="SESSION_SECRET")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/test_config.py -v
# Expected: 2 passed
```

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "Phase 0: pydantic Settings loaded from env"
```

### Task 0.7: Minimal bootable FastAPI app

**Files:** Create `app/main.py`, `tests/test_main_boot.py`

- [ ] **Step 1: Write test**

```python
# tests/test_main_boot.py
from fastapi.testclient import TestClient


def test_healthz_returns_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "s" * 32)

    from app.config import get_settings
    get_settings.cache_clear()

    from app.main import create_app
    client = TestClient(create_app())

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert "uptime_s" in body
    assert body["model_loaded"] is False  # not loaded in test
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_main_boot.py -v
# Expected: ImportError for app.main
```

- [ ] **Step 3: Write `app/main.py`**

```python
# app/main.py
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
```

- [ ] **Step 4: Run test, confirm pass**

```bash
pytest tests/test_main_boot.py -v
# Expected: 1 passed
```

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_main_boot.py
git commit -m "Phase 0: bootable FastAPI with /healthz"
```

### Task 0.8: `README.md` skeleton

**Files:** Create `README.md`

- [ ] **Step 1: Write file**

```markdown
# EchoChat Demo

Web demo for echocardiography AI (Report, Measurement, Disease, VQA) on DICOM uploads. See `docs/target.md` for product requirements and `docs/superpowers/specs/2026-04-18-echochat-demo-design.md` for the design.

## Development

```bash
cp .env.example .env
# edit .env with real values (model path, data dir, password, session secret)
conda env create -f environment.yml
conda activate echochat-demo
pytest
uvicorn app.main:app --reload --port 12345
```

## Deployment

```bash
bash scripts/deploy.sh
ssh eez195 'cd /nfs/usrhome2/EchoChatDemo && bash scripts/run_prod.sh'
```

## Layout

```
app/         FastAPI application
constants/   Authoritative capability lists
templates/   Jinja2 HTML
static/      CSS/JS/images
scripts/     Run + deploy shell helpers
tests/       Pytest suite
docs/        Spec and plan
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Phase 0: README skeleton"
```

**Milestone Phase 0:** `pytest` green, `uvicorn app.main:app --port 12345` starts and `/healthz` responds.

---

# Phase 1 — Data layer

### Task 1.1: `constants/diseases.py`

**Files:** Create `constants/diseases.py`, `tests/test_constants.py`

- [ ] **Step 1: Write test**

```python
# tests/test_constants.py
def test_diseases_mirror_docs():
    from constants.diseases import SUPPORT_DISEASES
    assert len(SUPPORT_DISEASES) == 28
    assert "Aortic regurgitation" in SUPPORT_DISEASES
    assert "Pericardial effusion" in SUPPORT_DISEASES
    # no duplicates
    assert len(set(SUPPORT_DISEASES)) == len(SUPPORT_DISEASES)
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest tests/test_constants.py::test_diseases_mirror_docs -v
# Expected: ModuleNotFoundError
```

- [ ] **Step 3: Write `constants/diseases.py`**

```python
# constants/diseases.py
"""Supported diseases for the Disease Diagnosis task.

Source of truth: docs/disease(1).py. Keep this file in sync.
"""

SUPPORT_DISEASES: list[str] = [
    "Aortic regurgitation",
    "Aortic stenosis",
    "Bicuspid aortic valve",
    "Aortic root dilation",
    "Tricuspid Regurgitation",
    "Mitral regurgitation",
    "Mitral stenosis",
    "Pulmonary regurgitation",
    "Pulmonary artery dilation",
    "Pulmonary hypertension",
    "Left-atrial Dilation",
    "Right-atrial Dilation",
    "Atrial septal defect",
    "Left-ventricular Dilation",
    "Left-ventricular apical aneurysm",
    "Left-ventricular diastolic dysfunction",
    "Left-ventricular systolic dysfunction",
    "Right-ventricular Dilation",
    "Right-ventricular systolic dysfunction",
    "Ventricular septal defect",
    "Inferior vena cava dilation",
    "Hypertrophic cardiomyopathy",
    "Segmental wall-motion abnormality",
    "Pacemaker in situ",
    "Status post aortic-valve replacement",
    "Status post mitral-valve replacement",
    "Mechanical prosthetic valve (post valve replacement)",
    "Pericardial effusion",
]
```

- [ ] **Step 4: Run, confirm pass; commit**

```bash
pytest tests/test_constants.py::test_diseases_mirror_docs -v
git add constants/diseases.py tests/test_constants.py
git commit -m "Phase 1: SUPPORT_DISEASES constant (28 items)"
```

### Task 1.2: `constants/measurements.py`

**Files:** Create `constants/measurements.py`, extend `tests/test_constants.py`

- [ ] **Step 1: Add test**

```python
# append to tests/test_constants.py
def test_measurements_mirror_docs():
    from constants.measurements import SUPPORT_MEASUREMENTS
    assert len(SUPPORT_MEASUREMENTS) == 22
    assert "LV ejection fraction" in SUPPORT_MEASUREMENTS
    assert "Peak LVOT velocity" in SUPPORT_MEASUREMENTS
    assert len(set(SUPPORT_MEASUREMENTS)) == len(SUPPORT_MEASUREMENTS)
```

- [ ] **Step 2: Write `constants/measurements.py`**

```python
# constants/measurements.py
"""Supported measurements for the Measurement task.

Source of truth: docs/measurement(1).py.
"""

SUPPORT_MEASUREMENTS: list[str] = [
    "LV ejection fraction",
    "Left-atrial antero-posterior dimension",
    "Left-atrial volume at end-systole",
    "Left-ventricular posterior-wall thickness in diastole",
    "Aortic-root diameter",
    "Interventricular-septal thickness in diastole",
    "Mitral inflow A-wave peak velocity",
    "Mitral inflow E-wave peak velocity",
    "Peak transmitral velocity",
    "Peak transmitral pressure gradient",
    "Septal mitral-annulus early-diastolic tissue velocity (E')",
    "Peak systolic velocity across the aortic valve",
    "Peak systolic pressure gradient across the aortic valve",
    "Peak velocity across pulmonary valve",
    "Peak pressure gradient across pulmonary valve",
    "Tricuspid-annular-plane systolic excursion",
    "Right-ventricular lateral-annulus systolic tissue velocity (S')",
    "Avg E/E'",
    "Peak velocity of TR jet",
    "Peak pressure gradient of TR jet",
    "Peak pressure gradient across LVOT",
    "Peak LVOT velocity",
]
```

- [ ] **Step 3: Run, pass, commit**

```bash
pytest tests/test_constants.py::test_measurements_mirror_docs -v
git add constants/measurements.py tests/test_constants.py
git commit -m "Phase 1: SUPPORT_MEASUREMENTS constant (22 items)"
```

### Task 1.3: `constants/report_sections.py`

**Files:** Create `constants/report_sections.py`

- [ ] **Step 1: Add test**

```python
# append to tests/test_constants.py
def test_report_sections():
    from constants.report_sections import REPORT_SECTIONS
    assert REPORT_SECTIONS == [
        "Aortic Valve",
        "Atria",
        "Great Vessels",
        "Left Ventricle",
        "Mitral Valve",
        "Pericardium Pleural",
        "Pulmonic Valve",
        "Right Ventricle",
        "Tricuspid Valve",
        "Summary",
    ]
```

- [ ] **Step 2: Write**

```python
# constants/report_sections.py
"""Ordered report sections matching docs/report(1).py system prompt."""

REPORT_SECTIONS: list[str] = [
    "Aortic Valve",
    "Atria",
    "Great Vessels",
    "Left Ventricle",
    "Mitral Valve",
    "Pericardium Pleural",
    "Pulmonic Valve",
    "Right Ventricle",
    "Tricuspid Valve",
    "Summary",
]
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_constants.py::test_report_sections -v
git add constants/report_sections.py tests/test_constants.py
git commit -m "Phase 1: REPORT_SECTIONS constant (10 items)"
```

### Task 1.4: `constants/view_labels.py`

**Files:** Create `constants/view_labels.py`

- [ ] **Step 1: Add test**

```python
# append to tests/test_constants.py
def test_view_labels():
    from constants.view_labels import VIEW_LABELS, view_group, is_doppler
    assert len(VIEW_LABELS) == 38
    assert "Apical 4C 2D" in VIEW_LABELS
    assert view_group("Apical 4C 2D") == "a4c"
    assert view_group("Parasternal Long Axis 2D") == "plax"
    assert view_group("Parasternal Mitral Valve Short Axis") == "psax"
    assert view_group("Subxiphoid IVC 2D") == "ivc"
    assert view_group("Suprasternal Notch") == "ssn"
    assert view_group("Apical 2C 2D") == "a2c"
    assert is_doppler("Parasternal Short Axis Tricuspid Regurgitation CW") is True
    assert is_doppler("Apical 4C 2D") is False
```

- [ ] **Step 2: Write file**

```python
# constants/view_labels.py
"""38 view classes as emitted by EchoView38Classifier on port 8995.

Groups are coarse buckets the UI uses to render the Study Panel badges.
Doppler/spectrum variants are NOT subclassified (per product decision):
the UI rolls them into a single 'Doppler / Spectrum' chip count.
"""

VIEW_LABELS: list[str] = [
    "Apical 4C 2D",
    "Apical 2C 2D",
    "Apical 3C 2D",
    "Parasternal Mitral Valve Short Axis",
    "Parasternal Papillary Muscle Short Axis 2D",
    "Parasternal Apical Short Axis 2D",
    "Apical 5C AV CW",
    "Subxiphoid IVC 2D",
    "Parasternal Long Axis 2D",
    "Parasternal Long Axis of the Pulmonary Artery",
    "Parasternal Short Axis Tricuspid PW",
    "Apical 4C Color",
    "Apical 4C MV Annulus TDI PW",
    "Parasternal Long Axis Color",
    "Apical 4C MV PW",
    "Parasternal Short Axis Tricuspid Color",
    "Parasternal Long Axis M-mode",
    "Apical 4C Right Ventricular Focus",
    "Parasternal Right Ventricular Outflow Tract Color",
    "Parasternal Right Ventricular Outflow Tract RVOT PW",
    "Parasternal Right Ventricular Outflow Tract PA PW",
    "Apical 4C TV Annulus TDI PW",
    "Apical 4C TAPSE",
    "Parasternal Right Ventricular Inflow Tract Color",
    "Parasternal Right Ventricular Inflow Tract 2D",
    "Parasternal Short Axis Tricuspid Regurgitation CW",
    "Subxiphoid IVC M-mode",
    "Suprasternal Notch",
    "Parasternal Right Ventricular Outflow Tract PR CW",
    "A4C LVO",
    "A3C LVO",
    "A4C MCE",
    "A3C MCE",
    "A2C MCE",
    "A2C LVO",
    "A4C MCE Flash",
    "A2C MCE Flash",
    "A3C MCE Flash",
]

# Doppler / M-mode / spectrum tokens — any label containing one of these
# counts as "spectrum" per the product rule.
_SPECTRUM_TOKENS = ("PW", "CW", "TDI", "M-mode", "Color", "LVO", "MCE")


def is_doppler(label: str) -> bool:
    return any(tok in label for tok in _SPECTRUM_TOKENS)


_GROUP_PREFIXES: list[tuple[str, str]] = [
    ("Apical 4C", "a4c"),
    ("A4C", "a4c"),
    ("Apical 2C", "a2c"),
    ("A2C", "a2c"),
    ("Apical 3C", "a3c"),
    ("A3C", "a3c"),
    ("Apical 5C", "a5c"),
    ("Parasternal Long Axis of the Pulmonary Artery", "pala"),
    ("Parasternal Long Axis", "plax"),
    ("Parasternal Mitral Valve Short Axis", "psax_mv"),
    ("Parasternal Papillary Muscle Short Axis", "psax_pm"),
    ("Parasternal Apical Short Axis", "psax_apex"),
    ("Parasternal Short Axis", "psax"),
    ("Parasternal Right Ventricular Outflow Tract", "rvot"),
    ("Parasternal Right Ventricular Inflow Tract", "rvit"),
    ("Subxiphoid IVC", "ivc"),
    ("Suprasternal Notch", "ssn"),
]


def view_group(label: str) -> str:
    """Map a raw label to a coarse group key.

    Groups named 'psax_*' collapse to 'psax' for simple 'has-PSAX' checks
    via view_coarse_group.
    """
    for prefix, key in _GROUP_PREFIXES:
        if label.startswith(prefix):
            return key
    return "other"


def view_coarse_group(label: str) -> str:
    g = view_group(label)
    if g.startswith("psax"):
        return "psax"
    return g
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_constants.py::test_view_labels -v
git add constants/view_labels.py tests/test_constants.py
git commit -m "Phase 1: VIEW_LABELS + view_group + is_doppler helpers"
```

### Task 1.5: `constants/prompts.py` (task prompt templates)

**Files:** Create `constants/prompts.py`

- [ ] **Step 1: Write test**

```python
# append to tests/test_constants.py
def test_prompts_contain_templates():
    from constants.prompts import REPORT_PROMPT, MEASUREMENT_PROMPT, DISEASE_PROMPT, VQA_PROMPT
    assert "echo report" in REPORT_PROMPT.system.lower() or "echo" in REPORT_PROMPT.system.lower()
    assert "<measure>" in MEASUREMENT_PROMPT.query_template
    assert "<disease>" in DISEASE_PROMPT.query_template
    assert VQA_PROMPT.query_template == "{question}"
```

- [ ] **Step 2: Write file**

```python
# constants/prompts.py
"""System prompts and query templates per task.

Source of truth: docs/{report,measurement,disease}(1).py.
VQA had no prior prompt; we pick a bounded system prompt here.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    query_template: str  # may contain substitution tokens described per-task


REPORT_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Your task is to write a sectioned "
        "echo report including sections of: Aortic Valve, Atria, Great Vessels, "
        "Left Ventricle, Mitral Valve, Pericardium Pleural, Pulmonic Valve, "
        "Right Ventricle, Tricuspid Valve, and Summary, from the given "
        "echocardiography."
    ),
    query_template="Write a report from the echocardiography.",
)

MEASUREMENT_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Your task is to measure heart "
        "parameters from the given echocardiography."
    ),
    query_template="Please measure the <measure>.",
)

DISEASE_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Your task is to diagnose heart "
        "conditions from the given echocardiography. Answer yes or no."
    ),
    query_template="Based on the echocardiography, does the patient have <disease>?",
)

VQA_PROMPT = PromptTemplate(
    system=(
        "You are an expert echocardiologist. Answer the user's question about "
        "the provided echocardiography concisely and clinically. If the "
        "question cannot be answered from the given images, state that "
        "explicitly. Do not answer non-echocardiography questions."
    ),
    query_template="{question}",
)
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_constants.py::test_prompts_contain_templates -v
git add constants/prompts.py tests/test_constants.py
git commit -m "Phase 1: PromptTemplate per task (report/measurement/disease/vqa)"
```

### Task 1.6: `constants/presets.py`

**Files:** Create `constants/presets.py`

- [ ] **Step 1: Add test**

```python
# append to tests/test_constants.py
def test_presets_subset_of_supported():
    from constants.diseases import SUPPORT_DISEASES
    from constants.measurements import SUPPORT_MEASUREMENTS
    from constants.presets import (
        MEASUREMENT_PRESETS, DISEASE_PRESETS, VQA_EXAMPLES,
    )

    for preset in MEASUREMENT_PRESETS.values():
        for item in preset:
            assert item in SUPPORT_MEASUREMENTS, f"{item} not in SUPPORT_MEASUREMENTS"

    for preset in DISEASE_PRESETS.values():
        for item in preset:
            assert item in SUPPORT_DISEASES, f"{item} not in SUPPORT_DISEASES"

    # VQA examples is a list of short strings. Placeholder empty allowed.
    assert isinstance(VQA_EXAMPLES, list)
    for q in VQA_EXAMPLES:
        assert isinstance(q, str) and 5 <= len(q) <= 160
```

- [ ] **Step 2: Write file**

```python
# constants/presets.py
"""Quick-pick presets used by the Measurement and Disease tabs.

Keys are human-readable labels shown as buttons in the UI. Every item must
exist in the corresponding SUPPORT_* list; the test asserts this invariant.
"""

MEASUREMENT_PRESETS: dict[str, list[str]] = {
    "Basic 5": [
        "LV ejection fraction",
        "Left-atrial antero-posterior dimension",
        "Aortic-root diameter",
        "Interventricular-septal thickness in diastole",
        "Left-ventricular posterior-wall thickness in diastole",
    ],
    "Valvular Doppler": [
        "Peak transmitral velocity",
        "Peak transmitral pressure gradient",
        "Peak systolic velocity across the aortic valve",
        "Peak systolic pressure gradient across the aortic valve",
        "Peak velocity of TR jet",
        "Peak pressure gradient of TR jet",
    ],
    "LV Function": [
        "LV ejection fraction",
        "Mitral inflow E-wave peak velocity",
        "Mitral inflow A-wave peak velocity",
        "Septal mitral-annulus early-diastolic tissue velocity (E')",
        "Avg E/E'",
    ],
    "RV Function": [
        "Tricuspid-annular-plane systolic excursion",
        "Right-ventricular lateral-annulus systolic tissue velocity (S')",
        "Peak velocity of TR jet",
    ],
}

DISEASE_PRESETS: dict[str, list[str]] = {
    "Valvular": [
        "Aortic regurgitation",
        "Aortic stenosis",
        "Mitral regurgitation",
        "Mitral stenosis",
        "Tricuspid Regurgitation",
        "Pulmonary regurgitation",
    ],
    "LV Function": [
        "Left-ventricular Dilation",
        "Left-ventricular systolic dysfunction",
        "Left-ventricular diastolic dysfunction",
        "Left-ventricular apical aneurysm",
        "Segmental wall-motion abnormality",
    ],
    "RV & Pulmonary": [
        "Right-ventricular Dilation",
        "Right-ventricular systolic dysfunction",
        "Pulmonary hypertension",
        "Pulmonary artery dilation",
    ],
    "Structural": [
        "Atrial septal defect",
        "Ventricular septal defect",
        "Bicuspid aortic valve",
        "Hypertrophic cardiomyopathy",
        "Pericardial effusion",
    ],
}

# Placeholder: domain expert fills this later via /api/constants (see spec §11).
# Keep as empty list until content arrives; the UI hides the dropdown if empty.
VQA_EXAMPLES: list[str] = []
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_constants.py::test_presets_subset_of_supported -v
git add constants/presets.py tests/test_constants.py
git commit -m "Phase 1: Measurement & Disease presets; VQA examples placeholder"
```

### Task 1.7: Pydantic schemas — `app/models/study.py`

**Files:** Create `app/models/study.py`, `tests/test_models.py`

- [ ] **Step 1: Write test**

```python
# tests/test_models.py
from datetime import datetime
from app.models.study import Clip, Study, TasksAvailability


def test_clip_roundtrip():
    c = Clip(
        file_id="f1",
        original_filename="IM_1234",
        kind="dicom",
        raw_path="/x/raw/f1.dcm",
        converted_path="/x/conv/f1.mp4",
        view="Apical 4C 2D",
        user_view=None,
        confidence=0.87,
        is_video=True,
    )
    s = c.model_dump_json()
    c2 = Clip.model_validate_json(s)
    assert c2 == c


def test_tasks_availability_shape():
    ta = TasksAvailability(
        report=True, measurement=True, disease=True, vqa=True,
        missing_groups=["a2c", "ivc"],
    )
    assert ta.report is True
    assert "a2c" in ta.missing_groups


def test_study_roundtrip(tmp_path):
    s = Study(
        study_id="sid1",
        created_at=datetime(2026, 4, 18, 12, 0, 0),
        clips=[],
        tasks=TasksAvailability(report=True, measurement=True, disease=True,
                                vqa=True, missing_groups=[]),
    )
    s2 = Study.model_validate_json(s.model_dump_json())
    assert s2 == s
```

- [ ] **Step 2: Write `app/models/study.py`**

```python
# app/models/study.py
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


ClipKind = Literal["dicom", "image", "video", "unknown"]


class Clip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    original_filename: str
    kind: ClipKind
    raw_path: str
    converted_path: Optional[str] = None
    view: Optional[str] = None        # classifier output (one of VIEW_LABELS or None)
    user_view: Optional[str] = None   # user override (wins over `view`)
    confidence: Optional[float] = None
    is_video: bool = False

    @property
    def effective_view(self) -> Optional[str]:
        return self.user_view or self.view


class TasksAvailability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: bool
    measurement: bool
    disease: bool
    vqa: bool
    missing_groups: list[str]


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid")

    study_id: str
    created_at: datetime
    clips: list[Clip]
    tasks: TasksAvailability
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_models.py -v
git add app/models/study.py tests/test_models.py
git commit -m "Phase 1: Clip / TasksAvailability / Study Pydantic schemas"
```

### Task 1.8: Pydantic schemas — `app/models/task.py`

**Files:** Create `app/models/task.py`, extend `tests/test_models.py`

- [ ] **Step 1: Extend test**

```python
# append to tests/test_models.py
def test_task_schemas():
    from app.models.task import (
        TaskKind, TaskStatus, ReportResult, ReportSection,
        MeasurementItem, MeasurementResult,
        DiseaseItem, DiseaseResult,
        VQAMessage, VQAResult,
    )

    r = ReportResult(
        status="done",
        sections=[ReportSection(name="Left Ventricle", content="...", edited=False)],
    )
    assert r.sections[0].name == "Left Ventricle"

    m = MeasurementResult(
        status="done",
        items=[MeasurementItem(name="LV ejection fraction", value="55", unit="%", raw="...")],
    )
    assert m.items[0].unit == "%"

    d = DiseaseResult(
        status="done",
        items=[DiseaseItem(name="Aortic stenosis", answer="no", raw="No.")],
    )
    assert d.items[0].answer == "no"

    v = VQAResult(
        status="done",
        messages=[VQAMessage(role="user", content="What view?"),
                  VQAMessage(role="assistant", content="A4C.")],
    )
    assert v.messages[-1].role == "assistant"

    # Enum-ish literals round-trip
    assert TaskKind.REPORT == "report"
    assert TaskStatus.RUNNING == "running"
```

- [ ] **Step 2: Write `app/models/task.py`**

```python
# app/models/task.py
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class TaskKind(str, Enum):
    REPORT = "report"
    MEASUREMENT = "measurement"
    DISEASE = "disease"
    VQA = "vqa"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    content: str
    edited: bool = False


class ReportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    sections: list[ReportSection] = []
    error: Optional[str] = None


class MeasurementItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    raw: str


class MeasurementResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    items: list[MeasurementItem] = []
    error: Optional[str] = None


class DiseaseItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    answer: Literal["yes", "no", "unknown"]
    raw: str


class DiseaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    items: list[DiseaseItem] = []
    error: Optional[str] = None


class VQAMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class VQAResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: TaskStatus
    messages: list[VQAMessage] = []
    error: Optional[str] = None
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_models.py -v
git add app/models/task.py tests/test_models.py
git commit -m "Phase 1: task schemas (Report/Measurement/Disease/VQA)"
```

### Task 1.9: `app/storage.py` — per-study filesystem accessor

**Files:** Create `app/storage.py`, `tests/test_storage.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_storage.py
import json
from datetime import datetime
from pathlib import Path

import pytest

from app.models.study import Clip, Study, TasksAvailability
from app.storage import Storage


@pytest.fixture
def store(tmp_path):
    return Storage(base=tmp_path)


def test_new_study_creates_directories(store, tmp_path):
    sid = store.new_study_id()
    store.ensure_study(sid)
    root = tmp_path / "sessions" / sid
    assert (root / "raw").is_dir()
    assert (root / "converted").is_dir()
    assert (root / "results").is_dir()
    assert (root / "exports").is_dir()


def test_save_and_load_study(store):
    sid = store.new_study_id()
    store.ensure_study(sid)
    s = Study(
        study_id=sid,
        created_at=datetime(2026, 4, 18, 12),
        clips=[Clip(file_id="f1", original_filename="x.dcm", kind="dicom",
                    raw_path="/tmp/x.dcm")],
        tasks=TasksAvailability(report=True, measurement=True, disease=True,
                                vqa=True, missing_groups=[]),
    )
    store.save_study(s)
    loaded = store.load_study(sid)
    assert loaded == s


def test_raw_converted_paths(store):
    sid = store.new_study_id()
    store.ensure_study(sid)
    p = store.raw_path(sid, "f1", ".dcm")
    assert str(p).endswith("raw/f1.dcm")
    q = store.converted_path(sid, "f1", ".mp4")
    assert str(q).endswith("converted/f1.mp4")


def test_save_result_json(store):
    sid = store.new_study_id()
    store.ensure_study(sid)
    store.save_result(sid, "report.json", {"sections": [{"name": "LV", "content": "ok"}]})
    data = json.loads(Path(store.result_path(sid, "report.json")).read_text())
    assert data["sections"][0]["name"] == "LV"
```

- [ ] **Step 2: Write `app/storage.py`**

```python
# app/storage.py
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.study import Study


class Storage:
    """Filesystem accessor for /<base>/sessions/<study_id>/ layout."""

    def __init__(self, base: Path):
        self.base = Path(base)
        (self.base / "sessions").mkdir(parents=True, exist_ok=True)

    # --- ids ---
    def new_study_id(self) -> str:
        return uuid.uuid4().hex[:16]

    # --- paths ---
    def study_root(self, study_id: str) -> Path:
        return self.base / "sessions" / study_id

    def ensure_study(self, study_id: str) -> Path:
        root = self.study_root(study_id)
        for sub in ("raw", "converted", "results", "exports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        meta = root / "meta.json"
        if not meta.exists():
            initial = Study(
                study_id=study_id,
                created_at=datetime.utcnow(),
                clips=[],
                tasks=_default_availability(),
            )
            meta.write_text(initial.model_dump_json(indent=2))
        return root

    def raw_path(self, study_id: str, file_id: str, suffix: str) -> Path:
        return self.study_root(study_id) / "raw" / f"{file_id}{suffix}"

    def converted_path(self, study_id: str, file_id: str, suffix: str) -> Path:
        return self.study_root(study_id) / "converted" / f"{file_id}{suffix}"

    def thumbnail_path(self, study_id: str, file_id: str) -> Path:
        return self.study_root(study_id) / "converted" / f"{file_id}.thumb.png"

    def result_path(self, study_id: str, filename: str) -> Path:
        return self.study_root(study_id) / "results" / filename

    def export_path(self, study_id: str, filename: str) -> Path:
        return self.study_root(study_id) / "exports" / filename

    # --- meta ---
    def load_study(self, study_id: str) -> Study:
        meta = self.study_root(study_id) / "meta.json"
        return Study.model_validate_json(meta.read_text())

    def save_study(self, study: Study) -> None:
        self.ensure_study(study.study_id)
        meta = self.study_root(study.study_id) / "meta.json"
        meta.write_text(study.model_dump_json(indent=2))

    def save_result(self, study_id: str, filename: str, data: Any) -> None:
        p = self.result_path(study_id, filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _default_availability():
    from app.models.study import TasksAvailability
    return TasksAvailability(
        report=False, measurement=False, disease=False, vqa=False, missing_groups=[],
    )
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_storage.py -v
git add app/storage.py tests/test_storage.py
git commit -m "Phase 1: Storage accessor (study root/raw/converted/results paths)"
```

**Milestone Phase 1:** `pytest` green across constants, models, storage. System now has type-safe building blocks.

---

# Phase 2 — Inference clients

### Task 2.1: DICOM synth fixture helper

**Files:** Create `tests/fixtures/make_dicom.py`

- [ ] **Step 1: Write helper**

```python
# tests/fixtures/make_dicom.py
"""Synthesize tiny DICOM files for testing the pipeline.

We cannot ship real patient DICOMs; instead we synthesize minimal ones
with pydicom + numpy so tests stay fully in-tree and deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _empty_ds(path: Path) -> FileDataset:
    meta = Dataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Echo"
    ds.PatientID = "TEST0001"
    ds.Modality = "US"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def make_still_dicom(path: Path, shape=(64, 64)) -> Path:
    ds = _empty_ds(path)
    arr = (np.random.default_rng(0).integers(0, 255, size=shape, dtype=np.uint8))
    ds.Rows, ds.Columns = shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)
    return path


def make_cine_dicom(path: Path, frames: int = 5, shape=(32, 32)) -> Path:
    ds = _empty_ds(path)
    arr = np.random.default_rng(1).integers(0, 255, size=(frames, *shape), dtype=np.uint8)
    ds.Rows, ds.Columns = shape
    ds.NumberOfFrames = frames
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(path), write_like_original=False)
    return path


def make_non_dicom(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)
    return path
```

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/make_dicom.py
git commit -m "Phase 2: DICOM synthesis helper for test fixtures"
```

### Task 2.2: `dicom_pipeline.py` — magic-byte sniff

**Files:** Create `app/services/dicom_pipeline.py`, `tests/test_dicom_pipeline.py`

- [ ] **Step 1: Write test**

```python
# tests/test_dicom_pipeline.py
import pytest

from tests.fixtures.make_dicom import make_still_dicom, make_non_dicom
from app.services.dicom_pipeline import looks_like_dicom


def test_recognizes_dicom_by_magic_bytes(tmp_path):
    p = make_still_dicom(tmp_path / "renamed_no_ext")
    assert looks_like_dicom(p) is True


def test_renamed_to_png_still_recognized(tmp_path):
    p = make_still_dicom(tmp_path / "looks_like.png")
    assert looks_like_dicom(p) is True


def test_non_dicom_rejected(tmp_path):
    p = make_non_dicom(tmp_path / "real.png")
    assert looks_like_dicom(p) is False


def test_zero_byte_file_rejected(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    assert looks_like_dicom(p) is False
```

- [ ] **Step 2: Write minimal pipeline**

```python
# app/services/dicom_pipeline.py
from __future__ import annotations

from pathlib import Path


DICM_OFFSET = 128
DICM_MAGIC = b"DICM"


def looks_like_dicom(path: Path) -> bool:
    """Check DICOM preamble magic bytes (offset 128, 'DICM').

    Tolerates any filename/extension. Zero-byte files are rejected.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return False
    if size < DICM_OFFSET + len(DICM_MAGIC):
        return False
    with p.open("rb") as f:
        f.seek(DICM_OFFSET)
        return f.read(len(DICM_MAGIC)) == DICM_MAGIC
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_dicom_pipeline.py -v
git add app/services/dicom_pipeline.py tests/test_dicom_pipeline.py
git commit -m "Phase 2: DICOM magic-byte sniff (filename-agnostic)"
```

### Task 2.3: DICOM → mp4 / png conversion

**Files:** Extend `app/services/dicom_pipeline.py`, `tests/test_dicom_pipeline.py`

- [ ] **Step 1: Extend tests**

```python
# append to tests/test_dicom_pipeline.py
from app.services.dicom_pipeline import convert_dicom, ConvertResult


def test_convert_still(tmp_path):
    src = make_still_dicom(tmp_path / "s.dcm")
    out = tmp_path / "out"
    out.mkdir()
    result = convert_dicom(src, out / "target")
    assert isinstance(result, ConvertResult)
    assert result.is_video is False
    assert result.output_path.suffix == ".png"
    assert result.output_path.exists()
    assert result.thumbnail_path.exists()


def test_convert_cine(tmp_path):
    src = make_cine_dicom(tmp_path / "c.dcm", frames=5)
    out = tmp_path / "out"
    out.mkdir()
    result = convert_dicom(src, out / "target")
    assert result.is_video is True
    assert result.output_path.suffix == ".mp4"
    assert result.output_path.exists()
    assert result.frame_count == 5
    assert result.thumbnail_path.exists()


def test_convert_rejects_non_dicom(tmp_path):
    src = make_non_dicom(tmp_path / "bad")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="not a DICOM"):
        convert_dicom(src, out / "target")


from tests.fixtures.make_dicom import make_cine_dicom  # noqa: E402
```

- [ ] **Step 2: Extend implementation**

```python
# append to app/services/dicom_pipeline.py
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pydicom
import cv2


@dataclass(frozen=True)
class ConvertResult:
    output_path: Path
    thumbnail_path: Path
    is_video: bool
    frame_count: int
    width: int
    height: int


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Return an 8-bit BGR image from an arbitrary DICOM frame."""
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        if hi > lo:
            f = (f - lo) / (hi - lo) * 255.0
        frame = f.astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _make_thumbnail(bgr: np.ndarray, out: Path, max_side: int = 240) -> Path:
    h, w = bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), bgr)
    return out


def convert_dicom(src: Path, target_stem: Path, *, fps: int = 20) -> ConvertResult:
    """Convert a DICOM at `src` into mp4 (cine) or png (still).

    `target_stem` is a path *without* extension; we add .mp4 / .png.
    A sibling `<stem>.thumb.png` is also produced.

    Raises ValueError if the file is not a DICOM.
    """
    src = Path(src)
    target_stem = Path(target_stem)
    if not looks_like_dicom(src):
        raise ValueError(f"not a DICOM: {src}")

    ds = pydicom.dcmread(str(src))
    pixels = ds.pixel_array  # shape: (H,W) or (F,H,W) or (H,W,3) etc.

    thumb_path = target_stem.with_suffix("").with_name(target_stem.name + ".thumb.png")

    if pixels.ndim == 2 or (pixels.ndim == 3 and pixels.shape[-1] in (3, 4)):
        # Still frame
        frame = _normalize_frame(pixels)
        out = target_stem.with_suffix(".png")
        cv2.imwrite(str(out), frame)
        _make_thumbnail(frame, thumb_path)
        h, w = frame.shape[:2]
        return ConvertResult(out, thumb_path, is_video=False, frame_count=1, width=w, height=h)

    if pixels.ndim == 3:
        # (F, H, W) multi-frame
        frames = pixels
    elif pixels.ndim == 4:
        frames = pixels
    else:
        raise ValueError(f"unexpected pixel shape {pixels.shape}")

    h, w = frames.shape[1], frames.shape[2]
    out = target_stem.with_suffix(".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, float(fps), (w, h))
    try:
        for i in range(frames.shape[0]):
            writer.write(_normalize_frame(frames[i]))
    finally:
        writer.release()
    _make_thumbnail(_normalize_frame(frames[0]), thumb_path)
    return ConvertResult(out, thumb_path, is_video=True, frame_count=int(frames.shape[0]),
                         width=w, height=h)
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_dicom_pipeline.py -v
git add app/services/dicom_pipeline.py tests/test_dicom_pipeline.py
git commit -m "Phase 2: DICOM -> mp4/png + thumbnail conversion"
```

### Task 2.4: View classifier HTTP client

**Files:** Create `app/services/view_classifier.py`, `tests/test_view_classifier.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_view_classifier.py
import httpx
import pytest
import respx

from app.services.view_classifier import ViewClassifier, ClassifyOutcome


@pytest.fixture
def vc():
    return ViewClassifier(base_url="http://vc.test", timeout=0.5)


@respx.mock
async def test_classify_known_view(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)

    respx.post("http://vc.test/classify").mock(
        return_value=httpx.Response(
            200, json={"class_name": "Apical 4C 2D", "confidence": 0.91}
        )
    )
    out = await vc.classify(fake)
    assert isinstance(out, ClassifyOutcome)
    assert out.view == "Apical 4C 2D"
    assert out.confidence == pytest.approx(0.91)


@respx.mock
async def test_classify_unknown_view(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    respx.post("http://vc.test/classify").mock(
        return_value=httpx.Response(200, json={"class_name": "Something Weird", "confidence": 0.2})
    )
    out = await vc.classify(fake)
    assert out.view is None        # not in VIEW_LABELS -> fall back to None
    assert out.confidence == pytest.approx(0.2)


@respx.mock
async def test_classify_timeout(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    respx.post("http://vc.test/classify").mock(side_effect=httpx.ReadTimeout("slow"))
    out = await vc.classify(fake)
    assert out.view is None
    assert out.error == "timeout"


@respx.mock
async def test_classify_http_error(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    respx.post("http://vc.test/classify").mock(return_value=httpx.Response(500))
    out = await vc.classify(fake)
    assert out.view is None
    assert out.error and "500" in out.error
```

- [ ] **Step 2: Write implementation**

```python
# app/services/view_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from constants.view_labels import VIEW_LABELS


@dataclass(frozen=True)
class ClassifyOutcome:
    view: Optional[str]       # canonical label from VIEW_LABELS, or None
    confidence: Optional[float]
    raw_class: Optional[str]
    error: Optional[str] = None


class ViewClassifier:
    """Async client for the EchoView38 service (POST /classify)."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def classify(self, abs_path: Path) -> ClassifyOutcome:
        url = f"{self.base_url}/classify"
        payload = {"path": str(Path(abs_path).resolve()), "topk": 1}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, json=payload)
        except httpx.TimeoutException:
            return ClassifyOutcome(view=None, confidence=None, raw_class=None, error="timeout")
        except httpx.HTTPError as e:
            return ClassifyOutcome(view=None, confidence=None, raw_class=None, error=str(e))

        if r.status_code != 200:
            return ClassifyOutcome(
                view=None, confidence=None, raw_class=None,
                error=f"http {r.status_code}",
            )
        body = r.json()
        raw = body.get("class_name") or body.get("detected_view_type") or body.get("top1")
        confidence = body.get("confidence")
        view = raw if raw in VIEW_LABELS else None
        return ClassifyOutcome(view=view, confidence=confidence, raw_class=raw)
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_view_classifier.py -v
git add app/services/view_classifier.py tests/test_view_classifier.py
git commit -m "Phase 2: async view classifier HTTP client w/ error paths"
```

### Task 2.5: `echochat_engine.py` — engine wrapper (no real model in unit tests)

**Files:** Create `app/services/echochat_engine.py`, `tests/test_echochat_engine.py`

- [ ] **Step 1: Write unit tests (protocol + locking)**

```python
# tests/test_echochat_engine.py
import asyncio

import pytest

from app.services.echochat_engine import EchoChatEngine


class FakeBackend:
    def __init__(self, response: str = "stub response"):
        self.response = response
        self.calls: list[dict] = []

    def infer_sync(self, system, query, images, videos):
        self.calls.append(
            dict(system=system, query=query, images=list(images), videos=list(videos))
        )
        return self.response


async def test_engine_delegates_to_backend():
    be = FakeBackend("hello")
    eng = EchoChatEngine(backend=be)
    out = await eng.infer(system="S", query="Q", images=["a.png"], videos=["b.mp4"])
    assert out == "hello"
    assert be.calls[0]["system"] == "S"
    assert be.calls[0]["query"] == "Q"


async def test_engine_serializes_concurrent_calls():
    order: list[str] = []

    class SeqBackend:
        def __init__(self):
            self.i = 0
        def infer_sync(self, system, query, images, videos):
            self.i += 1
            order.append(f"start-{self.i}")
            # Simulate work
            import time
            time.sleep(0.02)
            order.append(f"end-{self.i}")
            return str(self.i)

    eng = EchoChatEngine(backend=SeqBackend())
    results = await asyncio.gather(
        eng.infer(system="s", query="q", images=[], videos=[]),
        eng.infer(system="s", query="q", images=[], videos=[]),
        eng.infer(system="s", query="q", images=[], videos=[]),
    )
    # Each call must fully finish before the next starts.
    assert order == ["start-1", "end-1", "start-2", "end-2", "start-3", "end-3"]
    assert results == ["1", "2", "3"]
```

- [ ] **Step 2: Write implementation (real backend gated)**

```python
# app/services/echochat_engine.py
from __future__ import annotations

import asyncio
from typing import Protocol, Sequence


class EngineBackend(Protocol):
    def infer_sync(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ) -> str: ...


class EchoChatEngine:
    """Async wrapper with a global lock so the single GPU model runs one job at a time."""

    def __init__(self, backend: EngineBackend):
        self._backend = backend
        self._lock = asyncio.Lock()

    async def infer(
        self,
        system: str,
        query: str,
        images: Sequence[str],
        videos: Sequence[str],
    ) -> str:
        async with self._lock:
            return await asyncio.to_thread(
                self._backend.infer_sync, system, query, images, videos
            )


class SwiftPtBackend:
    """Real backend using ms_swift PtEngine. Imported lazily so unit tests never load torch."""

    def __init__(self, model_path: str):
        from swift.llm import PtEngine, get_template

        self.engine = PtEngine(
            model_path, torch_dtype="bfloat16", attn_impl="flash_attn", use_hf=True
        )
        self.template = get_template(
            self.engine.model_meta.template,
            self.engine.tokenizer,
            default_system=None,
        )

    def infer_sync(
        self,
        system: str,
        query: str,
        images,
        videos,
    ) -> str:
        from swift.llm import InferRequest, RequestConfig

        placeholders = " ".join(["<video>"] * len(videos) + ["<image>"] * len(images))
        prompt = (placeholders + " " + query).strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
        req = InferRequest(messages=messages, images=list(images), videos=list(videos))
        cfg = RequestConfig(max_tokens=1024, temperature=0, stream=False)
        gen = self.engine.infer([req], cfg)
        return gen[0].choices[0].message.content
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_echochat_engine.py -v
git add app/services/echochat_engine.py tests/test_echochat_engine.py
git commit -m "Phase 2: EchoChatEngine async wrapper with global lock; SwiftPtBackend"
```

### Task 2.6: Opt-in real model smoke test

**Files:** Create `tests/test_real_inference.py`

- [ ] **Step 1: Write a slow marker test**

```python
# tests/test_real_inference.py
"""Real-model integration test. Requires GPU + model path.

Run with:  pytest -m slow tests/test_real_inference.py -v
"""
import os
from pathlib import Path

import pytest


@pytest.mark.slow
def test_real_model_returns_non_empty():
    model_path = os.environ.get("ECHOCHAT_MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        pytest.skip("ECHOCHAT_MODEL_PATH not set or missing")

    from app.services.echochat_engine import EchoChatEngine, SwiftPtBackend

    be = SwiftPtBackend(model_path)
    eng = EchoChatEngine(backend=be)

    import asyncio
    out = asyncio.run(eng.infer(
        system="You are a helpful assistant.",
        query="Hello.",
        images=[],
        videos=[],
    ))
    assert isinstance(out, str) and len(out) > 0
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_real_inference.py
git commit -m "Phase 2: opt-in real-model smoke (-m slow)"
```

**Milestone Phase 2:** All fast unit tests pass. DICOM pipeline converts files, view classifier client handles success/unknown/timeout/HTTP error, EchoChat engine serializes concurrent calls. Real model path verified separately under `-m slow`.

---

# Phase 3 — API and auth

### Task 3.1: Progress pub/sub

**Files:** Create `app/services/progress.py`, `tests/test_progress.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_progress.py
import asyncio

import pytest

from app.services.progress import ProgressHub, ProgressEvent


async def test_publish_and_subscribe():
    hub = ProgressHub()
    events: list[ProgressEvent] = []

    async def reader():
        async for evt in hub.subscribe("task-1"):
            events.append(evt)
            if evt.kind == "done":
                break

    r = asyncio.create_task(reader())
    await asyncio.sleep(0)    # let subscribe register
    await hub.publish("task-1", ProgressEvent(kind="phase", data={"phase": "start"}))
    await hub.publish("task-1", ProgressEvent(kind="done", data={}))
    await asyncio.wait_for(r, timeout=1.0)
    assert [e.kind for e in events] == ["phase", "done"]


async def test_late_subscriber_still_gets_end():
    hub = ProgressHub()
    # Publish before any subscriber — should buffer
    await hub.publish("t2", ProgressEvent(kind="phase", data={"phase": "x"}))
    await hub.publish("t2", ProgressEvent(kind="done", data={}))

    seen: list[str] = []
    async for evt in hub.subscribe("t2"):
        seen.append(evt.kind)
        if evt.kind == "done":
            break
    assert seen == ["phase", "done"]
```

- [ ] **Step 2: Write implementation**

```python
# app/services/progress.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ProgressEvent:
    kind: str          # "phase" | "partial" | "item" | "message" | "done" | "error" | "clip"
    data: dict[str, Any] = field(default_factory=dict)


class _Channel:
    def __init__(self):
        self.queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()
        self.buffer: list[ProgressEvent] = []   # replayed to late subscribers
        self.done: bool = False


class ProgressHub:
    """Per-task in-memory pub/sub.

    Publishers push events; subscribers receive them in order. Late subscribers
    get a replay of everything already emitted plus live events until a 'done'
    or 'error' event arrives.

    Single-process only.
    """

    def __init__(self):
        self._channels: dict[str, _Channel] = {}

    def _chan(self, task_id: str) -> _Channel:
        if task_id not in self._channels:
            self._channels[task_id] = _Channel()
        return self._channels[task_id]

    async def publish(self, task_id: str, evt: ProgressEvent) -> None:
        ch = self._chan(task_id)
        ch.buffer.append(evt)
        await ch.queue.put(evt)
        if evt.kind in ("done", "error"):
            ch.done = True

    async def subscribe(self, task_id: str) -> AsyncIterator[ProgressEvent]:
        ch = self._chan(task_id)
        for evt in list(ch.buffer):
            yield evt
            if evt.kind in ("done", "error"):
                return
        if ch.done:
            return
        while True:
            evt = await ch.queue.get()
            yield evt
            if evt.kind in ("done", "error"):
                return


# Singleton used by routers
hub = ProgressHub()
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_progress.py -v
git add app/services/progress.py tests/test_progress.py
git commit -m "Phase 3: ProgressHub pub/sub with late-subscriber replay"
```

### Task 3.2: Auth middleware + login/logout

**Files:** Create `app/auth.py`, `tests/test_auth.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_auth.py
from fastapi.testclient import TestClient


def _make_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    return TestClient(create_app())


def test_requires_login_to_access_home(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/home", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_login_wrong_password_rejected(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/login", data={"password": "wrong"}, follow_redirects=False)
    assert r.status_code == 401


def test_login_correct_password_sets_cookie_and_redirects(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.post("/login", data={"password": "pw"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/home"
    assert "set-cookie" in r.headers


def test_authed_request_reaches_home(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    r = c.get("/home")
    assert r.status_code == 200


def test_logout_clears_cookie(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    c.post("/logout")
    r = c.get("/home", follow_redirects=False)
    assert r.status_code == 303


def test_healthz_is_public(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/healthz")
    assert r.status_code == 200
```

- [ ] **Step 2: Write `app/auth.py`**

```python
# app/auth.py
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, URLSafeSerializer
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings


_COOKIE_NAME = "echochat_session"
_PUBLIC_PATHS = {"/login", "/logout", "/healthz", "/static", "/favicon.ico"}


def _serializer() -> URLSafeSerializer:
    return URLSafeSerializer(get_settings().session_secret, salt="echochat-auth")


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(_COOKIE_NAME)
    if not token:
        return False
    try:
        data = _serializer().loads(token)
    except BadSignature:
        return False
    return data.get("ok") is True


class RequireAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in _PUBLIC_PATHS):
            return await call_next(request)
        if path == "/" or is_authenticated(request):
            return await call_next(request)
        return RedirectResponse("/login", status_code=303)


router = APIRouter()


@router.post("/login")
def login(password: str = Form(...)):
    if password != get_settings().shared_password:
        raise HTTPException(status_code=401, detail="Invalid password.")
    token = _serializer().dumps({"ok": True})
    resp = RedirectResponse("/home", status_code=303)
    resp.set_cookie(
        _COOKIE_NAME, token, httponly=True, samesite="lax", max_age=24 * 3600
    )
    return resp


@router.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(_COOKIE_NAME)
    return resp
```

- [ ] **Step 3: Wire middleware + login/logout + stub pages in `app/main.py`**

Replace `create_app` in `app/main.py`:

```python
# app/main.py (update)
import time
from fastapi import FastAPI
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
        # Temporary stub until Phase 5 installs templates
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
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_auth.py -v
git add app/auth.py app/main.py tests/test_auth.py
git commit -m "Phase 3: shared-password auth middleware + login/logout + stub pages"
```

### Task 3.3: Constants + healthz router

**Files:** Create `app/routers/meta.py`, `tests/test_meta_router.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_meta_router.py
from fastapi.testclient import TestClient


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    c = TestClient(create_app())
    c.post("/login", data={"password": "pw"})
    return c


def test_constants_has_all_lists(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.get("/api/constants")
    assert r.status_code == 200
    body = r.json()
    assert len(body["diseases"]) == 28
    assert len(body["measurements"]) == 22
    assert len(body["report_sections"]) == 10
    assert len(body["views"]) == 38
    assert "measurements" in body["presets"]
    assert "diseases" in body["presets"]
    assert isinstance(body["presets"]["vqa_examples"], list)


def test_constants_requires_auth(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    c = TestClient(create_app())
    r = c.get("/api/constants", follow_redirects=False)
    assert r.status_code == 303
```

- [ ] **Step 2: Write `app/routers/meta.py`**

```python
# app/routers/meta.py
from fastapi import APIRouter

from constants.diseases import SUPPORT_DISEASES
from constants.measurements import SUPPORT_MEASUREMENTS
from constants.report_sections import REPORT_SECTIONS
from constants.view_labels import VIEW_LABELS
from constants.presets import DISEASE_PRESETS, MEASUREMENT_PRESETS, VQA_EXAMPLES

router = APIRouter()


@router.get("/api/constants")
def constants():
    return {
        "diseases": SUPPORT_DISEASES,
        "measurements": SUPPORT_MEASUREMENTS,
        "report_sections": REPORT_SECTIONS,
        "views": VIEW_LABELS,
        "presets": {
            "measurements": MEASUREMENT_PRESETS,
            "diseases": DISEASE_PRESETS,
            "vqa_examples": VQA_EXAMPLES,
        },
    }
```

- [ ] **Step 3: Wire into `app/main.py`**

In `create_app`, after `app.include_router(auth_router)`, add:

```python
from app.routers.meta import router as meta_router  # at top
app.include_router(meta_router)
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_meta_router.py -v
git add app/routers/meta.py app/main.py tests/test_meta_router.py
git commit -m "Phase 3: /api/constants endpoint with authoritative lists"
```

### Task 3.4: Upload router — create study + upload files

**Files:** Create `app/routers/upload.py`, `tests/test_upload_router.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_upload_router.py
from fastapi.testclient import TestClient

from tests.fixtures.make_dicom import make_still_dicom, make_non_dicom


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    c = TestClient(create_app())
    c.post("/login", data={"password": "pw"})
    return c


def test_create_study_returns_id(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/study")
    assert r.status_code == 200
    assert isinstance(r.json()["study_id"], str) and len(r.json()["study_id"]) >= 8


def test_upload_dicom_accepted(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "real.dcm")
    with src.open("rb") as f:
        r = c.post(f"/api/study/{sid}/upload", files={"file": ("weird.png", f, "application/octet-stream")})
    assert r.status_code == 200
    j = r.json()
    assert j["kind"] == "dicom"
    assert j["filename"] == "weird.png"
    assert j["file_id"]


def test_upload_rejects_non_dicom(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    sid = c.post("/api/study").json()["study_id"]
    src = make_non_dicom(tmp_path / "not.dcm")
    with src.open("rb") as f:
        r = c.post(f"/api/study/{sid}/upload", files={"file": ("x.dcm", f, "application/octet-stream")})
    assert r.status_code == 400
    assert "DICOM" in r.json()["detail"]
```

- [ ] **Step 2: Write `app/routers/upload.py`**

```python
# app/routers/upload.py
from __future__ import annotations

import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path

from app.config import get_settings
from app.services.dicom_pipeline import looks_like_dicom
from app.storage import Storage

router = APIRouter()


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


@router.post("/api/study")
def create_study():
    s = _store()
    sid = s.new_study_id()
    s.ensure_study(sid)
    return {"study_id": sid}


@router.post("/api/study/{sid}/upload")
async def upload_file(sid: str, file: UploadFile = File(...)):
    s = _store()
    s.ensure_study(sid)

    file_id = uuid.uuid4().hex[:16]
    original_ext = Path(file.filename or "").suffix or ""
    raw = s.raw_path(sid, file_id, original_ext or "")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(data)

    if not looks_like_dicom(raw):
        raw.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is not DICOM.")

    # Store skeleton clip entry; view classification happens in /process.
    study = s.load_study(sid)
    from app.models.study import Clip
    study.clips.append(
        Clip(
            file_id=file_id,
            original_filename=file.filename or file_id,
            kind="dicom",
            raw_path=str(raw),
        )
    )
    s.save_study(study)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "kind": "dicom",
        "size": len(data),
        "saved_as": str(raw),
    }
```

- [ ] **Step 3: Wire in `app/main.py`**

```python
from app.routers.upload import router as upload_router
app.include_router(upload_router)
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_upload_router.py -v
git add app/routers/upload.py app/main.py tests/test_upload_router.py
git commit -m "Phase 3: POST /api/study and /api/study/<sid>/upload"
```

### Task 3.5: Process SSE — convert + classify

**Files:** Extend `app/routers/upload.py`, extend tests

- [ ] **Step 1: Write test with mocked classifier**

```python
# append to tests/test_upload_router.py
import respx
import httpx
from tests.fixtures.make_dicom import make_cine_dicom


@respx.mock
def test_process_sse_classifies_and_converts(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    sid = c.post("/api/study").json()["study_id"]

    src = make_cine_dicom(tmp_path / "cine.dcm", frames=3, shape=(16, 16))
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("weird_cine", f, "application/octet-stream")})

    respx.post("http://127.0.0.1:8995/classify").mock(
        return_value=httpx.Response(200, json={"class_name": "Apical 4C 2D", "confidence": 0.8})
    )

    # TestClient.stream
    with c.stream("GET", f"/api/study/{sid}/process") as r:
        body = b"".join(r.iter_bytes())

    text = body.decode()
    assert "event: phase" in text
    assert "event: clip" in text
    assert "event: done" in text
    # After process, study should have converted_path + view
    study = c.get(f"/api/study/{sid}").json()
    assert study["clips"][0]["converted_path"]
    assert study["clips"][0]["view"] == "Apical 4C 2D"
    assert study["clips"][0]["confidence"] == 0.8
```

- [ ] **Step 2: Extend `app/routers/upload.py`**

```python
# append to app/routers/upload.py
import asyncio
import json

from fastapi import Request
from sse_starlette.sse import EventSourceResponse

from app.services.dicom_pipeline import convert_dicom
from app.services.view_classifier import ViewClassifier
from app.models.study import Clip
from app.services.progress import ProgressEvent, hub


def _view_client() -> ViewClassifier:
    return ViewClassifier(base_url=get_settings().view_classifier_url)


@router.get("/api/study/{sid}/process")
async def process_sse(sid: str, request: Request):
    s = _store()

    async def gen():
        study = s.load_study(sid)
        vc = _view_client()

        yield {"event": "phase", "data": json.dumps({"phase": "start"})}

        for clip in study.clips:
            if clip.converted_path:
                continue  # idempotent: skip already processed
            raw = clip.raw_path

            # Convert
            yield {"event": "phase",
                   "data": json.dumps({"phase": "parsing_dicom", "file_id": clip.file_id})}
            try:
                result = await asyncio.to_thread(
                    convert_dicom,
                    raw,
                    s.converted_path(sid, clip.file_id, ""),
                )
            except Exception as e:
                yield {"event": "error",
                       "data": json.dumps({"file_id": clip.file_id, "reason": str(e)})}
                continue

            clip.converted_path = str(result.output_path)
            clip.is_video = result.is_video

            # Classify
            yield {"event": "phase",
                   "data": json.dumps({"phase": "classifying", "file_id": clip.file_id})}
            outcome = await vc.classify(result.output_path)
            clip.view = outcome.view
            clip.confidence = outcome.confidence

            yield {"event": "clip", "data": json.dumps({
                "file_id": clip.file_id,
                "view": clip.view,
                "confidence": clip.confidence,
                "is_video": clip.is_video,
                "raw_class": outcome.raw_class,
            })}

            # Persist after each clip
            s.save_study(study)

        yield {"event": "phase", "data": json.dumps({"phase": "done"})}
        yield {"event": "done", "data": json.dumps({})}

    return EventSourceResponse(gen())
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_upload_router.py -v
git add app/routers/upload.py tests/test_upload_router.py
git commit -m "Phase 3: SSE /process: convert DICOM, classify view, stream clip events"
```

### Task 3.6: Study router — get state, patch clip, delete, thumbnail, video

**Files:** Create `app/routers/study.py`, `tests/test_study_router.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_study_router.py
import httpx
import respx
from fastapi.testclient import TestClient

from tests.fixtures.make_dicom import make_still_dicom


def _authed_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app
    c = TestClient(create_app())
    c.post("/login", data={"password": "pw"})
    return c


def _new_study_with_clip(c, tmp_path):
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "s.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("s.dcm", f, "application/octet-stream")})
    return sid


@respx.mock
def test_get_study_returns_tasks_availability(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    respx.post("http://127.0.0.1:8995/classify").mock(
        return_value=httpx.Response(200, json={"class_name": "Apical 4C 2D", "confidence": 0.7})
    )
    with c.stream("GET", f"/api/study/{sid}/process") as r:
        _ = b"".join(r.iter_bytes())
    body = c.get(f"/api/study/{sid}").json()
    assert body["tasks"]["report"] is True
    assert isinstance(body["tasks"]["missing_groups"], list)


def test_patch_clip_user_view(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    fid = c.get(f"/api/study/{sid}").json()["clips"][0]["file_id"]

    r = c.patch(f"/api/study/{sid}/clip/{fid}", json={"user_view": "Apical 2C 2D"})
    assert r.status_code == 200
    assert r.json()["user_view"] == "Apical 2C 2D"


def test_patch_clip_rejects_unknown_view(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    fid = c.get(f"/api/study/{sid}").json()["clips"][0]["file_id"]

    r = c.patch(f"/api/study/{sid}/clip/{fid}", json={"user_view": "Not A View"})
    assert r.status_code == 400


def test_delete_clip(monkeypatch, tmp_path):
    c = _authed_client(monkeypatch, tmp_path)
    sid = _new_study_with_clip(c, tmp_path)
    fid = c.get(f"/api/study/{sid}").json()["clips"][0]["file_id"]
    assert c.delete(f"/api/study/{sid}/clip/{fid}").status_code == 204
    assert c.get(f"/api/study/{sid}").json()["clips"] == []
```

- [ ] **Step 2: Write `app/routers/study.py`**

```python
# app/routers/study.py
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Response
from fastapi.responses import FileResponse

from app.config import get_settings
from app.storage import Storage
from app.models.study import TasksAvailability
from constants.view_labels import VIEW_LABELS, view_coarse_group
from constants.measurements import SUPPORT_MEASUREMENTS
from constants.diseases import SUPPORT_DISEASES


router = APIRouter()


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


def _recompute_availability(study) -> TasksAvailability:
    """Very coarse availability:
    - report: always True if any clip present.
    - measurement / disease: True if at least one clip present.
    - vqa: True if any clip present.
    - missing_groups: coarse groups not seen in any clip.
    """
    if not study.clips:
        return TasksAvailability(report=False, measurement=False, disease=False,
                                 vqa=False, missing_groups=["any"])

    seen_groups: set[str] = set()
    for c in study.clips:
        v = c.effective_view
        if v:
            seen_groups.add(view_coarse_group(v))

    baseline = {"a4c", "plax", "psax"}
    missing = sorted(baseline - seen_groups)

    return TasksAvailability(
        report=True, measurement=True, disease=True, vqa=True, missing_groups=missing,
    )


@router.get("/api/study/{sid}")
def get_study(sid: str):
    s = _store()
    study = s.load_study(sid)
    study.tasks = _recompute_availability(study)
    s.save_study(study)
    return study.model_dump()


@router.patch("/api/study/{sid}/clip/{fid}")
def patch_clip(sid: str, fid: str, body: dict = Body(...)):
    if "user_view" not in body:
        raise HTTPException(status_code=400, detail="user_view required")
    uv = body["user_view"]
    if uv is not None and uv not in VIEW_LABELS:
        raise HTTPException(status_code=400, detail=f"user_view must be in VIEW_LABELS or null")

    s = _store()
    study = s.load_study(sid)
    for clip in study.clips:
        if clip.file_id == fid:
            clip.user_view = uv
            study.tasks = _recompute_availability(study)
            s.save_study(study)
            return clip.model_dump()
    raise HTTPException(status_code=404, detail="clip not found")


@router.delete("/api/study/{sid}/clip/{fid}", status_code=204)
def delete_clip(sid: str, fid: str):
    s = _store()
    study = s.load_study(sid)
    before = len(study.clips)
    study.clips = [c for c in study.clips if c.file_id != fid]
    if len(study.clips) == before:
        raise HTTPException(status_code=404, detail="clip not found")
    # Remove files (best-effort)
    for pat in ("raw/*", "converted/*"):
        for p in Path(s.study_root(sid)).glob(pat):
            if fid in p.name:
                p.unlink(missing_ok=True)
    study.tasks = _recompute_availability(study)
    s.save_study(study)
    return Response(status_code=204)


@router.get("/api/study/{sid}/clip/{fid}/thumbnail")
def thumbnail(sid: str, fid: str):
    s = _store()
    p = s.thumbnail_path(sid, fid)
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(p), media_type="image/png")


@router.get("/api/study/{sid}/clip/{fid}/video")
def clip_video(sid: str, fid: str):
    s = _store()
    study = s.load_study(sid)
    match = next((c for c in study.clips if c.file_id == fid), None)
    if not match or not match.converted_path:
        raise HTTPException(status_code=404)
    p = Path(match.converted_path)
    if not p.exists():
        raise HTTPException(status_code=404)
    mt = "video/mp4" if match.is_video else "image/png"
    return FileResponse(str(p), media_type=mt)
```

- [ ] **Step 3: Wire in `app/main.py`**

```python
from app.routers.study import router as study_router
app.include_router(study_router)
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_study_router.py -v
git add app/routers/study.py app/main.py tests/test_study_router.py
git commit -m "Phase 3: study GET/PATCH/DELETE + thumbnail/video endpoints"
```

**Milestone Phase 3:** `pytest` green. The system supports full upload + process + study state + clip editing via curl. No templates yet; frontend is stub HTML.

---

# Phase 4 — Task orchestrators

### Task 4.1: Task registry + shared helpers

**Files:** Create `app/services/tasks/base.py`, `tests/test_tasks_base.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_tasks_base.py
from app.services.tasks.base import collect_media, split_value_unit


def test_collect_media_splits_videos_and_images():
    clips = [
        type("C", (), {"converted_path": "/a/b.mp4", "is_video": True})(),
        type("C", (), {"converted_path": "/a/c.png", "is_video": False})(),
        type("C", (), {"converted_path": None, "is_video": False})(),
    ]
    imgs, vids = collect_media(clips)
    assert imgs == ["/a/c.png"]
    assert vids == ["/a/b.mp4"]


def test_split_value_unit_simple_number():
    v, u = split_value_unit("55 %")
    assert v == "55"
    assert u == "%"


def test_split_value_unit_plain_number():
    v, u = split_value_unit("1.23")
    assert v == "1.23"
    assert u is None


def test_split_value_unit_with_word_unit():
    v, u = split_value_unit("12 cm/s")
    assert v == "12"
    assert u == "cm/s"


def test_split_value_unit_none_when_non_numeric():
    v, u = split_value_unit("Unable to measure")
    assert v is None
    assert u is None
```

- [ ] **Step 2: Write `app/services/tasks/base.py`**

```python
# app/services/tasks/base.py
from __future__ import annotations

import re
from typing import Iterable


def collect_media(clips) -> tuple[list[str], list[str]]:
    """Split usable clips into (images, videos) based on the flag+path."""
    images: list[str] = []
    videos: list[str] = []
    for c in clips:
        if not getattr(c, "converted_path", None):
            continue
        if getattr(c, "is_video", False):
            videos.append(c.converted_path)
        else:
            images.append(c.converted_path)
    return images, videos


_NUMERIC = re.compile(r"(-?\d+(?:\.\d+)?)\s*([%A-Za-z/\u00b5°][\w\s/\u00b5\-\^]*)?")


def split_value_unit(s: str) -> tuple[str | None, str | None]:
    """Best-effort extractor: from 'EF is 55 %' pull ('55', '%').

    Returns (None, None) if no number is found. Keep simple — model outputs
    vary, and the raw response is shown beside the parsed value.
    """
    if not s:
        return None, None
    m = _NUMERIC.search(s)
    if not m:
        return None, None
    value = m.group(1)
    unit = (m.group(2) or "").strip() or None
    # Guard against trailing noise like "value 55 which indicates..."
    if unit:
        unit = unit.split()[0].rstrip(".,;:").strip() or None
    return value, unit
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_tasks_base.py -v
git add app/services/tasks/base.py tests/test_tasks_base.py
git commit -m "Phase 4: shared task helpers (media split + value/unit parsing)"
```

### Task 4.2: Report task orchestrator

**Files:** Create `app/services/tasks/report.py`, `tests/test_tasks_report.py`

- [ ] **Step 1: Write test with fake engine**

```python
# tests/test_tasks_report.py
import pytest

from app.services.tasks.report import run_report
from app.services.progress import ProgressHub
from app.models.study import Clip


class FakeEngine:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def infer(self, system, query, images, videos):
        self.calls += 1
        return self.text


async def test_report_emits_sections(tmp_path):
    hub = ProgressHub()
    engine = FakeEngine(
        "Aortic Valve: normal.\n"
        "Atria: normal.\n"
        "Great Vessels: normal.\n"
        "Left Ventricle: EF 55.\n"
        "Mitral Valve: normal.\n"
        "Pericardium Pleural: no effusion.\n"
        "Pulmonic Valve: normal.\n"
        "Right Ventricle: normal.\n"
        "Tricuspid Valve: trivial TR.\n"
        "Summary: unremarkable study."
    )
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]

    events: list[dict] = []
    task_id = "tid1"

    async def reader():
        async for evt in hub.subscribe(task_id):
            events.append({"kind": evt.kind, **evt.data})
            if evt.kind in ("done", "error"):
                break

    import asyncio
    r = asyncio.create_task(reader())
    result = await run_report(task_id=task_id, clips=clips, engine=engine, hub=hub)
    await asyncio.wait_for(r, timeout=1.0)

    assert result.status == "done"
    assert len(result.sections) == 10
    assert result.sections[0].name == "Aortic Valve"
    assert "normal" in result.sections[0].content.lower()

    assert any(e.get("section") == "Left Ventricle" for e in events if e["kind"] == "partial")
    assert any(e["kind"] == "done" for e in events)


async def test_report_missing_section_preserves_placeholder(tmp_path):
    hub = ProgressHub()
    engine = FakeEngine("Summary: minimal response only.")
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]

    result = await run_report(task_id="t2", clips=clips, engine=engine, hub=hub)
    # Sections still all present, but most empty-ish
    names = [s.name for s in result.sections]
    assert names.count("Summary") == 1
    summary = next(s for s in result.sections if s.name == "Summary")
    assert "minimal" in summary.content.lower()
```

- [ ] **Step 2: Write `app/services/tasks/report.py`**

```python
# app/services/tasks/report.py
from __future__ import annotations

import re
from typing import Sequence

from app.models.study import Clip
from app.models.task import ReportResult, ReportSection, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.prompts import REPORT_PROMPT
from constants.report_sections import REPORT_SECTIONS


def _split_sections(text: str) -> dict[str, str]:
    """Parse a model response with 'Section Name: content' lines.

    Tolerant of multi-line sections: a line that starts with a known section
    label begins a new section; subsequent lines belong to it until the next
    label. Unknown headings are kept under the previous section.
    """
    out: dict[str, str] = {s: "" for s in REPORT_SECTIONS}
    pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(s) for s in REPORT_SECTIONS) + r")\s*[:\-\u2014]\s*(.*)$",
        re.IGNORECASE,
    )

    current: str | None = None
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            name = next(s for s in REPORT_SECTIONS if s.lower() == m.group(1).lower())
            current = name
            out[current] = (out[current] + ("\n" if out[current] else "") + m.group(2)).strip()
        elif current and line.strip():
            out[current] = (out[current] + "\n" + line).strip()
    return out


async def run_report(
    *, task_id: str, clips: Sequence[Clip], engine, hub: ProgressHub
) -> ReportResult:
    images, videos = collect_media(clips)

    await hub.publish(task_id, ProgressEvent(kind="phase",
                     data={"phase": "preparing_context"}))
    try:
        await hub.publish(task_id, ProgressEvent(kind="phase",
                         data={"phase": "inference"}))
        raw = await engine.infer(
            system=REPORT_PROMPT.system,
            query=REPORT_PROMPT.query_template,
            images=images,
            videos=videos,
        )
    except Exception as e:
        res = ReportResult(status=TaskStatus.ERROR, error=str(e))
        await hub.publish(task_id, ProgressEvent(kind="error",
                         data={"reason": str(e)}))
        return res

    sections_map = _split_sections(raw)
    sections = [ReportSection(name=name, content=sections_map.get(name, "").strip())
                for name in REPORT_SECTIONS]

    for s in sections:
        await hub.publish(task_id, ProgressEvent(kind="partial",
                         data={"section": s.name, "content": s.content}))

    result = ReportResult(status=TaskStatus.DONE, sections=sections)
    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return result
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_tasks_report.py -v
git add app/services/tasks/report.py tests/test_tasks_report.py
git commit -m "Phase 4: Report orchestrator (single call, split into 10 sections, SSE events)"
```

### Task 4.3: Measurement task orchestrator

**Files:** Create `app/services/tasks/measurement.py`, `tests/test_tasks_measurement.py`

- [ ] **Step 1: Write test**

```python
# tests/test_tasks_measurement.py
import asyncio

from app.services.tasks.measurement import run_measurement
from app.services.progress import ProgressHub
from app.models.study import Clip


class SeqEngine:
    def __init__(self, answers):
        self.answers = list(answers)
    async def infer(self, system, query, images, videos):
        return self.answers.pop(0)


async def test_measurement_emits_items_and_persists_order():
    hub = ProgressHub()
    engine = SeqEngine(["55 %", "38 mm"])
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]

    events: list[dict] = []
    async def reader():
        async for evt in hub.subscribe("m1"):
            events.append({"kind": evt.kind, **evt.data})
            if evt.kind in ("done", "error"):
                break
    r = asyncio.create_task(reader())
    result = await run_measurement(
        task_id="m1",
        clips=clips,
        items=["LV ejection fraction", "Left-atrial antero-posterior dimension"],
        engine=engine,
        hub=hub,
    )
    await asyncio.wait_for(r, timeout=1.0)

    assert result.status == "done"
    names = [i.name for i in result.items]
    assert names == ["LV ejection fraction", "Left-atrial antero-posterior dimension"]
    assert result.items[0].value == "55"
    assert result.items[0].unit == "%"
    assert result.items[1].unit == "mm"
    item_events = [e for e in events if e["kind"] == "item"]
    assert len(item_events) == 2


async def test_measurement_rejects_unknown_item():
    hub = ProgressHub()
    engine = SeqEngine([])
    import pytest
    with pytest.raises(ValueError):
        await run_measurement(task_id="m2", clips=[], items=["Bogus"], engine=engine, hub=hub)
```

- [ ] **Step 2: Write `app/services/tasks/measurement.py`**

```python
# app/services/tasks/measurement.py
from __future__ import annotations

from typing import Sequence

from app.models.study import Clip
from app.models.task import MeasurementItem, MeasurementResult, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media, split_value_unit
from constants.measurements import SUPPORT_MEASUREMENTS
from constants.prompts import MEASUREMENT_PROMPT


async def run_measurement(
    *, task_id: str, clips: Sequence[Clip], items: Sequence[str], engine, hub: ProgressHub,
) -> MeasurementResult:
    for it in items:
        if it not in SUPPORT_MEASUREMENTS:
            raise ValueError(f"unsupported measurement: {it}")

    images, videos = collect_media(clips)
    out: list[MeasurementItem] = []
    total = len(items)
    for i, name in enumerate(items, start=1):
        await hub.publish(task_id, ProgressEvent(kind="phase",
                         data={"phase": "during", "i": i, "n": total, "name": name}))
        query = MEASUREMENT_PROMPT.query_template.replace("<measure>", name)
        try:
            raw = await engine.infer(
                system=MEASUREMENT_PROMPT.system, query=query,
                images=images, videos=videos,
            )
        except Exception as e:
            await hub.publish(task_id, ProgressEvent(kind="error",
                             data={"reason": str(e), "name": name}))
            return MeasurementResult(status=TaskStatus.ERROR, items=out, error=str(e))

        value, unit = split_value_unit(raw)
        item = MeasurementItem(name=name, value=value, unit=unit, raw=raw.strip())
        out.append(item)
        await hub.publish(task_id, ProgressEvent(kind="item", data=item.model_dump()))

    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return MeasurementResult(status=TaskStatus.DONE, items=out)
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_tasks_measurement.py -v
git add app/services/tasks/measurement.py tests/test_tasks_measurement.py
git commit -m "Phase 4: Measurement orchestrator (per-item sequential, SSE item events)"
```

### Task 4.4: Disease task orchestrator

**Files:** Create `app/services/tasks/disease.py`, `tests/test_tasks_disease.py`

- [ ] **Step 1: Write test**

```python
# tests/test_tasks_disease.py
import asyncio

from app.services.tasks.disease import run_disease
from app.services.progress import ProgressHub
from app.models.study import Clip


class SeqEngine:
    def __init__(self, answers): self.answers = list(answers)
    async def infer(self, system, query, images, videos):
        return self.answers.pop(0)


async def test_disease_parses_yes_no_and_unknown():
    hub = ProgressHub()
    engine = SeqEngine([
        "Yes, there is aortic stenosis.",
        "No, not present.",
        "I cannot determine based on the images.",
    ])
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]
    events: list[dict] = []
    async def reader():
        async for e in hub.subscribe("d1"):
            events.append({"kind": e.kind, **e.data})
            if e.kind in ("done", "error"):
                break
    r = asyncio.create_task(reader())
    result = await run_disease(
        task_id="d1", clips=clips,
        items=["Aortic stenosis", "Mitral regurgitation", "Hypertrophic cardiomyopathy"],
        engine=engine, hub=hub,
    )
    await asyncio.wait_for(r, timeout=1.0)

    answers = [i.answer for i in result.items]
    assert answers == ["yes", "no", "unknown"]
    assert len([e for e in events if e["kind"] == "item"]) == 3
```

- [ ] **Step 2: Write `app/services/tasks/disease.py`**

```python
# app/services/tasks/disease.py
from __future__ import annotations

import re
from typing import Sequence

from app.models.study import Clip
from app.models.task import DiseaseItem, DiseaseResult, TaskStatus
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.diseases import SUPPORT_DISEASES
from constants.prompts import DISEASE_PROMPT


_YES = re.compile(r"^\s*(yes|yeah|present|affirmative)\b", re.IGNORECASE)
_NO = re.compile(r"^\s*(no|not present|negative|absent)\b", re.IGNORECASE)


def _parse_yn(text: str) -> str:
    if _YES.search(text):
        return "yes"
    if _NO.search(text):
        return "no"
    # Fall back: scan first sentence
    first = text.strip().split("\n", 1)[0]
    if _YES.search(first):
        return "yes"
    if _NO.search(first):
        return "no"
    return "unknown"


async def run_disease(
    *, task_id: str, clips: Sequence[Clip], items: Sequence[str], engine, hub: ProgressHub,
) -> DiseaseResult:
    for it in items:
        if it not in SUPPORT_DISEASES:
            raise ValueError(f"unsupported disease: {it}")

    images, videos = collect_media(clips)
    out: list[DiseaseItem] = []
    for i, name in enumerate(items, start=1):
        await hub.publish(task_id, ProgressEvent(kind="phase",
                         data={"phase": "during", "i": i, "n": len(items), "name": name}))
        query = DISEASE_PROMPT.query_template.replace("<disease>", name)
        try:
            raw = await engine.infer(
                system=DISEASE_PROMPT.system, query=query,
                images=images, videos=videos,
            )
        except Exception as e:
            await hub.publish(task_id, ProgressEvent(kind="error",
                             data={"reason": str(e), "name": name}))
            return DiseaseResult(status=TaskStatus.ERROR, items=out, error=str(e))

        item = DiseaseItem(name=name, answer=_parse_yn(raw), raw=raw.strip())
        out.append(item)
        await hub.publish(task_id, ProgressEvent(kind="item", data=item.model_dump()))

    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return DiseaseResult(status=TaskStatus.DONE, items=out)
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_tasks_disease.py -v
git add app/services/tasks/disease.py tests/test_tasks_disease.py
git commit -m "Phase 4: Disease orchestrator (yes/no parse, SSE item events)"
```

### Task 4.5: VQA task orchestrator

**Files:** Create `app/services/tasks/vqa.py`, `tests/test_tasks_vqa.py`

- [ ] **Step 1: Write test**

```python
# tests/test_tasks_vqa.py
import asyncio

from app.services.tasks.vqa import run_vqa
from app.services.progress import ProgressHub
from app.models.study import Clip


class OneShotEngine:
    def __init__(self, resp): self.resp = resp
    async def infer(self, system, query, images, videos): return self.resp


async def test_vqa_returns_assistant_message():
    hub = ProgressHub()
    clips = [Clip(file_id="f1", original_filename="x", kind="dicom",
                  raw_path="/x", converted_path="/a.mp4", is_video=True)]
    events: list[dict] = []
    async def reader():
        async for e in hub.subscribe("v1"):
            events.append({"kind": e.kind, **e.data})
            if e.kind in ("done", "error"):
                break
    r = asyncio.create_task(reader())
    result = await run_vqa(
        task_id="v1", clips=clips,
        question="What view is shown?",
        engine=OneShotEngine("It is A4C."),
        hub=hub,
    )
    await asyncio.wait_for(r, timeout=1.0)

    assert result.status == "done"
    assert result.messages[-1].role == "assistant"
    assert "A4C" in result.messages[-1].content
    assert any(e["kind"] == "message" for e in events)
```

- [ ] **Step 2: Write `app/services/tasks/vqa.py`**

```python
# app/services/tasks/vqa.py
from __future__ import annotations

from typing import Sequence

from app.models.study import Clip
from app.models.task import TaskStatus, VQAMessage, VQAResult
from app.services.progress import ProgressEvent, ProgressHub
from app.services.tasks.base import collect_media
from constants.prompts import VQA_PROMPT


async def run_vqa(
    *, task_id: str, clips: Sequence[Clip], question: str, engine, hub: ProgressHub,
) -> VQAResult:
    if not question.strip():
        raise ValueError("question is empty")
    images, videos = collect_media(clips)

    await hub.publish(task_id, ProgressEvent(kind="phase", data={"phase": "thinking"}))
    try:
        raw = await engine.infer(
            system=VQA_PROMPT.system,
            query=VQA_PROMPT.query_template.format(question=question),
            images=images, videos=videos,
        )
    except Exception as e:
        await hub.publish(task_id, ProgressEvent(kind="error", data={"reason": str(e)}))
        return VQAResult(status=TaskStatus.ERROR, error=str(e))

    msgs = [
        VQAMessage(role="user", content=question.strip()),
        VQAMessage(role="assistant", content=raw.strip()),
    ]
    await hub.publish(task_id, ProgressEvent(kind="message", data=msgs[-1].model_dump()))
    await hub.publish(task_id, ProgressEvent(kind="done", data={"task_id": task_id}))
    return VQAResult(status=TaskStatus.DONE, messages=msgs)
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_tasks_vqa.py -v
git add app/services/tasks/vqa.py tests/test_tasks_vqa.py
git commit -m "Phase 4: VQA orchestrator (bounded system prompt, single turn)"
```

### Task 4.6: Tasks router + persistence + SSE stream

**Files:** Create `app/routers/tasks.py`, `tests/test_tasks_router.py`

- [ ] **Step 1: Write test with fake engine injected via dependency override**

```python
# tests/test_tasks_router.py
import json
import asyncio

import httpx
import respx
from fastapi.testclient import TestClient

from tests.fixtures.make_dicom import make_still_dicom


class FakeEngine:
    def __init__(self, text="Aortic Valve: normal. Summary: ok."):
        self.text = text
    async def infer(self, system, query, images, videos):
        return self.text


def _client(monkeypatch, tmp_path, engine=None):
    monkeypatch.setenv("ECHOCHAT_MODEL_PATH", str(tmp_path / "m"))
    monkeypatch.setenv("ECHOCHAT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SHARED_PASSWORD", "pw")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)
    from app.config import get_settings
    get_settings.cache_clear()

    from app.main import create_app
    from app.routers.tasks import get_engine
    app = create_app()
    if engine is not None:
        app.dependency_overrides[get_engine] = lambda: engine
    c = TestClient(app)
    c.post("/login", data={"password": "pw"})
    return c


@respx.mock
def test_start_report_and_stream(monkeypatch, tmp_path):
    engine = FakeEngine(
        "\n".join([
            f"{s}: ok." for s in
            ["Aortic Valve", "Atria", "Great Vessels", "Left Ventricle",
             "Mitral Valve", "Pericardium Pleural", "Pulmonic Valve",
             "Right Ventricle", "Tricuspid Valve", "Summary"]
        ])
    )
    c = _client(monkeypatch, tmp_path, engine=engine)
    sid = c.post("/api/study").json()["study_id"]

    src = make_still_dicom(tmp_path / "a.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload",
               files={"file": ("a.dcm", f, "application/octet-stream")})

    respx.post("http://127.0.0.1:8995/classify").mock(
        return_value=httpx.Response(200, json={"class_name": "Apical 4C 2D", "confidence": 0.9})
    )
    with c.stream("GET", f"/api/study/{sid}/process") as r:
        _ = b"".join(r.iter_bytes())

    tid = c.post(f"/api/study/{sid}/task/report", json={}).json()["task_id"]

    with c.stream("GET", f"/api/task/{tid}/stream") as r:
        text = b"".join(r.iter_bytes()).decode()

    assert "event: done" in text
    result = c.get(f"/api/task/{tid}").json()
    assert result["status"] == "done"
    assert len(result["sections"]) == 10
```

- [ ] **Step 2: Write `app/routers/tasks.py`**

```python
# app/routers/tasks.py
from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Body, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.models.task import TaskKind, TaskStatus
from app.services.progress import hub, ProgressEvent
from app.services.tasks.report import run_report
from app.services.tasks.measurement import run_measurement
from app.services.tasks.disease import run_disease
from app.services.tasks.vqa import run_vqa
from app.storage import Storage

router = APIRouter()

# ---- Engine injection ----
# The real engine is constructed during FastAPI startup; tests override via
# app.dependency_overrides[get_engine] = lambda: fake.
_engine_holder: dict = {"engine": None}


def set_engine(engine) -> None:
    _engine_holder["engine"] = engine


def get_engine():
    engine = _engine_holder["engine"]
    if engine is None:
        raise HTTPException(status_code=503,
                            detail="Model is still warming up. Try again shortly.")
    return engine


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


# In-memory task registry (single-process)
_tasks: dict[str, dict] = {}


def _register_task(tid: str, kind: TaskKind, sid: str) -> None:
    _tasks[tid] = {"kind": kind, "study_id": sid, "result": None,
                   "status": TaskStatus.QUEUED}


def _set_result(tid: str, result) -> None:
    rec = _tasks.get(tid)
    if not rec:
        return
    rec["result"] = result.model_dump() if hasattr(result, "model_dump") else result
    rec["status"] = (result.status if hasattr(result, "status") else TaskStatus.DONE)


async def _run_and_persist(tid: str, kind: TaskKind, sid: str, coro_factory):
    _tasks[tid]["status"] = TaskStatus.RUNNING
    try:
        result = await coro_factory()
    except Exception as e:
        await hub.publish(tid, ProgressEvent(kind="error", data={"reason": str(e)}))
        _tasks[tid]["status"] = TaskStatus.ERROR
        return
    _set_result(tid, result)
    # persist to disk
    s = _store()
    filename = {
        TaskKind.REPORT: "report.json",
        TaskKind.MEASUREMENT: "measurements.json",
        TaskKind.DISEASE: "diseases.json",
        TaskKind.VQA: "vqa.json",
    }[kind]
    s.save_result(sid, filename, result.model_dump())


@router.post("/api/study/{sid}/task/report")
async def start_report(sid: str, engine=Depends(get_engine)):
    s = _store()
    study = s.load_study(sid)
    if not study.clips:
        raise HTTPException(status_code=400, detail="Study has no clips.")
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.REPORT, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.REPORT, sid,
        lambda: run_report(task_id=tid, clips=study.clips, engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.post("/api/study/{sid}/task/measurement")
async def start_measurement(sid: str, body: dict = Body(...), engine=Depends(get_engine)):
    items = body.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items[] required")
    s = _store()
    study = s.load_study(sid)
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.MEASUREMENT, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.MEASUREMENT, sid,
        lambda: run_measurement(task_id=tid, clips=study.clips, items=items,
                                engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.post("/api/study/{sid}/task/disease")
async def start_disease(sid: str, body: dict = Body(...), engine=Depends(get_engine)):
    items = body.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="items[] required")
    s = _store()
    study = s.load_study(sid)
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.DISEASE, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.DISEASE, sid,
        lambda: run_disease(task_id=tid, clips=study.clips, items=items,
                            engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.post("/api/study/{sid}/task/vqa")
async def start_vqa(sid: str, body: dict = Body(...), engine=Depends(get_engine)):
    q = (body.get("question") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question required")
    s = _store()
    study = s.load_study(sid)
    tid = uuid.uuid4().hex[:16]
    _register_task(tid, TaskKind.VQA, sid)
    asyncio.create_task(_run_and_persist(
        tid, TaskKind.VQA, sid,
        lambda: run_vqa(task_id=tid, clips=study.clips, question=q,
                        engine=engine, hub=hub),
    ))
    return {"task_id": tid}


@router.get("/api/task/{tid}/stream")
async def stream_task(tid: str):
    if tid not in _tasks:
        raise HTTPException(status_code=404)

    async def gen():
        async for evt in hub.subscribe(tid):
            yield {"event": evt.kind, "data": json.dumps(evt.data)}

    return EventSourceResponse(gen())


@router.get("/api/task/{tid}")
def get_task(tid: str):
    rec = _tasks.get(tid)
    if not rec:
        raise HTTPException(status_code=404)
    if rec["result"] is None:
        return {"status": rec["status"].value if hasattr(rec["status"], "value") else rec["status"]}
    return rec["result"]
```

- [ ] **Step 3: Wire in `app/main.py` + startup hook**

In `app/main.py`, add engine loading at startup:

```python
# app/main.py (extend create_app)
from app.routers.tasks import router as tasks_router, set_engine

# in create_app(), after including routers:
app.include_router(tasks_router)

@app.on_event("startup")
async def _load_engine():
    global _model_ready
    import os, asyncio
    if os.environ.get("ECHOCHAT_SKIP_MODEL", "").lower() in ("1", "true", "yes"):
        return  # tests + dev without GPU
    from app.services.echochat_engine import EchoChatEngine, SwiftPtBackend

    def _build():
        be = SwiftPtBackend(str(settings.model_path))
        return EchoChatEngine(backend=be)

    eng = await asyncio.to_thread(_build)
    set_engine(eng)
    globals()["_model_ready"] = True
```

- [ ] **Step 4: Pass + commit**

```bash
ECHOCHAT_SKIP_MODEL=1 pytest tests/test_tasks_router.py -v
git add app/routers/tasks.py app/main.py tests/test_tasks_router.py
git commit -m "Phase 4: tasks router + startup model loader (skippable via env)"
```

### Task 4.7: Report section edit + regenerate

**Files:** Create `app/routers/report_io.py`, add test to `tests/test_tasks_router.py`

- [ ] **Step 1: Add test**

```python
# append to tests/test_tasks_router.py
def test_patch_report_section(monkeypatch, tmp_path):
    engine = FakeEngine(
        "\n".join([f"{s}: ok." for s in [
            "Aortic Valve","Atria","Great Vessels","Left Ventricle","Mitral Valve",
            "Pericardium Pleural","Pulmonic Valve","Right Ventricle","Tricuspid Valve","Summary",
        ]])
    )
    c = _client(monkeypatch, tmp_path, engine=engine)
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "a.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload", files={"file": ("a", f, "application/octet-stream")})

    # No classify needed for report generation in this test; we bypass /process
    # by marking clip processed manually via patch (for simplicity)
    study = c.get(f"/api/study/{sid}").json()
    # run report directly (no preview needed in test)
    tid = c.post(f"/api/study/{sid}/task/report", json={}).json()["task_id"]
    import time; time.sleep(0.1)

    # patch section
    r = c.patch(f"/api/study/{sid}/report/section",
                json={"section": "Left Ventricle", "content": "EF 55 (edited)"})
    assert r.status_code == 200
    assert r.json()["edited"] is True

    # GET report back from disk via /api/task would still show original;
    # verify persistence by reading the results file path directly
    from pathlib import Path
    import json as J
    p = Path(tmp_path) / "sessions" / sid / "results" / "report.json"
    assert p.exists()
    data = J.loads(p.read_text())
    lv = next(s for s in data["sections"] if s["name"] == "Left Ventricle")
    assert lv["content"] == "EF 55 (edited)"
    assert lv["edited"] is True
```

- [ ] **Step 2: Write `app/routers/report_io.py`**

```python
# app/routers/report_io.py
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException

from app.config import get_settings
from app.storage import Storage
from constants.report_sections import REPORT_SECTIONS

router = APIRouter()


def _store() -> Storage:
    return Storage(base=get_settings().data_dir)


@router.patch("/api/study/{sid}/report/section")
def patch_section(sid: str, body: dict = Body(...)):
    s = _store()
    name = body.get("section")
    content = body.get("content", "")
    if name not in REPORT_SECTIONS:
        raise HTTPException(status_code=400, detail="invalid section name")
    p = s.result_path(sid, "report.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail="report not generated yet")
    data = json.loads(Path(p).read_text())
    found = False
    for sec in data.get("sections", []):
        if sec["name"] == name:
            sec["content"] = content
            sec["edited"] = True
            found = True
            break
    if not found:
        data.setdefault("sections", []).append(
            {"name": name, "content": content, "edited": True}
        )
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"name": name, "content": content, "edited": True}
```

- [ ] **Step 3: Wire in `app/main.py`**

```python
from app.routers.report_io import router as report_io_router
app.include_router(report_io_router)
```

- [ ] **Step 4: Pass + commit**

```bash
ECHOCHAT_SKIP_MODEL=1 pytest tests/test_tasks_router.py::test_patch_report_section -v
git add app/routers/report_io.py app/main.py tests/test_tasks_router.py
git commit -m "Phase 4: PATCH /api/study/<sid>/report/section persists edits"
```

**Milestone Phase 4:** `pytest` green with fake engine. End-to-end via curl: create study → upload → process → run report/measurement/disease/vqa → stream events → read final result → edit a section.

---

# Phase 5 — Frontend

The frontend is server-rendered Jinja2 with Tailwind CSS via CDN, Inter font, lucide icons via inline SVG. No build step. Tests here are manual (load the page; verify the described state) plus a small number of integration tests verifying that routes return 200 and contain expected strings.

### Task 5.1: Base template + design tokens

**Files:** Create `templates/base.html`, `static/css/app.css`

- [ ] **Step 1: Write `templates/base.html`**

```html
{# templates/base.html #}
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{% block title %}EchoChat Demo{% endblock %}</title>
  <link rel="icon" href="/static/img/logo.svg"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"/>
  <link rel="stylesheet" href="/static/css/app.css"/>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          colors: {
            primary: '#0E6AAD',
            accent:  '#1F9D8F',
            warning: '#E09B3A',
            danger:  '#C0392B',
            surface: '#FFFFFF',
            page:    '#F7F9FB',
            ink:     '#1F2937',
            muted:   '#6B7280',
          },
          fontFamily: {
            sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
            mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
          },
          boxShadow: {
            card: '0 1px 3px rgba(16,24,40,0.04)',
            cardHover: '0 4px 12px rgba(16,24,40,0.08)',
          },
          borderRadius: {
            xl2: '12px',
          },
        }
      }
    }
  </script>
</head>
<body class="bg-page text-ink font-sans antialiased">
  {% block body %}{% endblock %}
</body>
</html>
```

- [ ] **Step 2: Write `static/css/app.css`**

```css
/* static/css/app.css — layered on top of Tailwind. */
:root {
  --transition-medium: 200ms ease-out;
}

/* Tabular digits for value tables */
.tabular-nums { font-variant-numeric: tabular-nums; }

/* Card */
.card {
  background: #FFFFFF;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
  transition: box-shadow var(--transition-medium);
}
.card:hover {
  box-shadow: 0 4px 12px rgba(16, 24, 40, 0.08);
}

/* Tab underline */
.tab-item {
  position: relative;
  padding: 10px 16px;
  color: #6B7280;
  transition: color var(--transition-medium);
  cursor: pointer;
}
.tab-item[aria-selected="true"] {
  color: #0E6AAD;
}
.tab-item::after {
  content: '';
  position: absolute;
  left: 16px;
  right: 16px;
  bottom: 0;
  height: 2px;
  background: #0E6AAD;
  transform: scaleX(0);
  transform-origin: left center;
  transition: transform var(--transition-medium);
}
.tab-item[aria-selected="true"]::after { transform: scaleX(1); }

/* View badge */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid transparent;
}
.badge--ok      { background: rgba(31,157,143,0.08); color: #1F9D8F; border-color: rgba(31,157,143,0.25); }
.badge--warn    { background: rgba(224,155,58,0.10); color: #8A5A1A; border-color: rgba(224,155,58,0.30); }
.badge--unknown { background: rgba(107,114,128,0.08); color: #4B5563; border-color: rgba(107,114,128,0.20); }

/* Drop zone */
.dropzone {
  border: 2px dashed #CBD5E1;
  border-radius: 16px;
  transition: all var(--transition-medium);
}
.dropzone--hover { border-color: #0E6AAD; background: rgba(14,106,173,0.04); }

/* Pulse for drag */
@keyframes subtle-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(14,106,173,0.0); }
  50%     { box-shadow: 0 0 0 4px rgba(14,106,173,0.10); }
}
.dropzone--hover { animation: subtle-pulse 1.2s ease-in-out infinite; }

/* Report section editable */
[contenteditable="true"] { outline: none; }
.report-section {
  padding: 12px 16px;
  border-radius: 8px;
  transition: background var(--transition-medium);
}
.report-section:focus-within {
  background: rgba(14,106,173,0.04);
}
```

- [ ] **Step 3: Commit**

```bash
git add templates/base.html static/css/app.css
git commit -m "Phase 5: base Jinja template + Tailwind config + design tokens CSS"
```

### Task 5.2: Login page

**Files:** Create `templates/login.html`, update `app/main.py` to render templates

- [ ] **Step 1: Write `templates/login.html`**

```html
{# templates/login.html #}
{% extends "base.html" %}
{% block title %}Sign in · EchoChat Demo{% endblock %}
{% block body %}
<main class="min-h-screen grid place-items-center px-6">
  <div class="w-full max-w-sm">
    <div class="flex items-center gap-3 justify-center mb-10">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#0E6AAD" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h3l2-7 3 14 2-10 3 7h5"/></svg>
      <span class="text-xl font-semibold tracking-tight">EchoChat Demo</span>
    </div>
    <div class="card p-8">
      <h1 class="text-lg font-semibold mb-1">Sign in</h1>
      <p class="text-sm text-muted mb-6">Enter the shared demo password to continue.</p>
      {% if error %}
      <div class="mb-4 px-3 py-2 rounded-lg bg-red-50 text-danger text-sm">{{ error }}</div>
      {% endif %}
      <form method="post" action="/login" class="space-y-4">
        <label class="block">
          <span class="text-sm font-medium">Password</span>
          <input type="password" name="password" autofocus required
                 class="mt-1 w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/40" />
        </label>
        <button type="submit" class="w-full py-2 rounded-lg bg-primary text-white font-medium hover:bg-primary/90 transition">
          Continue
        </button>
      </form>
    </div>
    <p class="text-xs text-muted text-center mt-6">For reviewer use only.</p>
  </div>
</main>
{% endblock %}
```

- [ ] **Step 2: Update `app/main.py` to use templates + static mount**

```python
# at top of app/main.py, add:
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

# inside create_app(), replace the existing /login GET and / GET:
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/home", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str | None = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

# static mount
app.mount("/static", StaticFiles(directory="static"), name="static")
```

And import `Request` from fastapi at the top.

- [ ] **Step 3: Integration test**

```python
# append to tests/test_auth.py
def test_login_page_renders_english_form(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    r = c.get("/login")
    assert r.status_code == 200
    assert "Sign in" in r.text
    assert "password" in r.text.lower()
    assert "<html lang=\"en\"" in r.text
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_auth.py::test_login_page_renders_english_form -v
git add templates/login.html app/main.py tests/test_auth.py
git commit -m "Phase 5: login page (Tailwind, Inter, clinical aesthetic)"
```

### Task 5.3: Home page

**Files:** Create `templates/home.html`

- [ ] **Step 1: Write template**

```html
{# templates/home.html #}
{% extends "base.html" %}
{% block title %}EchoChat · Home{% endblock %}
{% block body %}
<header class="sticky top-0 z-10 bg-white/90 backdrop-blur border-b border-slate-200">
  <div class="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
    <div class="flex items-center gap-2">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0E6AAD" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h3l2-7 3 14 2-10 3 7h5"/></svg>
      <span class="font-semibold tracking-tight">EchoChat Demo</span>
    </div>
    <form method="post" action="/logout"><button class="text-sm text-muted hover:text-ink transition">Sign out</button></form>
  </div>
</header>

<main class="max-w-6xl mx-auto px-6 py-14">
  <section class="mb-14">
    <h1 class="text-4xl font-semibold tracking-tight">Echocardiography AI, end to end.</h1>
    <p class="mt-3 text-muted max-w-2xl">
      Upload a complete echo study and review four AI-powered tasks: sectioned report generation,
      structured measurements, disease evaluation, and visual question answering.
      All capabilities are bounded and traceable.
    </p>
    <a href="/upload" class="inline-flex items-center gap-2 mt-8 px-5 py-3 rounded-xl bg-primary text-white font-medium hover:bg-primary/90 transition">
      Start new study
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
    </a>
  </section>

  <section class="grid grid-cols-1 md:grid-cols-2 gap-5">
    {% for card in cards %}
    <article class="card p-6">
      <div class="flex items-start gap-3">
        <div class="w-10 h-10 rounded-xl bg-primary/10 text-primary grid place-items-center">
          {{ card.icon | safe }}
        </div>
        <div class="flex-1">
          <h3 class="font-semibold text-lg">{{ card.title }}</h3>
          <p class="text-sm text-muted mt-1">{{ card.desc }}</p>
          <ul class="mt-3 text-xs text-muted space-y-1">
            {% for req in card.inputs %}<li>· {{ req }}</li>{% endfor %}
          </ul>
        </div>
        <span class="text-xs text-muted">{{ card.stat }}</span>
      </div>
    </article>
    {% endfor %}
  </section>
</main>
{% endblock %}
```

- [ ] **Step 2: Update home route in `app/main.py`**

```python
# replace home_page in app/main.py
@app.get("/home", response_class=HTMLResponse)
def home_page(request: Request):
    cards = [
        {
            "title": "Report Generation",
            "desc": "Generate a 10-section echocardiography report (AV, Atria, GV, LV, MV, Pericardium, PV, RV, TV, Summary).",
            "inputs": ["Full echo study with at least A4C + PLAX views recommended."],
            "stat": "10 sections",
            "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
        },
        {
            "title": "Measurement",
            "desc": "Select structured echo parameters to measure, with clinical units.",
            "inputs": ["B-mode 2D clips + Doppler/spectrum clips as applicable."],
            "stat": "22 measurements",
            "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 3H3v18h18z"/><path d="M7 7v10"/><path d="M11 10v7"/><path d="M15 13v4"/></svg>',
        },
        {
            "title": "Disease Diagnosis",
            "desc": "Evaluate presence of supported cardiac conditions from the uploaded study.",
            "inputs": ["Full echo study; views vary per condition."],
            "stat": "28 conditions",
            "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-6 8-12a8 8 0 1 0-16 0c0 6 8 12 8 12z"/></svg>',
        },
        {
            "title": "Visual Question Answering",
            "desc": "Ask focused clinical questions about the uploaded study; responses are bounded to echocardiography.",
            "inputs": ["At least one clip; questions must be echo-related."],
            "stat": "Bounded input",
            "icon": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/></svg>',
        },
    ]
    return templates.TemplateResponse("home.html", {"request": request, "cards": cards})
```

- [ ] **Step 3: Integration test**

```python
# append to tests/test_auth.py
def test_home_page_has_task_cards(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    r = c.get("/home")
    assert r.status_code == 200
    assert "Report Generation" in r.text
    assert "Measurement" in r.text
    assert "Disease Diagnosis" in r.text
    assert "Visual Question Answering" in r.text
    assert "Start new study" in r.text
```

- [ ] **Step 4: Pass + commit**

```bash
pytest tests/test_auth.py::test_home_page_has_task_cards -v
git add templates/home.html app/main.py tests/test_auth.py
git commit -m "Phase 5: home page with 4 task cards"
```

### Task 5.4: Upload page + JS

**Files:** Create `templates/upload.html`, `static/js/upload.js`

- [ ] **Step 1: Write `templates/upload.html`**

```html
{# templates/upload.html #}
{% extends "base.html" %}
{% block title %}Upload · EchoChat{% endblock %}
{% block body %}
<header class="sticky top-0 z-10 bg-white/90 backdrop-blur border-b border-slate-200">
  <div class="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
    <div class="flex items-center gap-2">
      <a href="/home" class="flex items-center gap-2 text-muted hover:text-ink transition">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M11 18l-6-6 6-6"/></svg>
        <span class="text-sm">Home</span>
      </a>
    </div>
    <div class="text-sm text-muted">New study</div>
  </div>
</header>

<main class="max-w-4xl mx-auto px-6 py-14">
  <h1 class="text-2xl font-semibold tracking-tight">Upload an echocardiography study</h1>
  <p class="mt-2 text-muted">Drop one or more DICOM files. Filenames need not be standard; the system detects DICOM by content.</p>

  <section id="dropzone" class="dropzone mt-8 py-16 px-8 text-center">
    <svg class="mx-auto mb-4 text-primary" width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
    <p class="font-medium">Drag DICOM files here, or <label class="text-primary cursor-pointer">browse<input id="file-input" type="file" multiple class="hidden"/></label></p>
    <p class="text-xs text-muted mt-2">Any extension accepted. Non-DICOM files are rejected.</p>
  </section>

  <section id="file-list" class="mt-8 space-y-2"></section>

  <section class="mt-8 flex gap-3">
    <button id="continue-btn" disabled class="px-4 py-2 rounded-xl bg-primary text-white font-medium disabled:bg-slate-300 disabled:cursor-not-allowed transition">Continue to workspace</button>
    <button id="new-study-btn" class="px-4 py-2 rounded-xl border border-slate-200 text-muted hover:bg-slate-50 transition">Start over</button>
  </section>
</main>
<script src="/static/js/upload.js"></script>
{% endblock %}
```

- [ ] **Step 2: Write `static/js/upload.js`**

```javascript
// static/js/upload.js
(async function () {
  const dz = document.getElementById('dropzone');
  const input = document.getElementById('file-input');
  const list = document.getElementById('file-list');
  const continueBtn = document.getElementById('continue-btn');
  const newBtn = document.getElementById('new-study-btn');

  let studyId = null;
  const fileRows = new Map();

  async function ensureStudy() {
    if (studyId) return studyId;
    const r = await fetch('/api/study', { method: 'POST' });
    studyId = (await r.json()).study_id;
    return studyId;
  }

  function row(name) {
    const el = document.createElement('div');
    el.className = 'card px-4 py-3 flex items-center justify-between';
    el.innerHTML = `
      <div class="min-w-0">
        <div class="font-medium truncate">${name}</div>
        <div class="text-xs text-muted mt-0.5" data-status>Queued</div>
      </div>
      <div class="w-32 bg-slate-100 rounded-full h-1 overflow-hidden">
        <div class="h-1 bg-primary" style="width:0%" data-bar></div>
      </div>
    `;
    list.appendChild(el);
    return el;
  }

  function setStatus(el, text) { el.querySelector('[data-status]').textContent = text; }
  function setBar(el, pct) { el.querySelector('[data-bar]').style.width = pct + '%'; }

  async function sniffDicom(file) {
    // Read bytes 128..132 check 'DICM'
    if (file.size < 132) return false;
    const buf = await file.slice(128, 132).arrayBuffer();
    const s = new TextDecoder().decode(buf);
    return s === 'DICM';
  }

  async function uploadOne(file) {
    if (!await sniffDicom(file)) {
      const el = row(file.name);
      setStatus(el, 'Rejected (not DICOM)');
      return;
    }
    const el = row(file.name);
    setStatus(el, 'Uploading…');
    const sid = await ensureStudy();
    const form = new FormData();
    form.append('file', file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setBar(el, (e.loaded / e.total * 100) | 0);
    };
    await new Promise((resolve) => {
      xhr.onload = () => {
        if (xhr.status === 200) {
          const j = JSON.parse(xhr.responseText);
          fileRows.set(j.file_id, el);
          setStatus(el, 'Uploaded, awaiting classification…');
        } else {
          setStatus(el, 'Failed: ' + xhr.status);
        }
        resolve();
      };
      xhr.open('POST', `/api/study/${sid}/upload`);
      xhr.send(form);
    });
  }

  async function runProcess() {
    if (!studyId) return;
    const es = new EventSource(`/api/study/${studyId}/process`);
    es.addEventListener('phase', (evt) => {
      const data = JSON.parse(evt.data);
      if (data.file_id && fileRows.has(data.file_id)) {
        const el = fileRows.get(data.file_id);
        setStatus(el, `${data.phase.replace('_', ' ')}…`);
      }
    });
    es.addEventListener('clip', (evt) => {
      const data = JSON.parse(evt.data);
      const el = fileRows.get(data.file_id);
      if (el) setStatus(el, `View: ${data.view || 'Unknown'}`);
    });
    es.addEventListener('error', (evt) => { es.close(); });
    es.addEventListener('done', () => {
      es.close();
      continueBtn.disabled = false;
    });
  }

  async function handleFiles(files) {
    for (const f of files) await uploadOne(f);
    await runProcess();
  }

  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dropzone--hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dropzone--hover'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dropzone--hover');
    if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  });
  input.addEventListener('change', () => handleFiles(input.files));

  continueBtn.addEventListener('click', () => {
    if (studyId) location.href = `/workspace/${studyId}`;
  });
  newBtn.addEventListener('click', () => location.reload());
})();
```

- [ ] **Step 3: Wire route**

```python
# app/main.py (add)
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})
```

- [ ] **Step 4: Integration test**

```python
# append to tests/test_auth.py
def test_upload_page_renders(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    r = c.get("/upload")
    assert r.status_code == 200
    assert "Drag DICOM" in r.text
    assert "upload.js" in r.text
```

- [ ] **Step 5: Pass + commit**

```bash
pytest tests/test_auth.py::test_upload_page_renders -v
git add templates/upload.html static/js/upload.js app/main.py tests/test_auth.py
git commit -m "Phase 5: upload page + drag-drop JS with DICOM sniff + SSE classify"
```

### Task 5.5: Workspace page + JS + editor

**Files:** Create `templates/workspace.html`, `static/js/workspace.js`, `static/js/editor.js`

- [ ] **Step 1: Write `templates/workspace.html`**

```html
{# templates/workspace.html #}
{% extends "base.html" %}
{% block title %}Workspace · EchoChat{% endblock %}
{% block body %}
<header class="sticky top-0 z-10 bg-white/90 backdrop-blur border-b border-slate-200">
  <div class="max-w-[1400px] mx-auto px-6 h-14 flex items-center justify-between">
    <a href="/home" class="flex items-center gap-2 text-muted hover:text-ink transition">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 12H5M11 18l-6-6 6-6"/></svg>
      <span class="text-sm">Home</span>
    </a>
    <div class="text-xs text-muted font-mono">study {{ study_id }}</div>
  </div>
</header>

<main class="max-w-[1400px] mx-auto px-6 py-6 grid grid-cols-12 gap-6">
  <!-- Study panel -->
  <aside class="col-span-12 lg:col-span-4 xl:col-span-3 space-y-4">
    <div class="card p-5">
      <h2 class="font-semibold mb-3">Study</h2>
      <div id="clips-list" class="space-y-2 max-h-[400px] overflow-auto pr-1"></div>
      <a href="/upload" class="mt-4 inline-block text-primary text-sm hover:underline">+ Add more clips</a>
    </div>
    <div class="card p-5">
      <h2 class="font-semibold mb-3">Capabilities</h2>
      <div id="tasks-avail" class="space-y-2 text-sm"></div>
    </div>
  </aside>

  <!-- Task tabs -->
  <section class="col-span-12 lg:col-span-8 xl:col-span-9 card overflow-hidden">
    <nav class="flex border-b border-slate-100" role="tablist">
      <button class="tab-item" data-tab="report" aria-selected="true">Report</button>
      <button class="tab-item" data-tab="measurement" aria-selected="false">Measurement</button>
      <button class="tab-item" data-tab="disease" aria-selected="false">Disease</button>
      <button class="tab-item" data-tab="vqa" aria-selected="false">VQA</button>
    </nav>

    {% include 'components/task_tab_report.html' %}
    {% include 'components/task_tab_measurement.html' %}
    {% include 'components/task_tab_disease.html' %}
    {% include 'components/task_tab_vqa.html' %}
  </section>
</main>

<script>
  window.ECHOCHAT_STUDY_ID = {{ study_id | tojson }};
</script>
<script src="/static/js/editor.js"></script>
<script src="/static/js/workspace.js"></script>
{% endblock %}
```

- [ ] **Step 2: Write component partials**

`templates/components/task_tab_report.html`:

```html
<div class="tab-panel p-6" data-panel="report">
  <div class="flex items-center justify-between mb-5">
    <div>
      <h3 class="font-semibold text-lg">Report</h3>
      <p class="text-sm text-muted">Generate a sectioned echocardiography report. You can edit any section after generation.</p>
    </div>
    <div class="flex items-center gap-2">
      <button id="report-run" class="px-4 py-2 rounded-lg bg-primary text-white font-medium hover:bg-primary/90 transition">Generate report</button>
      <a id="report-pdf" href="#" class="hidden px-3 py-2 rounded-lg border border-slate-200 text-sm hover:bg-slate-50 transition">Export PDF</a>
      <a id="report-docx" href="#" class="hidden px-3 py-2 rounded-lg border border-slate-200 text-sm hover:bg-slate-50 transition">Export DOCX</a>
    </div>
  </div>
  <div id="report-progress" class="text-sm text-muted mb-3"></div>
  <div id="report-sections" class="space-y-1"></div>
</div>
```

`templates/components/task_tab_measurement.html`:

```html
<div class="tab-panel p-6 hidden" data-panel="measurement">
  <div class="flex items-start justify-between mb-5 gap-4">
    <div>
      <h3 class="font-semibold text-lg">Measurement</h3>
      <p class="text-sm text-muted">Select items and run. Results stream in below.</p>
    </div>
    <div>
      <button id="measure-run" class="px-4 py-2 rounded-lg bg-primary text-white font-medium hover:bg-primary/90 transition">Run selected</button>
    </div>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div>
      <div id="measure-presets" class="flex flex-wrap gap-2 mb-3"></div>
      <div id="measure-items" class="max-h-[420px] overflow-auto border border-slate-100 rounded-lg divide-y"></div>
    </div>
    <div>
      <h4 class="text-sm font-semibold text-muted mb-2">Results</h4>
      <div class="border border-slate-100 rounded-lg overflow-hidden">
        <table class="w-full text-sm">
          <thead class="bg-slate-50 text-muted">
            <tr><th class="text-left px-3 py-2">Measurement</th><th class="text-right px-3 py-2">Value</th><th class="text-left px-3 py-2">Unit</th></tr>
          </thead>
          <tbody id="measure-results" class="divide-y"></tbody>
        </table>
      </div>
    </div>
  </div>
  <div id="measure-progress" class="text-sm text-muted mt-3"></div>
</div>
```

`templates/components/task_tab_disease.html`:

```html
<div class="tab-panel p-6 hidden" data-panel="disease">
  <div class="flex items-start justify-between mb-5 gap-4">
    <div>
      <h3 class="font-semibold text-lg">Disease Diagnosis</h3>
      <p class="text-sm text-muted">Select conditions to evaluate. Answers are bounded yes/no/unknown.</p>
    </div>
    <div>
      <button id="disease-run" class="px-4 py-2 rounded-lg bg-primary text-white font-medium hover:bg-primary/90 transition">Evaluate selected</button>
    </div>
  </div>
  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div>
      <div id="disease-presets" class="flex flex-wrap gap-2 mb-3"></div>
      <div id="disease-items" class="max-h-[420px] overflow-auto border border-slate-100 rounded-lg divide-y"></div>
    </div>
    <div>
      <h4 class="text-sm font-semibold text-muted mb-2">Results</h4>
      <div id="disease-results" class="border border-slate-100 rounded-lg divide-y"></div>
    </div>
  </div>
  <div id="disease-progress" class="text-sm text-muted mt-3"></div>
</div>
```

`templates/components/task_tab_vqa.html`:

```html
<div class="tab-panel p-6 hidden" data-panel="vqa">
  <h3 class="font-semibold text-lg">Visual Question Answering</h3>
  <p class="text-sm text-muted mb-4">Ask a focused clinical question about the uploaded study.</p>
  <select id="vqa-preset" class="w-full md:w-96 px-3 py-2 border border-slate-200 rounded-lg mb-3 hidden">
    <option value="">— Example questions —</option>
  </select>
  <div id="vqa-log" class="space-y-3 mb-4 max-h-[420px] overflow-auto pr-1"></div>
  <form id="vqa-form" class="flex items-center gap-2">
    <input id="vqa-q" type="text" placeholder="e.g. Describe the wall motion of the LV." class="flex-1 px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/30"/>
    <button class="px-4 py-2 rounded-lg bg-primary text-white font-medium hover:bg-primary/90 transition">Send</button>
  </form>
  <div id="vqa-progress" class="text-sm text-muted mt-3"></div>
</div>
```

- [ ] **Step 3: Write `static/js/editor.js`**

```javascript
// static/js/editor.js — contenteditable report section with blur autosave.
window.attachSectionEditor = function (el, studyId, sectionName) {
  el.setAttribute('contenteditable', 'true');
  el.setAttribute('spellcheck', 'false');
  el.addEventListener('blur', async () => {
    const content = el.innerText.trim();
    const resp = await fetch(`/api/study/${studyId}/report/section`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ section: sectionName, content }),
    });
    if (resp.ok) {
      el.dataset.edited = '1';
    }
  });
};
```

- [ ] **Step 4: Write `static/js/workspace.js`**

```javascript
// static/js/workspace.js — orchestrates the whole workspace page.
(async function () {
  const sid = window.ECHOCHAT_STUDY_ID;

  // --- Tabs ---
  document.querySelectorAll('.tab-item').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-item').forEach((b) => b.setAttribute('aria-selected', 'false'));
      btn.setAttribute('aria-selected', 'true');
      const key = btn.dataset.tab;
      document.querySelectorAll('.tab-panel').forEach((p) => {
        p.classList.toggle('hidden', p.dataset.panel !== key);
      });
    });
  });

  // --- Constants (diseases, measurements, presets) ---
  const constants = await (await fetch('/api/constants')).json();

  // ---- Study panel ----
  async function renderStudy() {
    const s = await (await fetch(`/api/study/${sid}`)).json();

    // Clips
    const clipsEl = document.getElementById('clips-list');
    clipsEl.innerHTML = '';
    for (const c of s.clips) {
      const view = c.user_view || c.view || null;
      const row = document.createElement('div');
      row.className = 'flex items-center gap-3 p-2 rounded-lg hover:bg-slate-50 transition';
      const badgeCls = view ? 'badge badge--ok' : 'badge badge--unknown';
      row.innerHTML = `
        <img src="/api/study/${sid}/clip/${c.file_id}/thumbnail" alt="" class="w-12 h-12 rounded-md object-cover bg-slate-100"/>
        <div class="min-w-0 flex-1">
          <div class="text-sm font-medium truncate">${c.original_filename}</div>
          <div class="mt-1 flex items-center gap-2">
            <span class="${badgeCls}">${view || 'Unknown'}</span>
            ${c.user_view ? '<span class="text-xs text-muted">(manual)</span>' : ''}
          </div>
        </div>
        <button class="text-xs text-muted hover:text-danger transition" data-delete="${c.file_id}">Remove</button>
      `;
      // Click-to-edit view
      row.querySelector('span.badge').addEventListener('click', async () => {
        const choice = prompt('Set view (exact label from VIEW_LABELS, or blank for unknown):', view || '');
        if (choice === null) return;
        const body = choice.trim() ? { user_view: choice.trim() } : { user_view: null };
        const r = await fetch(`/api/study/${sid}/clip/${c.file_id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (r.ok) renderStudy();
      });
      row.querySelector('[data-delete]').addEventListener('click', async () => {
        if (!confirm(`Remove ${c.original_filename}?`)) return;
        await fetch(`/api/study/${sid}/clip/${c.file_id}`, { method: 'DELETE' });
        renderStudy();
      });
      clipsEl.appendChild(row);
    }

    // Availability
    const avail = document.getElementById('tasks-avail');
    const missing = s.tasks.missing_groups || [];
    avail.innerHTML = `
      <div>Report: ${s.tasks.report ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      <div>Measurement: ${s.tasks.measurement ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      <div>Disease: ${s.tasks.disease ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      <div>VQA: ${s.tasks.vqa ? '<span class="badge badge--ok">ready</span>' : '<span class="badge badge--warn">no clips</span>'}</div>
      ${missing.length ? `<div class="pt-2 text-xs text-muted">Missing views: ${missing.join(', ')} (soft warning)</div>` : ''}
    `;
  }
  await renderStudy();

  // ---- Measurement tab ----
  const measPresetsEl = document.getElementById('measure-presets');
  const measItemsEl = document.getElementById('measure-items');
  for (const [name, items] of Object.entries(constants.presets.measurements)) {
    const b = document.createElement('button');
    b.className = 'px-3 py-1.5 text-xs rounded-full border border-slate-200 hover:bg-slate-50 transition';
    b.textContent = name;
    b.addEventListener('click', () => {
      measItemsEl.querySelectorAll('input').forEach((inp) => {
        inp.checked = items.includes(inp.value);
      });
    });
    measPresetsEl.appendChild(b);
  }
  for (const m of constants.measurements) {
    const row = document.createElement('label');
    row.className = 'flex items-center gap-2 px-3 py-2 hover:bg-slate-50 transition cursor-pointer';
    row.innerHTML = `<input type="checkbox" value="${m}"/><span class="text-sm">${m}</span>`;
    measItemsEl.appendChild(row);
  }

  // ---- Disease tab ----
  const disPresetsEl = document.getElementById('disease-presets');
  const disItemsEl = document.getElementById('disease-items');
  for (const [name, items] of Object.entries(constants.presets.diseases)) {
    const b = document.createElement('button');
    b.className = 'px-3 py-1.5 text-xs rounded-full border border-slate-200 hover:bg-slate-50 transition';
    b.textContent = name;
    b.addEventListener('click', () => {
      disItemsEl.querySelectorAll('input').forEach((inp) => { inp.checked = items.includes(inp.value); });
    });
    disPresetsEl.appendChild(b);
  }
  for (const d of constants.diseases) {
    const row = document.createElement('label');
    row.className = 'flex items-center gap-2 px-3 py-2 hover:bg-slate-50 transition cursor-pointer';
    row.innerHTML = `<input type="checkbox" value="${d}"/><span class="text-sm">${d}</span>`;
    disItemsEl.appendChild(row);
  }

  // ---- VQA preset ----
  if (constants.presets.vqa_examples && constants.presets.vqa_examples.length) {
    const sel = document.getElementById('vqa-preset');
    sel.classList.remove('hidden');
    for (const q of constants.presets.vqa_examples) {
      const o = document.createElement('option'); o.value = q; o.textContent = q; sel.appendChild(o);
    }
    sel.addEventListener('change', () => {
      if (sel.value) document.getElementById('vqa-q').value = sel.value;
    });
  }

  // ---- Task runners ----
  function streamTask(taskId, onEvent) {
    const es = new EventSource(`/api/task/${taskId}/stream`);
    for (const kind of ['phase', 'partial', 'item', 'message', 'error', 'done']) {
      es.addEventListener(kind, (evt) => onEvent(kind, JSON.parse(evt.data)));
    }
    es.addEventListener('done', () => es.close());
    es.addEventListener('error', () => es.close());
    return es;
  }

  // Report
  const reportRun = document.getElementById('report-run');
  const reportSections = document.getElementById('report-sections');
  const reportProgress = document.getElementById('report-progress');
  const reportPdf = document.getElementById('report-pdf');
  const reportDocx = document.getElementById('report-docx');

  async function runReport() {
    if (reportSections.children.length && !confirm('This will discard your current edits. Continue?')) return;
    reportSections.innerHTML = '';
    reportProgress.textContent = 'Preparing echocardiography context…';
    const r = await (await fetch(`/api/study/${sid}/task/report`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'phase') {
        if (data.phase === 'inference') reportProgress.textContent = 'Generating report…';
        if (data.phase === 'preparing_context') reportProgress.textContent = 'Preparing echocardiography context…';
      } else if (kind === 'partial') {
        const wrap = document.createElement('div');
        wrap.className = 'report-section border border-slate-100 rounded-lg';
        wrap.innerHTML = `
          <div class="flex items-center justify-between px-4 py-2 text-sm text-muted">
            <span class="font-medium text-ink">${data.section}</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
          </div>
          <div class="px-4 pb-3 text-sm leading-relaxed whitespace-pre-wrap" data-section-body></div>
        `;
        wrap.querySelector('[data-section-body]').textContent = data.content;
        window.attachSectionEditor(wrap.querySelector('[data-section-body]'), sid, data.section);
        reportSections.appendChild(wrap);
      } else if (kind === 'done') {
        reportProgress.textContent = 'Report generated. You can edit any section above.';
        reportPdf.href = `/api/study/${sid}/report/export?format=pdf`;
        reportDocx.href = `/api/study/${sid}/report/export?format=docx`;
        reportPdf.classList.remove('hidden');
        reportDocx.classList.remove('hidden');
      } else if (kind === 'error') {
        reportProgress.textContent = 'Generation failed: ' + data.reason;
      }
    });
  }
  reportRun.addEventListener('click', runReport);

  // Measurement
  document.getElementById('measure-run').addEventListener('click', async () => {
    const items = [...measItemsEl.querySelectorAll('input:checked')].map((i) => i.value);
    if (!items.length) return;
    const tbody = document.getElementById('measure-results');
    tbody.innerHTML = '';
    const progress = document.getElementById('measure-progress');
    const r = await (await fetch(`/api/study/${sid}/task/measurement`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'phase' && data.phase === 'during') {
        progress.textContent = `Measuring ${data.name} (${data.i}/${data.n})…`;
      } else if (kind === 'item') {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td class="px-3 py-2">${data.name}</td><td class="px-3 py-2 text-right tabular-nums font-mono">${data.value ?? '—'}</td><td class="px-3 py-2 text-muted">${data.unit ?? ''}</td>`;
        tbody.appendChild(tr);
      } else if (kind === 'done') {
        progress.textContent = `${items.length} measurements complete.`;
      } else if (kind === 'error') {
        progress.textContent = 'Failed: ' + data.reason;
      }
    });
  });

  // Disease
  document.getElementById('disease-run').addEventListener('click', async () => {
    const items = [...disItemsEl.querySelectorAll('input:checked')].map((i) => i.value);
    if (!items.length) return;
    const results = document.getElementById('disease-results');
    results.innerHTML = '';
    const progress = document.getElementById('disease-progress');
    const r = await (await fetch(`/api/study/${sid}/task/disease`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items }),
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'phase' && data.phase === 'during') {
        progress.textContent = `Evaluating ${data.name} (${data.i}/${data.n})…`;
      } else if (kind === 'item') {
        const row = document.createElement('details');
        row.className = 'p-3';
        const cls = data.answer === 'yes' ? 'badge--warn'
                  : data.answer === 'no' ? 'badge--ok' : 'badge--unknown';
        row.innerHTML = `
          <summary class="flex items-center gap-3 cursor-pointer">
            <span class="badge ${cls}">${data.answer}</span>
            <span class="text-sm">${data.name}</span>
          </summary>
          <pre class="mt-2 px-3 py-2 bg-slate-50 rounded text-xs whitespace-pre-wrap">${data.raw}</pre>
        `;
        results.appendChild(row);
      } else if (kind === 'done') {
        progress.textContent = `${items.length} conditions evaluated.`;
      } else if (kind === 'error') {
        progress.textContent = 'Failed: ' + data.reason;
      }
    });
  });

  // VQA
  document.getElementById('vqa-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = document.getElementById('vqa-q').value.trim();
    if (!q) return;
    const log = document.getElementById('vqa-log');
    const ubub = document.createElement('div');
    ubub.className = 'flex justify-end';
    ubub.innerHTML = `<div class="max-w-[80%] bg-primary text-white rounded-2xl rounded-br-sm px-4 py-2 text-sm">${q}</div>`;
    log.appendChild(ubub);
    document.getElementById('vqa-q').value = '';
    const progress = document.getElementById('vqa-progress');
    progress.textContent = 'Thinking…';
    const r = await (await fetch(`/api/study/${sid}/task/vqa`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    })).json();
    streamTask(r.task_id, (kind, data) => {
      if (kind === 'message') {
        const bub = document.createElement('div');
        bub.className = 'flex';
        bub.innerHTML = `<div class="max-w-[80%] bg-slate-100 rounded-2xl rounded-bl-sm px-4 py-2 text-sm whitespace-pre-wrap">${data.content}</div>`;
        log.appendChild(bub);
      } else if (kind === 'done') {
        progress.textContent = '';
      } else if (kind === 'error') {
        progress.textContent = 'Failed: ' + data.reason;
      }
    });
  });
})();
```

- [ ] **Step 5: Wire route**

```python
# app/main.py (add)
@app.get("/workspace/{study_id}", response_class=HTMLResponse)
def workspace_page(request: Request, study_id: str):
    return templates.TemplateResponse(
        "workspace.html", {"request": request, "study_id": study_id},
    )
```

- [ ] **Step 6: Integration test**

```python
# append to tests/test_auth.py
def test_workspace_page_renders(monkeypatch, tmp_path):
    c = _make_client(monkeypatch, tmp_path)
    c.post("/login", data={"password": "pw"})
    r = c.get("/workspace/some-study-id")
    assert r.status_code == 200
    assert "Report" in r.text
    assert "Measurement" in r.text
    assert "Disease" in r.text
    assert "VQA" in r.text
    assert "workspace.js" in r.text
    assert "window.ECHOCHAT_STUDY_ID" in r.text
```

- [ ] **Step 7: Pass + commit**

```bash
pytest tests/test_auth.py::test_workspace_page_renders -v
git add templates/workspace.html templates/components/ static/js/workspace.js static/js/editor.js app/main.py tests/test_auth.py
git commit -m "Phase 5: workspace page with study panel, 4 task tabs, SSE consumers, editor"
```

**Milestone Phase 5:** Manual smoke-test — open `/login`, sign in, `/upload`, drop a test DICOM, `/workspace/<sid>`, see clip pill, tabs, and run a task with the engine stubbed via `ECHOCHAT_SKIP_MODEL=1` and an injected fake engine (the tasks won't actually produce output; checks only that the UI wires events).

---

# Phase 6 — Export (PDF + DOCX)

### Task 6.1: `export.py` — render report to PDF / DOCX

**Files:** Create `app/services/export.py`, `tests/test_export.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_export.py
import io
from pathlib import Path

from app.services.export import render_report_pdf, render_report_docx


def _sample_data():
    return {
        "sections": [
            {"name": "Aortic Valve", "content": "Normal.", "edited": False},
            {"name": "Left Ventricle", "content": "EF 55%.", "edited": True},
            {"name": "Summary", "content": "Unremarkable.", "edited": False},
        ]
    }


def test_render_pdf_writes_non_empty_pdf(tmp_path):
    out = tmp_path / "r.pdf"
    render_report_pdf(_sample_data(), out)
    data = out.read_bytes()
    assert data[:4] == b"%PDF"
    assert len(data) > 1000


def test_render_docx_writes_non_empty_docx(tmp_path):
    out = tmp_path / "r.docx"
    render_report_docx(_sample_data(), out)
    # DOCX is a zip
    assert out.read_bytes()[:2] == b"PK"
```

- [ ] **Step 2: Write implementation**

```python
# app/services/export.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from weasyprint import HTML
from docx import Document
from docx.shared import Pt


_CSS = """
  @page { size: A4; margin: 22mm 18mm; }
  body { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; color: #1F2937; font-size: 11pt; }
  h1 { font-size: 18pt; margin-bottom: 0; color: #0E6AAD; }
  h2 { font-size: 12pt; margin-top: 16px; color: #0E6AAD; border-bottom: 1px solid #E5E7EB; padding-bottom: 4px; }
  p  { white-space: pre-wrap; line-height: 1.5; }
  .edited { color: #1F9D8F; font-size: 9pt; text-transform: uppercase; letter-spacing: 0.04em; }
  .meta { color: #6B7280; font-size: 9pt; margin-bottom: 18px; }
"""


def render_report_pdf(data: dict, out_path: Path) -> None:
    sections = data.get("sections", [])
    html_parts = [
        "<html><head><meta charset='utf-8'><style>", _CSS,
        "</style></head><body>",
        "<h1>Echocardiography Report</h1>",
        "<div class='meta'>Generated by EchoChat Demo</div>",
    ]
    for sec in sections:
        html_parts.append(f"<h2>{sec['name']}"
                          f"{' <span class=edited>(edited)</span>' if sec.get('edited') else ''}</h2>")
        content = (sec.get("content") or "").strip() or "Not reported."
        html_parts.append(f"<p>{_escape(content)}</p>")
    html_parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string="".join(html_parts)).write_pdf(str(out_path))


def render_report_docx(data: dict, out_path: Path) -> None:
    doc = Document()
    title = doc.add_heading("Echocardiography Report", level=1)
    meta = doc.add_paragraph("Generated by EchoChat Demo")
    for run in meta.runs:
        run.font.size = Pt(9)
    for sec in data.get("sections", []):
        h = doc.add_heading(sec["name"], level=2)
        if sec.get("edited"):
            r = doc.add_paragraph()
            er = r.add_run("(edited)")
            er.italic = True
            er.font.size = Pt(9)
        doc.add_paragraph((sec.get("content") or "").strip() or "Not reported.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))
```

- [ ] **Step 3: Pass + commit**

```bash
pytest tests/test_export.py -v
git add app/services/export.py tests/test_export.py
git commit -m "Phase 6: report PDF (weasyprint) and DOCX (python-docx) renderers"
```

### Task 6.2: Export route

**Files:** Extend `app/routers/report_io.py`, `tests/test_tasks_router.py`

- [ ] **Step 1: Extend test**

```python
# append to tests/test_tasks_router.py
def test_export_pdf(monkeypatch, tmp_path):
    engine = FakeEngine(
        "\n".join([f"{s}: ok." for s in [
            "Aortic Valve","Atria","Great Vessels","Left Ventricle","Mitral Valve",
            "Pericardium Pleural","Pulmonic Valve","Right Ventricle","Tricuspid Valve","Summary",
        ]])
    )
    c = _client(monkeypatch, tmp_path, engine=engine)
    sid = c.post("/api/study").json()["study_id"]
    src = make_still_dicom(tmp_path / "a.dcm")
    with src.open("rb") as f:
        c.post(f"/api/study/{sid}/upload", files={"file": ("a", f, "application/octet-stream")})

    tid = c.post(f"/api/study/{sid}/task/report", json={}).json()["task_id"]
    import time; time.sleep(0.2)

    r = c.get(f"/api/study/{sid}/report/export?format=pdf")
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
    r = c.get(f"/api/study/{sid}/report/export?format=docx")
    assert r.status_code == 200
    assert r.content[:2] == b"PK"
```

- [ ] **Step 2: Extend `app/routers/report_io.py`**

```python
# append to app/routers/report_io.py
from fastapi import Query
from fastapi.responses import FileResponse

from app.services.export import render_report_pdf, render_report_docx


@router.get("/api/study/{sid}/report/export")
def export_report(sid: str, format: str = Query(..., pattern="^(pdf|docx)$")):
    import json
    s = _store()
    p = s.result_path(sid, "report.json")
    if not p.exists():
        raise HTTPException(status_code=404, detail="report not generated yet")
    data = json.loads(p.read_text())

    out_name = f"report.{format}"
    out_path = s.export_path(sid, out_name)
    if format == "pdf":
        render_report_pdf(data, out_path)
        media = "application/pdf"
    else:
        render_report_docx(data, out_path)
        media = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    return FileResponse(str(out_path), media_type=media, filename=out_name)
```

- [ ] **Step 3: Pass + commit**

```bash
ECHOCHAT_SKIP_MODEL=1 pytest tests/test_tasks_router.py::test_export_pdf -v
git add app/routers/report_io.py tests/test_tasks_router.py
git commit -m "Phase 6: GET /api/study/<sid>/report/export?format=pdf|docx"
```

**Milestone Phase 6:** Full server-side flow works. `pytest` all green; report exports downloadable.

---

# Phase 7 — Deploy + smoke

### Task 7.1: `scripts/run_dev.sh`

**Files:** Create `scripts/run_dev.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
# scripts/run_dev.sh — local hot-reload dev server.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ECHOCHAT_SKIP_MODEL="${ECHOCHAT_SKIP_MODEL:-1}"  # dev by default skips heavy load
uvicorn app.main:app --reload --host "${ECHOCHAT_HOST:-127.0.0.1}" \
  --port "${ECHOCHAT_PORT:-12345}"
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x scripts/run_dev.sh
git add scripts/run_dev.sh
git commit -m "Phase 7: run_dev.sh with hot-reload and ECHOCHAT_SKIP_MODEL default on"
```

### Task 7.2: `scripts/run_prod.sh`

**Files:** Create `scripts/run_prod.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
# scripts/run_prod.sh — start the server under tmux session 'echochat-demo'.
# Run from /nfs/usrhome2/EchoChatDemo on eez195.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

SESSION="echochat-demo"
LOG_DIR="./logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/service.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-20971520}"
export VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-13107}"
export FPS="${FPS:-1}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Stop it first with: tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -dA -s "$SESSION" \
  "exec gunicorn -k uvicorn.workers.UvicornWorker \
       -b ${ECHOCHAT_HOST:-0.0.0.0}:${ECHOCHAT_PORT:-12345} \
       --workers 1 --timeout 600 \
       app.main:app 2>&1 | tee -a '$LOG_FILE'"

echo "Started tmux session '$SESSION'. Tail log: tail -f $LOG_FILE"
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x scripts/run_prod.sh
git add scripts/run_prod.sh
git commit -m "Phase 7: run_prod.sh launches gunicorn under tmux with logs"
```

### Task 7.3: `scripts/deploy.sh`

**Files:** Create `scripts/deploy.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
# scripts/deploy.sh — sync local repo to the deployment target.
# Expects SSH config or explicit host. Usage: bash scripts/deploy.sh [user@host]
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE="${1:-jguoaz@eez195.ece.ust.hk}"
REMOTE_PATH="/nfs/usrhome2/EchoChatDemo"

rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.ruff_cache' \
  --exclude 'node_modules' \
  --exclude 'data' \
  --exclude 'logs' \
  --exclude '*.log' \
  --exclude 'venv' \
  --exclude '.venv' \
  ./ "${REMOTE}:${REMOTE_PATH}/"

echo "Deployed to ${REMOTE}:${REMOTE_PATH}"
echo "On the server, run:   bash scripts/run_prod.sh"
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x scripts/deploy.sh
git add scripts/deploy.sh
git commit -m "Phase 7: deploy.sh rsyncs to /nfs/usrhome2/EchoChatDemo"
```

### Task 7.4: `scripts/smoke.sh`

**Files:** Create `scripts/smoke.sh`

- [ ] **Step 1: Write**

```bash
#!/usr/bin/env bash
# scripts/smoke.sh — manual end-to-end smoke test. Run AFTER server is up.
# Requires: curl, jq, at least one real DICOM file path as $1.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-12345}"
BASE="http://${HOST}:${PORT}"
DICOM="${1:?usage: $0 /path/to/file.dcm}"
PASSWORD="${SHARED_PASSWORD:-echochat}"

COOKIE="$(mktemp)"
trap 'rm -f "$COOKIE"' EXIT

echo "1) Login"
curl -s -c "$COOKIE" -d "password=${PASSWORD}" -o /dev/null -w "%{http_code}\n" \
  "${BASE}/login"

echo "2) Create study"
SID=$(curl -s -b "$COOKIE" -X POST "${BASE}/api/study" | jq -r .study_id)
echo "   study_id=$SID"

echo "3) Upload DICOM"
curl -s -b "$COOKIE" -F "file=@${DICOM}" "${BASE}/api/study/${SID}/upload" | jq .

echo "4) Process (SSE, first 40 events)"
curl -s -b "$COOKIE" -N "${BASE}/api/study/${SID}/process" | head -n 40

echo "5) Study state"
curl -s -b "$COOKIE" "${BASE}/api/study/${SID}" | jq '{clips: .clips[].view, tasks: .tasks}'

echo "6) Start report"
TID=$(curl -s -b "$COOKIE" -X POST -H 'Content-Type: application/json' \
  -d '{}' "${BASE}/api/study/${SID}/task/report" | jq -r .task_id)
echo "   task_id=$TID"

echo "7) Stream task (first 60 events)"
curl -s -b "$COOKIE" -N "${BASE}/api/task/${TID}/stream" | head -n 60

echo "8) Final result"
curl -s -b "$COOKIE" "${BASE}/api/task/${TID}" | jq '.status, .sections[0]'

echo "9) Export PDF"
curl -s -b "$COOKIE" -o /tmp/report.pdf "${BASE}/api/study/${SID}/report/export?format=pdf"
echo "   saved /tmp/report.pdf ($(stat -c %s /tmp/report.pdf) bytes)"
```

- [ ] **Step 2: chmod + commit**

```bash
chmod +x scripts/smoke.sh
git add scripts/smoke.sh
git commit -m "Phase 7: smoke.sh end-to-end curl script"
```

### Task 7.5: README final polish + push to GitHub

**Files:** Update `README.md`

- [ ] **Step 1: Rewrite README**

```markdown
# EchoChat Demo

Web demo showcasing four echocardiography AI tasks — **Report generation**, **Measurement**, **Disease diagnosis**, **Visual question answering** — on uploaded DICOM studies.

- Product spec: `docs/target.md`
- Design: `docs/superpowers/specs/2026-04-18-echochat-demo-design.md`
- Implementation plan: `docs/superpowers/plans/2026-04-18-echochat-demo.md`

## Prerequisites

- Python 3.10+
- GPU with sufficient VRAM for `echochatv1.5` (used on `cuda:2` by default)
- View classifier service (EchoView38) reachable at `http://127.0.0.1:8995`
- NFS-writable `ECHOCHAT_DATA_DIR`

## Local dev

```bash
cp .env.example .env            # edit values as needed
conda env create -f environment.yml
conda activate echochat-demo
pytest                          # fast unit suite
bash scripts/run_dev.sh         # uvicorn --reload, model loading skipped
```

Open `http://127.0.0.1:12345/login`. The password comes from `SHARED_PASSWORD` (default: `echochat`).

## Production (on eez195)

```bash
bash scripts/deploy.sh                 # rsync to /nfs/usrhome2/EchoChatDemo
ssh jguoaz@eez195 'cd /nfs/usrhome2/EchoChatDemo && bash scripts/run_prod.sh'
```

The service listens on `0.0.0.0:12345`. The domain owner maps `echochat.micro-heart.com` → `127.0.0.1:12345` via Nginx.

## Smoke test

```bash
bash scripts/smoke.sh /path/to/example.dcm
```

## Repository layout

See the file tree in `docs/superpowers/plans/2026-04-18-echochat-demo.md` under **File Structure**.

## Adding a supported measurement / disease / view

1. Add the item to the relevant constants file (`constants/measurements.py`, `constants/diseases.py`, or `constants/view_labels.py`).
2. Update presets in `constants/presets.py` if it should appear in a quick-pick set.
3. Unit tests under `tests/test_constants.py` enforce integrity.
4. No frontend changes required — `/api/constants` serves the updated list and the workspace page picks it up on next load.

## Extending VQA example questions

Populate `VQA_EXAMPLES` in `constants/presets.py`. The dropdown auto-hides while the list is empty.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Phase 7: README with full local + prod + smoke instructions"
```

- [ ] **Step 3: Push to GitHub (confirm with user before executing)**

```bash
# Confirm the user has configured credentials first.
# Then:
git push -u origin main
```

If push fails on auth, stop and ask the user for a Personal Access Token or SSH key setup.

**Milestone Phase 7:** Code pushed to `JR-Guo/EchoChatDemo`. Service is deployable via `deploy.sh` + `run_prod.sh`. Smoke script confirms the full path through upload → classification → report → export.

---

## Spec coverage check

| Spec section | Tasks |
|---|---|
| §1 goals + §1 non-goals | Phase 0.1–0.8 scaffold; Phase 1 constants; all task code English-only |
| §2 architecture (FastAPI, single worker, cuda:2, cuda:3 classifier) | 0.6 config, 0.7 main, 2.5 engine, 3.5 process SSE |
| §3.1 home | 5.3 |
| §3.2 upload | 5.4, 2.2/2.3 DICOM pipeline, 3.4 POST upload, 3.5 process SSE |
| §3.3 workspace (study panel, 4 tabs) | 5.5, 4.6 task router, 4.2–4.5 orchestrators |
| §3.4 soft warnings | 3.6 `_recompute_availability`, 5.5 badges, no hard blocks anywhere |
| §4 visual tokens, Tailwind, Inter, lucide | 5.1 base template + CSS, 5.3–5.5 pages |
| §5 code layout | every phase creates files in the planned places |
| §6.1 pages | 5.2–5.5 |
| §6.2 study + upload API | 3.4, 3.5, 3.6 |
| §6.3 task API | 4.6 |
| §6.4 report edit/export | 4.7 + 6.2 |
| §6.5 /api/constants + /healthz | 3.3 meta router + 0.7 healthz |
| §6.6 SSE event payloads | 3.5, 4.2–4.5, 4.6 |
| §7 progress copy | 4.2–4.5 phases, 5.5 JS text |
| §7 errors + edge cases | 3.4 empty/non-DICOM, 4.6 503 on warm-up, 5.5 regenerate confirm |
| §8 dependencies | 0.3 requirements.txt |
| §9 deployment | 7.1–7.3 scripts |
| §10 testing strategy | every backend task ships with pytest; 2.6 slow marker; 7.4 smoke |
| §11 open items (VQA presets, view-task mapping) | 1.6 placeholder list + 5.5 empty-list hides dropdown |
| §12 traceability | this coverage table closes the loop |

**Type consistency check:** `ClipKind`, `TaskKind`, `TaskStatus`, `ProgressEvent.kind` strings are reused verbatim across backend and frontend; `view_group` vs `view_coarse_group` both exported from `constants/view_labels.py`; session cookie name `echochat_session` and result filenames `report.json` / `measurements.json` / `diseases.json` / `vqa.json` are written by the same module that reads them.

**No placeholders:** every step has complete code or an exact command. The only intentionally blank piece is `constants/presets.VQA_EXAMPLES = []`, which is documented in the README and the spec's Open Items.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-18-echochat-demo.md`. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent implements each task; I review between tasks; fast iteration, less context pollution in this session.

**2. Inline Execution** — execute tasks in this session using superpowers:executing-plans; batch execution with checkpoints for review.

**Which approach?**

