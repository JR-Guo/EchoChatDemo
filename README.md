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
