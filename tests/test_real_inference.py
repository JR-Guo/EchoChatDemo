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
