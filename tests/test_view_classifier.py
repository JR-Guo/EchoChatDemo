import httpx
import pytest
import respx

from app.services.view_classifier import ViewClassifier, ClassifyOutcome


@pytest.fixture
def vc():
    return ViewClassifier(base_url="http://vc.test", api_key="test-key", timeout=0.5)


@respx.mock
async def test_classify_known_view(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)

    respx.post("http://vc.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "classification": {
                    "original_view_name": "Apical 4C 2D",
                    "detected_view_type": "echo_a4c",
                    "confidence": 0.91,
                }
            },
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
    respx.post("http://vc.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"classification": {"original_view_name": "Unknown Variant", "confidence": 0.2}},
        )
    )
    out = await vc.classify(fake)
    assert out.view is None
    assert out.confidence == pytest.approx(0.2)


@respx.mock
async def test_classify_from_choices_content(vc, tmp_path):
    # Service omits the top-level `classification` block; parse from the
    # assistant message JSON instead.
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    respx.post("http://vc.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"original_view_name": "Parasternal Long Axis 2D", "confidence": 0.7}'}}
                ]
            },
        )
    )
    out = await vc.classify(fake)
    assert out.view == "Parasternal Long Axis 2D"
    assert out.confidence == pytest.approx(0.7)


@respx.mock
async def test_classify_timeout(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    respx.post("http://vc.test/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("slow")
    )
    out = await vc.classify(fake)
    assert out.view is None
    assert out.error == "timeout"


@respx.mock
async def test_classify_http_error(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    respx.post("http://vc.test/v1/chat/completions").mock(
        return_value=httpx.Response(500)
    )
    out = await vc.classify(fake)
    assert out.view is None
    assert out.error and "500" in out.error


@respx.mock
async def test_classify_sends_bearer_header(vc, tmp_path):
    fake = tmp_path / "x.dcm"
    fake.write_bytes(b"0" * 300)
    route = respx.post("http://vc.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"classification": {"original_view_name": "Apical 4C 2D", "confidence": 0.5}}
        )
    )
    await vc.classify(fake)
    assert route.called
    sent = route.calls.last.request
    assert sent.headers.get("authorization") == "Bearer test-key"
