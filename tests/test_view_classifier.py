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
    assert out.view is None
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
