"""awvision 0.3.3: the crop-friendly b64 ask, the pinned resolver, the api_key_file bearer.

Every test patches urlopen, so a hidden network call fails instead of passing silently.
"""
import io
import json
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import awvision.vision as v  # noqa: E402
import pytest  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _reply(text="ok"):
    return _Resp(json.dumps({"choices": [{"message": {"content": text}}]}).encode())


@pytest.fixture
def clean(monkeypatch, tmp_path):
    for k in ("AWVISION_URL", "AWVISION_MODEL", "AWVISION_CA_BUNDLE", "AWVISION_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWVISION_CONFIG", str(tmp_path / "absent.json"))
    v._resolved.clear()
    v._pinned.clear()

    def _no_net(*a, **k):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(v, "urlopen", _no_net)
    yield
    v._pinned.clear()


def test_b64_payload_shape_and_timeout(clean, monkeypatch):
    seen = {}

    def fake(req, timeout=None, context=None):
        seen["timeout"] = timeout
        seen["body"] = json.loads(req.data.decode())
        seen["headers"] = dict(req.header_items())
        return _reply('{"role":"button"}')

    monkeypatch.setattr(v, "urlopen", fake)
    out = v.get_vision_response_b64("QUJD", "image/png", "what?", "http://h:1", "m",
                                    max_tokens=96, timeout=7.5)
    assert out == '{"role":"button"}'
    assert seen["timeout"] == 7.5
    body = seen["body"]
    assert body["model"] == "m" and body["max_tokens"] == 96
    parts = body["messages"][0]["content"]
    assert parts[0]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert parts[1]["text"] == "what?"
    assert not any(k.lower() == "authorization" for k in seen["headers"])


def test_bearer_comes_from_api_key_file(clean, monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("tok-123\n", encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"endpoint": "http://h:1", "api_key_file": str(key)}),
                   encoding="utf-8")
    monkeypatch.setenv("AWVISION_CONFIG", str(cfg))
    seen = {}

    def fake(req, timeout=None, context=None):
        seen["auth"] = req.get_header("Authorization")
        return _reply()

    monkeypatch.setattr(v, "urlopen", fake)
    v.get_vision_response_b64("QUJD", "image/png", "q", "http://h:1", "m")
    assert seen["auth"] == "Bearer tok-123"


def test_require_scheduler_refuses_mesh_fallback(clean, monkeypatch):
    probed = []

    def lm(ep, timeout=10.0):
        probed.append(ep)
        return {"gemma4-12b": True} if ep.startswith("http://") else None

    monkeypatch.setattr(v, "_list_models", lm)
    with pytest.raises(v.VisionUnavailable) as ei:
        v.resolve_pinned(require_scheduler=True)
    assert all(p.startswith("https://") for p in probed)
    assert "plain-http" in str(ei.value)


def test_explicit_endpoint_is_honoured(clean, monkeypatch):
    monkeypatch.setenv("AWVISION_URL", "http://127.0.0.1:9/")
    monkeypatch.setenv("AWVISION_MODEL", "vlm")
    assert v.resolve_pinned() == ("http://127.0.0.1:9", "vlm")


def test_eight_threads_probe_once(clean, monkeypatch):
    calls = []

    def lm(ep, timeout=10.0):
        calls.append(ep)
        time.sleep(0.05)
        return {"bonsai2-27b": True}

    monkeypatch.setattr(v, "_list_models", lm)
    out = []
    ts = [threading.Thread(target=lambda: out.append(v.resolve_pinned())) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1
    assert out == [(v._CANDIDATES[0], "bonsai2-27b")] * 8
