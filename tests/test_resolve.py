"""Endpoint/model auto-resolution: works without AWVISION_URL on a machine where the mesh
default is unreachable, and never names a vision model the endpoint has parked."""
import awvision.vision as v


def _reset(monkeypatch, listing):
    monkeypatch.delenv("AWVISION_URL", raising=False)
    monkeypatch.delenv("AWVISION_MODEL", raising=False)
    v._resolved.clear()
    monkeypatch.setattr(v, "_list_models", lambda ep, timeout=10.0: listing.get(ep))


def test_first_reachable_candidate_wins(monkeypatch):
    _reset(monkeypatch, {v._CANDIDATES[0]: {"gemma4-12b": True}})
    assert v.resolve_endpoint() == v._CANDIDATES[0]


def test_unreachable_everything_keeps_the_last_candidate(monkeypatch):
    _reset(monkeypatch, {})
    assert v.resolve_endpoint() == v._CANDIDATES[-1]


def test_parked_gemma_falls_to_an_available_vision_model(monkeypatch):
    _reset(monkeypatch, {v._CANDIDATES[0]: {"gemma4-12b": False, "bonsai2-27b": True}})
    ep = v.resolve_endpoint()
    assert v.resolve_model(ep) == "bonsai2-27b"


def test_env_overrides_everything(monkeypatch):
    _reset(monkeypatch, {v._CANDIDATES[0]: {"bonsai2-27b": True}})
    monkeypatch.setenv("AWVISION_URL", "http://x:1")
    monkeypatch.setenv("AWVISION_MODEL", "m")
    assert v.resolve_endpoint() == "http://x:1"
    assert v.resolve_model("http://x:1") == "m"
