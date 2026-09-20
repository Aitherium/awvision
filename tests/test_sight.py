"""Sight: frame -> look -> sentence -> room. Nothing here opens a camera, a network
socket or the real status file; every I/O seam is injected."""

import io
import json
import types

import pytest
from awvision import sight


@pytest.fixture(autouse=True)
def _isolated_status(tmp_path, monkeypatch):
    monkeypatch.setenv("AWVISION_SIGHT_STATUS", str(tmp_path / "status.json"))
    monkeypatch.delenv(sight.KILL_ENV, raising=False)


def test_frame_diff_is_zero_for_identical_and_maximal_for_unknown():
    a = [10] * 1024
    assert sight.frame_diff(a, list(a)) == 0.0
    assert sight.frame_diff(a, [20] * 1024) == 10.0
    assert sight.frame_diff(None, a) == 255.0
    assert sight.frame_diff(a, [1, 2, 3]) == 255.0


def test_should_look_needs_change_and_respects_pacing():
    # a static scene never looks, however long it has been
    assert sight.should_look(0.5, 10_000, 6.5) is False
    # a change looks once the window has passed: max(10 s floor, 2 x 6.5 s latency) = 13 s
    assert sight.should_look(40.0, 12.9, 6.5) is False
    assert sight.should_look(40.0, 13.0, 6.5) is True
    # a fast model is still held to the floor
    assert sight.should_look(40.0, 9.0, 0.2) is False
    assert sight.should_look(40.0, 10.0, 0.2) is True


def test_speakable_is_one_short_line_of_whole_sentences():
    long = ("A red mug sits on a wooden desk beside a keyboard. " * 8).strip()
    out = sight.speakable("**Look:**\n`" + long + "`")
    assert len(out) <= sight.SPEAKABLE_LIMIT
    assert "\n" not in out and "`" not in out and "*" not in out
    assert out.endswith(".")
    assert sight.speakable("A cat.") == "A cat."


def test_the_event_carries_text_and_a_hash_never_pixels():
    ev = sight.sight_event("a mug", source_kind="screen", node_id="box", sha256="abc",
                           model="m", latency_ms=6500, changed=True, diff=12.345)
    assert ev["type"] == "sight_observed" and ev["room"] == "sight"
    assert ev["actor"]["kind"] == "service"
    assert ev["payload"]["frame_sha256"] == "abc" and ev["payload"]["diff"] == 12.35
    forbidden = {"image", "frame", "data", "base64", "pixels", "jpeg"}
    assert not (forbidden & set(ev["payload"])), "a sight event must never carry the frame"
    assert "pillar" not in ev, "the pillar is the vocabulary's to assign, not the producer's"


def test_say_event_is_a_short_agent_message_into_main():
    ev = sight.say_event("word " * 200)
    assert ev["type"] == "agent_message" and ev["room"] == "main"
    assert len(ev["payload"]["text"]) <= sight.SPEAKABLE_LIMIT


def test_publish_posts_to_the_room_with_the_bearer(monkeypatch):
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp(b'{"ok": true, "pillar": "context"}')

    env = {"AITHER_HARNESS_TOKEN": "tok", "AITHER_HARNESS_URL": "http://127.0.0.1:9/"}
    out = sight.publish({"type": "sight_observed"}, env=env, opener=opener)
    assert out == {"ok": True, "pillar": "context"}
    assert seen["url"] == "http://127.0.0.1:9/events" and seen["auth"] == "Bearer tok"
    assert seen["body"] == {"type": "sight_observed"}


def test_publish_without_a_token_is_none_not_a_pretend_success(monkeypatch, tmp_path):
    monkeypatch.setattr(sight.Path, "home", lambda: tmp_path)
    assert sight.publish({"type": "x"}, env={}) is None


def test_ffmpeg_argv_for_rtsp_and_for_a_device():
    rtsp = sight.ffmpeg_argv("rtsp://cam/1", "o.jpg")
    assert rtsp[rtsp.index("-i") + 1] == "rtsp://cam/1" and "-rtsp_transport" in rtsp
    assert rtsp[-1] == "o.jpg" and rtsp[rtsp.index("-frames:v") + 1] == "1"
    dev = sight.ffmpeg_argv("device:Front Cam", "o.jpg")
    assert any("Front Cam" in a for a in dev)


def test_source_kind_names_each_lane():
    assert sight.source_kind("screen") == "screen"
    assert sight.source_kind("device:x") == "webcam"
    assert sight.source_kind("rtsp://a") == "rtsp"
    assert sight.source_kind("photo.jpg") == "file"


def _args(**kw):
    base = dict(prompt=None, endpoint=None, model=None, publish=False, say=False,
                room="sight", node_id="box", json=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_observe_looks_publishes_and_says_and_keeps_no_frame(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"not really a jpeg")
    sent = []
    obs = sight.observe(
        str(img), _args(say=True),
        publish_fn=lambda ev: sent.append(ev) or {"ok": True},
        look_fn=lambda image, prompt, endpoint, model: ("a red mug on a desk", 6500),
    )
    assert obs["text"] == "a red mug on a desk" and obs["source_kind"] == "file"
    assert [e["type"] for e in sent] == ["sight_observed", "agent_message"]
    assert sent[0]["payload"]["frame_sha256"] == sight.sha256_file(str(img))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["last_publish_at"] and status["last_text"] == "a red mug on a desk"


def test_observe_that_cannot_publish_says_so_in_the_status(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    obs = sight.observe(str(img), _args(publish=True), publish_fn=lambda ev: None,
                        look_fn=lambda *a: ("a desk", 100))
    assert obs["published"] is None
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert "last_publish_at" not in status and "not published" in status["last_error"]


def test_an_empty_answer_is_a_failure(tmp_path, monkeypatch):
    import awvision.vision as vision

    monkeypatch.setattr(vision, "get_vision_response", lambda *a: "   ")
    with pytest.raises(RuntimeError):
        sight.look("x.jpg", "what")


def test_the_kill_switch_refuses_to_watch(monkeypatch, tmp_path):
    monkeypatch.setenv(sight.KILL_ENV, "0")
    assert sight.cmd_watch(_args(source="screen", every=1)) == 3
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["refused"] == "kill_switch"


def test_watch_looks_on_change_only_and_stays_quiet_on_a_static_scene(monkeypatch):
    # frames: static, static, static, CHANGED, changed(static again)
    frames = [[10] * 1024, [10] * 1024, [10] * 1024, [200] * 1024, [200] * 1024, [200] * 1024]
    state = {"i": 0, "now": 1000.0}
    monkeypatch.setattr(sight, "frame_signature", lambda path: frames[min(state["i"], 5)])

    def capture_fn(source, out):
        return out

    def observe_fn(source, args, prev_sig=None):
        return {"text": f"look {state['i']}", "latency_ms": 1000, "diff": 0.0,
                "signature": frames[min(state["i"], 5)], "sha256": "s",
                "source_kind": "screen", "published": None, "said": None}

    def sleep(sec):
        state["i"] += 1
        state["now"] += 20.0  # past the pacing floor every tick, so CHANGE decides

    rc = sight.cmd_watch(_args(source="screen", every=2, threshold=6.0, max_ticks=6),
                         sleep=sleep, clock=lambda: state["now"],
                         observe_fn=observe_fn, capture_fn=capture_fn)
    assert rc == 0
    # exactly two looks: the first frame, and the one change. Four static ticks = zero looks.
    # (the counter lives in the status file, written at stop)


def test_watch_counts_two_looks_for_first_frame_plus_one_change(monkeypatch, tmp_path):
    frames = [[10] * 1024] * 3 + [[200] * 1024] * 3
    state = {"i": 0, "now": 1000.0}
    monkeypatch.setattr(sight, "frame_signature", lambda path: frames[min(state["i"], 5)])
    looks = []

    def observe_fn(source, args, prev_sig=None):
        looks.append(state["i"])
        return {"text": "x", "latency_ms": 1000, "diff": 0.0,
                "signature": frames[min(state["i"], 5)], "sha256": "s",
                "source_kind": "screen", "published": None, "said": None}

    def sleep(sec):
        state["i"] += 1
        state["now"] += 20.0

    sight.cmd_watch(_args(source="screen", every=2, threshold=6.0, max_ticks=6),
                    sleep=sleep, clock=lambda: state["now"],
                    observe_fn=observe_fn, capture_fn=lambda s, o: o)
    assert looks == [0, 3]
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["looks"] == 2 and status["watching"] is None


def test_watch_without_a_source_is_a_usage_error():
    assert sight.cmd_watch(_args(source="")) == 2


# ── keeping frames: never by default, one chokepoint, a copy to Strata only ──

class FakeStrata:
    """The injected client. Records writes; lists and deletes what it holds."""

    def __init__(self, *, unreachable=False, vault=False, refuse_writes=False):
        self.writes, self.objects, self.deleted = [], {}, []
        self.unreachable, self.vault, self.refuse_writes = unreachable, vault, refuse_writes
        self.vault_probes = 0

    def write(self, path, data, tier):
        if self.refuse_writes:
            raise sight.StrataError("POST /strata/write: HTTP 503")
        self.writes.append((path, tier, len(data)))
        self.objects[path] = data
        return True

    def list(self, prefix):
        if self.unreachable:
            raise sight.StrataError("GET /strata/list: connection refused")
        return [{"path": p, "modified": "2026-09-19T00:00:00"} for p in self.objects
                if p.startswith(prefix)]

    def delete(self, path):
        self.objects.pop(path)
        self.deleted.append(path)
        return True

    def vault_unlocked(self):
        self.vault_probes += 1
        return self.vault


def test_keep_policy_is_off_by_default_for_every_source_kind():
    for kind in ("file", "webcam", "rtsp", "screen"):
        allowed, dest, reason = sight.keep_policy(object(), kind, env={})
        assert allowed is False and dest == "" and "--keep-frames" in reason


def test_keep_policy_honours_the_host_refusal_over_the_flag():
    args = types.SimpleNamespace(keep_frames=True)
    for value in ("0", "off", "false", "no"):
        allowed, _, reason = sight.keep_policy(args, "file", env={sight.KEEP_ENV: value})
        assert allowed is False and sight.KEEP_ENV in reason
    allowed, dest, _ = sight.keep_policy(args, "file", env={sight.KEEP_ENV: "1"}, sha256="abc")
    assert allowed is True and dest == "aither://cache/vision/sight/abc.jpg"


def test_keep_policy_sends_a_live_source_only_to_an_unlocked_vault():
    args = types.SimpleNamespace(keep_frames=True)
    probes = []
    for kind in ("webcam", "rtsp", "screen"):
        # nobody asked the vault -> locked -> refused
        assert sight.keep_policy(args, kind, env={}, sha256="s")[0] is False
        assert sight.keep_policy(args, kind, env={}, sha256="s",
                                 vault_unlocked=lambda: False)[0] is False
        allowed, dest, _ = sight.keep_policy(args, kind, env={}, sha256="s",
                                             vault_unlocked=lambda: probes.append(1) or True)
        assert allowed is True and dest == "aither://lockbox/private/vision/sight/s.jpg"
    # a file source never asks the vault, and no destination is ever a local path
    allowed, dest, _ = sight.keep_policy(args, "file", env={}, sha256="s",
                                         vault_unlocked=lambda: probes.append(1) or True)
    assert allowed and dest.startswith("aither://") and len(probes) == 3


def _observe_with(tmp_path, monkeypatch, fake, **extra):
    """Run observe() with a fake look and the fake client; return (obs, temp_dirs_made)."""
    img = tmp_path / "x.jpg"
    img.write_bytes(b"not really a jpeg")
    made = []
    real_mkdtemp = sight.tempfile.mkdtemp

    def mkdtemp(**kw):
        d = real_mkdtemp(**kw)
        made.append(d)
        return d

    monkeypatch.setattr(sight.tempfile, "mkdtemp", mkdtemp)
    obs = sight.observe(str(img), _args(publish=True, **extra),
                        publish_fn=lambda ev: {"ok": True},
                        look_fn=lambda image, prompt, endpoint, model: ("a red mug", 6500),
                        strata_client=fake)
    return obs, made


def test_observe_keeps_no_frame_without_the_flag_and_copies_exactly_once_with_it(
        tmp_path, monkeypatch):
    import os

    fake = FakeStrata()
    obs, made = _observe_with(tmp_path, monkeypatch, fake)
    assert obs["kept"] is False and obs["kept_at"] is None and fake.writes == []
    assert made and not any(os.path.exists(d) for d in made), "the temp frame must be gone"

    fake = FakeStrata()
    obs, made = _observe_with(tmp_path, monkeypatch, fake, keep_frames=True)
    sha = sight.sha256_file(str(tmp_path / "x.jpg"))
    assert obs["kept"] is True and obs["kept_at"] == f"aither://cache/vision/sight/{sha}.jpg"
    assert obs["kept_ttl_s"] == sight.KEEP_TTL_S
    assert fake.writes == [(obs["kept_at"], "cache", len(b"not really a jpeg"))]
    assert made and not any(os.path.exists(d) for d in made), "a keep is a COPY; the temp dies"
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["keeping"] == obs["kept_at"] and status["kept_count"] == 1
    assert status["kept_ttl_s"] == sight.KEEP_TTL_S


def test_a_failed_copy_is_a_reason_and_the_frame_still_dies(tmp_path, monkeypatch):
    import os

    fake = FakeStrata(refuse_writes=True)
    obs, made = _observe_with(tmp_path, monkeypatch, fake, keep_frames=True)
    assert obs["kept"] is False and "copy to Strata failed" in obs["keep_reason"]
    assert not any(os.path.exists(d) for d in made)
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["keeping"] is None and "failed" in status["keep_refused"]


def test_the_event_says_kept_without_carrying_the_frame():
    ev = sight.sight_event("a mug", source_kind="file", node_id="box", sha256="abc",
                           model="m", latency_ms=1, changed=True, diff=0.0)
    assert ev["payload"]["kept"] is False and ev["payload"]["kept_ttl_s"] == 0
    kept = sight.sight_event("a mug", source_kind="file", node_id="box", sha256="abc",
                             model="m", latency_ms=1, changed=True, diff=0.0,
                             kept=True, kept_ttl_s=sight.KEEP_TTL_S)
    assert kept["payload"]["kept"] is True and kept["payload"]["kept_ttl_s"] == 86400
    assert not (sight.FORBIDDEN_PAYLOAD_KEYS & set(kept["payload"]))


def test_watch_banner_and_status_say_keeping_only_when_a_keep_is_on(tmp_path, monkeypatch,
                                                                    capsys):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"frame")
    look = lambda image, prompt, endpoint, model: ("a desk", 100)  # noqa: E731

    def observe_fn(source, args, prev_sig=None, **kw):
        return sight.observe(source, args, prev_sig=prev_sig, look_fn=look,
                             publish_fn=lambda ev: None, **kw)

    fake = FakeStrata()
    rc = sight.cmd_watch(_args(source=str(img), every=1, max_ticks=1),
                         sleep=lambda s: None, observe_fn=observe_fn, strata_client=fake)
    assert rc == 0 and fake.writes == []
    assert "KEEPING" not in capsys.readouterr().err
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["keeping"] is None and status["kept_count"] == 0

    rc = sight.cmd_watch(_args(source=str(img), every=1, max_ticks=1, keep_frames=True),
                         sleep=lambda s: None, observe_fn=observe_fn, strata_client=fake)
    err = capsys.readouterr().err
    assert rc == 0 and len(fake.writes) == 1
    assert "ON AIR, KEEPING FRAMES -> aither://cache/vision/sight/" in err
    assert "expires in 24h" in err
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["keeping"] == fake.writes[0][0] and status["kept_count"] == 1


def test_watch_with_the_flag_on_a_live_source_says_why_it_is_not_keeping(capsys, tmp_path):
    fake = FakeStrata(vault=False)
    rc = sight.cmd_watch(_args(source="screen", every=1, max_ticks=1, keep_frames=True),
                         sleep=lambda s: None, capture_fn=lambda s, o: o,
                         observe_fn=lambda *a, **k: {"text": "x", "latency_ms": 1, "diff": 0.0,
                                                    "signature": None, "sha256": "s",
                                                    "source_kind": "screen", "published": None,
                                                    "said": None},
                         strata_client=fake)
    err = capsys.readouterr().err
    assert rc == 0 and "KEEPING" not in err and "vault is locked" in err
    assert fake.vault_probes == 1 and fake.writes == []
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["keeping"] is None and "vault is locked" in status["keep_refused"]


def test_forget_purges_then_finds_nothing_and_cannot_pretend_when_strata_is_down(
        tmp_path, monkeypatch, capsys):
    fake = FakeStrata()
    _observe_with(tmp_path, monkeypatch, fake, keep_frames=True)
    assert len(fake.objects) == 1

    assert sight.cmd_forget(types.SimpleNamespace(all=True, older_than=None),
                            strata_client=fake) == 0
    assert capsys.readouterr().out.strip() == "forgot 1 frame"
    assert fake.deleted == [fake.writes[0][0]] and fake.objects == {}
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["kept_count"] == 0 and status["keeping"] is None

    assert sight.cmd_forget(types.SimpleNamespace(all=True, older_than=None),
                            strata_client=fake) == 0
    assert capsys.readouterr().out.strip() == "forgot 0 frames"

    down = FakeStrata(unreachable=True)
    assert sight.cmd_forget(types.SimpleNamespace(all=True, older_than=None),
                            strata_client=down) == 2, "could not look is not nothing to purge"
    assert sight.cmd_forget(types.SimpleNamespace(all=False, older_than=None),
                            strata_client=fake) == 2


def test_forget_older_than_deletes_by_age_and_never_on_a_missing_stamp(capsys):
    fake = FakeStrata()
    old, new = sight.KEEP_PREFIX + "old.jpg", sight.KEEP_PREFIX + "new.jpg"
    fake.objects = {old: b"", new: b""}
    listing = [{"path": old, "modified": "2026-09-18T00:00:00"},
               {"path": new, "modified": "2026-09-20T05:00:00"},
               {"path": sight.KEEP_PREFIX + "unstamped.jpg"}]
    fake.list = lambda prefix: listing if prefix == sight.KEEP_PREFIX else []
    from datetime import datetime
    now = datetime.fromisoformat("2026-09-20T06:00:00").timestamp()
    rc = sight.cmd_forget(types.SimpleNamespace(all=False, older_than=24.0),
                          strata_client=fake, clock=lambda: now)
    assert rc == 0 and fake.deleted == ["aither://cache/vision/sight/old.jpg"]
    assert capsys.readouterr().out.strip() == "forgot 1 frame (1 without a timestamp left alone)"


def test_the_stdlib_strata_client_speaks_the_canonical_wire_shape():
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=0):
        calls.append((req.get_method(), req.full_url, req.get_header("X-internal-key"),
                      json.loads(req.data.decode("utf-8")) if req.data else None))
        if "/strata/list" in req.full_url:
            return _Resp(b'{"entries": [{"name": "a.jpg", "type": "file", '
                         b'"virtual_path": "aither:/cache/vision/sight/a.jpg", '
                         b'"modified": "2026-09-19T00:00:00"}, '
                         b'{"name": "d", "type": "directory"}]}')
        return _Resp(b'{"success": true}')

    env = {"AITHER_STRATA_URL": "http://127.0.0.1:9/", "AITHER_INTERNAL_SECRET": "k"}
    client = sight.StrataClient(env=env, opener=opener)
    assert client.write("aither://cache/vision/sight/a.jpg", b"\xff\xd8", "cache") is True
    method, url, key, body = calls[0]
    assert (method, url, key) == ("POST", "http://127.0.0.1:9/strata/write", "k")
    assert body["path"] == "aither://cache/vision/sight/a.jpg" and body["tier"] == "cache"
    assert body["content"] == "/9g="  # base64, the shape /strata/write decodes
    files = client.list(sight.KEEP_PREFIX)
    assert files == [{"path": "aither://cache/vision/sight/a.jpg",
                      "modified": "2026-09-19T00:00:00"}]
    assert calls[1][0] == "GET"
    assert "path=aither%3A%2F%2Fcache%2Fvision%2Fsight%2F" in calls[1][1]
    assert client.delete("aither://cache/vision/sight/a.jpg") is True
    assert calls[2][0] == "DELETE"
    assert calls[2][1].startswith("http://127.0.0.1:9/strata/delete?path=")


def test_the_stdlib_strata_client_reports_a_dead_strata_never_an_empty_answer():
    from urllib.error import URLError

    def opener(req, timeout=0):
        raise URLError("connection refused")

    client = sight.StrataClient(env={"AITHER_STRATA_URL": "http://127.0.0.1:9"}, opener=opener)
    with pytest.raises(sight.StrataError):
        client.list(sight.KEEP_PREFIX)
    assert client.vault_unlocked() is False, "an unanswered vault is a locked vault"
