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
