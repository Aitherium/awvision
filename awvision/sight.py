"""Sight: a frame -> one look -> a sentence -> (optionally) the room.

``awvision ask`` answers a question about an image you already have. ``see`` and
``watch`` are the other half: take a frame from a SOURCE you name (a file, the screen,
an RTSP camera, a capture device), ask the vision model what is there, and hand the
answer on as one ``sight_observed`` event -- text and a frame hash, never pixels -- so
anything listening to the room (a desk avatar that speaks, a shell that shows it, an
agent that can be told about it) learns what was seen.

The rules this module is built around:

* **The source is named by the caller, always.** Nothing here discovers a camera. The
  command line that starts a watch IS the consent record; ``AWVISION_SIGHT=0`` refuses
  to start one at all.
* **A model cannot look at every frame.** ``watch`` grabs cheaply and LOOKS only when
  the picture changed (32x32 grayscale mean-abs diff against the last frame that was
  actually sent) and never more often than ``max(floor, 2 x the last look's latency)``.
  Measured 2026-09-19 against a 12B vision model: 17 s cold, 6.5 s warm -- an unpaced
  loop would queue looks forever.
* **No frame is kept.** A captured frame lives in a temp file for the length of one
  look and is deleted; the event carries its sha256 so two observations of one frame
  can be told apart without the frame existing anywhere.
* **Silence is not a pass.** ``~/.aither/sight/status.json`` moves ``last_publish_at``
  only on a real publish, so a watcher that captures happily and publishes nothing reads
  as exactly that.

Standard library only. Pillow is OPTIONAL: it is needed for ``--screen`` and for the
perceptual frame diff; without it the diff degrades to "bytes differ", said out loud.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

KILL_ENV = "AWVISION_SIGHT"
DEFAULT_PROMPT = (
    "In one or two plain sentences, say what is in this picture. "
    "Name people, objects and any readable text. No preamble."
)
SPEAKABLE_LIMIT = 220  # the desk stage voices an agent_message only up to this length
DIFF_THRESHOLD = 6.0   # mean abs grayscale difference (0..255) that counts as "changed"
LOOK_FLOOR_S = 10.0


# ── pure helpers ─────────────────────────────────────────────────────────────

def kill_switch(env: Optional[dict] = None) -> bool:
    """True when sight is switched off for this host (``AWVISION_SIGHT=0``)."""
    env = os.environ if env is None else env
    return str(env.get(KILL_ENV, "")).strip().lower() in ("0", "off", "false", "no")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_signature(path: str) -> Optional[list]:
    """32x32 grayscale pixels, or None when Pillow is unavailable / the file is unreadable."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            return list(img.convert("L").resize((32, 32)).getdata())
    except (OSError, ValueError):
        return None


def frame_diff(a: Optional[list], b: Optional[list]) -> float:
    """Mean absolute grayscale difference, 0..255. Unknown vs anything = maximal."""
    if a is None or b is None or len(a) != len(b) or not a:
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b)) / float(len(a))


def should_look(diff: float, since_last_look_s: float, last_latency_s: float, *,
                threshold: float = DIFF_THRESHOLD, floor_s: float = LOOK_FLOOR_S) -> bool:
    """Look only when the picture changed AND the pacing window has passed."""
    if diff < threshold:
        return False
    return since_last_look_s >= max(floor_s, 2.0 * max(0.0, last_latency_s))


def speakable(text: str, limit: int = SPEAKABLE_LIMIT) -> str:
    """One line a voice can say: no newlines, no code marks, whole sentences up to ``limit``."""
    flat = re.sub(r"[`*_#>{}\[\]]", "", " ".join((text or "").split()))
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if stop >= 40:
        return cut[: stop + 1]
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "."


def sight_event(text: str, *, source_kind: str, node_id: str, sha256: str, model: str,
                latency_ms: int, changed: bool, diff: float, room: str = "sight") -> dict:
    """The room event. TEXT and a HASH only -- a payload key that could carry pixels
    (image/frame/data/base64) is never set here, and the test pins that."""
    return {
        "room": room,
        "type": "sight_observed",
        "actor": {"kind": "service", "id": "awvision", "name": "awvision"},
        "payload": {
            "text": text,
            "source_kind": source_kind,
            "node_id": node_id,
            "frame_sha256": sha256,
            "model": model,
            "latency_ms": int(latency_ms),
            "changed": bool(changed),
            "diff": round(float(diff), 2),
        },
    }


def say_event(text: str, room: str = "main") -> dict:
    """The short line a desk avatar voices. An ``agent_message`` from a service actor."""
    return {
        "room": room,
        "type": "agent_message",
        "actor": {"kind": "service", "id": "awvision", "name": "awvision"},
        "payload": {"text": speakable(text), "origin": "sight"},
    }


# ── the room ─────────────────────────────────────────────────────────────────

def harness_url(env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    return str(env.get("AITHER_HARNESS_URL") or "http://127.0.0.1:8362").rstrip("/")


def harness_token(env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    tok = str(env.get("AITHER_HARNESS_TOKEN") or "").strip()
    if tok:
        return tok
    try:
        return (Path.home() / ".aither" / "harness_token").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def publish(event: dict, env: Optional[dict] = None,
            opener: Callable = urlopen, timeout: float = 60.0) -> Optional[dict]:
    """POST one event to the room. None when there is no room to publish to (no token)
    or it refused -- the caller says so; it never pretends."""
    token = harness_token(env)
    if not token:
        return None
    req = Request(
        harness_url(env) + "/events",
        data=json.dumps(event).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except (HTTPError, URLError, OSError, ValueError):
        return None


def status_path() -> Path:
    return Path(os.environ.get("AWVISION_SIGHT_STATUS")
                or (Path.home() / ".aither" / "sight" / "status.json"))


def write_status(**fields) -> None:
    """Best-effort status file. ``last_publish_at`` is only ever passed on a real publish."""
    path = status_path()
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        current = {}
    current.update(fields)
    current["at"] = time.time()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


# ── capture ──────────────────────────────────────────────────────────────────

class CaptureError(RuntimeError):
    """A source could not produce a frame. The message names the fix."""


def capture_screen(out: str) -> str:
    try:
        from PIL import ImageGrab  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CaptureError("--screen needs Pillow: pip install pillow") from exc
    try:
        ImageGrab.grab().convert("RGB").save(out, "JPEG", quality=80)
    except (OSError, ValueError) as exc:
        raise CaptureError(f"could not grab the screen: {exc}") from exc
    return out


def ffmpeg_argv(source: str, out: str) -> list:
    """One frame from an RTSP URL or a capture device (``device:<name>``)."""
    if source.startswith("device:"):
        name = source[len("device:"):]
        if sys.platform == "win32":
            head = ["-f", "dshow", "-i", f"video={name}"]
        elif sys.platform == "darwin":
            head = ["-f", "avfoundation", "-i", name]
        else:
            head = ["-f", "v4l2", "-i", name]
    else:
        head = ["-rtsp_transport", "tcp", "-i", source]
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *head,
            "-frames:v", "1", "-q:v", "4", out]


def capture_ffmpeg(source: str, out: str, timeout: float = 20.0) -> str:
    if not shutil.which("ffmpeg"):
        raise CaptureError("ffmpeg is required for camera sources and is not on PATH")
    try:
        proc = subprocess.run(ffmpeg_argv(source, out), capture_output=True, timeout=timeout,
                              text=True, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureError(f"ffmpeg could not read {source}: {exc}") from exc
    if proc.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no frame produced"]
        raise CaptureError(f"no frame from {source}: {tail[0]}")
    return out


def source_kind(source: str) -> str:
    if source == "screen":
        return "screen"
    if source.startswith("device:"):
        return "webcam"
    if source.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return "rtsp"
    return "file"


def capture(source: str, out: str) -> str:
    """One frame from ``source`` into ``out`` (or the file itself for a file source)."""
    kind = source_kind(source)
    if kind == "screen":
        return capture_screen(out)
    if kind in ("webcam", "rtsp"):
        return capture_ffmpeg(source, out)
    if not os.path.exists(source):
        raise CaptureError(f"no such image: {source}")
    return source


# ── look ─────────────────────────────────────────────────────────────────────

def look(image: str, prompt: str, endpoint=None, model=None) -> tuple:
    """``(text, latency_ms)``. An EMPTY answer is a failure, never a quiet success."""
    from awvision.vision import get_vision_response

    t0 = time.time()
    text = (get_vision_response(image, prompt, endpoint, model) or "").strip()
    latency_ms = int((time.time() - t0) * 1000)
    if not text:
        raise RuntimeError("the vision model answered with nothing")
    return text, latency_ms


def _node_id(args) -> str:
    return (getattr(args, "node_id", "") or os.environ.get("AITHER_NODE_ID")
            or os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "this-host")


def observe(source: str, args, *, prev_sig=None, publish_fn: Callable = publish,
            look_fn: Callable = look, capture_fn: Callable = capture) -> dict:
    """Capture -> look -> (publish). Returns the observation; the temp frame is deleted."""
    tmp_dir = tempfile.mkdtemp(prefix="awvision-sight-")
    try:
        frame = capture_fn(source, os.path.join(tmp_dir, "frame.jpg"))
        sig = frame_signature(frame)
        diff = frame_diff(prev_sig, sig) if prev_sig is not None else 255.0
        text, latency_ms = look_fn(frame, getattr(args, "prompt", None) or DEFAULT_PROMPT,
                                   getattr(args, "endpoint", None), getattr(args, "model", None))
        obs = {
            "text": text, "latency_ms": latency_ms, "diff": diff, "signature": sig,
            "sha256": sha256_file(frame), "source_kind": source_kind(source),
            "published": None, "said": None,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    model = getattr(args, "model", None) or os.environ.get("AWVISION_MODEL", "gemma4-12b")
    if getattr(args, "publish", False) or getattr(args, "say", False):
        event = sight_event(text, source_kind=obs["source_kind"], node_id=_node_id(args),
                            sha256=obs["sha256"], model=model, latency_ms=latency_ms,
                            changed=True, diff=diff, room=getattr(args, "room", "sight"))
        obs["published"] = publish_fn(event)
        if getattr(args, "say", False):
            obs["said"] = publish_fn(say_event(text))
        if obs["published"] is not None:
            write_status(last_publish_at=time.time(), last_text=speakable(text),
                         source_kind=obs["source_kind"], last_error=None)
        else:
            write_status(last_error="not published: no harness token or the room refused")
    return obs


def _print_observation(obs: dict, args) -> None:
    if getattr(args, "json", False):
        out = {k: v for k, v in obs.items() if k != "signature"}
        print(json.dumps(out, indent=1))
        return
    print(obs["text"])
    wants = getattr(args, "publish", False) or getattr(args, "say", False)
    if wants and obs["published"] is None:
        print("(not published: no harness token, or the room refused -- "
              "is `adk harness serve` running?)", file=sys.stderr)


def cmd_see(args) -> int:
    source = ("screen" if getattr(args, "screen", False)
              else getattr(args, "rtsp", None) or
              (f"device:{args.device}" if getattr(args, "device", None) else None)
              or getattr(args, "image", None))
    if not source:
        print("see: name a source -- an image path, --screen, --rtsp URL or --device NAME",
              file=sys.stderr)
        return 2
    try:
        obs = observe(source, args)
    except (CaptureError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    _print_observation(obs, args)
    return 0


def cmd_watch(args, *, sleep: Callable = time.sleep, clock: Callable = time.time,
              observe_fn: Callable = observe, capture_fn: Callable = capture) -> int:
    """Grab every ``--every`` seconds; LOOK only on change, paced by the last latency."""
    if kill_switch():
        write_status(refused="kill_switch")
        print(f"sight is switched off on this host ({KILL_ENV}=0); not watching.", file=sys.stderr)
        return 3
    source = getattr(args, "source", "") or ""
    if not source:
        print("watch: --source is required (screen | rtsp://... | device:NAME | a file)",
              file=sys.stderr)
        return 2
    every = max(0.5, float(getattr(args, "every", 2.0)))
    threshold = float(getattr(args, "threshold", DIFF_THRESHOLD))
    max_looks = int(getattr(args, "max_looks", 0) or 0)
    max_ticks = int(getattr(args, "max_ticks", 0) or 0)
    print(f"watching {source} -- ON AIR (Ctrl+C stops; {KILL_ENV}=0 refuses to start)",
          file=sys.stderr)
    write_status(watching=source, refused=None, started_at=clock())
    last_sig, last_look_at, last_latency, looks, ticks, errors = None, 0.0, 0.0, 0, 0, 0
    try:
        while True:
            ticks += 1
            tmp_dir = tempfile.mkdtemp(prefix="awvision-watch-")
            try:
                frame = capture_fn(source, os.path.join(tmp_dir, "frame.jpg"))
                sig = frame_signature(frame)
                errors = 0
            except CaptureError as exc:
                errors += 1
                write_status(last_error=str(exc))
                print(f"! {exc}", file=sys.stderr)
                sig = None
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if sig is not None or last_sig is None:
                diff = frame_diff(last_sig, sig) if last_sig is not None else 255.0
                first = last_look_at == 0.0
                if errors == 0 and (first or should_look(diff, clock() - last_look_at,
                                                         last_latency, threshold=threshold)):
                    try:
                        obs = observe_fn(source, args, prev_sig=last_sig)
                        last_sig = obs.get("signature") or sig
                        last_latency = obs["latency_ms"] / 1000.0
                        last_look_at = clock()
                        looks += 1
                        _print_observation(obs, args)
                    except (CaptureError, RuntimeError) as exc:
                        write_status(last_error=str(exc))
                        print(f"! {exc}", file=sys.stderr)
            if (max_looks and looks >= max_looks) or (max_ticks and ticks >= max_ticks):
                break
            # Back off on a dead source instead of hammering it: 2 s, 4 s, ... capped at 60 s.
            sleep(min(60.0, every * (2 ** min(errors, 5))) if errors else every)
    except KeyboardInterrupt:
        pass
    write_status(watching=None, looks=looks)
    print(f"stopped after {looks} look(s) in {ticks} tick(s) -- OFF AIR", file=sys.stderr)
    return 0
