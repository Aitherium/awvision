"""Core vision API client logic."""

import base64
import json
import os
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_image_as_base64(image_path: str) -> str:
    """Load image file and encode as base64."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {image_path}")
    with open(path, 'rb') as f:
        image_data = f.read()
    if not image_data:
        raise IOError(f"Image file is empty: {image_path}")
    return base64.b64encode(image_data).decode('utf-8')


def get_media_type(image_path: str) -> str:
    """Determine media type from file extension."""
    suffix = Path(image_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    return media_type_map.get(suffix, 'image/jpeg')


# Endpoint/model resolution when the caller names neither. The fixed default (a mesh
# address) is unreachable from most machines, and the model it names may be parked;
# so: AWVISION_URL if set, else the first candidate that answers /v1/models, and the
# requested model if that endpoint serves it, else the first available vision model.
_CANDIDATES = ("https://127.0.0.1:8150", "http://100.64.0.38:8124")
_VISION_MODELS = ("gemma4-12b", "bonsai2-27b")
_resolved: dict = {}


def _list_models(endpoint: str, timeout: float = 10.0):
    try:
        with urlopen(f"{endpoint.rstrip('/')}/v1/models", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - an unreachable candidate is simply skipped
        return None
    out = {}
    for m in (data or {}).get("data") or []:
        a = m.get("aither") or {}
        healthy = a.get("backend_healthy", True) is not False
        out[m.get("id")] = bool(a.get("available", True)) and healthy
    return out


def resolve_endpoint() -> str:
    env = os.getenv("AWVISION_URL", "")
    if env:
        return env
    if "endpoint" not in _resolved:
        _resolved["endpoint"] = _CANDIDATES[-1]
        for c in _CANDIDATES:
            models = _list_models(c)
            if models is not None:
                _resolved["endpoint"], _resolved["models"] = c, models
                break
    return _resolved["endpoint"]


def resolve_model(endpoint: str) -> str:
    env = os.getenv("AWVISION_MODEL", "")
    if env:
        return env
    if _resolved.get("endpoint") == endpoint:
        models = _resolved.get("models")
    else:
        models = _list_models(endpoint)
    if models:
        for m in _VISION_MODELS:
            if models.get(m):
                return m
    return _VISION_MODELS[0]


def get_vision_response(image_path: str, question: str, endpoint=None, model=None):
    """Send image + question to vision model and get response."""
    if endpoint is None:
        endpoint = resolve_endpoint()
    if model is None:
        model = resolve_model(endpoint)
    image_b64 = load_image_as_base64(image_path)
    media_type = get_media_type(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    }
    return _make_vision_request(endpoint, model, payload)


def compare_vision_images(image_a: str, image_b: str, endpoint=None, model=None):
    """Compare two images using a vision model."""
    if endpoint is None:
        endpoint = resolve_endpoint()
    if model is None:
        model = resolve_model(endpoint)
    img_a_b64 = load_image_as_base64(image_a)
    img_b_b64 = load_image_as_base64(image_b)
    media_type_a = get_media_type(image_a)
    media_type_b = get_media_type(image_b)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare these two images. First image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type_a};base64,{img_a_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Second image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type_b};base64,{img_b_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What are the key differences and similarities?"
                    }
                ]
            }
        ]
    }
    return _make_vision_request(endpoint, model, payload)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _make_vision_request(endpoint: str, model: str, payload: dict) -> str:
    """Make a request to the vision API.

    A reasoning-capable vision model (Bonsai-2) thinks before it answers, so an
    uncapped request through MicroScheduler got max_tokens=1500 and outran the old fixed
    60 s timeout (measured 2026-10-01). Both are now bounded and configurable:
    AWVISION_MAX_TOKENS (default 512) and AWVISION_TIMEOUT seconds (default 180).
    """
    payload.setdefault("max_tokens", int(_env_float("AWVISION_MAX_TOKENS", 512)))
    # Describing pixels needs no chain-of-thought; a thinking model answered the same
    # image in 1 s with it off versus ~58 s on. AWVISION_THINKING=1 restores it.
    if os.getenv("AWVISION_THINKING", "") not in ("1", "true", "on", "yes"):
        payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
    timeout = _env_float("AWVISION_TIMEOUT", 180.0)
    url = f"{endpoint}/v1/chat/completions"
    req = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
    except (TimeoutError, socket.timeout):
        raise RuntimeError(
            f"Vision service at {endpoint} did not answer within {timeout:.0f}s. "
            "Raise AWVISION_TIMEOUT, or lower AWVISION_MAX_TOKENS for a reasoning model"
        )
    except HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Vision endpoint not found at {url}\nMake sure the vision service is running at {endpoint}"
            )
        elif e.code == 401:
            raise RuntimeError(f"Authentication failed at {url}\nCheck your credentials or API key")
        else:
            raise RuntimeError(f"Vision service returned HTTP {e.code}\nURL: {url}")
    except URLError:
        raise RuntimeError(f"Cannot reach vision service at {endpoint}\nMake sure the service is running and accessible")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Vision service returned invalid JSON\nResponse was not valid JSON: {e}")
    if not isinstance(response_data, dict):
        raise RuntimeError(f"Vision service returned unexpected response type: {type(response_data)}\nExpected JSON object")
    if "error" in response_data:
        error_msg = response_data.get("error", {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get("message", str(error_msg))
        raise RuntimeError(f"Vision service error: {error_msg}")
    choices = response_data.get('choices')
    if not choices or not isinstance(choices, list) or len(choices) == 0:
        raise RuntimeError(
            f"Model '{model}' at {endpoint} returned an empty choices list.\nThis typically means the model does not support vision or the request was invalid."
        )
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"Model '{model}' returned unexpected choice structure")
    message = first_choice.get('message')
    if not message or not isinstance(message, dict):
        raise RuntimeError(f"Model '{model}' returned no message in response.\nThis typically means the model does not support vision or cannot process images.")
    content = message.get('content', '')
    if not content or (isinstance(content, str) and content.isspace()):
        raise RuntimeError(
            f"Model '{model}' at {endpoint} returned empty content.\nThis indicates the model does not support vision or cannot process image_url content blocks.\nVerify that:\n  1. The model '{model}' is vision-capable\n  2. The endpoint supports the image_url format\n  3. The image was properly encoded as a data URL"
        )
    return content
