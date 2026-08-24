"""Core vision API client logic."""

import base64
import json
import os
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


def get_vision_response(image_path: str, question: str, endpoint=None, model=None):
    """Send image + question to vision model and get response."""
    if endpoint is None:
        endpoint = os.getenv('AWVISION_URL', 'http://localhost:8150')
    if model is None:
        model = os.getenv('AWVISION_MODEL', 'gpt-4-vision')
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
        endpoint = os.getenv('AWVISION_URL', 'http://localhost:8150')
    if model is None:
        model = os.getenv('AWVISION_MODEL', 'gpt-4-vision')
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


def _make_vision_request(endpoint: str, model: str, payload: dict) -> str:
    """Make a request to the vision API."""
    url = f"{endpoint}/v1/chat/completions"
    req = Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urlopen(req, timeout=60) as response:
            response_data = json.loads(response.read().decode('utf-8'))
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
