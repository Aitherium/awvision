# awvision

Ask questions about images using a vision-capable model.

## What it does

awvision is a lightweight client for vision-capable, OpenAI-compatible LLM services. Send an image and a question, get an answer.

The contract: **Ask a question about an image and get an answer.**

## Installation

```bash
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

### Ask a question about an image

```bash
awvision ask image.png "What is in this image?"
```

### Describe an image

```bash
awvision describe image.jpg
```

### Compare two images

```bash
awvision compare image1.png image2.png
```

## Configuration

Set environment variables to customize the endpoint and model:

```bash
export AWVISION_URL=http://localhost:8150
export AWVISION_MODEL=gpt-4-vision
```

Or pass them as options:

```bash
awvision --endpoint http://localhost:8150 --model gpt-4-vision ask image.png "What's this?"
```

## Self-test

Run the self-tests to verify awvision is working correctly:

```bash
awvision --self-test
```

This will:
- Test image loading and encoding
- Verify media type detection
- Check endpoint configuration
- Validate error handling

## Requirements

- Python 3.10+
- A vision-capable LLM service (e.g., OpenAI-compatible API)
- The service must support the `image_url` content block format

## How it works

awvision:
1. Loads the image file and encodes it as base64
2. Wraps it in an OpenAI-compatible chat message with `image_url` format
3. Sends it to the configured vision service
4. Returns the model's response

### Important: Empty response detection

If a text-only model receives an image, it typically returns HTTP 200 with an empty response. awvision detects this and raises a clear error:

```
ERROR: Model 'gpt-4-vision' at http://localhost:8150 returned empty content.
This indicates the model does not support vision or cannot process image_url content blocks.
Verify that:
  1. The model 'gpt-4-vision' is vision-capable
  2. The endpoint supports the image_url format
  3. The image was properly encoded as a data URL
```

This prevents silent failures where an empty response looks like a successful query.

## API

For programmatic use:

```python
from awvision import ask_vision, describe_image, compare_images

# Ask a question
answer = ask_vision("image.png", "What color is this?")
print(answer)

# Describe an image
description = describe_image("image.jpg")
print(description)

# Compare two images
comparison = compare_images("image1.png", "image2.png")
print(comparison)
```

## Error handling

awvision provides clear error messages for common issues:

- **FileNotFoundError**: Image file not found
- **IOError**: Image file is empty or unreadable
- **RuntimeError**: Vision service is unavailable, model has no vision, or response is empty

All error messages include guidance on what to check.

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Supported image formats

- PNG (image/png)
- JPEG (image/jpeg)
- GIF (image/gif)
- WebP (image/webp)

## License

MIT
