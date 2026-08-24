"""awvision - Ask questions about images using a vision-capable model."""

__version__ = "0.1.0"
__all__ = ["ask_vision", "describe_image", "compare_images"]


def ask_vision(image_path: str, question: str, endpoint=None, model=None):
    """Ask a question about an image.

    Args:
        image_path: Path to the image file
        question: Question to ask about the image
        endpoint: Vision API endpoint (default: env AWVISION_URL or http://localhost:8150)
        model: Model name (default: env AWVISION_MODEL or gpt-4-vision)

    Returns:
        The model's response

    Raises:
        FileNotFoundError: If image file not found
        RuntimeError: If vision service unavailable or model lacks vision
    """
    from awvision.vision import get_vision_response
    return get_vision_response(image_path, question, endpoint, model)


def describe_image(image_path: str, endpoint=None, model=None):
    """Describe an image.

    Args:
        image_path: Path to the image file
        endpoint: Vision API endpoint
        model: Model name

    Returns:
        Description of the image
    """
    from awvision.vision import get_vision_response
    return get_vision_response(image_path, "Describe this image in detail.", endpoint, model)


def compare_images(image_a: str, image_b: str, endpoint=None, model=None):
    """Compare two images.

    Args:
        image_a: Path to first image
        image_b: Path to second image
        endpoint: Vision API endpoint
        model: Model name

    Returns:
        Comparison of the two images
    """
    from awvision.vision import compare_vision_images
    return compare_vision_images(image_a, image_b, endpoint, model)
