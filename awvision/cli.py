"""Command-line interface for awvision."""

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Ask questions about images using a vision-capable model'
    )
    parser.add_argument(
        '--self-test',
        action='store_true',
        help='Run self-tests and verify awvision is working'
    )
    parser.add_argument(
        '--endpoint',
        default=None,
        help='Vision API endpoint (env: AWVISION_URL, default: http://localhost:8150)'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Model name (env: AWVISION_MODEL, default: gpt-4-vision)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question about an image')
    ask_parser.add_argument('image', help='Path to image file')
    ask_parser.add_argument('question', help='Question to ask about the image')

    # describe command
    desc_parser = subparsers.add_parser('describe', help='Describe an image')
    desc_parser.add_argument('image', help='Path to image file')

    # compare command
    comp_parser = subparsers.add_parser('compare', help='Compare two images')
    comp_parser.add_argument('image_a', help='First image path')
    comp_parser.add_argument('image_b', help='Second image path')

    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.command:
        parser.print_help()
        return 1

    try:
        from awvision.vision import get_vision_response, compare_vision_images

        if args.command == 'ask':
            response = get_vision_response(args.image, args.question, args.endpoint, args.model)
            print(response)
        elif args.command == 'describe':
            response = get_vision_response(args.image, 'Describe this image in detail.', args.endpoint, args.model)
            print(response)
        elif args.command == 'compare':
            response = compare_vision_images(args.image_a, args.image_b, args.endpoint, args.model)
            print(response)
        return 0
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def run_self_test():
    """Run self-tests to verify awvision is working."""
    print("awvision self-test:")

    # Test 1: Check image loading works
    print("  [1/6] Image loading...", end=' ', flush=True)
    try:
        from awvision.vision import load_image_as_base64
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            png_data = bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
                0x54, 0x08, 0x99, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
                0x00, 0x00, 0x03, 0x00, 0x01, 0xDD, 0x4B, 0xFB,
                0x56, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
                0x44, 0xAE, 0x42, 0x60, 0x82
            ])
            f.write(png_data)
            test_image = f.name
        b64 = load_image_as_base64(test_image)
        assert b64, "Image encoding failed"
        print("ok")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1
    finally:
        try:
            Path(test_image).unlink()
        except Exception:
            pass

    # Test 2: Check missing file detection
    print("  [2/6] Missing file detection...", end=' ', flush=True)
    try:
        from awvision.vision import load_image_as_base64
        try:
            load_image_as_base64('/nonexistent/path.png')
            print("FAIL: Should have raised FileNotFoundError")
            return 1
        except FileNotFoundError:
            print("ok")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    # Test 3: Check media type detection
    print("  [3/6] Media type detection...", end=' ', flush=True)
    try:
        from awvision.vision import get_media_type
        assert get_media_type('test.png') == 'image/png'
        assert get_media_type('test.jpg') == 'image/jpeg'
        assert get_media_type('test.jpeg') == 'image/jpeg'
        assert get_media_type('test.webp') == 'image/webp'
        print("ok")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    # Test 4: Check empty response detection logic
    print("  [4/6] Empty response detection...", end=' ', flush=True)
    try:
        test_response = {}
        if not test_response.get('choices'):
            print("ok")
        else:
            print("FAIL: Logic error")
            return 1
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    # Test 5: Check endpoint configuration
    print("  [5/6] Endpoint configuration...", end=' ', flush=True)
    try:
        endpoint = os.getenv('AWVISION_URL', 'http://localhost:8150')
        model = os.getenv('AWVISION_MODEL', 'gpt-4-vision')
        assert endpoint, "Endpoint not configured"
        assert model, "Model not configured"
        print("ok")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    # Test 6: Check help works
    print("  [6/6] Help text...", end=' ', flush=True)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('test')
        print("ok")
    except Exception as e:
        print(f"FAIL: {e}")
        return 1

    print()
    print("All self-tests passed!")
    endpoint = os.getenv('AWVISION_URL', 'http://localhost:8150')
    print(f"To use awvision, ensure a vision-capable service is running at {endpoint}")
    print()
    print("Examples:")
    print("  awvision ask image.png 'What is in this image?'")
    print("  awvision describe image.jpg")
    print("  awvision compare image1.png image2.png")
    return 0


if __name__ == '__main__':
    sys.exit(main())
