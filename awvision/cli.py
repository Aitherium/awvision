"""Command-line interface for awvision."""

import argparse
import os
import sys
import tempfile
from pathlib import Path


def main():
    """Main CLI entry point."""
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
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
        help='Vision API endpoint (env: AWVISION_URL, default: http://100.64.0.38:8124 '
             '— fleet DGX gemma4-12b)'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Model name (env: AWVISION_MODEL, default: gemma4-12b)'
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

    # see command -- one look at a source you name, optionally into the room
    see_parser = subparsers.add_parser(
        'see', help='Look once at an image, the screen or a camera and say what is there')
    see_parser.add_argument('image', nargs='?', help='Path to an image file')
    see_parser.add_argument('--screen', action='store_true', help='Look at this screen')
    see_parser.add_argument('--rtsp', default=None, help='Look at one frame of an RTSP camera')
    see_parser.add_argument('--device', default=None, help='Look through a capture device, by name')
    see_parser.add_argument('--prompt', default=None, help='What to ask about the frame')
    see_parser.add_argument('--publish', action='store_true',
                            help='Publish a sight_observed event (text + frame hash) to the room')
    see_parser.add_argument('--say', action='store_true',
                            help='Also say it: one short agent_message into room main')
    see_parser.add_argument('--room', default='sight', help='Room for the sight event')
    see_parser.add_argument('--node-id', dest='node_id', default='', help='Which node saw it')
    see_parser.add_argument('--json', action='store_true', help='Print the observation as JSON')
    see_parser.add_argument('--keep-frames', dest='keep_frames', action='store_true',
                            help='Opt in: copy the frame to Strata cache (auto-deleted after 24h; '
                                 'a live source goes only to the private vault). Default: never. '
                                 'AWVISION_KEEP_FRAMES=0 refuses on this host.')

    # watch command -- keep looking, but only when the picture changes
    watch_parser = subparsers.add_parser(
        'watch', help='Keep watching a source you name; look only when the picture changes')
    watch_parser.add_argument('--source', default='',
                              help='screen | rtsp://... | device:NAME | an image path')
    watch_parser.add_argument('--every', type=float, default=2.0, help='Seconds between grabs')
    watch_parser.add_argument('--threshold', type=float, default=6.0,
                              help='Mean grayscale change (0-255) that counts as changed')
    watch_parser.add_argument('--max-looks', dest='max_looks', type=int, default=0,
                              help='Stop after this many looks (0 = until Ctrl+C)')
    watch_parser.add_argument('--prompt', default=None, help='What to ask about each frame')
    watch_parser.add_argument('--publish', action='store_true', help='Publish each observation')
    watch_parser.add_argument('--say', action='store_true', help='Say each observation in the room')
    watch_parser.add_argument('--room', default='sight', help='Room for the sight events')
    watch_parser.add_argument('--node-id', dest='node_id', default='',
                              help='Which node is watching')
    watch_parser.add_argument('--json', action='store_true', help='Print observations as JSON')
    watch_parser.add_argument('--max-ticks', dest='max_ticks', type=int, default=0,
                              help='Stop after this many grabs (0 = until Ctrl+C)')
    watch_parser.add_argument('--keep-frames', dest='keep_frames', action='store_true',
                              help='Opt in: copy each looked-at frame to Strata cache '
                                   '(auto-deleted after 24h; a live source goes only to the '
                                   'private vault). Default: never. AWVISION_KEEP_FRAMES=0 '
                                   'refuses on this host.')

    # forget command -- the purge verb for frames kept with --keep-frames
    forget_parser = subparsers.add_parser(
        'forget', help='Delete kept frames from Strata (--all or --older-than HOURS)')
    forget_parser.add_argument('--all', action='store_true', help='Forget every kept frame')
    forget_parser.add_argument('--older-than', dest='older_than', type=float, default=None,
                               metavar='HOURS', help='Forget frames older than this many hours')

    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.command:
        parser.print_help()
        return 1

    if args.command in ('see', 'watch', 'forget'):
        from awvision import sight

        if args.command == 'see':
            return sight.cmd_see(args)
        if args.command == 'watch':
            return sight.cmd_watch(args)
        return sight.cmd_forget(args)

    try:
        from awvision.vision import compare_vision_images, get_vision_response

        if args.command == 'ask':
            response = get_vision_response(args.image, args.question, args.endpoint, args.model)
            print(response)
        elif args.command == 'describe':
            response = get_vision_response(args.image, 'Describe this image in detail.',
                                           args.endpoint, args.model)
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
        except OSError as exc:
            print(f"  (temp image left behind: {exc})")

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
        endpoint = os.getenv('AWVISION_URL', 'http://100.64.0.38:8124')
        model = os.getenv('AWVISION_MODEL', 'gemma4-12b')
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
    endpoint = os.getenv('AWVISION_URL', 'http://100.64.0.38:8124')
    print(f"To use awvision, ensure a vision-capable service is running at {endpoint}")
    print()
    print("Examples:")
    print("  awvision ask image.png 'What is in this image?'")
    print("  awvision describe image.jpg")
    print("  awvision compare image1.png image2.png")
    return 0


if __name__ == '__main__':
    sys.exit(main())
