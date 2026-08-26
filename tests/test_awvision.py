"""Tests for awvision."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from awvision.vision import (
    compare_vision_images,
    get_media_type,
    get_vision_response,
    load_image_as_base64,
)


class TestImageLoading:
    """Test image loading functionality."""

    def test_load_valid_png_image(self):
        """Test loading a valid PNG image."""
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
        try:
            b64 = load_image_as_base64(test_image)
            assert b64
            assert len(b64) > 0
        finally:
            Path(test_image).unlink()

    def test_missing_file_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_as_base64('/nonexistent/path.png')

    def test_empty_file_raises_error(self):
        """Test that empty file raises IOError."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            test_image = f.name
        try:
            with pytest.raises(IOError):
                load_image_as_base64(test_image)
        finally:
            Path(test_image).unlink()


class TestMediaType:
    """Test media type detection."""

    def test_png_media_type(self):
        """Test PNG media type detection."""
        assert get_media_type('image.png') == 'image/png'

    def test_jpg_media_types(self):
        """Test JPG/JPEG media type detection."""
        assert get_media_type('image.jpg') == 'image/jpeg'
        assert get_media_type('image.jpeg') == 'image/jpeg'

    def test_gif_media_type(self):
        """Test GIF media type detection."""
        assert get_media_type('image.gif') == 'image/gif'

    def test_webp_media_type(self):
        """Test WebP media type detection."""
        assert get_media_type('image.webp') == 'image/webp'

    def test_unknown_extension_defaults_to_jpeg(self):
        """Test that unknown extensions default to JPEG."""
        assert get_media_type('image.unknown') == 'image/jpeg'


class TestVisionResponse:
    """Test vision response handling."""

    @patch('awvision.vision.urlopen')
    def test_valid_response(self, mock_urlopen):
        """Test handling valid vision response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "This is a test image."
                    }
                }
            ]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

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
        try:
            result = get_vision_response(test_image, "Test question")
            assert result == "This is a test image."
        finally:
            Path(test_image).unlink()

    @patch('awvision.vision.urlopen')
    def test_empty_content_raises_error(self, mock_urlopen):
        """Test that empty content raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": ""
                    }
                }
            ]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

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
        try:
            with pytest.raises(RuntimeError) as exc_info:
                get_vision_response(test_image, "Test question")
            assert "empty content" in str(exc_info.value).lower()
            assert "vision" in str(exc_info.value).lower()
        finally:
            Path(test_image).unlink()

    @patch('awvision.vision.urlopen')
    def test_no_choices_raises_error(self, mock_urlopen):
        """Test that response with no choices raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": []
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

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
        try:
            with pytest.raises(RuntimeError) as exc_info:
                get_vision_response(test_image, "Test question")
            assert "choices" in str(exc_info.value).lower()
        finally:
            Path(test_image).unlink()


class TestComparison:
    """Test image comparison functionality."""

    @patch('awvision.vision.urlopen')
    def test_compare_two_images(self, mock_urlopen):
        """Test comparing two images."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [
                {
                    "message": {
                        "content": "These images are similar."
                    }
                }
            ]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

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

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f1:
            f1.write(png_data)
            img1 = f1.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f2:
            f2.write(png_data)
            img2 = f2.name

        try:
            result = compare_vision_images(img1, img2)
            assert result == "These images are similar."
        finally:
            Path(img1).unlink()
            Path(img2).unlink()
