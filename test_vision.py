"""Tests for the Gemini vision-based dimension estimation."""

import os
from unittest.mock import MagicMock, patch

import pytest

from vision import (
    _parse_dimensions,
    estimate_dimensions,
    sort_with_fallback,
)


# -------------------------------------------------------------------
# TestParseDimensions — pure unit tests, no mocking needed
# -------------------------------------------------------------------


class TestParseDimensions:
    """Tests for the _parse_dimensions helper."""

    def test_valid_json(self):
        text = '{"width": 30, "height": 20, "length": 40}'
        result = _parse_dimensions(text)
        assert result == {
            "width": 30.0,
            "height": 20.0,
            "length": 40.0,
        }

    def test_valid_json_with_code_fences(self):
        text = '```json\n{"width": 30, "height": 20, "length": 40}\n```'
        result = _parse_dimensions(text)
        assert result == {
            "width": 30.0,
            "height": 20.0,
            "length": 40.0,
        }

    def test_valid_json_with_bare_code_fences(self):
        text = '```\n{"width": 10, "height": 10, "length": 10}\n```'
        result = _parse_dimensions(text)
        assert result == {
            "width": 10.0,
            "height": 10.0,
            "length": 10.0,
        }

    def test_float_values(self):
        text = '{"width": 30.5, "height": 20.3, "length": 40.1}'
        result = _parse_dimensions(text)
        assert result == {
            "width": 30.5,
            "height": 20.3,
            "length": 40.1,
        }

    def test_integers_converted_to_float(self):
        result = _parse_dimensions(
            '{"width": 1, "height": 2, "length": 3}'
        )
        for value in result.values():
            assert isinstance(value, float)

    def test_extra_keys_ignored(self):
        text = (
            '{"width": 10, "height": 20, '
            '"length": 30, "confidence": 0.9}'
        )
        result = _parse_dimensions(text)
        assert "confidence" not in result
        assert result["width"] == 10.0

    def test_error_response_raises(self):
        text = '{"error": "no package detected"}'
        with pytest.raises(
            ValueError, match="could not estimate"
        ):
            _parse_dimensions(text)

    def test_error_blurry_raises(self):
        text = '{"error": "image too blurry"}'
        with pytest.raises(
            ValueError, match="image too blurry"
        ):
            _parse_dimensions(text)

    def test_error_dark_raises(self):
        text = '{"error": "image too dark"}'
        with pytest.raises(
            ValueError, match="image too dark"
        ):
            _parse_dimensions(text)

    def test_error_multiple_packages_raises(self):
        text = '{"error": "multiple packages detected"}'
        with pytest.raises(
            ValueError, match="multiple packages"
        ):
            _parse_dimensions(text)

    def test_error_bad_angle_raises(self):
        text = '{"error": "cannot determine dimensions from this angle"}'
        with pytest.raises(
            ValueError, match="cannot determine"
        ):
            _parse_dimensions(text)

    def test_missing_width_raises(self):
        text = '{"height": 20, "length": 40}'
        with pytest.raises(ValueError, match="Missing 'width'"):
            _parse_dimensions(text)

    def test_missing_height_raises(self):
        text = '{"width": 30, "length": 40}'
        with pytest.raises(ValueError, match="Missing 'height'"):
            _parse_dimensions(text)

    def test_missing_length_raises(self):
        text = '{"width": 30, "height": 20}'
        with pytest.raises(ValueError, match="Missing 'length'"):
            _parse_dimensions(text)

    def test_negative_value_raises(self):
        text = '{"width": -5, "height": 20, "length": 40}'
        with pytest.raises(ValueError, match="must be positive"):
            _parse_dimensions(text)

    def test_zero_value_raises(self):
        text = '{"width": 0, "height": 20, "length": 40}'
        with pytest.raises(ValueError, match="must be positive"):
            _parse_dimensions(text)

    def test_non_numeric_value_raises(self):
        text = '{"width": "big", "height": 20, "length": 40}'
        with pytest.raises(ValueError, match="must be a number"):
            _parse_dimensions(text)

    def test_invalid_json_raises(self):
        with pytest.raises(
            ValueError, match="Could not parse"
        ):
            _parse_dimensions("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_dimensions("")


# -------------------------------------------------------------------
# Fixtures for mocked Gemini API tests
# -------------------------------------------------------------------


@pytest.fixture()
def dummy_image_path(tmp_path):
    """Create a temporary dummy JPEG file."""
    path = tmp_path / "test_box.jpg"
    # Minimal bytes so the file exists; PIL is mocked.
    path.write_bytes(b"\xff\xd8\xff\xe0")
    return str(path)


@pytest.fixture()
def mock_gemini():
    """Mock the Gemini client and PIL.Image.open."""
    mock_client = MagicMock()
    mock_image = MagicMock()

    with patch("vision._get_client", return_value=mock_client), \
         patch("vision.Image") as mock_pil:
        mock_pil.open.return_value = mock_image
        yield mock_client, mock_image


def _make_response(text):
    """Create a mock Gemini response with the given text."""
    response = MagicMock()
    response.text = text
    return response


# -------------------------------------------------------------------
# TestEstimateDimensions — mocked API tests
# -------------------------------------------------------------------


class TestEstimateDimensions:
    """Tests for estimate_dimensions with mocked Gemini API."""

    def test_successful_estimation(
        self, dummy_image_path, mock_gemini
    ):
        mock_client, _ = mock_gemini
        mock_client.models.generate_content.return_value = (
            _make_response(
                '{"width": 45, "height": 30, "length": 60}'
            )
        )

        result = estimate_dimensions(dummy_image_path)

        assert result == {
            "width": 45.0,
            "height": 30.0,
            "length": 60.0,
        }

    def test_file_not_found_raises(self):
        with pytest.raises(
            FileNotFoundError, match="not found"
        ):
            estimate_dimensions("/no/such/file.jpg")

    def test_unsupported_format_raises(self, tmp_path):
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("hello")
        with pytest.raises(
            ValueError, match="Unsupported image format"
        ):
            estimate_dimensions(str(txt_file))

    def test_api_key_missing_raises(self, dummy_image_path):
        with patch("vision._get_api_key") as mock_get_key, \
             patch("vision.Image"):
            mock_get_key.side_effect = EnvironmentError(
                "GOOGLE_API_KEY not found"
            )
            with pytest.raises(
                EnvironmentError, match="GOOGLE_API_KEY"
            ):
                estimate_dimensions(dummy_image_path)

    def test_api_error_response_raises(
        self, dummy_image_path, mock_gemini
    ):
        mock_client, _ = mock_gemini
        mock_client.models.generate_content.return_value = (
            _make_response('{"error": "no package detected"}')
        )

        with pytest.raises(
            ValueError, match="could not estimate"
        ):
            estimate_dimensions(dummy_image_path)

    def test_api_garbage_response_raises(
        self, dummy_image_path, mock_gemini
    ):
        mock_client, _ = mock_gemini
        mock_client.models.generate_content.return_value = (
            _make_response("I cannot process this image")
        )

        with pytest.raises(
            ValueError, match="Could not parse"
        ):
            estimate_dimensions(dummy_image_path)

    def test_model_name_passed(
        self, dummy_image_path, mock_gemini
    ):
        mock_client, _ = mock_gemini
        mock_client.models.generate_content.return_value = (
            _make_response(
                '{"width": 10, "height": 10, "length": 10}'
            )
        )

        estimate_dimensions(dummy_image_path)

        call_kwargs = (
            mock_client.models.generate_content.call_args
        )
        assert call_kwargs.kwargs["model"] == "gemini-2.5-flash"

    def test_supported_extensions(self, tmp_path, mock_gemini):
        mock_client, _ = mock_gemini
        mock_client.models.generate_content.return_value = (
            _make_response(
                '{"width": 10, "height": 10, "length": 10}'
            )
        )

        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = tmp_path / f"box{ext}"
            path.write_bytes(b"\x00")
            result = estimate_dimensions(str(path))
            assert "width" in result


# -------------------------------------------------------------------
# TestSortWithFallback — fallback flow + edge cases
# -------------------------------------------------------------------


class TestSortWithFallback:
    """Tests for the sort_with_fallback function."""

    _EXPECTED_KEYS = {
        "stack", "source", "dimensions", "volume_cm3",
        "mass_kg", "is_bulky", "is_heavy",
    }

    def test_manual_dims_returns_all_keys(self):
        result = sort_with_fallback(
            mass=5.0, width=10, height=10, length=10,
        )
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_manual_dims_source_is_manual(self):
        result = sort_with_fallback(
            mass=5.0, width=10, height=10, length=10,
        )
        assert result["source"] == "manual"

    @patch("vision.estimate_dimensions")
    def test_manual_dims_skips_gemini(self, mock_estimate):
        sort_with_fallback(
            mass=5.0, width=10, height=10, length=10,
        )
        mock_estimate.assert_not_called()

    def test_manual_dims_standard(self):
        result = sort_with_fallback(
            mass=5.0, width=10, height=10, length=10,
        )
        assert result["stack"] == "STANDARD"

    def test_manual_dims_special_heavy(self):
        result = sort_with_fallback(
            mass=25.0, width=10, height=10, length=10,
        )
        assert result["stack"] == "SPECIAL"
        assert result["is_heavy"] is True

    def test_manual_dims_rejected(self):
        result = sort_with_fallback(
            mass=25.0, width=200, height=200, length=200,
        )
        assert result["stack"] == "REJECTED"

    @patch("vision.estimate_dimensions")
    def test_gemini_fallback_called(self, mock_estimate):
        mock_estimate.return_value = {
            "width": 10.0,
            "height": 10.0,
            "length": 10.0,
        }
        result = sort_with_fallback(
            mass=5.0, image_path="box.jpg",
        )
        mock_estimate.assert_called_once_with("box.jpg")
        assert result["source"] == "gemini"
        assert result["stack"] == "STANDARD"

    @patch("vision.estimate_dimensions")
    def test_gemini_fallback_result_fields(self, mock_estimate):
        mock_estimate.return_value = {
            "width": 45.0,
            "height": 30.0,
            "length": 60.0,
        }
        result = sort_with_fallback(
            mass=12.5, image_path="box.jpg",
        )
        assert result["dimensions"] == {
            "width": 45.0,
            "height": 30.0,
            "length": 60.0,
        }
        assert result["volume_cm3"] == 81000.0
        assert result["mass_kg"] == 12.5

    def test_partial_dims_routes_to_special(self):
        result = sort_with_fallback(
            mass=5.0, width=10, height=None, length=10,
        )
        assert result["stack"] == "SPECIAL"
        assert result["source"] == "error"
        assert result["dimensions"] is None

    def test_partial_dims_one_provided_routes_to_special(self):
        result = sort_with_fallback(mass=5.0, width=10)
        assert result["stack"] == "SPECIAL"
        assert result["source"] == "error"

    def test_no_dims_no_image_routes_to_special(self):
        result = sort_with_fallback(mass=5.0)
        assert result["stack"] == "SPECIAL"
        assert result["source"] == "error"

    @patch("vision.estimate_dimensions")
    def test_gemini_failure_routes_to_special(self, mock_estimate):
        mock_estimate.side_effect = ValueError("bad image")
        result = sort_with_fallback(
            mass=5.0, image_path="box.jpg",
        )
        assert result["stack"] == "SPECIAL"
        assert result["source"] == "error"

    def test_invalid_mass_routes_to_special(self):
        result = sort_with_fallback(
            mass=-1, width=10, height=10, length=10,
        )
        assert result["stack"] == "SPECIAL"
        assert result["source"] == "manual"
