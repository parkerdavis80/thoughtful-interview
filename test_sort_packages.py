"""Tests for the package sorting function."""

import pytest

from sort_packages import sort, sort_with_details


class TestStandard:
    """Packages that are neither bulky nor heavy go to STANDARD."""

    def test_small_light_package(self):
        assert sort(10, 10, 10, 5) == "STANDARD"

    def test_just_under_volume_threshold(self):
        # 99 * 100 * 100 = 990,000 < 1,000,000
        assert sort(99, 100, 100, 19) == "STANDARD"

    def test_just_under_dimension_threshold(self):
        # 149 * 50 * 50 = 372,500 < 1,000,000
        assert sort(149, 50, 50, 19) == "STANDARD"

    def test_just_under_mass_threshold(self):
        assert sort(50, 50, 50, 19.9) == "STANDARD"

    def test_minimal_package(self):
        assert sort(1, 1, 1, 0.1) == "STANDARD"


class TestSpecial:
    """Packages that are bulky OR heavy (not both) go to SPECIAL."""

    def test_heavy_only(self):
        assert sort(10, 10, 10, 20) == "SPECIAL"

    def test_bulky_by_volume_only(self):
        # 100 * 100 * 100 = 1,000,000 (exactly at threshold)
        assert sort(100, 100, 100, 10) == "SPECIAL"

    def test_bulky_by_single_dimension_width(self):
        assert sort(150, 1, 1, 1) == "SPECIAL"

    def test_bulky_by_single_dimension_height(self):
        assert sort(1, 150, 1, 1) == "SPECIAL"

    def test_bulky_by_single_dimension_length(self):
        assert sort(1, 1, 150, 1) == "SPECIAL"

    def test_heavy_at_exact_threshold(self):
        assert sort(10, 10, 10, 20) == "SPECIAL"

    def test_volume_at_exact_threshold(self):
        assert sort(100, 100, 100, 19) == "SPECIAL"

    def test_dimension_at_exact_threshold(self):
        assert sort(150, 10, 10, 19) == "SPECIAL"


class TestRejected:
    """Packages that are both bulky AND heavy go to REJECTED."""

    def test_heavy_and_bulky_by_volume(self):
        assert sort(100, 100, 100, 20) == "REJECTED"

    def test_heavy_and_bulky_by_dimension(self):
        assert sort(150, 10, 10, 20) == "REJECTED"

    def test_very_heavy_and_very_bulky(self):
        assert sort(200, 200, 200, 100) == "REJECTED"

    def test_bulky_by_volume_and_heavy(self):
        # 1000 * 1000 * 1 = 1,000,000
        assert sort(1000, 1000, 1, 25) == "REJECTED"


class TestEdgeCases:
    """Boundary conditions and invalid inputs."""

    def test_float_dimensions(self):
        assert sort(99.5, 100.5, 100.5, 19.5) == "SPECIAL"

    def test_very_large_values(self):
        assert sort(1e6, 1e6, 1e6, 1e6) == "REJECTED"

    def test_fractional_mass(self):
        assert sort(10, 10, 10, 0.01) == "STANDARD"

    def test_negative_dimension_raises(self):
        with pytest.raises(ValueError, match="width must be positive"):
            sort(-1, 10, 10, 5)

    def test_zero_dimension_raises(self):
        with pytest.raises(ValueError, match="height must be positive"):
            sort(10, 0, 10, 5)

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            sort(10, 10, 10, -1)

    def test_zero_mass_raises(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            sort(10, 10, 10, 0)

    def test_string_input_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            sort("ten", 10, 10, 5)

    def test_none_input_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            sort(10, None, 10, 5)

    def test_bool_input_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            sort(True, 10, 10, 5)

    def test_nan_input_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            sort(10, float("nan"), 10, 5)

    def test_inf_input_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            sort(10, 10, float("inf"), 5)

    def test_negative_inf_input_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            sort(10, 10, 10, float("-inf"))


class TestSortWithDetails:
    """Tests for the sort_with_details function."""

    _EXPECTED_KEYS = {
        "stack", "dimensions", "volume_cm3",
        "mass_kg", "is_bulky", "is_heavy",
    }

    def test_returns_all_keys(self):
        result = sort_with_details(10, 10, 10, 5)
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_standard_flags(self):
        result = sort_with_details(10, 10, 10, 5)
        assert result["stack"] == "STANDARD"
        assert result["is_bulky"] is False
        assert result["is_heavy"] is False

    def test_special_heavy_flags(self):
        result = sort_with_details(10, 10, 10, 20)
        assert result["stack"] == "SPECIAL"
        assert result["is_bulky"] is False
        assert result["is_heavy"] is True

    def test_special_bulky_flags(self):
        result = sort_with_details(100, 100, 100, 10)
        assert result["stack"] == "SPECIAL"
        assert result["is_bulky"] is True
        assert result["is_heavy"] is False

    def test_rejected_flags(self):
        result = sort_with_details(100, 100, 100, 20)
        assert result["stack"] == "REJECTED"
        assert result["is_bulky"] is True
        assert result["is_heavy"] is True

    def test_volume_calculation(self):
        result = sort_with_details(10, 20, 30, 5)
        assert result["volume_cm3"] == 6000.0

    def test_dimensions_in_result(self):
        result = sort_with_details(10, 20, 30, 5)
        assert result["dimensions"] == {
            "width": 10.0,
            "height": 20.0,
            "length": 30.0,
        }

    def test_mass_in_result(self):
        result = sort_with_details(10, 10, 10, 7.5)
        assert result["mass_kg"] == 7.5

    def test_negative_mass_routes_to_special(self):
        result = sort_with_details(10, 10, 10, -1)
        assert result["stack"] == "SPECIAL"
        assert result["volume_cm3"] is None
        assert result["is_bulky"] is None
        assert result["is_heavy"] is None

    def test_nan_dimension_routes_to_special(self):
        result = sort_with_details(float("nan"), 10, 10, 5)
        assert result["stack"] == "SPECIAL"

    def test_bool_input_routes_to_special(self):
        result = sort_with_details(True, 10, 10, 5)
        assert result["stack"] == "SPECIAL"

    def test_string_input_routes_to_special(self):
        result = sort_with_details("big", 10, 10, 5)
        assert result["stack"] == "SPECIAL"
