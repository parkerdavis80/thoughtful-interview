"""Package sorting module for Thoughtful's robotic automation factory."""

import math


def sort(width: float, height: float, length: float, mass: float) -> str:
    """Dispatch a package to the correct stack based on dimensions and mass.

    Args:
        width: Package width in centimeters.
        height: Package height in centimeters.
        length: Package length in centimeters.
        mass: Package mass in kilograms.

    Returns:
        "STANDARD", "SPECIAL", or "REJECTED".

    Raises:
        ValueError: If any dimension or mass is not a positive number.
    """
    params = [
        ("width", width),
        ("height", height),
        ("length", length),
        ("mass", mass),
    ]

    for name, value in params:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be a number, got {type(value).__name__}"
            )
        if math.isnan(value):
            raise ValueError(f"{name} must be a number, got nan")
        if math.isinf(value):
            raise ValueError(
                f"{name} must be finite, got {value}"
            )
        if value <= 0:
            raise ValueError(
                f"{name} must be positive, got {value}"
            )

    volume = width * height * length
    bulky = volume >= 1_000_000 or max(width, height, length) >= 150
    heavy = mass >= 20

    return "REJECTED" if (bulky and heavy) else "SPECIAL" if (bulky or heavy) else "STANDARD"


def sort_with_details(width, height, length, mass):
    """Sort a package and return a detailed result breakdown.

    If the inputs fail validation, the package is routed to
    SPECIAL instead of raising an exception.

    Args:
        width: Package width in centimeters.
        height: Package height in centimeters.
        length: Package length in centimeters.
        mass: Package mass in kilograms.

    Returns:
        A dict with keys:
            stack: "STANDARD", "SPECIAL", or "REJECTED".
            dimensions: {"width": ..., "height": ..., "length": ...}.
            volume_cm3: float or None if inputs are invalid.
            mass_kg: the mass value as given.
            is_bulky: bool or None if inputs are invalid.
            is_heavy: bool or None if inputs are invalid.
    """
    try:
        stack = sort(width, height, length, mass)
    except (ValueError, TypeError):
        return {
            "stack": "SPECIAL",
            "dimensions": {
                "width": width,
                "height": height,
                "length": length,
            },
            "volume_cm3": None,
            "mass_kg": mass,
            "is_bulky": None,
            "is_heavy": None,
        }

    volume = width * height * length
    return {
        "stack": stack,
        "dimensions": {
            "width": float(width),
            "height": float(height),
            "length": float(length),
        },
        "volume_cm3": float(volume),
        "mass_kg": float(mass),
        "is_bulky": volume >= 1_000_000 or max(width, height, length) >= 150,
        "is_heavy": mass >= 20,
    }
