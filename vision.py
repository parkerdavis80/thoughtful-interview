"""Vision-based package dimension estimation using Google Gemini."""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types
from PIL import Image

from sort_packages import sort_with_details

_SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
}

_PROMPT = (
    "You are analyzing an image of a shipping package or box. "
    "Estimate the width, height, and length of the package "
    "in centimeters.\n\n"
    "IMPORTANT RULES:\n"
    "- Only estimate dimensions if you can clearly see a "
    "single package and can reasonably judge its size.\n"
    "- Do NOT guess or make up numbers. If the image is "
    "blurry, too dark, taken from an angle that hides a "
    "dimension, or otherwise unclear, return an error.\n"
    "- If there are multiple packages, return an error.\n"
    "- If the object is not a shipping package or box, "
    "return an error.\n\n"
    "RESPONSE FORMAT — respond with ONLY a JSON object, "
    "no other text:\n\n"
    "On success:\n"
    '{"width": <number>, "height": <number>, '
    '"length": <number>}\n\n'
    "On failure (use the most specific reason):\n"
    '{"error": "no package detected"}\n'
    '{"error": "image too blurry"}\n'
    '{"error": "image too dark"}\n'
    '{"error": "multiple packages detected"}\n'
    '{"error": "cannot determine dimensions from this angle"}\n'
)


_SECRET_NAME = "GOOGLE_API_KEY"


def _get_api_key():
    """Retrieve the Gemini API key from the best available source.

    Resolution order:
        1. .env file (local development)
        2. AWS Secrets Manager (production)
        3. os.environ fallback

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the key cannot be found anywhere.
    """
    # 1. .env file — keeps key out of shell env / process table
    from dotenv import dotenv_values
    env = dotenv_values()
    api_key = env.get(_SECRET_NAME)
    if api_key:
        logger.info("API key loaded from .env file")
        return api_key

    # 2. AWS Secrets Manager — production path
    try:
        import boto3
        client = boto3.client("secretsmanager")
        resp = client.get_secret_value(SecretId=_SECRET_NAME)
        api_key = resp["SecretString"]
        if api_key:
            logger.info("API key loaded from AWS Secrets Manager")
            return api_key
    except Exception:
        logger.debug("AWS Secrets Manager unavailable, trying env")

    # 3. os.environ — CI / container-injected secrets
    api_key = os.environ.get(_SECRET_NAME)
    if api_key:
        logger.info("API key loaded from environment variable")
        return api_key

    raise EnvironmentError(
        f"{_SECRET_NAME} not found in .env, AWS Secrets "
        "Manager, or environment variables."
    )


def _get_client():
    """Build a Gemini client from the best available key source.

    Returns:
        A google.genai.Client instance.

    Raises:
        EnvironmentError: If no API key can be found.
    """
    api_key = _get_api_key()
    client = genai.Client(api_key=api_key)
    del api_key
    return client


def _parse_dimensions(response_text):
    """Extract width, height, and length from a JSON response.

    Handles markdown code fences and extra whitespace.

    Args:
        response_text: Raw text from the Gemini API response.

    Returns:
        A dict with "width", "height", "length" as floats.

    Raises:
        ValueError: If the response cannot be parsed or is invalid.
    """
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse Gemini response as JSON: "
            f"{response_text!r}"
        ) from exc

    if "error" in data:
        raise ValueError(
            "Gemini could not estimate dimensions: "
            f"{data['error']}"
        )

    for key in ("width", "height", "length"):
        if key not in data:
            raise ValueError(
                f"Missing '{key}' in Gemini response: "
                f"{response_text!r}"
            )
        value = data[key]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"'{key}' must be a number, "
                f"got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(
                f"'{key}' must be positive, got {value}"
            )

    return {
        "width": float(data["width"]),
        "height": float(data["height"]),
        "length": float(data["length"]),
    }


def estimate_dimensions(image_path):
    """Send an image to Gemini and return estimated box dimensions.

    Args:
        image_path: Path to an image file
            (JPEG, PNG, WebP, GIF, BMP).

    Returns:
        A dict with "width", "height", "length" in centimeters.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file type is unsupported or the
            response cannot be parsed into valid dimensions.
        EnvironmentError: If GOOGLE_API_KEY is not set.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    image = Image.open(path)
    client = _get_client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, _PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return _parse_dimensions(response.text)


def sort_with_fallback(
    mass,
    width=None,
    height=None,
    length=None,
    image_path=None,
):
    """Sort a package using manual dimensions or Gemini fallback.

    Tries manual dimensions first. If any dimension is None,
    falls back to Gemini vision to estimate dimensions from an
    image. If dimensions cannot be determined at all (no dims,
    no image, Gemini failure, partial dims), the package is
    routed to SPECIAL with source="error".

    Args:
        mass: Package mass in kilograms.
        width: Package width in cm, or None to use Gemini.
        height: Package height in cm, or None to use Gemini.
        length: Package length in cm, or None to use Gemini.
        image_path: Path to an image file (used when
            dimensions are None).

    Returns:
        A dict with keys: stack, source, dimensions, volume_cm3,
        mass_kg, is_bulky, is_heavy.
    """
    def _error_result():
        return {
            "stack": "SPECIAL",
            "source": "error",
            "dimensions": None,
            "volume_cm3": None,
            "mass_kg": mass,
            "is_bulky": None,
            "is_heavy": None,
        }

    dims = [width, height, length]
    provided = [d is not None for d in dims]

    if all(provided):
        source = "manual"
        logger.debug("Using manual dimensions")
    elif not any(provided):
        if image_path is None:
            logger.warning("No dimensions or image provided")
            return _error_result()
        try:
            logger.info("Falling back to Gemini for dimensions")
            estimated = estimate_dimensions(image_path)
        except (ValueError, FileNotFoundError, EnvironmentError) as exc:
            logger.error("Gemini fallback failed: %s", exc)
            return _error_result()
        width = estimated["width"]
        height = estimated["height"]
        length = estimated["length"]
        source = "gemini"
    else:
        logger.warning("Partial dimensions provided, routing to SPECIAL")
        return _error_result()

    result = sort_with_details(width, height, length, mass)
    result["source"] = source
    return result
