# Package Sorting System

A package dispatching function for Thoughtful's robotic automation factory. Sorts packages into the correct stack based on their volume, dimensions, and mass.

## Sorting Rules

| Stack      | Condition                          |
|------------|------------------------------------|
| STANDARD   | Neither bulky nor heavy            |
| SPECIAL    | Either bulky or heavy (not both)   |
| REJECTED   | Both bulky and heavy               |

- **Bulky**: volume (W x H x L) >= 1,000,000 cm³ or any single dimension >= 150 cm
- **Heavy**: mass >= 20 kg

## Usage

```python
from sort_packages import sort

sort(10, 10, 10, 5)       # "STANDARD"
sort(150, 10, 10, 5)      # "SPECIAL" (bulky by dimension)
sort(100, 100, 100, 25)   # "REJECTED" (bulky and heavy)
```

## Detailed Sorting

`sort_with_details()` returns a full breakdown of the sorting decision:

```python
from sort_packages import sort_with_details

result = sort_with_details(100, 100, 100, 25)
# {
#     "stack": "REJECTED",
#     "dimensions": {"width": 100.0, "height": 100.0, "length": 100.0},
#     "volume_cm3": 1000000.0,
#     "mass_kg": 25.0,
#     "is_bulky": True,
#     "is_heavy": True,
# }
```

## Vision-Based Sorting with Fallback

`sort_with_fallback()` tries manual dimensions first, then falls
back to Google Gemini vision if no dimensions are provided.

### Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root (already gitignored):

```
GOOGLE_API_KEY=your-api-key-here
```

In production, store the key in **AWS Secrets Manager** under the
name `GOOGLE_API_KEY` — the code picks it up automatically. The
resolution order is: `.env` file, AWS Secrets Manager, then
`os.environ` as a last resort.

### Usage

```python
from vision import sort_with_fallback

# Manual dimensions — Gemini is never called
result = sort_with_fallback(mass=12.5, width=45, height=30, length=60)
# {"stack": "STANDARD", "source": "manual", ...}

# No dimensions — falls back to Gemini vision
result = sort_with_fallback(mass=12.5, image_path="photo_of_box.jpg")
# {"stack": "STANDARD", "source": "gemini", ...}
```

The result dict always includes: `stack`, `source`, `dimensions`,
`volume_cm3`, `mass_kg`, `is_bulky`, `is_heavy`.

Note: Mass must still be provided manually (Gemini estimates
dimensions only).

## Running Tests

```bash
pip install -r requirements.txt
python3 -m pytest test_sort_packages.py test_vision.py -v
```
