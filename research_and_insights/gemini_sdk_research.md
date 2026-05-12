# Gemini SDK Research — Image-to-3D Pipeline
<!-- Last updated: May 12, 2026 -->

---

## Table of Contents

1. [Recommended SDK (2026)](#1-recommended-sdk-2026)
2. [Installation and GitHub](#2-installation-and-github)
3. [API Key Setup and Client Initialization](#3-api-key-setup-and-client-initialization)
4. [Current Model IDs](#4-current-model-ids)
5. [Vision / Image Input](#5-vision--image-input)
6. [Structured JSON Output with Pydantic](#6-structured-json-output-with-pydantic)
7. [Bounding Box / Spatial Detection](#7-bounding-box--spatial-detection)
8. [Web Search Grounding](#8-web-search-grounding)
9. [Rate Limits and Quotas](#9-rate-limits-and-quotas)
10. [Migration Notes: google-generativeai → google-genai](#10-migration-notes-google-generativeai--google-genai)

---

## 1. Recommended SDK (2026)

**Use `google-genai`.** Do NOT use `google-generativeai`.

| Package | Status |
|---|---|
| `google-genai` | **Recommended** — GA (General Availability), actively maintained, all new features |
| `google-generativeai` | **Deprecated** as of November 30, 2025. Repo renamed to `deprecated-generative-ai-python`. Not actively maintained. |

### Why the change?
With Gemini 2.0, Google consolidated all developer-facing SDKs into a single unified "Google Gen AI SDK." The new SDK:
- Supports Gemini Developer API (ai.google.dev) and Vertex AI with the same client
- Uses a centralized `Client` object for all services (models, files, caching, tuning, chat)
- Is the only SDK that receives new features (structured output enhancements, native tools, etc.)
- Is required for all Gemini 3.x model access

**Deprecated repo:** https://github.com/google-gemini/deprecated-generative-ai-python

---

## 2. Installation and GitHub

### Install

```bash
pip install google-genai
```

Or with uv (faster):
```bash
uv pip install google-genai
```

**Latest version as of May 12, 2026:** `v2.1.0` (released May 12, 2026)

**Requirements:** Python >= 3.10

### GitHub

- **Python SDK:** https://github.com/googleapis/python-genai
- **JavaScript SDK:** https://github.com/googleapis/js-genai
- **Go SDK:** https://github.com/googleapis/go-genai
- **Java SDK:** https://github.com/googleapis/java-genai

### API Reference Docs

https://googleapis.github.io/python-genai/

### PyPI

https://pypi.org/project/google-genai/

---

## 3. API Key Setup and Client Initialization

### Environment Variables

Two env vars are recognized (set only one):

| Variable | Notes |
|---|---|
| `GEMINI_API_KEY` | Recommended for Gemini Developer API |
| `GOOGLE_API_KEY` | Also supported; takes precedence if BOTH are set |

Get a free API key at: https://aistudio.google.com/apikey

> **Security note (June 2026):** Starting June 19, 2026, Google will discontinue support for unrestricted traffic keys. Apply API key restrictions in the Google Cloud Console before that date.

### Client Initialization

**Auto-detect from environment variable (recommended):**
```python
from google import genai

# Automatically reads GEMINI_API_KEY (or GOOGLE_API_KEY) from environment
client = genai.Client()
```

**Explicit API key:**
```python
from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
```

**Setting the env var in shell:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Setting in Python before client creation:**
```python
import os
os.environ["GEMINI_API_KEY"] = "your-api-key-here"

from google import genai
client = genai.Client()
```

### For Vertex AI (enterprise/production alternative)

```python
from google import genai

client = genai.Client(
    vertexai=True,
    project="your-gcp-project-id",
    location="us-central1",
)
```

---

## 4. Current Model IDs

> Use these exact strings in `model=` parameter of API calls.

### Gemini 2.5 Series — STABLE (Recommended for production)

| Model ID | Description | Vision Input | Best For |
|---|---|---|---|
| `gemini-2.5-flash` | Best price-performance | Yes | High-volume, low-latency tasks, **default choice for this pipeline** |
| `gemini-2.5-flash-lite` | Fastest, cheapest | Yes | Very high throughput |
| `gemini-2.5-pro` | Highest reasoning capability | Yes | Complex reasoning, accuracy-critical tasks |
| `gemini-2.5-flash-image` | Image generation (Nano Banana) | Yes | Image generation/editing only |

**Context windows for 2.5 Flash and 2.5 Pro:**
- Input token limit: **1,048,576 tokens** (1M+)
- Output token limit: **65,536 tokens**

### Gemini 3.x Series — PREVIEW (Use with caution in production)

| Model ID | Status | Notes |
|---|---|---|
| `gemini-3.1-flash-lite` | **Stable** | Frontier-class, cost-efficient |
| `gemini-3-flash-preview` | Preview | High performance |
| `gemini-3.1-pro-preview` | Preview | Advanced reasoning, agentic |
| `gemini-3.1-flash-lite-preview` | Preview | — |
| `gemini-3.1-flash-live-preview` | Preview | Real-time dialogue |
| `gemini-3-pro-preview` | **Shut down March 9, 2026** | Migrate to `gemini-3.1-pro-preview` |

### Specialized Models

| Model ID | Purpose |
|---|---|
| `gemini-2.5-computer-use-preview-10-2025` | UI automation |
| `gemini-embedding-2` | Multimodal embeddings |
| `gemini-robotics-er-1.6-preview` | Embodied reasoning |

### Recommendation for this pipeline

Use **`gemini-2.5-flash`** for the main image-to-3D pipeline (bounding box detection, object identification, dimension lookup). It is:
- Stable (not preview)
- Multimodal (supports image input natively)
- 10x cheaper than 2.5 Pro
- 1M token context window handles large images easily

Use **`gemini-2.5-pro`** only for tasks requiring highest accuracy (e.g., complex scene understanding or STEM reasoning about object geometry).

---

## 5. Vision / Image Input

All Gemini 2.5+ and 3.x models support multimodal image input. Supported formats: **PNG, JPEG, WebP, HEIC, HEIF**.

### Method 1: PIL Image (direct — SDK auto-converts)

```python
from google import genai
from PIL import Image

client = genai.Client()
image = Image.open("path/to/image.png")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image, "Describe this image in detail."]
)
print(response.text)
```

### Method 2: Raw bytes (most explicit control)

```python
from google import genai
from google.genai import types

client = genai.Client()

with open("path/to/image.jpg", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "What objects are visible in this image?"
    ]
)
print(response.text)
```

### Method 3: PIL image → bytes (for PIL images not loaded from file)

```python
from google import genai
from google.genai import types
from PIL import Image
import io

client = genai.Client()

pil_image = Image.open("image.png")  # or any PIL image in memory
buf = io.BytesIO()
pil_image.save(buf, format="PNG")
image_bytes = buf.getvalue()

image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[image_part, "Analyze this image."]
)
print(response.text)
```

### Method 4: URL (fetch and pass as bytes)

```python
import requests
from google import genai
from google.genai import types

client = genai.Client()

image_url = "https://example.com/product.jpg"
image_bytes = requests.get(image_url).content

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        "Identify all objects in this image."
    ]
)
print(response.text)
```

### Method 5: Files API (for large/reusable images — avoids re-uploading)

```python
from google import genai

client = genai.Client()

# Upload once, reuse across multiple requests
my_file = client.files.upload(file="path/to/large_image.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[my_file, "Describe this image."]
)
print(response.text)
```

### Method 6: Google Cloud Storage URI

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "What is in this image?",
        types.Part.from_uri(
            file_uri="gs://your-bucket/image.jpg",
            mime_type="image/jpeg"
        )
    ]
)
print(response.text)
```

### Multiple images in one request

```python
from google import genai
from google.genai import types

client = genai.Client()

with open("front.jpg", "rb") as f:
    front_bytes = f.read()
with open("side.jpg", "rb") as f:
    side_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=front_bytes, mime_type="image/jpeg"),
        types.Part.from_bytes(data=side_bytes, mime_type="image/jpeg"),
        "Compare these two views of the same object."
    ]
)
print(response.text)
```

---

## 6. Structured JSON Output with Pydantic

The SDK has two related parameters for structured output:
- **`response_schema`** — pass a Pydantic model class directly (SDK converts to JSON Schema internally)
- **`response_json_schema`** — pass a raw JSON Schema dict (or `MyModel.model_json_schema()`)
- **`response_mime_type="application/json"`** — always required alongside either schema param

### Basic example: Pydantic model as response_schema

```python
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional

client = genai.Client()

class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient.")
    quantity: str = Field(description="Quantity including units.")

class Recipe(BaseModel):
    recipe_name: str = Field(description="The name of the recipe.")
    prep_time_minutes: Optional[int] = Field(description="Prep time in minutes.")
    ingredients: List[Ingredient]
    instructions: List[str]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Give me a recipe for chocolate chip cookies.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe,          # Pass the class itself, not an instance
    ),
)

# Parse the validated response
recipe = Recipe.model_validate_json(response.text)
print(recipe.recipe_name)
print(recipe.ingredients)
```

### Alternative: pass JSON Schema dict directly

```python
from google import genai
from google.genai import types
from pydantic import BaseModel

client = genai.Client()

class ProductInfo(BaseModel):
    product_name: str
    category: str
    estimated_dimensions_cm: dict  # {"width": float, "height": float, "depth": float}
    material: str

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Identify the product in this image and estimate its dimensions.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=ProductInfo.model_json_schema(),
    ),
)

import json
data = json.loads(response.text)
product = ProductInfo(**data)
```

### List of Pydantic objects

```python
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
import json

client = genai.Client()

class DetectedObject(BaseModel):
    label: str
    confidence: float
    description: str

# Use list[DetectedObject] — note lowercase 'list', not List
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Identify all objects in this image.", image_part],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[DetectedObject],
    ),
)

objects = [DetectedObject(**obj) for obj in json.loads(response.text)]
```

### Important notes on Pydantic + Gemini

- The SDK converts Pydantic models to JSON Schema automatically when using `response_schema`
- As of November 2025, the API supports `additionalProperties` JSON Schema keyword
- Google added support for `anyOf`, `$ref`, and property ordering (keys returned in schema definition order)
- Use `Field(description=...)` on Pydantic fields — descriptions improve output quality significantly
- Nested Pydantic models work (as of SDK v1.x+, earlier versions had issues with nesting — Issue #60 on GitHub)

---

## 7. Bounding Box / Spatial Detection

### Coordinate Format — CONFIRMED CURRENT

The format is: **`box_2d = [ymin, xmin, ymax, xmax]`** normalized to **0–1000**.

This has been consistent since Gemini 1.5 and is still used in all 2.5 and 3.x model documentation as of 2026. The model is highly sensitive to this exact format — changing to `[xmin, ymin, xmax, ymax]` degrades performance significantly.

### How to convert to pixel coordinates

```python
# normalized_value / 1000 * image_dimension = pixel_coordinate
abs_y_min = int(box_2d[0] / 1000 * image_height)
abs_x_min = int(box_2d[1] / 1000 * image_width)
abs_y_max = int(box_2d[2] / 1000 * image_height)
abs_x_max = int(box_2d[3] / 1000 * image_width)
```

### Prompt format for bounding boxes

The model expects the format to be spelled out explicitly in the prompt. Recommended prompt:

```
Detect all prominent items in the image.
Return a JSON array where each item has:
  - "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000
  - "label": a descriptive string for the object

Limit to 25 objects. If an object appears multiple times, give each instance
a unique label based on its position or distinguishing characteristics
(e.g., "red cup on left", "red cup on right").
```

### System instruction (for persistent configuration)

```python
system_instruction = """
Return bounding boxes as a JSON array with labels. Never return masks or 
segmentation data unless explicitly requested. Limit to 25 objects maximum.
If an object is present multiple times, give each object a unique label 
according to its distinct characteristics (color, position, size).
Coordinates must be in [ymin, xmin, ymax, xmax] format normalized to 0-1000.
"""
```

### Complete production-ready bounding box example

```python
from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
)
from pydantic import BaseModel
from PIL import Image
import json

client = genai.Client()

class BoundingBox(BaseModel):
    box_2d: list[int]   # [ymin, xmin, ymax, xmax], values 0-1000
    label: str

def detect_objects(image_path: str) -> list[BoundingBox]:
    """Detect objects in an image and return bounding boxes."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_part = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    config = GenerateContentConfig(
        system_instruction=(
            "Return bounding boxes as an array with labels. Never return masks. "
            "Limit to 25 objects. If an object appears multiple times, give each "
            "a unique label by position or distinguishing characteristics."
        ),
        temperature=0.5,        # Non-zero temperature recommended for detection tasks
        safety_settings=[
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ],
        response_mime_type="application/json",
        response_schema=list[BoundingBox],
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image_part,
            "Detect all prominent items. box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.",
        ],
        config=config,
    )

    return response.parsed   # SDK auto-validates against response_schema when parsed attr used


def convert_to_pixels(
    box_2d: list[int],
    image_width: int,
    image_height: int
) -> tuple[int, int, int, int]:
    """Convert normalized [0-1000] box_2d to pixel coordinates."""
    ymin, xmin, ymax, xmax = box_2d
    return (
        int(xmin / 1000 * image_width),   # pixel_xmin
        int(ymin / 1000 * image_height),  # pixel_ymin
        int(xmax / 1000 * image_width),   # pixel_xmax
        int(ymax / 1000 * image_height),  # pixel_ymax
    )


# Usage
image_path = "product.jpg"
boxes = detect_objects(image_path)

img = Image.open(image_path)
width, height = img.size

for box in boxes:
    x1, y1, x2, y2 = convert_to_pixels(box.box_2d, width, height)
    print(f"{box.label}: ({x1}, {y1}) -> ({x2}, {y2})")
```

### Gemini 2.5+ Segmentation Masks (bonus capability)

Starting with Gemini 2.5, the model can also return segmentation masks alongside bounding boxes:

```python
# Each detected item can have:
# {
#   "box_2d": [y0, x0, y1, x1],   <- normalized 0-1000, [ymin,xmin,ymax,xmax]
#   "label": "object name",
#   "mask": "<base64_encoded_png>" <- segmentation mask within the bounding box
# }
```

Request masks explicitly in the prompt: `"Also provide a segmentation mask for each object as a base64-encoded PNG."`

### Tips for better bounding box results

- **Use temperature > 0** (0.5 is recommended) — zero temperature reduces detection diversity
- **Limit to 25 objects** in system instructions to prevent looping
- **Be specific** about what to detect in the prompt
- **Use descriptive labels** — ask for position-based labels for duplicate objects
- The model format `[ymin, xmin, ymax, xmax]` is fixed and critical — do not change the order

---

## 8. Web Search Grounding

Google Search grounding lets Gemini access real-time web content. For this pipeline it is useful for looking up product dimensions, materials, or specifications.

### How it works

1. Application sends prompt with `google_search` tool enabled
2. Model decides if a search would improve the answer
3. If yes, model auto-generates and executes search queries
4. Model processes results and returns grounded response with `groundingMetadata` (citations, sources)

### Basic grounded search

```python
from google import genai
from google.genai import types

client = genai.Client()

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the standard dimensions of an IKEA KALLAX shelf unit?",
    config=config,
)

print(response.text)

# Access grounding metadata (sources, citations)
if response.candidates[0].grounding_metadata:
    for source in response.candidates[0].grounding_metadata.grounding_chunks:
        print(f"Source: {source.web.title} — {source.web.uri}")
```

### Grounded search for product dimension lookup (pipeline use case)

```python
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Optional

client = genai.Client()

class ProductDimensions(BaseModel):
    product_name: str
    width_cm: Optional[float]
    height_cm: Optional[float]
    depth_cm: Optional[float]
    weight_kg: Optional[float]
    source_url: Optional[str]

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

def lookup_product_dimensions(product_description: str) -> str:
    """Use grounded search to find real-world product dimensions."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Search for the real-world physical dimensions (width, height, depth in cm) "
            f"and weight of this product: {product_description}. "
            f"Return the most commonly cited dimensions from manufacturer specs."
        ),
        config=types.GenerateContentConfig(
            tools=[grounding_tool],
        ),
    )
    return response.text


# Example usage
dims = lookup_product_dimensions("Apple MacBook Pro 14-inch M4")
print(dims)
```

### Grounding with structured output (combined)

> **Note:** As of early 2026, combining `tools=[grounding_tool]` with `response_schema` in the same request may not be fully supported (the model may ignore the schema when search grounding is active). Do grounded search in one call, then extract structured data in a second call:

```python
# Step 1: Grounded search to get raw text
search_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Find dimensions of: {product_name}",
    config=types.GenerateContentConfig(tools=[grounding_tool]),
)
raw_text = search_response.text

# Step 2: Extract structured data from raw text
extraction_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"Extract product dimensions from this text as JSON:\n\n{raw_text}",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ProductDimensions,
    ),
)
dimensions = ProductDimensions.model_validate_json(extraction_response.text)
```

### Billing note

Each search query the model executes is billed separately. If the model runs multiple queries for one prompt, each counts as a billable event. For budget-critical pipelines, consider caching grounded results.

### Model support for grounding

All Gemini 2.5 and 3.x models support Google Search grounding.

---

## 9. Rate Limits and Quotas

Rate limits are **per-project** (not per API key) and reset at **midnight Pacific time** for daily limits. Limits are enforced across RPM, TPM, and RPD simultaneously — hitting any one limit triggers a 429 error.

### Free Tier

| Model | RPM | TPM | RPD |
|---|---|---|---|
| `gemini-2.5-pro` | 5 | 250,000 | 100 |
| `gemini-2.5-flash` | 10 | 250,000 | 250 |
| `gemini-2.5-flash-lite` | 15 | 250,000 | 1,000 |

> **Warning:** Free tier quotas were reduced 50-80% in December 2025 due to abuse. Current free tier numbers are conservative. Always check your live limits in [Google AI Studio Rate Limits dashboard](https://aistudio.google.com/rate-limit).

### Paid Tiers

| Tier | Qualification | Flash RPM | Pro RPM |
|---|---|---|---|
| Tier 1 | Set up billing (≤$250/mo) | 300 | 150 |
| Tier 2 | $100+ spent + 3 days ($2,000 cap) | 1,000 | 1,000 |
| Tier 3 | $1,000+ spent + 30 days ($20K–$100K+) | 4,000+ | 4,000+ |

All paid tiers get **1,000,000 TPM** for both Flash and Pro.

### Pipeline implications (multiple calls per image)

For a pipeline making ~4 calls per image (bounding box detection + object ID + grounded dimension lookup + structured extraction), the effective throughput is:

**Free tier with gemini-2.5-flash (10 RPM):**
- Maximum: 10 / 4 = **~2.5 images/minute** on free tier
- Daily limit: 250 / 4 = **~62 images/day** on free tier

**Tier 1 with gemini-2.5-flash (300 RPM):**
- Maximum: 300 / 4 = **~75 images/minute**
- No hard RPD cap at Tier 1

**Recommended strategy for this pipeline:**
1. Use `gemini-2.5-flash` (not Pro) for all non-critical calls to stay within cheaper tier
2. Add exponential backoff on 429 errors
3. Cache grounded search results (product dimensions don't change often)
4. Batch image preprocessing before API calls to avoid idle time between requests
5. Upgrade to Tier 1 billing for any production workload exceeding a few dozen images

### Error handling for rate limits

```python
import time
import google.api_core.exceptions
from google import genai

client = genai.Client()

def generate_with_retry(model: str, contents, config=None, max_retries: int = 5):
    """Generate content with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + 1   # 2s, 5s, 9s, 17s, ...
            print(f"Rate limited. Retrying in {wait_time}s... (attempt {attempt+1})")
            time.sleep(wait_time)
```

---

## 10. Migration Notes: google-generativeai → google-genai

### Key architecture change

**Old (`google-generativeai`, deprecated):**
```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Tell me a story")
print(response.text)
```

**New (`google-genai`, current):**
```python
from google import genai

client = genai.Client()  # reads GEMINI_API_KEY from env
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tell me a story"
)
print(response.text)
```

### Migration summary table

| Feature | Old (google-generativeai) | New (google-genai) |
|---|---|---|
| Install | `pip install google-generativeai` | `pip install google-genai` |
| Import | `import google.generativeai as genai` | `from google import genai` |
| Auth | `genai.configure(api_key=...)` | `genai.Client(api_key=...)` or env var |
| Model init | `genai.GenerativeModel(model_name)` | `client.models.generate_content(model=...)` |
| Config | `generation_config={}` dict | `config=types.GenerateContentConfig(...)` |
| Structured output | `generation_config={"response_schema": ...}` | `config=types.GenerateContentConfig(response_schema=...)` |
| Files | `genai.upload_file(...)` | `client.files.upload(...)` |
| Chat | `model.start_chat()` | `client.chats.create(model=...)` |

### Install package for migration (PyPI)

```bash
pip uninstall google-generativeai
pip install google-genai
```

---

## Summary: Recommended Setup for Image-to-3D Pipeline

```python
# requirements.txt additions:
# google-genai>=2.1.0
# pydantic>=2.0.0

import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional
import io
from PIL import Image

# --- Client setup ---
# Set GEMINI_API_KEY env var before running
client = genai.Client()

# --- Models to use ---
VISION_MODEL = "gemini-2.5-flash"        # bounding boxes, object detection
REASONING_MODEL = "gemini-2.5-pro"       # complex scene understanding (if needed)
SEARCH_MODEL = "gemini-2.5-flash"        # grounded dimension lookup

# --- Typical call pattern for the pipeline ---

class BoundingBox(BaseModel):
    box_2d: list[int]   # [ymin, xmin, ymax, xmax] normalized 0-1000
    label: str

class ObjectDimensions(BaseModel):
    label: str
    estimated_real_width_cm: Optional[float]
    estimated_real_height_cm: Optional[float]
    estimated_real_depth_cm: Optional[float]
    material: Optional[str]

def analyze_image(pil_image: Image.Image) -> list[BoundingBox]:
    """Step 1: Detect objects and their bounding boxes."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")

    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            "Detect all prominent objects. box_2d=[ymin, xmin, ymax, xmax] normalized to 0-1000.",
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[BoundingBox],
            temperature=0.5,
        ),
    )
    return response.parsed


def lookup_dimensions(object_label: str) -> str:
    """Step 2: Ground-truth dimension lookup via Google Search."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    response = client.models.generate_content(
        model=SEARCH_MODEL,
        contents=f"What are the standard physical dimensions (cm) of: {object_label}?",
        config=types.GenerateContentConfig(tools=[grounding_tool]),
    )
    return response.text
```

---

## Sources

- [Gemini API Libraries (official)](https://ai.google.dev/gemini-api/docs/libraries)
- [google-genai PyPI](https://pypi.org/project/google-genai/)
- [googleapis/python-genai GitHub](https://github.com/googleapis/python-genai)
- [google-genai SDK Reference Docs](https://googleapis.github.io/python-genai/)
- [Deprecated SDK repo](https://github.com/google-gemini/deprecated-generative-ai-python)
- [Migration guide: generativeai → genai](https://ai.google.dev/gemini-api/docs/migrate)
- [Gemini Structured Output docs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Gemini Image Understanding docs](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Bounding Box Detection (Vertex AI)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection)
- [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search)
- [Gemini API Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Gemini Model IDs](https://ai.google.dev/gemini-api/docs/models)
- [Gemini 2.5 Flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash)
- [Gemini 2.5 Pro model page](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-pro)
- [Gemini API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- [Gemini API Key setup](https://ai.google.dev/gemini-api/docs/api-key)
- [Rate limits explained (YingTu)](https://yingtu.ai/en/blog/gemini-api-rate-limits-explained)
- [Bounding Boxes - Gemini by Example](https://geminibyexample.com/007-bounding-boxes/)
- [Simon Willison - Gemini bounding boxes](https://simonwillison.net/2024/Aug/26/gemini-bounding-box-visualization/)
