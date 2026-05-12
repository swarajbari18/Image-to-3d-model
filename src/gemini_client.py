"""
gemini_client.py — All Gemini 2.5 Flash API interactions.
==========================================================
Architecture:
  Uses the NEW 'google-genai' SDK (v2.1+).  The deprecated
  'google-generativeai' package reached EOL November 2025.

  SDK pattern:
      from google import genai
      client = genai.Client()           # reads GEMINI_API_KEY from env

  Four public methods (each is a separate API call or multi-call sequence):

  1. parse_composite(image_path)
     Gemini sees the raw marketing composite and returns structured JSON:
     bounding boxes for front+back views, occlusion metadata.
     Bbox format: [ymin, xmin, ymax, xmax] normalized 0-1000.

  2. validate_masks(overlay_image_path, view_labels)
     Gemini sees the SAM mask overlay and returns structured validation
     JSON — issues + suggested fixes (negative points, bbox adjustments).

  3. validate_amodal(completed_image_path, object_label)
     Two-call approach:
       Call 1 → generate domain-appropriate validation criteria
       Call 2 → validate the pix2gestalt completion against those criteria

  4. lookup_dimensions(object_label)
     Two-call approach:
       Call 1 → grounded Google Search for product specs (raw text)
       Call 2 → extract structured ProductDimensions from the raw text
     (search + schema in one call is not reliably supported as of 2026)

Rate-limit note: free tier is 10 RPM for gemini-2.5-flash.  This module
adds exponential back-off on 429 errors.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import List, Optional

from google import genai  # type: ignore[import]
from google.genai import types  # type: ignore[import]
from pydantic import BaseModel, Field

from src.config import cfg

# ---------------------------------------------------------------------------
# Pydantic models — define the exact JSON shapes Gemini must return
# ---------------------------------------------------------------------------


class ObjectView(BaseModel):
    """One detected view (front or back) within the composite image."""

    label: str = Field(
        description="Short semantic label, e.g. 'front' or 'back'."
    )
    box_2d: List[int] = Field(
        description=(
            "Bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000. "
            "Values must be integers in [0, 1000]."
        )
    )
    is_occluded: bool = Field(
        description="True if any portion of this view is hidden behind another object."
    )
    occlusion_description: Optional[str] = Field(
        default=None,
        description=(
            "If is_occluded=True, describe which part is hidden and what is occluding it."
        ),
    )


class CompositeParseResult(BaseModel):
    """
    Result of parsing a marketing composite image.

    The composite is expected to contain exactly two views of the same product
    (front panel and back panel), optionally side-by-side or overlapping.
    """

    product_label: str = Field(
        description="Short description of the product, e.g. 'Samsung Galaxy S25 Ultra'."
    )
    views: List[ObjectView] = Field(
        description="List of detected product views.  Expect exactly 2: front and back."
    )
    composite_width: int = Field(
        description="Estimated pixel width of the full composite image."
    )
    composite_height: int = Field(
        description="Estimated pixel height of the full composite image."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any other observations relevant to 3D reconstruction.",
    )


class MaskIssue(BaseModel):
    """A single issue found in a SAM segmentation mask."""

    mask: str = Field(description="Which mask has the issue: 'front' or 'back'.")
    issue_type: str = Field(
        description=(
            "One of: shadow_included | mask_bleed | mask_incomplete | "
            "label_swap | spurious_region | occlusion_boundary_wrong"
        )
    )
    description: str = Field(
        description="Human-readable explanation of the issue."
    )
    add_negative_points: Optional[List[List[int]]] = Field(
        default=None,
        description=(
            "List of [x, y] pixel coordinates to add as SAM negative-prompt points "
            "to exclude regions bleeding into the mask."
        ),
    )
    tighten_bbox: Optional[List[int]] = Field(
        default=None,
        description=(
            "Adjusted [ymin, xmin, ymax, xmax] bounding box in 0-1000 normalized coords. "
            "Only set if the fix requires a tighter box."
        ),
    )


class MaskValidationResult(BaseModel):
    """Validation result for the SAM mask overlay."""

    validation_passed: bool = Field(
        description="True if both masks correctly cover their respective views."
    )
    issues: List[MaskIssue] = Field(
        default_factory=list,
        description="List of issues found.  Empty if validation_passed=True.",
    )
    overall_quality: str = Field(
        description="One of: excellent | good | acceptable | poor"
    )


class AmodalCriteria(BaseModel):
    """Domain-specific validation criteria for amodal completion."""

    expected_features: List[str] = Field(
        description=(
            "List of features that the completed back panel should show, "
            "e.g. 'camera island with 4 lenses', 'smooth matte black finish'."
        )
    )
    forbidden_features: List[str] = Field(
        description=(
            "List of features that must NOT appear in the completion, "
            "e.g. 'USB-C port on the back surface'."
        )
    )


class AmodalValidationResult(BaseModel):
    """Result of validating a pix2gestalt amodal completion."""

    score: int = Field(
        description="Quality score 1-10.  ≥7 is considered acceptable for 3D reconstruction."
    )
    passed: bool = Field(
        description="True if score >= 7 and no critical issues."
    )
    present_features: List[str] = Field(
        description="Expected features that are correctly present."
    )
    missing_features: List[str] = Field(
        description="Expected features that are missing or incorrect."
    )
    hallucinated_features: List[str] = Field(
        description="Features present that should NOT be there."
    )
    inpaint_suggestion: Optional[str] = Field(
        default=None,
        description=(
            "If passed=False, a targeted SD-inpainting prompt to fix the worst issue."
        ),
    )


class ProductDimensions(BaseModel):
    """Real-world physical dimensions of the product, in millimeters."""

    product_name: str = Field(description="Full product name from specs.")
    width_mm: Optional[float] = Field(
        default=None, description="Width in millimeters (shorter horizontal dimension)."
    )
    height_mm: Optional[float] = Field(
        default=None, description="Height in millimeters (longer vertical dimension)."
    )
    depth_mm: Optional[float] = Field(
        default=None, description="Depth / thickness in millimeters."
    )
    source_note: Optional[str] = Field(
        default=None, description="URL or source cited for these dimensions."
    )


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


class GeminiClient:
    """
    Wrapper around google-genai v2.1+ for all pipeline Gemini calls.

    All calls include exponential back-off on HTTP 429 (rate-limited).
    The client reads GEMINI_API_KEY automatically from the environment
    (set by config.py during import).
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._client = genai.Client()
        self.model = model or cfg.GEMINI_MODEL

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_part(self, image_path: str | Path) -> types.Part:
        """Load an image from disk and wrap it in a Gemini Part."""
        path = Path(image_path)
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".webp": "image/webp"}
        mime = mime_map.get(path.suffix.lower(), "image/jpeg")
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)

    def _pil_part(self, pil_image, fmt: str = "PNG") -> types.Part:
        """Convert a PIL Image to a Gemini Part."""
        buf = io.BytesIO()
        pil_image.save(buf, format=fmt)
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        return types.Part.from_bytes(data=buf.getvalue(), mime_type=mime)

    def _generate(
        self,
        contents: list,
        config: types.GenerateContentConfig,
        max_retries: int = 5,
    ) -> genai.types.GenerateContentResponse:
        """
        Call generate_content with exponential back-off on rate-limit errors.

        Waits:  2s, 5s, 9s, 17s, 33s between retries.
        """
        import google.api_core.exceptions  # type: ignore[import]

        for attempt in range(max_retries):
            try:
                return self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except google.api_core.exceptions.ResourceExhausted:
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + 1
                print(f"[Gemini] Rate-limited — retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def parse_composite(self, image_path: str | Path) -> CompositeParseResult:
        """
        Parse a marketing composite image and return structured bounding boxes.

        Bbox format returned by Gemini: [ymin, xmin, ymax, xmax] normalized 0-1000.
        Use segmentation.gemini_box_to_sam() to convert to SAM pixel coords.

        Args:
            image_path: Path to the composite JPEG/PNG image.

        Returns:
            CompositeParseResult with front+back view bboxes and occlusion info.
        """
        img_part = self._image_part(image_path)

        prompt = (
            "This image is a product marketing composite showing the same smartphone "
            "from multiple angles — typically front panel and back panel side-by-side "
            "or overlapping.\n\n"
            "Your task:\n"
            "1. Identify EACH distinct view of the device (front face / back face).\n"
            "2. For each view, output a bounding box in [ymin, xmin, ymax, xmax] format, "
            "   values normalized to 0-1000.\n"
            "3. Note whether any view is partially occluded by another view and describe "
            "   which region is hidden.\n\n"
            "Use label 'front' for the display/screen side, 'back' for the camera/rear side.\n"
            "Be conservative with bounding boxes — include a small margin but exclude "
            "drop shadows and reflections."
        )

        config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=CompositeParseResult,
        )

        response = self._generate([img_part, prompt], config)
        result = CompositeParseResult.model_validate_json(response.text)
        return result

    def validate_masks(
        self,
        overlay_image_path: str | Path,
        view_labels: List[str],
    ) -> MaskValidationResult:
        """
        Validate SAM segmentation masks by inspecting a colored overlay image.

        The overlay should show:
          - Front mask in blue  (RGB 0, 120, 255)
          - Back mask in orange (RGB 255, 80, 0)

        Gemini answers an easier question than the original parse: does the
        colored region correctly cover the intended view?

        Args:
            overlay_image_path: Path to the mask-overlay image.
            view_labels:        List of view names shown, e.g. ['front', 'back'].

        Returns:
            MaskValidationResult — validation_passed + list of issues with fix hints.
        """
        img_part = self._image_part(overlay_image_path)
        label_str = " and ".join(view_labels)

        prompt = (
            f"This image shows a smartphone marketing composite with colored "
            f"segmentation masks overlaid:\n"
            f"  - Blue region  = front display panel mask\n"
            f"  - Orange region = back camera panel mask\n\n"
            f"Please check EACH mask for the following issues:\n"
            f"  1. shadow_included: does the mask extend into a soft gradient drop-shadow?\n"
            f"  2. mask_bleed: does the front mask bleed into the back panel or vice versa?\n"
            f"  3. mask_incomplete: is part of the visible panel NOT covered by its mask?\n"
            f"  4. label_swap: is the blue region covering the back panel (or vice versa)?\n"
            f"  5. spurious_region: is there a third colored region from a reflection?\n"
            f"  6. occlusion_boundary_wrong: at the overlap boundary, is the cut incorrect?\n\n"
            f"For each issue found, suggest a fix:\n"
            f"  - add_negative_points: [[x,y], ...] pixel coords inside the spurious area\n"
            f"  - tighten_bbox: new [ymin,xmin,ymax,xmax] in 0-1000 normalized coords\n\n"
            f"Set validation_passed=true ONLY if both masks are correct with no issues."
        )

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=MaskValidationResult,
        )

        response = self._generate([img_part, prompt], config)
        return MaskValidationResult.model_validate_json(response.text)

    def validate_amodal(
        self,
        completed_image_path: str | Path,
        object_label: str,
    ) -> AmodalValidationResult:
        """
        Validate a pix2gestalt amodal completion in two API calls.

        Call 1: Ask Gemini to generate domain-appropriate validation criteria
                (what features should the completed back panel have?).
        Call 2: Show the completed image and validate against those criteria.

        Args:
            completed_image_path: Path to the pix2gestalt RGBA output image.
            object_label:         Product description, e.g. 'Samsung Galaxy S25 Ultra back panel'.

        Returns:
            AmodalValidationResult with score (1-10) and per-feature breakdown.
        """
        # Call 1 — generate criteria (text only, no image)
        criteria_prompt = (
            f"You are a product quality assessor for 3D model generation.\n"
            f"The object is: {object_label}\n\n"
            f"List the features that a correctly completed image of the FULL (unoccluded) "
            f"back panel of this device should have.  Also list features that should NOT "
            f"appear (forbidden features such as screen/display elements on the back, "
            f"wrong port placement, etc.).\n\n"
            f"Be specific and based on real product knowledge."
        )
        criteria_config = types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=AmodalCriteria,
        )
        criteria_resp = self._generate([criteria_prompt], criteria_config)
        criteria = AmodalCriteria.model_validate_json(criteria_resp.text)

        # Call 2 — validate image against criteria
        img_part = self._image_part(completed_image_path)
        validation_prompt = (
            f"Evaluate this image of the completed back panel of a {object_label}.\n\n"
            f"Expected features to check for:\n"
            + "\n".join(f"  - {f}" for f in criteria.expected_features)
            + "\n\nForbidden features (must NOT be present):\n"
            + "\n".join(f"  - {f}" for f in criteria.forbidden_features)
            + "\n\n"
            f"Give a quality score from 1 (terrible) to 10 (perfect).  "
            f"Set passed=true if score >= 7 and no critical issues.  "
            f"If passed=false, provide a targeted inpaint_suggestion prompt to fix "
            f"the single most important issue."
        )
        validation_config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=AmodalValidationResult,
        )
        validation_resp = self._generate([img_part, validation_prompt], validation_config)
        return AmodalValidationResult.model_validate_json(validation_resp.text)

    def lookup_dimensions(self, object_label: str) -> ProductDimensions:
        """
        Look up real-world product dimensions using Google Search grounding.

        Implemented as two API calls because combining grounding + response_schema
        in a single call is unreliable as of 2026:

        Call 1 (grounded):  Google Search → raw text with dimensions
        Call 2 (structured): Extract ProductDimensions from raw text

        Args:
            object_label: Product name, e.g. 'Samsung Galaxy S25 Ultra'.

        Returns:
            ProductDimensions with width_mm, height_mm, depth_mm populated.
        """
        # Call 1 — grounded search
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        search_config = types.GenerateContentConfig(tools=[grounding_tool])
        search_prompt = (
            f"Find the official physical dimensions (width, height, thickness/depth) "
            f"in millimeters for: {object_label}.  "
            f"Return the manufacturer's specification values.  "
            f"Include the source URL if available."
        )
        search_resp = self._generate([search_prompt], search_config)
        raw_text = search_resp.text

        # Call 2 — structured extraction
        extract_prompt = (
            f"Extract the product dimensions from the following text and return "
            f"them as structured JSON.\n\nText:\n{raw_text}"
        )
        extract_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ProductDimensions,
        )
        extract_resp = self._generate([extract_prompt], extract_config)
        return ProductDimensions.model_validate_json(extract_resp.text)
