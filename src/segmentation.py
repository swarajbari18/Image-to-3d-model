"""
segmentation.py — SAM2 segmentation with Gemini feedback loop.
==============================================================
Architecture:
  1. gemini_client.parse_composite() gives bbox [ymin,xmin,ymax,xmax] × 1000
  2. gemini_box_to_sam() converts to [xmin,ymin,xmax,ymax] in pixels
     (SAM uses (x,y) not (y,x) — these are opposite conventions)
  3. set_image() is called ONCE → encoder runs once, predictions are cheap
  4. predict() is called for front AND back, reusing the cached embedding
  5. render_mask_overlay() creates a colored composite (blue=front, orange=back)
  6. gemini_client.validate_masks() inspects the overlay → returns issues
  7. Apply fixes (negative points, tightened bbox) → re-predict → repeat ≤ 3×
  8. On convergence (or max iterations): extract RGBA crops and return result

VRAM: SAM2 Hiera-Large = ~1.5 GB.  Loaded via ModelManager context manager.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.config import cfg
from src.gemini_client import CompositeParseResult, GeminiClient, ObjectView
from src.model_manager import manager


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def gemini_box_to_sam(
    box_2d: List[int], img_h: int, img_w: int
) -> np.ndarray:
    """
    Convert Gemini's [ymin, xmin, ymax, xmax] (normalized 0-1000)
    to SAM's [xmin, ymin, xmax, ymax] (pixel coordinates).

    Gemini encodes boxes as (y, x) — SAM expects (x, y).

    Args:
        box_2d: Gemini bounding box [ymin, xmin, ymax, xmax] in 0-1000.
        img_h:  Image height in pixels.
        img_w:  Image width in pixels.

    Returns:
        np.ndarray of shape (4,): [xmin_px, ymin_px, xmax_px, ymax_px]
    """
    ymin, xmin, ymax, xmax = box_2d
    return np.array([
        int(xmin / 1000 * img_w),   # xmin pixel
        int(ymin / 1000 * img_h),   # ymin pixel
        int(xmax / 1000 * img_w),   # xmax pixel
        int(ymax / 1000 * img_h),   # ymax pixel
    ])


# ---------------------------------------------------------------------------
# Mask overlay rendering (for Gemini validation)
# ---------------------------------------------------------------------------


def render_mask_overlay(
    image_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.5,
) -> Image.Image:
    """
    Render colored mask overlays on the composite image.

    Color convention (matches Gemini validation prompt):
      - Front panel → blue  (0, 120, 255)
      - Back panel  → orange (255, 80, 0)

    Args:
        image_rgb: np.ndarray (H, W, 3) uint8 — the original composite.
        masks:     Dict mapping view label (str) → binary mask (H, W) bool.
        alpha:     Blend factor in [0, 1].  0 = no overlay, 1 = solid color.

    Returns:
        PIL Image (RGB) with colored overlays blended in.
    """
    colors = {
        "front": np.array([0, 120, 255]),
        "back":  np.array([255, 80, 0]),
    }
    overlay = image_rgb.astype(float)
    for label, mask in masks.items():
        color = colors.get(label, np.array([128, 128, 128]))
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )
    return Image.fromarray(overlay.astype(np.uint8))


# ---------------------------------------------------------------------------
# RGBA crop extraction
# ---------------------------------------------------------------------------


def extract_rgba_crop(image_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Extract a tight RGBA crop of the masked region.

    The alpha channel is derived directly from the binary mask (255 = visible,
    0 = transparent).  The crop is tight around the bounding box of the mask.

    Args:
        image_rgb: np.ndarray (H, W, 3) uint8.
        mask:      np.ndarray (H, W) bool — True = part of object.

    Returns:
        PIL Image in RGBA mode, cropped to the mask bounding box.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return Image.fromarray(rgba, "RGBA")  # empty mask — return full

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return Image.fromarray(rgba[rmin : rmax + 1, cmin : cmax + 1], "RGBA")


# ---------------------------------------------------------------------------
# Dataclass for segmentation results
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """
    Output of the full SAM2 + Gemini segmentation pipeline.

    Attributes:
        front_mask:      Binary mask for the front panel (H, W) bool.
        back_mask_modal: Binary mask for the VISIBLE portion of the back panel.
                         Has a 'bite' out of it at the occlusion boundary.
        front_rgba:      RGBA PIL Image crop of the front panel.
        back_rgba_modal: RGBA PIL Image crop of the visible back panel portion.
        overlay_image:   Colored overlay PIL Image for inspection / debugging.
        iterations_used: Number of Gemini feedback iterations executed.
        validation_notes: Final Gemini validation result summary.
    """

    front_mask: np.ndarray
    back_mask_modal: np.ndarray
    front_rgba: Image.Image
    back_rgba_modal: Image.Image
    overlay_image: Image.Image
    iterations_used: int
    validation_notes: str


# ---------------------------------------------------------------------------
# SAM2 loader (used by ModelManager)
# ---------------------------------------------------------------------------


def _build_sam2_loader(checkpoint: str, config_name: str, device: str):
    """Return a zero-argument lambda that builds and returns the SAM2 predictor."""

    def _load():
        import os
        import sam2 as _sam2_module
        from hydra import initialize_config_dir  # type: ignore[import]
        from hydra.core.global_hydra import GlobalHydra  # type: ignore[import]
        from sam2.build_sam import build_sam2  # type: ignore[import]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import]

        sam2_root = os.path.dirname(_sam2_module.__file__)

        # config_name may be "sam2.1/sam2.1_hiera_l.yaml" (subdir/filename) or
        # a plain filename.  Split so we can point Hydra at the subdir via a
        # real filesystem path — pkg://sam2 cannot resolve "sam2.1/" because
        # the dot makes it an invalid Python package identifier.
        if "/" in config_name:
            subdir, fname = config_name.split("/", 1)
            config_dir = os.path.abspath(os.path.join(sam2_root, "configs", subdir))
        else:
            fname = config_name
            config_dir = os.path.abspath(os.path.join(sam2_root, "configs"))

        GlobalHydra.instance().clear()
        print(f"[segmentation] Building SAM2 from {checkpoint}")
        with initialize_config_dir(config_dir=config_dir, job_name="sam2_load"):
            model = build_sam2(config_file=fname, ckpt_path=checkpoint, device=device)

        return SAM2ImagePredictor(model)

    return _load


# ---------------------------------------------------------------------------
# Core segmentation function
# ---------------------------------------------------------------------------


def run_segmentation(
    composite_path: str | Path,
    parse_result: CompositeParseResult,
    gemini: Optional[GeminiClient] = None,
    max_feedback_iters: int = 3,
    overlay_save_path: Optional[str | Path] = None,
) -> SegmentationResult:
    """
    Segment front and back panels from a marketing composite image.

    Steps:
      1. Load SAM2 Hiera-Large via ModelManager (auto-unloaded on exit).
      2. set_image() once — encodes the image at cost, reused for all predict() calls.
      3. Predict front mask and back mask from Gemini-provided bounding boxes.
      4. Render colored overlay → Gemini validate → apply fixes → repeat ≤ max_feedback_iters.
      5. Extract RGBA crops from best masks.
      6. Unload SAM2, return SegmentationResult.

    Args:
        composite_path:    Path to the raw composite image (JPEG/PNG).
        parse_result:      Result of GeminiClient.parse_composite().
        gemini:            Optional GeminiClient instance (creates one if None).
        max_feedback_iters: Max Gemini-SAM feedback iterations before accepting.
        overlay_save_path: If provided, save the final overlay image here.

    Returns:
        SegmentationResult containing masks and RGBA crops.

    Raises:
        ValueError: If parse_result does not contain both 'front' and 'back' views.
    """
    gemini = gemini or GeminiClient()

    # --- Validate parse result ---
    view_map: Dict[str, ObjectView] = {v.label.lower(): v for v in parse_result.views}
    if "front" not in view_map or "back" not in view_map:
        raise ValueError(
            f"parse_result must contain 'front' and 'back' views. "
            f"Got: {list(view_map.keys())}"
        )

    # --- Load composite image ---
    composite_pil = Image.open(composite_path).convert("RGB")
    composite_rgb = np.array(composite_pil)
    img_h, img_w = composite_rgb.shape[:2]

    # --- SAM2 boxes ---
    sam_box_front = gemini_box_to_sam(view_map["front"].box_2d, img_h, img_w)
    sam_box_back = gemini_box_to_sam(view_map["back"].box_2d, img_h, img_w)

    # --- Extra negative points from prior iterations (mutable) ---
    neg_points_front: List[List[int]] = []
    neg_points_back: List[List[int]] = []

    best_masks: Dict[str, np.ndarray] = {}
    iterations_used = 0
    validation_notes = ""

    # --- SAM2 load + predict loop ---
    loader = _build_sam2_loader(
        checkpoint=str(cfg.sam2_checkpoint_path),
        config_name=cfg.sam2_config_name,
        device=cfg.DEVICE,
    )

    with manager.use("sam2", loader) as predictor:
        predictor.set_image(composite_rgb)

        for iteration in range(max_feedback_iters + 1):
            # Build predict kwargs
            front_kwargs: dict = dict(box=sam_box_front, multimask_output=True)
            back_kwargs: dict = dict(box=sam_box_back, multimask_output=True)

            if neg_points_front:
                arr = np.array(neg_points_front)
                front_kwargs["point_coords"] = arr
                front_kwargs["point_labels"] = np.zeros(len(arr), dtype=int)
            if neg_points_back:
                arr = np.array(neg_points_back)
                back_kwargs["point_coords"] = arr
                back_kwargs["point_labels"] = np.zeros(len(arr), dtype=int)

            import torch  # deferred — SAM2 needs torch on the right device

            with torch.inference_mode():
                front_masks, front_scores, _ = predictor.predict(**front_kwargs)
                back_masks, back_scores, _ = predictor.predict(**back_kwargs)

            front_mask = front_masks[np.argmax(front_scores)]  # (H, W) bool
            back_mask = back_masks[np.argmax(back_scores)]

            best_masks = {"front": front_mask, "back": back_mask}

            if iteration == max_feedback_iters:
                validation_notes = (
                    f"Max iterations ({max_feedback_iters}) reached — using best masks."
                )
                break

            # --- Render overlay and ask Gemini to validate ---
            overlay_img = render_mask_overlay(composite_rgb, best_masks)
            overlay_path = cfg.OUTPUT_DIR / f"overlay_iter{iteration}.png"
            overlay_img.save(str(overlay_path))

            val_result = gemini.validate_masks(
                overlay_image_path=overlay_path,
                view_labels=["front", "back"],
            )
            iterations_used = iteration + 1

            if val_result.validation_passed:
                validation_notes = (
                    f"Gemini validation passed at iteration {iteration + 1}. "
                    f"Quality: {val_result.overall_quality}"
                )
                break

            # --- Apply Gemini-suggested fixes ---
            for issue in val_result.issues:
                target = issue.mask.lower()
                if issue.add_negative_points:
                    if target == "front":
                        neg_points_front.extend(issue.add_negative_points)
                    else:
                        neg_points_back.extend(issue.add_negative_points)
                if issue.tighten_bbox:
                    new_box = gemini_box_to_sam(issue.tighten_bbox, img_h, img_w)
                    if target == "front":
                        sam_box_front = new_box
                    else:
                        sam_box_back = new_box

            print(
                f"[segmentation] Iteration {iteration + 1}: "
                f"{len(val_result.issues)} issue(s) found — retrying with fixes."
            )

    # --- Extract final crops ---
    front_rgba = extract_rgba_crop(composite_rgb, best_masks["front"])
    back_rgba = extract_rgba_crop(composite_rgb, best_masks["back"])
    final_overlay = render_mask_overlay(composite_rgb, best_masks)

    if overlay_save_path:
        final_overlay.save(str(overlay_save_path))

    return SegmentationResult(
        front_mask=best_masks["front"],
        back_mask_modal=best_masks["back"],
        front_rgba=front_rgba,
        back_rgba_modal=back_rgba,
        overlay_image=final_overlay,
        iterations_used=iterations_used,
        validation_notes=validation_notes,
    )
