"""
amodal_completion.py — pix2gestalt amodal completion + SD inpainting fallback.
===============================================================================
Architecture:
  SAM2 produces a MODAL mask of the back panel — it only covers VISIBLE pixels.
  Because the front panel occludes ~20% of the back panel's upper-left corner,
  the back-panel mask has a phone-shaped 'bite' out of it.

  Feeding this incomplete mask directly to Hunyuan3D would produce a 3-D mesh
  with a literal concave notch at the occlusion boundary.

  pix2gestalt solves this: given the whole composite image + the modal mask,
  it generates an AMODAL completion — the full back panel as if the front
  panel were removed.

  Strategy:
    1. Run pix2gestalt with 4 different random seeds → 4 candidate completions.
    2. Use GeminiClient.validate_amodal() to score each candidate (1-10).
    3. Pick the highest-scoring candidate.
    4. If the winner scores < 7: apply targeted SD inpainting on the worst region
       using Gemini's inpaint_suggestion prompt.

  VRAM:
    pix2gestalt: ~10 GB (must assert this before loading)
    SD inpainting: ~3.5 GB with CPU offload (loaded separately after pix2gestalt unloads)

  Imports for pix2gestalt work via sys.path.insert because it is a cloned repo
  with no pip package.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from src.config import cfg
from src.gemini_client import AmodalValidationResult, GeminiClient, ProductDimensions
from src.model_manager import manager


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AmodalResult:
    """
    Output of the amodal completion stage.

    Attributes:
        best_candidate:     PIL Image (RGBA) — the best completed back panel.
        all_candidates:     All 4 pix2gestalt outputs (useful for side-by-side display).
        validation_result:  Gemini's score + feature breakdown for the winner.
        seed_used:          Which seed (1-4) produced the winning candidate.
        sd_inpaint_applied: True if SD inpainting was used to fix the winner.
        dimensions:         Real-world dimensions fetched by Gemini grounded search.
    """

    best_candidate: Image.Image
    all_candidates: List[Image.Image]
    validation_result: AmodalValidationResult
    seed_used: int
    sd_inpaint_applied: bool
    dimensions: Optional[ProductDimensions] = None


# ---------------------------------------------------------------------------
# pix2gestalt runner
# ---------------------------------------------------------------------------


def _run_pix2gestalt_single(
    composite_pil: Image.Image,
    modal_mask: np.ndarray,
    ckpt_path: str,
    seed: int,
    steps: int = 50,
    scale: float = 3.0,
) -> Image.Image:
    """
    Run one pix2gestalt inference pass and return the completed RGBA image.

    pix2gestalt is imported from the cloned repo (no pip package).
    sys.path.insert is done here rather than at module level so that
    config.py and model_manager.py can be imported without the repo present.

    Args:
        composite_pil: Full composite RGB PIL Image (context for completion).
        modal_mask:    np.ndarray (H, W) bool — SAM modal mask of back panel.
        ckpt_path:     Path to epoch=000005.ckpt (or epoch=000010.ckpt).
        seed:          Random seed for diffusion sampling (try 1, 2, 3, 4).
        steps:         Denoising steps.  50 = full quality, 20 = faster.
        scale:         CFG guidance scale.  3.0 is the paper's default.

    Returns:
        PIL Image (RGBA) — the completed back panel on white background.
    """
    repo = str(cfg.PIX2GESTALT_REPO_DIR)   # /content/pix2gestalt (repo root)
    # The import is `from pix2gestalt.inference import run_pix2gestalt`
    # which resolves because pix2gestalt/ is a subdir of the repo root.
    # Do NOT insert the inner pix2gestalt/ dir — that would break the import.
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from pix2gestalt.inference import run_pix2gestalt  # type: ignore[import]

    result_rgba: Image.Image = run_pix2gestalt(
        ckpt_path=ckpt_path,
        whole_image=composite_pil,      # PIL Image (RGB) — provides scene context
        mask=modal_mask,                # np.ndarray (H, W) bool
        seed=seed,
        steps=steps,
        scale=scale,
        device=cfg.DEVICE,
    )
    return result_rgba


# ---------------------------------------------------------------------------
# SD inpainting fallback
# ---------------------------------------------------------------------------


def _apply_sd_inpainting(
    base_image: Image.Image,
    prompt: str,
    object_label: str,
) -> Image.Image:
    """
    Apply Stable Diffusion inpainting to the worst region of a pix2gestalt completion.

    Uses the public mirror 'stable-diffusion-v1-5/stable-diffusion-inpainting'
    (no HF token required).  Runs with CPU offload to fit in ~3.5 GB VRAM.

    The region to inpaint is determined from Gemini's inpaint_suggestion prompt
    which describes what to fix.  We create a simple center-weighted mask as a
    heuristic — the previously occluded region is typically in the upper-left.

    Args:
        base_image:   PIL Image (RGB or RGBA) from pix2gestalt.
        prompt:       Gemini's suggested inpainting prompt (fix description).
        object_label: Used to enrich the negative prompt.

    Returns:
        PIL Image (RGB) with targeted inpainting applied.
    """
    from diffusers import StableDiffusionInpaintPipeline  # type: ignore[import]

    print("[amodal] Applying SD inpainting fallback...")

    def _load_sd():
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            cfg.SD_INPAINT_MODEL,
            torch_dtype=cfg.TORCH_DTYPE,
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing()
        return pipe

    with manager.use("sd_inpaint", _load_sd) as pipe:
        img_rgb = base_image.convert("RGB")
        w, h = img_rgb.size

        # Heuristic inpaint mask: upper-left quadrant (the typical occlusion zone)
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        mask_arr[: h // 2, : w // 2] = 255  # upper-left quarter = inpaint region
        mask_pil = Image.fromarray(mask_arr, "L")

        result = pipe(
            prompt=prompt,
            negative_prompt=f"artifacts, blurry, wrong details, distorted, low quality, {object_label} front face on back",
            image=img_rgb,
            mask_image=mask_pil,
            strength=0.80,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_amodal(
    composite_pil: Image.Image,
    back_modal_mask: np.ndarray,
    object_label: str,
    gemini: Optional[GeminiClient] = None,
    seeds: List[int] = None,
    steps: int = 50,
    apply_inpaint_if_needed: bool = True,
) -> AmodalResult:
    """
    Complete the occluded portion of the back panel using pix2gestalt.

    Runs pix2gestalt with multiple seeds, scores each with Gemini, picks the best.
    Falls back to SD inpainting if the best candidate scores < 7.

    Args:
        composite_pil:         Full composite PIL Image (RGB) — context for pix2gestalt.
        back_modal_mask:       SAM modal mask (H, W) bool — visible back-panel pixels.
        object_label:          Product description for Gemini validation.
        gemini:                Optional GeminiClient (creates one if None).
        seeds:                 List of random seeds to try. Default: [1, 2, 3, 4].
        steps:                 pix2gestalt denoising steps.  Reduce to 20 if VRAM is tight.
        apply_inpaint_if_needed: If True, run SD inpainting when best score < 7.

    Returns:
        AmodalResult with best_candidate RGBA PIL Image.
    """
    seeds = seeds or [1, 2, 3, 4]
    gemini = gemini or GeminiClient()
    ckpt_path = str(cfg.pix2gestalt_ckpt_path)

    # --- Pre-flight VRAM check ---
    manager.assert_vram_available(required_mb=9_500)

    all_candidates: List[Image.Image] = []
    all_scores: List[int] = []
    all_validations: List[AmodalValidationResult] = []

    print(f"[amodal] Running pix2gestalt with {len(seeds)} seeds ({steps} steps each)...")

    for seed in seeds:
        print(f"[amodal]   Seed {seed}...")
        # pix2gestalt loads its own models internally — we can't wrap in ModelManager
        # We call gc/empty_cache between seeds to keep peak VRAM low
        try:
            candidate = _run_pix2gestalt_single(
                composite_pil=composite_pil,
                modal_mask=back_modal_mask,
                ckpt_path=ckpt_path,
                seed=seed,
                steps=steps,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[amodal] OOM on seed {seed} — skipping. Try steps=20.")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

        all_candidates.append(candidate)

        # Save candidate for Gemini inspection
        candidate_path = cfg.OUTPUT_DIR / f"amodal_seed{seed}.png"
        candidate.save(str(candidate_path))

        # Gemini validation
        val = gemini.validate_amodal(
            completed_image_path=candidate_path,
            object_label=object_label,
        )
        all_scores.append(val.score)
        all_validations.append(val)
        print(f"[amodal]   Seed {seed} → score {val.score}/10")

        gc.collect()
        torch.cuda.empty_cache()

    if not all_candidates:
        raise RuntimeError("[amodal] All pix2gestalt seeds failed — likely OOM.")

    # --- Pick best candidate ---
    best_idx = int(np.argmax(all_scores))
    best_score = all_scores[best_idx]
    best_candidate = all_candidates[best_idx]
    best_validation = all_validations[best_idx]
    best_seed = seeds[best_idx]

    print(f"[amodal] Best candidate: seed={best_seed}, score={best_score}/10")

    # --- SD inpainting fallback ---
    sd_applied = False
    if apply_inpaint_if_needed and not best_validation.passed:
        if best_validation.inpaint_suggestion:
            print(
                f"[amodal] Score {best_score} < 7 — applying SD inpainting. "
                f"Prompt: '{best_validation.inpaint_suggestion}'"
            )
            best_candidate = _apply_sd_inpainting(
                base_image=best_candidate,
                prompt=best_validation.inpaint_suggestion,
                object_label=object_label,
            )
            sd_applied = True
        else:
            print(
                f"[amodal] Score {best_score} — no inpaint suggestion from Gemini, "
                f"proceeding with best available."
            )

    # Save final amodal result
    final_path = cfg.OUTPUT_DIR / "amodal_best.png"
    best_candidate.save(str(final_path))
    print(f"[amodal] Best amodal completion saved to {final_path}")

    return AmodalResult(
        best_candidate=best_candidate,
        all_candidates=all_candidates,
        validation_result=best_validation,
        seed_used=best_seed,
        sd_inpaint_applied=sd_applied,
    )
