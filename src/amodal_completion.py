"""
amodal_completion.py — SynergyAmodal amodal completion + Gemini feedback loop.
===============================================================================
Architecture:
  SAM2 produces a MODAL mask of the back panel — only VISIBLE pixels are marked.
  Because the front panel occludes ~20% of the back panel's upper-left corner,
  the back-panel mask has a phone-shaped 'bite' out of it.

  Feeding this incomplete mask to Hunyuan3D produces a mesh with a literal
  concave notch at the occlusion boundary.  SynergyAmodal fixes this:

  Given the whole composite image + the modal mask + a text caption, it runs a
  two-pass coarse-to-fine diffusion process to produce the AMODAL completion —
  the full back panel as if the front panel were removed.

  Inference API (ported from SynergyAmodal/app.py):
    Pass 1 — global 512×512 inference via system.inference()
    Pass 2 — crop around amodal region, refine with initial_latent from pass 1

  Key advantage over pix2gestalt:
    - Text conditioning: caption = object_label tells the model what to complete
    - Modern torch 2.2 / diffusers 0.31 — no dep conflicts
    - ldm.ckpt + vae.ckpt from HuggingFace (not a 15.5 GB Columbia download)
    - No SAM1 dependency (pix2gestalt imported segment_anything at module level)

  Feedback loop (per preliminary research §Component 3):
    1. Run SynergyAmodal × 4 seeds → 4 candidate completions.
    2. Preparatory Gemini call generates object-specific validation criteria.
    3. Gemini scores all 4 candidates; each issue carries affected_region bbox
       and suggested_inpainting_prompt.
    4. If best_score >= 7: done.
    5. Else — per-issue targeted remediation:
         shape/boundary issue → re-run SynergyAmodal with new seeds
         content/detail issue → SD inpainting on ONLY the affected_region bbox
    6. Re-validate. Max 2 outer iterations; use best seen if all fail.

  VRAM:
    SynergyAmodal (ldm + vae): ~8-12 GB — managed via ModelManager
    SD inpainting: ~3.5 GB with CPU offload — loaded only if needed
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import cfg
from src.gemini_client import AmodalValidationResult, GeminiClient
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
        all_candidates:     All seed outputs (useful for side-by-side display).
        validation_result:  Gemini's score + feature breakdown for the winner.
        seed_used:          Which seed (1-4) produced the winning candidate.
        sd_inpaint_applied: True if targeted SD inpainting was applied.
    """

    best_candidate: Image.Image
    all_candidates: List[Image.Image]
    validation_result: AmodalValidationResult
    seed_used: int
    sd_inpaint_applied: bool


# ---------------------------------------------------------------------------
# SynergyAmodal helpers (ported from SynergyAmodal/app.py)
# ---------------------------------------------------------------------------

# Morphological dilation in PyTorch — used for slight mask erosion
_dilate = lambda x, k=5: F.max_pool2d(x[None], kernel_size=k, stride=1, padding=k // 2)[0]


def _load_synergyamodal() -> Tuple:
    """
    Load SynergyAmodal's LDM + RGBA-decoder checkpoints.

    Returns (system, mask_decoder) tuple — both moved to cfg.DEVICE and
    set to eval mode with EMA weights applied (per app.py pattern).
    """
    repo = str(cfg.SYNERGYAMODAL_REPO_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from omegaconf import OmegaConf  # type: ignore[import]
    from trainer_deocclusion_pseudo import InpaintingTrainer  # type: ignore[import]
    from trainer_deocclusion_pseudo_decoder import RGBADecoderTrainer  # type: ignore[import]

    config_path = str(cfg.SYNERGYAMODAL_REPO_DIR / "configs" / "inpainter_ours_pseudo.yaml")
    opt = OmegaConf.load(config_path)

    # --- LDM (the main diffusion model) ---
    system = InpaintingTrainer.load_from_checkpoint(
        str(cfg.synergyamodal_ldm_ckpt),
        map_location=cfg.DEVICE,
        strict=True,
        opt=opt,
    ).to(cfg.DEVICE)
    system.model_ema.load_ema_params(system.unet)
    del system.model_ema
    system.eval()

    # --- RGBA decoder (VAE that outputs image + amodal mask) ---
    mask_decoder = RGBADecoderTrainer.load_from_checkpoint(
        str(cfg.synergyamodal_vae_ckpt),
        map_location=cfg.DEVICE,
        strict=True,
    ).to(cfg.DEVICE)
    mask_decoder.model_ema.load_ema_params(mask_decoder.vae.decoder)
    del mask_decoder.model_ema
    del mask_decoder.vae.encoder
    mask_decoder.eval()

    return system, mask_decoder


@torch.no_grad()
def _run_synergyamodal_single(
    composite_pil: Image.Image,
    modal_mask: np.ndarray,
    system,
    mask_decoder,
    caption: str,
    seed: int,
    cfg_image: float = 1.0,
    cfg_text: float = 1.0,
    local_noise_strength: float = 1.0,
) -> Image.Image:
    """
    Run one SynergyAmodal inference pass (two-pass coarse-to-fine).

    Ported directly from SynergyAmodal/app.py::deocclude_button_click().

    Args:
        composite_pil:        Full composite RGB PIL Image (scene context).
        modal_mask:           np.ndarray (H, W) bool — SAM2 visible-pixel mask.
        system:               Loaded InpaintingTrainer model.
        mask_decoder:         Loaded RGBADecoderTrainer model.
        caption:              Text description of the object (e.g. "rear panel of
                              Samsung Galaxy S25 Ultra, 4 camera lenses").
        seed:                 Random seed for diffusion sampling.
        cfg_image / cfg_text: Classifier-free guidance scales.
        local_noise_strength: Pass-2 noise strength (1.0 = full refinement).

    Returns:
        PIL Image (RGBA) — the completed back panel on transparent background.
    """
    from utils import move_to, crop_and_resize, put_back  # type: ignore[import]

    torch.manual_seed(seed)

    # --- postprocess helper (closes over mask_decoder) ---
    @torch.no_grad()
    def postprocess_latent(latent, *args):
        return mask_decoder.vae.decode(latent).sample.sigmoid().round(),

    image_size = 512
    image_np = np.array(composite_pil.convert("RGB"))
    H, W = image_np.shape[:2]

    # Convert to tensors
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    mask_t = torch.from_numpy(modal_mask.astype(np.float32))  # (H, W)
    # Slight erosion to clean mask boundary (erode_kernel_size=1 per app defaults)
    mask_t = 1 - _dilate(1 - mask_t[None], k=1)[0]

    # --- Resize + pad to 512×512 ---
    resize_scale = image_size / max(H, W)
    all_t = torch.cat([image, mask_t[None]], dim=0)
    all_t = F.interpolate(
        all_t[None],
        size=(int(H * resize_scale), int(W * resize_scale)),
        mode="bilinear",
        align_corners=False,
    )[0]
    all_t = F.pad(all_t, (0, image_size - all_t.shape[-1], 0, image_size - all_t.shape[-2]))
    image_t, mask_t = all_t.split([3, 1], dim=0)
    mask_t = (mask_t > 0.5).float()

    # --- Pass 1: global coarse inference ---
    fbatch = {"image": image_t[None], "modal_mask": mask_t[None], "caption": [caption]}
    fbatch = move_to(fbatch, cfg.DEVICE)
    amodal_img, amodal_msk = system.inference(
        fbatch, cfg_image=cfg_image, cfg_text=cfg_text, postprocess_latent=postprocess_latent
    )

    # Crop padding and restore to original resolution
    amodal_rgba = torch.cat([amodal_img, amodal_msk], dim=1)[0].cpu()
    amodal_rgba = amodal_rgba[:, : int(H * resize_scale), : int(W * resize_scale)]
    amodal_rgba = F.interpolate(amodal_rgba[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    amodal_img_out, amodal_msk_out = amodal_rgba.split([3, 1], dim=0)
    amodal_msk_out = (amodal_msk_out > 0.5).float()
    amodal_img_out = amodal_img_out * amodal_msk_out + 1.0 * (1 - amodal_msk_out)

    # --- Pass 2: coarse-to-fine crop refinement ---
    image_orig = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    mask_orig = torch.from_numpy(modal_mask.astype(np.float32))
    mask_orig = 1 - _dilate(1 - mask_orig[None], k=1)[0]

    cropped_all, bbox = crop_and_resize(
        torch.cat([image_orig, mask_orig[None], amodal_img_out], dim=0),
        amodal_msk_out[0],
        min_size=256,
        padding_value=0,
    )
    cropped_image, cropped_mask, cropped_amodal = cropped_all.split([3, 1, 3], dim=0)
    cropped_mask = (cropped_mask > 0.5).float()

    batch = {"image": cropped_image[None], "modal_mask": cropped_mask[None], "caption": [caption]}
    batch = move_to(batch, cfg.DEVICE)

    cropped_amodal, cropped_amodal_msk = system.inference(
        batch,
        cfg_image=cfg_image,
        cfg_text=cfg_text,
        postprocess_latent=postprocess_latent,
        strength=local_noise_strength,
        initial_latent=system.latent_scale_fn(
            system.vae.encode(
                cropped_amodal[None].contiguous().to(cfg.DEVICE) * 2 - 1
            )
        ),
    )

    cropped_rgba = torch.cat([cropped_amodal, cropped_amodal_msk], dim=1)[0].cpu()
    final_rgba = put_back(cropped_rgba, H, W, bbox)
    final_rgba[3] = (final_rgba[3] > 0.5).float()

    # Return as PIL RGBA
    result_np = final_rgba.float().permute(1, 2, 0).mul(255).numpy().astype(np.uint8)
    return Image.fromarray(result_np, "RGBA")


# ---------------------------------------------------------------------------
# Targeted SD inpainting (content/detail fix inside the feedback loop)
# ---------------------------------------------------------------------------


def _apply_targeted_sd_inpainting(
    base_image: Image.Image,
    prompt: str,
    affected_region: List[int],
    object_label: str,
) -> Image.Image:
    """
    Apply Stable Diffusion inpainting to ONE specific region identified by Gemini.

    This is NOT a primary amodal path — it is the content/detail fix step inside
    the Gemini feedback loop.  It runs only when Gemini detects issues like
    wrong_lens_count or seam_artifact in best_candidate.issues[].

    The mask is built from Gemini's affected_region [ymin, xmin, ymax, xmax]
    pixel coordinates (not a hardcoded heuristic).

    Args:
        base_image:       PIL Image (RGBA or RGB) — SynergyAmodal output.
        prompt:           Gemini's suggested_inpainting_prompt for this issue.
        affected_region:  [ymin, xmin, ymax, xmax] pixel bbox from Gemini.
        object_label:     Used to enrich the negative prompt.

    Returns:
        PIL Image (RGB) — image with the specific region inpainted.
    """
    from diffusers import StableDiffusionInpaintPipeline  # type: ignore[import]

    print(f"[amodal] Targeted SD inpainting on region {affected_region}: '{prompt}'")

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

        # Build mask from Gemini's exact affected_region bbox
        ymin, xmin, ymax, xmax = [int(v) for v in affected_region]
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        # Dilate region by 8px to handle VAE latent boundary contamination
        pad = 8
        mask_arr[max(0, ymin - pad): min(h, ymax + pad),
                 max(0, xmin - pad): min(w, xmax + pad)] = 255
        mask_pil = Image.fromarray(mask_arr, "L")

        result = pipe(
            prompt=prompt,
            negative_prompt=(
                f"artifacts, blurry, wrong details, distorted, low quality, "
                f"{object_label} front face on back"
            ),
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
    apply_inpaint_if_needed: bool = True,
) -> AmodalResult:
    """
    Complete the occluded portion of the back panel using SynergyAmodal.

    Implements the full Gemini↔amodal feedback loop from preliminary research
    §Component 3.  Runs SynergyAmodal × 4 seeds, Gemini scores all candidates,
    picks best.  If best < 7, targeted SD inpainting fixes critical issues per
    Gemini's affected_region + suggested_inpainting_prompt.  Max 2 outer loops.

    Args:
        composite_pil:    Full composite PIL Image (RGB) — scene context.
        back_modal_mask:  SAM2 modal mask (H, W) bool — visible back-panel pixels.
        object_label:     Product description, e.g. "rear panel of Samsung S25 Ultra".
        gemini:           Optional GeminiClient (creates one if None).
        seeds:            Random seeds to try. Default: [1, 2, 3, 4].
        apply_inpaint_if_needed: If True, run targeted SD inpainting for issues.

    Returns:
        AmodalResult with best_candidate RGBA PIL Image.
    """
    seeds = seeds or [1, 2, 3, 4]
    gemini = gemini or GeminiClient()

    # Build text caption for SynergyAmodal (text conditioning is a key advantage)
    caption = f"Complete full deoccluded view of {object_label}, remove occluder"

    # Pre-flight VRAM check (~10 GB needed for ldm + vae)
    manager.assert_vram_available(required_mb=9_500)

    all_candidates: List[Image.Image] = []
    all_scores: List[int] = []
    all_validations: List[AmodalValidationResult] = []

    best_seen: Optional[Image.Image] = None
    best_seen_score: int = 0
    best_seen_validation: Optional[AmodalValidationResult] = None
    sd_applied = False

    # --- Outer feedback loop (max 2 iterations per research) ---
    for iteration in range(2):
        iter_candidates: List[Image.Image] = []
        iter_scores: List[int] = []
        iter_validations: List[AmodalValidationResult] = []

        iter_seeds = seeds if iteration == 0 else [seeds[-1] + iteration * 10 + i for i in range(4)]
        print(f"[amodal] Iteration {iteration + 1}/2 — running SynergyAmodal with seeds {iter_seeds}")

        # Load SynergyAmodal once per outer iteration; unload after all seeds
        with manager.use("synergyamodal", _load_synergyamodal) as (system, mask_decoder):
            for seed in iter_seeds:
                print(f"[amodal]   Seed {seed}...")
                try:
                    candidate = _run_synergyamodal_single(
                        composite_pil=composite_pil,
                        modal_mask=back_modal_mask,
                        system=system,
                        mask_decoder=mask_decoder,
                        caption=caption,
                        seed=seed,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[amodal]   OOM on seed {seed} — skipping.")
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    raise

                iter_candidates.append(candidate)

                # Save for Gemini inspection
                candidate_path = cfg.OUTPUT_DIR / f"amodal_iter{iteration}_seed{seed}.png"
                candidate.save(str(candidate_path))

                val = gemini.validate_amodal(
                    completed_image_path=candidate_path,
                    object_label=object_label,
                )
                iter_scores.append(val.score)
                iter_validations.append(val)
                print(f"[amodal]   Seed {seed} → score {val.score}/10")

                gc.collect()
                torch.cuda.empty_cache()

        # SynergyAmodal is now unloaded (ModelManager __exit__)

        if not iter_candidates:
            print("[amodal] All seeds failed this iteration — aborting.")
            break

        all_candidates.extend(iter_candidates)
        all_scores.extend(iter_scores)
        all_validations.extend(iter_validations)

        # Pick best in this iteration
        best_idx = int(np.argmax(iter_scores))
        best_score = iter_scores[best_idx]
        best_candidate = iter_candidates[best_idx]
        best_validation = iter_validations[best_idx]
        best_seed = iter_seeds[best_idx]

        # Track overall best seen across iterations
        if best_score > best_seen_score:
            best_seen_score = best_score
            best_seen = best_candidate
            best_seen_validation = best_validation

        print(f"[amodal] Iter {iteration + 1} best: seed={best_seed}, score={best_score}/10")

        # --- Check if best passes threshold ---
        if best_validation.passed:
            print(f"[amodal] Score {best_score} >= 7 — validation passed ✅")
            best_seen.save(str(cfg.OUTPUT_DIR / "amodal_best.png"))
            return AmodalResult(
                best_candidate=best_seen,
                all_candidates=all_candidates,
                validation_result=best_seen_validation,
                seed_used=best_seed,
                sd_inpaint_applied=False,
            )

        # --- Apply targeted SD inpainting for critical content/detail issues ---
        if apply_inpaint_if_needed and hasattr(best_validation, "issues") and best_validation.issues:
            repaired = best_candidate
            for issue in best_validation.issues:
                if getattr(issue, "severity", "") == "critical" and getattr(issue, "affected_region", None):
                    repaired = _apply_targeted_sd_inpainting(
                        base_image=repaired,
                        prompt=issue.suggested_inpainting_prompt,
                        affected_region=issue.affected_region,
                        object_label=object_label,
                    )
                    sd_applied = True

            if sd_applied:
                repaired_path = cfg.OUTPUT_DIR / f"amodal_repaired_iter{iteration}.png"
                repaired.save(str(repaired_path))
                repaired_val = gemini.validate_amodal(
                    completed_image_path=repaired_path,
                    object_label=object_label,
                )
                print(f"[amodal] Post-inpaint score: {repaired_val.score}/10")

                if repaired_val.score > best_seen_score:
                    best_seen_score = repaired_val.score
                    best_seen = repaired
                    best_seen_validation = repaired_val

                if repaired_val.passed:
                    print("[amodal] Repaired image passed validation ✅")
                    best_seen.save(str(cfg.OUTPUT_DIR / "amodal_best.png"))
                    return AmodalResult(
                        best_candidate=best_seen,
                        all_candidates=all_candidates,
                        validation_result=best_seen_validation,
                        seed_used=best_seed,
                        sd_inpaint_applied=True,
                    )

        # Loop continues with new seeds if iteration < 2

    # All iterations exhausted — use best seen, flag it
    print(
        f"[amodal] Max iterations reached. Using best seen (score={best_seen_score}/10). "
        f"Flagged as 'validated with minor issues'."
    )
    best_seen.save(str(cfg.OUTPUT_DIR / "amodal_best.png"))
    return AmodalResult(
        best_candidate=best_seen,
        all_candidates=all_candidates,
        validation_result=best_seen_validation,
        seed_used=-1,
        sd_inpaint_applied=sd_applied,
    )
