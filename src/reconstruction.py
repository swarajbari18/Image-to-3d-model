"""
reconstruction.py — Hunyuan3D-2mv multi-view 3D reconstruction + TRELLIS fallback.
====================================================================================
Architecture:
  Two-stage pipeline:
    Stage 1 (shape):   Hunyuan3DDiTFlowMatchingPipeline
                       Inputs: front RGBA + back RGBA (multi-view)
                       Output: bare mesh (no texture), ~50-300K triangles
                       VRAM: ~6 GB — load, run, UNLOAD before stage 2
    Stage 2 (texture): Hunyuan3DPaintPipeline
                       Inputs: same front/back RGBA + stage-1 mesh
                       Output: textured GLB (PBR, baked UV)
                       VRAM: ~10-14 GB — separate load after stage 1 is unloaded

  The two stages must NEVER be in VRAM simultaneously on a 16 GB GPU.

  TRELLIS fallback:
    If the Hunyuan3D CUDA rasterizer compilation fails (common on Colab depending
    on CUDA toolkit version), TRELLIS is used instead.
    TRELLIS is MIT-licensed, pip-installable, single-view only, ~12-16 GB VRAM.
    It receives only the front RGBA (amodal back is still used for amodal completion
    but not for 3D reconstruction in fallback mode).

  Input preparation:
    Both pipelines expect RGBA 512×512 images on white background.
    prepare_input_image() handles resize + pad + alpha compositing.

  VRAM budget with ModelManager:
    shape stage: assert ≥ 6.5 GB → load → infer → unload
    texture stage: assert ≥ 14 GB → load → infer → unload
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.config import cfg
from src.model_manager import manager


# ---------------------------------------------------------------------------
# Input image preparation
# ---------------------------------------------------------------------------


def prepare_input_image(rgba_pil: Image.Image, target_size: int = 512) -> Image.Image:
    """
    Prepare an RGBA image for Hunyuan3D-2mv input.

    Hunyuan3D expects:
      - RGBA mode
      - Centered on white (255, 255, 255, 255) background
      - Padded to square (target_size × target_size)

    Args:
        rgba_pil:    Source image (any mode, will be converted to RGBA).
        target_size: Target square dimension in pixels.  Default: 512.

    Returns:
        PIL Image in RGBA mode, target_size × target_size.
    """
    img = rgba_pil.convert("RGBA")

    # Composite onto white background
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])

    # Thumbnail-resize preserving aspect ratio
    img_rgb = white_bg.convert("RGB")
    img_rgb.thumbnail((target_size, target_size), Image.LANCZOS)

    # Center-pad to square
    square = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 255))
    offset_x = (target_size - img_rgb.width) // 2
    offset_y = (target_size - img_rgb.height) // 2
    square.paste(img_rgb.convert("RGBA"), (offset_x, offset_y))
    return square


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionResult:
    """
    Output of the 3D reconstruction stage.

    Attributes:
        glb_path:         Path to the exported raw GLB file (unprocessed, arbitrary scale).
        backend_used:     'hunyuan3d' or 'trellis' — which model was used.
        octree_resolution: Octree resolution used for shape generation (256 or 128).
    """

    glb_path: Path
    backend_used: str
    octree_resolution: int = 256


# ---------------------------------------------------------------------------
# Hunyuan3D-2mv: shape stage
# ---------------------------------------------------------------------------


def _build_hunyuan_shape_loader():
    """Return a loader fn for the Hunyuan3D shape pipeline."""

    def _load():
        sys.path.insert(0, str(cfg.HUNYUAN3D_REPO_DIR))
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]

        print("[reconstruction] Loading Hunyuan3D shape pipeline (fp16)...")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv",
            torch_dtype=cfg.TORCH_DTYPE,
            cache_dir=str(cfg.HUNYUAN3D_CACHE_DIR),
            token=cfg.HF_TOKEN,
        )
        return pipeline.to(cfg.DEVICE)

    return _load


# ---------------------------------------------------------------------------
# Hunyuan3D-2mv: texture stage
# ---------------------------------------------------------------------------


def _build_hunyuan_texture_loader():
    """Return a loader fn for the Hunyuan3D texture pipeline."""

    def _load():
        sys.path.insert(0, str(cfg.HUNYUAN3D_REPO_DIR))
        from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import]

        print("[reconstruction] Loading Hunyuan3D texture pipeline (fp16)...")
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv",
            torch_dtype=cfg.TORCH_DTYPE,
            cache_dir=str(cfg.HUNYUAN3D_CACHE_DIR),
            token=cfg.HF_TOKEN,
        )
        return pipeline.to(cfg.DEVICE)

    return _load


# ---------------------------------------------------------------------------
# TRELLIS fallback
# ---------------------------------------------------------------------------


def _reconstruct_trellis(front_rgba: Image.Image, output_path: Path) -> Path:
    """
    Fallback 3D reconstruction using TRELLIS (MIT license, pip-installable).

    TRELLIS is single-view only — it receives just the front panel.
    Install: pip install git+https://github.com/microsoft/TRELLIS.git

    Args:
        front_rgba:  Prepared (512×512, white-bg) front RGBA PIL Image.
        output_path: Path to write the output GLB.

    Returns:
        Path to the exported GLB file.
    """
    from trellis.pipelines import TrellisImageTo3DPipeline  # type: ignore[import]

    print("[reconstruction] Using TRELLIS fallback (single-view)...")

    def _load():
        pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        return pipeline.cuda()

    with manager.use("trellis", _load) as pipeline:
        outputs = pipeline.run(
            front_rgba,
            seed=42,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 12, "cfg_strength": 3.0},
        )

        # Export the triangle mesh (not Gaussian or NeRF)
        mesh = outputs["mesh"]
        mesh.export(str(output_path))

    print(f"[reconstruction] TRELLIS GLB saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_reconstruction(
    front_rgba: Image.Image,
    back_rgba: Image.Image,
    output_name: str = "phone_raw.glb",
    octree_resolution: int = 256,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> ReconstructionResult:
    """
    Generate a textured 3D mesh from front + back RGBA panel images.

    Attempts Hunyuan3D-2mv multi-view reconstruction first.  Falls back to
    TRELLIS if Hunyuan3D's CUDA extensions are unavailable.

    VRAM flow (never exceeds 16 GB):
      1. assert ≥ 6.5 GB → load shape pipeline → infer → unload shape pipeline
      2. assert ≥ 14 GB  → load texture pipeline → infer → unload texture pipeline

    Args:
        front_rgba:          RGBA PIL Image of front panel (any size).
        back_rgba:           RGBA PIL Image of completed back panel (any size).
        output_name:         Filename for the output GLB (saved in cfg.OUTPUT_DIR).
        octree_resolution:   256 (full quality) or 128 (OOM mitigation).
        num_inference_steps: Diffusion steps for shape generation.
        guidance_scale:      CFG scale for shape generation.

    Returns:
        ReconstructionResult with path to the raw (unprocessed) GLB file.
    """
    output_path = cfg.OUTPUT_DIR / output_name

    # Prepare inputs: square RGBA on white background
    front_prep = prepare_input_image(front_rgba, target_size=512)
    back_prep = prepare_input_image(back_rgba, target_size=512)

    # --- Attempt Hunyuan3D-2mv ---
    try:
        manager.print_vram_stats("before shape")

        # Stage 1: Shape generation
        with manager.use("hunyuan3d_shape", _build_hunyuan_shape_loader()) as shape_pipe:
            manager.assert_vram_available(
                manager.VRAM_REQUIREMENTS["hunyuan3d_shape"]
            )
            print(
                f"[reconstruction] Generating shape mesh "
                f"(octree_resolution={octree_resolution}, steps={num_inference_steps})..."
            )
            try:
                mesh = shape_pipe(
                    image=front_prep,
                    image_back=back_prep,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution,
                    output_type="mesh",
                )
            except torch.cuda.OutOfMemoryError:
                if octree_resolution > 128:
                    print(
                        "[reconstruction] OOM at octree_resolution=256 — "
                        "retrying with 128..."
                    )
                    octree_resolution = 128
                    mesh = shape_pipe(
                        image=front_prep,
                        image_back=back_prep,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        octree_resolution=128,
                        output_type="mesh",
                    )
                else:
                    raise

        # stage 1 unloaded — check VRAM before stage 2
        manager.print_vram_stats("after shape unload")

        # Stage 2: Texture generation
        with manager.use(
            "hunyuan3d_texture", _build_hunyuan_texture_loader()
        ) as tex_pipe:
            print("[reconstruction] Applying texture (Hunyuan3D-Paint)...")
            textured_mesh = tex_pipe(
                mesh=mesh,
                image=front_prep,
                image_back=back_prep,
            )

        textured_mesh.export(str(output_path))
        manager.print_vram_stats("after texture unload")
        print(f"[reconstruction] Hunyuan3D GLB saved to {output_path}")

        return ReconstructionResult(
            glb_path=output_path,
            backend_used="hunyuan3d",
            octree_resolution=octree_resolution,
        )

    except Exception as hunyuan_error:
        print(
            f"[reconstruction] Hunyuan3D failed: {hunyuan_error}\n"
            f"[reconstruction] Falling back to TRELLIS (single-view)..."
        )
        manager.unload_all()  # ensure no stale GPU state

        fallback_path = cfg.OUTPUT_DIR / output_name.replace(".glb", "_trellis.glb")
        _reconstruct_trellis(front_prep, fallback_path)

        return ReconstructionResult(
            glb_path=fallback_path,
            backend_used="trellis",
            octree_resolution=0,
        )
