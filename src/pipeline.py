"""
pipeline.py — End-to-end Image-to-3D orchestrator.
====================================================
Architecture:
  Thin orchestrator that wires together all five pipeline stages in sequence.
  Each stage gets a fresh GPU (all prior models unloaded).  The orchestrator
  itself is stateless — all state is passed as dataclass return values.

  Stage sequence:
    1. Gemini parse composite         → CompositeParseResult
    2. SAM2 segmentation              → SegmentationResult  (front_mask, back_modal_mask, RGBAs)
    3. Gemini lookup dimensions       → ProductDimensions   (mm specs from web search)
    4. SynergyAmodal amodal completion → AmodalResult        (completed back RGBA)
    5. Hunyuan3D-2mv reconstruction   → ReconstructionResult (raw GLB)
    6. Mesh processing                → MeshProcessingResult (final phone_final.glb)

  Colab usage — each stage maps to one notebook cell (see COLAB_CELLS.md).

  Error handling philosophy:
    - Stages 1-4 raise on failure (segmentation / amodal errors are unrecoverable).
    - Stage 5 has an internal TRELLIS fallback.
    - Stage 6 logs non-fatal warnings for UV issues but always exports a GLB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.amodal_completion import AmodalResult, run_amodal
from src.config import cfg
from src.gemini_client import (
    CompositeParseResult,
    GeminiClient,
    ProductDimensions,
)
from src.mesh_processing import MeshProcessingResult, run_mesh_processing
from src.model_manager import manager
from src.reconstruction import ReconstructionResult, run_reconstruction
from src.segmentation import SegmentationResult, run_segmentation


@dataclass
class PipelineResult:
    """
    Aggregated result of the full Image-to-3D pipeline run.

    Attributes:
        parse_result:       Gemini's scene understanding of the composite.
        segmentation:       SAM2 masks and RGBA crops.
        dimensions:         Real-world mm dimensions from Gemini grounded search.
        amodal:             SynergyAmodal amodal completion of back panel.
        reconstruction:     Raw GLB from Hunyuan3D-2mv or TRELLIS.
        mesh:               Final cleaned, oriented, scaled GLB.
        elapsed_seconds:    Total wall-clock time for the pipeline run.
    """

    parse_result: CompositeParseResult
    segmentation: SegmentationResult
    dimensions: Optional[ProductDimensions]
    amodal: AmodalResult
    reconstruction: ReconstructionResult
    mesh: MeshProcessingResult
    elapsed_seconds: float


def run_pipeline(
    composite_image_path: str | Path,
    target_faces: int = 40_000,
    amodal_seeds: int = 4,
) -> PipelineResult:
    """
    Run the complete Image-to-3D pipeline on a marketing composite image.

    Args:
        composite_image_path: Path to the input image (JPEG or PNG).
                              Should show front + back panel side-by-side.
        target_faces:         Target triangle count for the final GLB.
        amodal_seeds:         Number of SynergyAmodal seeds to try (1-4).

    Returns:
        PipelineResult with all intermediate + final outputs.
    """
    t0 = time.time()
    composite_path = Path(composite_image_path)
    gemini = GeminiClient()

    # ── Stage 1: Parse composite ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 1: Gemini scene understanding")
    print("=" * 60)
    parse_result = gemini.parse_composite(composite_path)
    print(f"  Product: {parse_result.product_label}")
    for v in parse_result.views:
        print(f"  View '{v.label}': bbox={v.box_2d}  occluded={v.is_occluded}")

    # ── Stage 2: SAM2 segmentation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2: SAM2 segmentation with Gemini feedback loop")
    print("=" * 60)
    seg_result = run_segmentation(
        composite_path=composite_path,
        parse_result=parse_result,
        gemini=gemini,
        overlay_save_path=cfg.OUTPUT_DIR / "final_overlay.png",
    )
    print(
        f"  Segmentation done. Iterations: {seg_result.iterations_used}. "
        f"{seg_result.validation_notes}"
    )

    # Save RGBA crops
    front_rgba_path = cfg.OUTPUT_DIR / "front_rgba.png"
    back_rgba_path = cfg.OUTPUT_DIR / "back_modal_rgba.png"
    seg_result.front_rgba.save(str(front_rgba_path))
    seg_result.back_rgba_modal.save(str(back_rgba_path))

    # ── Stage 3: Dimension lookup ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 3: Gemini grounded dimension lookup")
    print("=" * 60)
    dimensions: Optional[ProductDimensions] = None
    try:
        dimensions = gemini.lookup_dimensions(parse_result.product_label)
        print(
            f"  Dimensions: "
            f"W={dimensions.width_mm} mm  "
            f"H={dimensions.height_mm} mm  "
            f"D={dimensions.depth_mm} mm"
        )
        if dimensions.source_note:
            print(f"  Source: {dimensions.source_note}")
    except Exception as e:
        print(f"  WARNING: Dimension lookup failed ({e}) — mesh will not be metric-scaled.")

    # ── Stage 4: Amodal completion ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 4: SynergyAmodal amodal completion")
    print("=" * 60)
    from PIL import Image

    composite_pil = Image.open(composite_path).convert("RGB")
    amodal_result = run_amodal(
        composite_pil=composite_pil,
        back_modal_mask=seg_result.back_mask_modal,
        object_label=parse_result.product_label,
        gemini=gemini,
        seeds=list(range(1, amodal_seeds + 1)),
    )
    amodal_result.dimensions = dimensions

    # Save amodal back RGBA for reconstruction input
    amodal_back_path = cfg.OUTPUT_DIR / "back_amodal_rgba.png"
    amodal_result.best_candidate.save(str(amodal_back_path))

    # ── Stage 5: 3D reconstruction ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 5: Hunyuan3D-2mv reconstruction")
    print("=" * 60)
    recon_result = run_reconstruction(
        front_rgba=seg_result.front_rgba,
        back_rgba=amodal_result.best_candidate,
        output_name="phone_raw.glb",
    )
    print(f"  Reconstruction complete. Backend: {recon_result.backend_used}")
    print(f"  Raw GLB: {recon_result.glb_path}")

    # ── Stage 6: Mesh processing ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 6: Mesh cleanup + orient + scale + export")
    print("=" * 60)
    mesh_result = run_mesh_processing(
        glb_path=recon_result.glb_path,
        dimensions=dimensions,
        target_faces=target_faces,
        output_name="phone_final.glb",
    )

    elapsed = time.time() - t0
    print(f"\n✅  Pipeline complete in {elapsed:.1f}s")
    print(f"   Final GLB: {mesh_result.final_glb_path}")
    print(mesh_result.stats_table)

    return PipelineResult(
        parse_result=parse_result,
        segmentation=seg_result,
        dimensions=dimensions,
        amodal=amodal_result,
        reconstruction=recon_result,
        mesh=mesh_result,
        elapsed_seconds=elapsed,
    )
