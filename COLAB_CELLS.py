# Image-to-3D Pipeline — Google Colab Run Cells
# ================================================
# Copy each cell block into a separate Colab code cell.
# Run them ONE AT A TIME, top to bottom.
# Each cell prints progress and saves outputs to /content/output/.

# =============================================================================
# CELL 1 — Mount Google Drive (persistent model cache)
# =============================================================================
from google.colab import drive
drive.mount("/content/drive")

import os
os.makedirs("/content/drive/MyDrive/model_cache", exist_ok=True)
print("✅ Google Drive mounted. Model cache at /content/drive/MyDrive/model_cache")


# =============================================================================
# CELL 2 — Set Colab Secrets
# =============================================================================
# Before running this cell, add your secrets in the left sidebar:
#   🔑  Key icon → Secrets
#   Add:  GEMINI_API_KEY  (from aistudio.google.com/apikey)
#   Add:  HF_TOKEN        (from huggingface.co/settings/tokens)

# CELL 2 — Load secrets from .env file
import os
from dotenv import load_dotenv

# Loads variables from .env into os.environ
load_dotenv(".env")  # adjust path to your cloned repo

os.environ["OUTPUT_DIR"] = "/content/output"
os.makedirs("/content/output", exist_ok=True)

print("✅ Secrets loaded.")
print(f"  GEMINI_API_KEY: {'set ✓' if os.environ.get('GEMINI_API_KEY') else 'MISSING ❌'}")
print(f"  HF_TOKEN:       {'set ✓' if os.environ.get('HF_TOKEN') else 'MISSING ❌'}")



# =============================================================================
# CELL 3 — Install Python dependencies (run once per session)
# =============================================================================
# Takes ~3-5 minutes on a fresh Colab session.

# Core HuggingFace + diffusers
get_ipython().system("pip install -q transformers diffusers accelerate safetensors huggingface_hub")

# Gemini SDK (NEW — google-generativeai is deprecated)
get_ipython().system("pip install -q google-genai python-dotenv pydantic rich")

# SAM2 (from PyPI)
get_ipython().system("pip install -q sam2")

# Mesh processing
get_ipython().system('pip install -q "trimesh[easy]" pymeshfix pymeshlab open3d xatlas')

# Image processing
get_ipython().system("pip install -q opencv-python-headless scikit-image scipy imageio matplotlib")

# Verify
import torch, transformers, diffusers, sam2, trimesh
print(f"✅ All dependencies installed.")
print(f"   torch={torch.__version__}  CUDA={'✓' if torch.cuda.is_available() else '✗ (CPU)'}")
get_ipython().system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU'")


# =============================================================================
# CELL 4 — Install pix2gestalt (clone + deps)
# =============================================================================
# RCA: The repo structure is:
#   /content/pix2gestalt/          ← repo root (README.md, LICENSE, assets/, .git/)
#   /content/pix2gestalt/pix2gestalt/  ← actual Python package (inference.py, requirements.txt)
#
# The requirements.txt is at  /content/pix2gestalt/pix2gestalt/requirements.txt
# NOT at the repo root — that is why the original cell failed.
#
# The README's install steps (cd pix2gestalt; pip install -r requirements.txt)
# assume you are already INSIDE the cloned directory when you run that command.
# taming-transformers and CLIP must be cloned INSIDE the pix2gestalt repo dir
# (i.e. /content/pix2gestalt/) per the official README.

import os, sys

# --- Clone pix2gestalt ---
if not os.path.exists("/content/pix2gestalt"):
    get_ipython().system("git clone https://github.com/cvlab-columbia/pix2gestalt.git /content/pix2gestalt")
    print("✅ pix2gestalt cloned.")
else:
    print("✅ pix2gestalt already present — skipping clone.")

# Verify repo structure
import os
items = os.listdir("/content/pix2gestalt")
print(f"   Repo root contents: {items}")
inner = os.listdir("/content/pix2gestalt/pix2gestalt") if os.path.isdir("/content/pix2gestalt/pix2gestalt") else []
print(f"   pix2gestalt/ subdir contents: {inner}")

# --- Install pix2gestalt Python dependencies (FILTERED) ---
# pix2gestalt's requirements.txt pins several packages that conflict with
# what Colab and our pipeline already have installed.  We filter them out:
#
#  SKIPPED (reason):
#   torch==1.12.1       — CUDA 11.3 wheel no longer on PyPI; Colab has torch 2.x ✓
#   torchvision==0.13.1 — paired with old torch; skip
#   diffusers==0.12.1   — would DOWNGRADE to 2022 version; we need >=0.36.0 for SD inpainting
#   transformers==4.22.2 — would DOWNGRADE; we need >=4.45.0 for SAM2/Hunyuan3D
#   Pillow==9.5.0       — Colab has newer; strict pin causes conflicts
#   opencv-python       — conflicts with opencv-python-headless already installed
#   gradio              — not needed in our pipeline
#   carvekit-colab      — not needed
#
#  INSTALLED (safe, no conflicts):
#   segment-anything==1.0   ← KEY: inference.py imports this at module level
#   pytorch-lightning, omegaconf, einops, kornia, torchmetrics, imageio,
#   test-tube, webdataset, fire, lovely-numpy, lovely-tensors, albumentations,
#   torch-fidelity, plotly, rich

print("Installing pix2gestalt safe dependencies (skipping conflicting packages)...")
get_ipython().system(
    "pip install -q "
    "segment-anything==1.0 "          # SAM1 — imported at module level in inference.py
    "omegaconf==2.1.1 "
    "einops==0.3.0 "
    "pytorch-lightning==1.4.2 "
    "kornia==0.6 "
    "torchmetrics==0.6.0 "
    "imageio==2.9.0 "
    "imageio-ffmpeg==0.4.2 "
    "test-tube "
    "webdataset==0.2.5 "
    "fire==0.4.0 "
    "albumentations==0.4.3 "
    "torch-fidelity==0.3.0 "
    "lovely-numpy "
    "lovely-tensors "
    "plotly==5.13.1 "
    "fastapi==0.103.2 "
    "rich"
)
print("✅ pix2gestalt safe deps installed.")


# =============================================================================
# CELL 5 — Install Hunyuan3D-2 (clone + compile CUDA rasterizer)
# =============================================================================
# ⚠️  CUDA compilation can take 5-10 minutes.
# ⚠️  If compilation fails, the pipeline will automatically fall back to TRELLIS.

import os

if not os.path.exists("/content/Hunyuan3D-2"):
    # Step 1: Login to HuggingFace (required for gated Hunyuan3D-2mv model)
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    # Step 2: Clone repo
    get_ipython().system("git clone https://github.com/Tencent/Hunyuan3D-2 /content/Hunyuan3D-2")

    # Step 3: Install requirements
    get_ipython().system("pip install -q -r /content/Hunyuan3D-2/requirements.txt")

    # Step 4: Compile custom CUDA rasterizer (REQUIRED for texture generation)
    try:
        get_ipython().system("cd /content/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python3 setup.py install")
        get_ipython().system("cd /content/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && bash compile_mesh_painter.sh")
        print("✅ Hunyuan3D CUDA rasterizer compiled successfully.")
    except Exception as e:
        print(f"⚠️  CUDA rasterizer compilation failed: {e}")
        print("   Pipeline will use TRELLIS fallback for 3D reconstruction.")
else:
    print("✅ Hunyuan3D-2 already cloned.")


# =============================================================================
# CELL 6 — Download model weights
# =============================================================================
# SAM2:      ~224 MB — Meta CDN, no token
# pix2gestalt: ~15.5 GB — Columbia server, no token  ⚠️ Takes 10-15 min
# Hunyuan3D: ~8-10 GB  — HuggingFace, HF_TOKEN required

import os, shutil
from pathlib import Path

os.makedirs("/content/models/sam2", exist_ok=True)
os.makedirs("/content/models/pix2gestalt", exist_ok=True)
os.makedirs("/content/models/hunyuan3d", exist_ok=True)

GDRIVE_CACHE = "/content/drive/MyDrive/model_cache"

# --- SAM2 Hiera-Large (~224 MB) ---
sam2_ckpt = "/content/models/sam2/sam2.1_hiera_large.pt"
sam2_gdrive = f"{GDRIVE_CACHE}/sam2/sam2.1_hiera_large.pt"
if not os.path.exists(sam2_ckpt):
    if os.path.exists(sam2_gdrive):
        print("📂 Copying SAM2 from Google Drive cache...")
        os.makedirs("/content/models/sam2", exist_ok=True)
        shutil.copy(sam2_gdrive, sam2_ckpt)
    else:
        print("⬇️  Downloading SAM2 Hiera-Large (~224 MB)...")
        get_ipython().system(f"wget -q -O {sam2_ckpt} https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")
        os.makedirs(f"{GDRIVE_CACHE}/sam2", exist_ok=True)
        shutil.copy(sam2_ckpt, sam2_gdrive)
print(f"✅ SAM2: {sam2_ckpt}")

# --- pix2gestalt (~15.5 GB) ---
p2g_ckpt = "/content/models/pix2gestalt/epoch=000005.ckpt"
p2g_gdrive = f"{GDRIVE_CACHE}/pix2gestalt/epoch=000005.ckpt"
if not os.path.exists(p2g_ckpt):
    if os.path.exists(p2g_gdrive):
        print("📂 Copying pix2gestalt checkpoint from Google Drive cache (~15.5 GB, may take 5 min)...")
        os.makedirs("/content/models/pix2gestalt", exist_ok=True)
        shutil.copy(p2g_gdrive, p2g_ckpt)
    else:
        print("⬇️  Downloading pix2gestalt epoch=000005 (~15.5 GB, may take 10-15 min)...")
        get_ipython().system(f"wget -c -O {p2g_ckpt} https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt")
        os.makedirs(f"{GDRIVE_CACHE}/pix2gestalt", exist_ok=True)
        shutil.copy(p2g_ckpt, p2g_gdrive)
        print("💾 Saved pix2gestalt checkpoint to Google Drive cache.")
print(f"✅ pix2gestalt: {p2g_ckpt}")

# --- Hunyuan3D-2mv (~8-10 GB) ---
hunyuan_local = "/content/models/hunyuan3d"
hunyuan_gdrive = f"{GDRIVE_CACHE}/hunyuan3d"
if not os.listdir(hunyuan_local):
    if os.path.exists(hunyuan_gdrive) and os.listdir(hunyuan_gdrive):
        print("📂 Copying Hunyuan3D-2mv from Google Drive cache...")
        shutil.copytree(hunyuan_gdrive, hunyuan_local, dirs_exist_ok=True)
    else:
        print("⬇️  Downloading Hunyuan3D-2mv from HuggingFace (requires HF_TOKEN + license acceptance)...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="tencent/Hunyuan3D-2mv",
            cache_dir=hunyuan_local,
            token=os.environ["HF_TOKEN"]
        )
        os.makedirs(hunyuan_gdrive, exist_ok=True)
        shutil.copytree(hunyuan_local, hunyuan_gdrive, dirs_exist_ok=True)
        print("💾 Saved Hunyuan3D to Google Drive cache.")
print(f"✅ Hunyuan3D: {hunyuan_local}")


# =============================================================================
# CELL 7 — Clone this repo + set paths
# =============================================================================

import sys, os

REPO_URL = "https://github.com/YOUR_USERNAME/Image-to-3d-model.git"  # ← update this
REPO_DIR = "/content/Image-to-3d-model"

if not os.path.exists(REPO_DIR):
    get_ipython().system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"✅ Cloned repo to {REPO_DIR}")
else:
    get_ipython().system(f"cd {REPO_DIR} && git pull")
    print(f"✅ Repo up-to-date at {REPO_DIR}")

# Add repo to Python path
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Set environment overrides for Colab paths
os.environ["SAM2_CHECKPOINT_DIR"]    = "/content/models/sam2"
os.environ["PIX2GESTALT_CKPT_DIR"]   = "/content/models/pix2gestalt"
os.environ["HUNYUAN3D_CACHE_DIR"]    = "/content/models/hunyuan3d"
os.environ["OUTPUT_DIR"]             = "/content/output"
os.environ["SAM2_VARIANT"]           = "large"
os.environ["PIX2GESTALT_EPOCH"]      = "000005"

print("✅ Environment configured.")


# =============================================================================
# CELL 8 — Stage 1: Gemini scene parse
# =============================================================================
# Upload your composite image to /content/ first (Files panel → Upload).

from IPython.display import display, Image as IPImage
from src.config import cfg
from src.gemini_client import GeminiClient

# --- Upload / path ---
COMPOSITE_IMAGE = "/content/composite.jpg"   # ← change if your filename differs

gemini = GeminiClient()
parse_result = gemini.parse_composite(COMPOSITE_IMAGE)

print(f"\n📦 Product detected: {parse_result.product_label}")
print(f"📐 Composite size (approx): {parse_result.composite_width}×{parse_result.composite_height}")
print("\n🔍 Views detected:")
for v in parse_result.views:
    occ = f"  ← {v.occlusion_description}" if v.is_occluded else ""
    print(f"  [{v.label}] bbox={v.box_2d}  occluded={v.is_occluded}{occ}")

# Visualize bboxes on image
from PIL import Image, ImageDraw
import numpy as np

img = Image.open(COMPOSITE_IMAGE).convert("RGB")
draw = ImageDraw.Draw(img)
iw, ih = img.size
colors = {"front": "blue", "back": "orange"}

for v in parse_result.views:
    ymin, xmin, ymax, xmax = v.box_2d
    px = (int(xmin/1000*iw), int(ymin/1000*ih), int(xmax/1000*iw), int(ymax/1000*ih))
    draw.rectangle(px, outline=colors.get(v.label, "red"), width=4)
    draw.text((px[0]+4, px[1]+4), v.label, fill=colors.get(v.label, "red"))

img.save("/content/output/parse_result.jpg")
display(img)
print("✅ Stage 1 complete. Bboxes shown above.")


# =============================================================================
# CELL 9 — Stage 2: SAM2 segmentation
# =============================================================================

from src.segmentation import run_segmentation
import numpy as np

seg_result = run_segmentation(
    composite_path=COMPOSITE_IMAGE,
    parse_result=parse_result,
    gemini=gemini,
    max_feedback_iters=3,
    overlay_save_path="/content/output/final_overlay.png",
)

print(f"\n✅ Segmentation complete:")
print(f"   Iterations: {seg_result.iterations_used}")
print(f"   {seg_result.validation_notes}")
print(f"   Front mask — pixels: {seg_result.front_mask.sum():,}")
print(f"   Back mask  — pixels: {seg_result.back_mask_modal.sum():,}")

display(seg_result.overlay_image)
display(seg_result.front_rgba)
display(seg_result.back_rgba_modal)


# =============================================================================
# CELL 10 — Stage 3: Gemini grounded dimension lookup
# =============================================================================

dimensions = gemini.lookup_dimensions(parse_result.product_label)
print(f"\n📏 Dimensions for {parse_result.product_label}:")
print(f"   Width  : {dimensions.width_mm} mm")
print(f"   Height : {dimensions.height_mm} mm")
print(f"   Depth  : {dimensions.depth_mm} mm")
if dimensions.source_note:
    print(f"   Source : {dimensions.source_note}")


# =============================================================================
# CELL 11 — Stage 4: pix2gestalt amodal completion
# =============================================================================
# ⚠️  SAM2 must be unloaded before this cell runs (ModelManager handles this automatically).
# ⚠️  This cell loads ~10 GB to VRAM — takes 3-5 min per seed × 4 seeds = ~15-20 min.

from PIL import Image
from src.amodal_completion import run_amodal

composite_pil = Image.open(COMPOSITE_IMAGE).convert("RGB")

amodal_result = run_amodal(
    composite_pil=composite_pil,
    back_modal_mask=seg_result.back_mask_modal,
    object_label=parse_result.product_label,
    gemini=gemini,
    seeds=[1, 2, 3, 4],
    steps=50,
    apply_inpaint_if_needed=True,
)

print(f"\n✅ Amodal completion:")
print(f"   Best seed: {amodal_result.seed_used}")
print(f"   Score: {amodal_result.validation_result.score}/10")
print(f"   Passed: {amodal_result.validation_result.passed}")
print(f"   SD inpainting applied: {amodal_result.sd_inpaint_applied}")

# Show all 4 candidates side by side
from PIL import Image as PILImage
import numpy as np

candidates = amodal_result.all_candidates
w_each = 256
strip = PILImage.new("RGBA", (w_each * len(candidates), w_each), (255, 255, 255, 255))
for i, c in enumerate(candidates):
    thumb = c.resize((w_each, w_each))
    strip.paste(thumb, (i * w_each, 0))

strip.save("/content/output/amodal_candidates.png")
display(strip)
display(amodal_result.best_candidate)


# =============================================================================
# CELL 12 — Stage 5: Hunyuan3D-2mv 3D reconstruction
# =============================================================================
# ⚠️  pix2gestalt must be unloaded. ModelManager handles this automatically.
# ⚠️  Shape stage: ~6 GB. Texture stage: ~10-14 GB. Never both at once.

from src.reconstruction import run_reconstruction

recon_result = run_reconstruction(
    front_rgba=seg_result.front_rgba,
    back_rgba=amodal_result.best_candidate,
    output_name="phone_raw.glb",
    octree_resolution=256,   # reduce to 128 if OOM
    num_inference_steps=50,
)

print(f"\n✅ Reconstruction complete:")
print(f"   Backend: {recon_result.backend_used}")
print(f"   Raw GLB: {recon_result.glb_path}")

# Quick VRAM check
import torch
if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"   Free VRAM after reconstruction: {free_gb:.1f} GB")


# =============================================================================
# CELL 13 — Stage 6: Mesh cleanup + scale + export final GLB
# =============================================================================
# CPU-only — no VRAM needed.

from src.mesh_processing import run_mesh_processing

mesh_result = run_mesh_processing(
    glb_path=recon_result.glb_path,
    dimensions=dimensions,
    target_faces=40_000,
    output_name="phone_final.glb",
)

print(mesh_result.stats_table)
print(f"🎉  Final GLB ready: {mesh_result.final_glb_path}")

# Download the GLB
from google.colab import files
files.download(str(mesh_result.final_glb_path))


# =============================================================================
# CELL 14 — (Optional) Run the full pipeline in one call
# =============================================================================
# Equivalent to running Cells 8-13 in sequence, with a single function call.

from src.pipeline import run_pipeline

result = run_pipeline(
    composite_image_path=COMPOSITE_IMAGE,
    target_faces=40_000,
    amodal_seeds=4,
    amodal_steps=50,
)

print(f"\n🎉  Full pipeline complete in {result.elapsed_seconds:.0f}s")
print(f"   Final GLB: {result.mesh.final_glb_path}")

from google.colab import files
files.download(str(result.mesh.final_glb_path))
