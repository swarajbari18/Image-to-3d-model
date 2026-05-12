# Hunyuan3D-2mv Research
<!-- Filled: 2026-05-12 -->

## Source
- GitHub: https://github.com/Tencent/Hunyuan3D-2
- HuggingFace (gated): https://huggingface.co/tencent/Hunyuan3D-2mv
- License: Tencent Community License (non-commercial, auto-approved)

---

## Critical Facts

1. **Gated on HuggingFace** — HF account + token + license acceptance required. No workaround.
2. **Requires cloning the repo** — custom CUDA rasterizer must be compiled. Not a standard pip package.
3. **Two-stage pipeline**: shape (Hunyuan3D-DiT) + texture (Hunyuan3D-Paint)
4. **VRAM**: shape stage ~6 GB; full pipeline with texture ~10–16 GB on Colab T4 (16 GB)

---

## Installation in Colab

```bash
# Step 1: Authenticate HuggingFace (must do before any model download)
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])  # from Colab Secrets

# Step 2: Clone the repo
!git clone https://github.com/Tencent/Hunyuan3D-2
%cd Hunyuan3D-2

# Step 3: Install requirements
!pip install -r requirements.txt

# Step 4: Compile custom CUDA rasterizer (REQUIRED for texture generation)
%cd hy3dgen/texgen/custom_rasterizer
!python3 setup.py install
%cd ../../../

# Step 5: Compile differentiable renderer
%cd hy3dgen/texgen/differentiable_renderer
!bash compile_mesh_painter.sh
%cd ../../..
```

**Known risk:** CUDA extension compilation may fail on Colab depending on CUDA toolkit version.
**Mitigation:** Have TRELLIS as fallback (MIT license, pip-installable, no custom CUDA).

---

## Model Download

```python
from huggingface_hub import snapshot_download
import os

# Download Hunyuan3D-2mv model weights (gated — HF_TOKEN required)
local_dir = snapshot_download(
    repo_id="tencent/Hunyuan3D-2mv",
    cache_dir="/content/models/hunyuan3d/",
    token=os.environ["HF_TOKEN"]
)
print(f"Model downloaded to: {local_dir}")
```

---

## Python Inference API

```python
import sys
sys.path.insert(0, "/content/Hunyuan3D-2")

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import torch
from PIL import Image

# --- Stage 1: Shape generation ---
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    torch_dtype=torch.float16,
)
shape_pipeline = shape_pipeline.to("cuda")

# Multi-view input: front + back (left/right can be None)
front_image = Image.open("front_input.png").convert("RGBA")
back_image  = Image.open("back_input.png").convert("RGBA")

# Generate shape mesh (bare, no texture)
mesh = shape_pipeline(
    image=front_image,
    image_back=back_image,    # Multi-view: back panel
    num_inference_steps=50,
    guidance_scale=7.5,
    octree_resolution=256,    # reduce to 128 if OOM: saves ~4 GB VRAM
    output_type="mesh",
)

# Save intermediate mesh
mesh.export("/content/phone_shape_raw.glb")

# Unload shape model before texture stage
del shape_pipeline
import gc
gc.collect()
torch.cuda.empty_cache()

# --- Stage 2: Texture generation (Hunyuan3D-Paint) ---
texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    torch_dtype=torch.float16,
)
texture_pipeline = texture_pipeline.to("cuda")

textured_mesh = texture_pipeline(
    mesh=mesh,
    image=front_image,
    image_back=back_image,
)

textured_mesh.export("/content/phone_raw.glb")

del texture_pipeline
gc.collect()
torch.cuda.empty_cache()
```

---

## VRAM Budget (16 GB Colab T4)

| Stage | VRAM | Action |
|---|---|---|
| SAM2 segmentation | 1.5 GB | Load → run → unload |
| pix2gestalt | 10 GB | Load → run → unload |
| Hunyuan3D shape | 6 GB | Load → run → unload |
| Hunyuan3D texture | 10–14 GB | Load → run → unload |
| SD inpainting (optional) | 3.5 GB (with offload) | Load → run → unload |

**Key constraint:** Never load shape + texture at same time — unload shape before loading texture.

---

## OOM Mitigation

```python
# Reduce octree resolution if OOM during shape stage
mesh = shape_pipeline(
    ...,
    octree_resolution=128,   # Down from 256 — saves ~4 GB VRAM
)

# Between shape and texture stages:
del shape_pipeline
gc.collect()
torch.cuda.empty_cache()
# Wait for VRAM to free before loading texture pipeline
```

---

## TRELLIS Fallback (if Hunyuan3D fails)

```bash
pip install git+https://github.com/microsoft/TRELLIS.git
```

```python
from trellis.pipelines import TrellisImageTo3DPipeline

pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline = pipeline.cuda()

outputs = pipeline.run(
    front_image,
    seed=42,
    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
    slat_sampler_params={"steps": 12, "cfg_strength": 3.0},
)

# outputs["gaussian"] — 3D Gaussians
# outputs["mesh"] — triangle mesh
# outputs["radiance_field"] — NeRF

glb = pipeline.export_video(outputs)  # or export mesh
```

TRELLIS is MIT licensed, publicly available, and doesn't require custom CUDA compilation.
VRAM: 12–16 GB. Single-view only (no native multi-view), but single front view is acceptable fallback.

---

## Input Image Preparation

Hunyuan3D expects clean RGBA images with white background:

```python
from PIL import Image
import numpy as np

def prepare_input_image(rgba_pil, target_size=512):
    """
    Resize + pad to square, ensure white background.
    """
    img = rgba_pil.convert("RGBA")
    # Paste on white background
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])
    img_rgb = white_bg.convert("RGB")

    # Resize with padding to square
    img_rgb.thumbnail((target_size, target_size), Image.LANCZOS)
    square = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    offset = ((target_size - img_rgb.width) // 2,
              (target_size - img_rgb.height) // 2)
    square.paste(img_rgb, offset)
    return square
```

---

## What the Output Looks Like

- **Raw mesh:** ~50K–300K triangles, arbitrary scale (unit cube), PBR textures baked
- **Common issues:** Inflated thickness (1.5–3×), camera bump merged into body, sides invented
- **For demo purposes:** Rough silhouette and dominant color are usually correct
- **Metric scaling** (Component 6) fixes the thickness issue automatically

---

## Summary

| Property | Value |
|---|---|
| Repo | https://github.com/Tencent/Hunyuan3D-2 |
| HF model | `tencent/Hunyuan3D-2mv` (gated) |
| HF token | **Required** — must accept Tencent license |
| Installation | Clone repo + compile CUDA rasterizer |
| Shape VRAM | ~6 GB fp16 |
| Texture VRAM | ~10–14 GB fp16 |
| Input | RGBA 512×512, white background |
| Multi-view | front + back (left/right optional/None) |
| Output | GLB with PBR texture |
| Fallback | TRELLIS (MIT, pip install, single-view) |
