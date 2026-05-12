# Pix2gestalt Amodal Completion Research
<!-- Filled: 2026-05-12 -->

## Source
- GitHub: https://github.com/cvlab-columbia/pix2gestalt
- Paper: CVPR 2024 Highlight, Columbia University
- Inference entry point: `pix2gestalt/inference.py` → `run_pix2gestalt()` method

---

## Critical Reality Check: VRAM Requirement

> The Gradio app uses **22–28 GB of VRAM** (stated in README).
> The `run_pix2gestalt()` function alone uses ~9–10 GB in fp16.
> This is tight on 16 GB. Must unload all other models before running.

---

## Installation

**No pip package.** Must clone from GitHub and install deps manually.

```bash
# Step 1: Clone repo
git clone https://github.com/cvlab-columbia/pix2gestalt.git
cd pix2gestalt

# Step 2: Install pix2gestalt requirements
pip install -r requirements.txt

# Step 3: Install taming-transformers (required dependency)
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/

# Step 4: Install OpenAI CLIP (required dependency)
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

**Python version:** Tested on Python 3.9. Works on 3.10 (Colab default).

---

## Checkpoint Download (NO HuggingFace token required)

```bash
# Primary checkpoint (epoch=000005 — default, best zero-shot generalization)
mkdir -p /content/pix2gestalt/ckpt
wget -c -P /content/pix2gestalt/ckpt \
  https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt

# Alternative: epoch=000010 (better for synthetic occlusion, worse zero-shot)
wget -c -P /content/pix2gestalt/ckpt \
  https://gestalt.cs.columbia.edu/assets/epoch=000010.ckpt

# Also via HuggingFace (public, no gating):
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="cvlab/pix2gestalt-weights",
    filename="epoch=000005.ckpt",
    local_dir="/content/pix2gestalt/ckpt/"
)
```

**Checkpoint size:** ~15.5 GB. Allow 10–15 min on Colab.

---

## Python Inference API

The key function is `run_pix2gestalt` in `pix2gestalt/inference.py`:

```python
import sys
sys.path.insert(0, "/content/pix2gestalt")

from pix2gestalt.inference import run_pix2gestalt
import numpy as np
from PIL import Image

# Inputs:
# - whole_image: PIL Image (RGB), the original composite image
# - modal_mask: numpy array (H, W) binary, the SAM modal mask of the occluded back panel

# Run amodal completion with multiple seeds (best-of-N)
results = []
for seed in [1, 2, 3, 4]:
    completed_rgba = run_pix2gestalt(
        ckpt_path="/content/pix2gestalt/ckpt/epoch=000005.ckpt",
        whole_image=whole_image,      # PIL Image (RGB)
        mask=modal_mask,              # numpy (H,W) bool
        seed=seed,
        steps=50,                     # denoising steps (50 default, 20 for speed)
        scale=3.0,                    # classifier-free guidance scale
        device="cuda",
    )
    results.append(completed_rgba)   # returns PIL Image (RGBA)

# results[i] is the complete back panel on white background
# Select best via Gemini validation or take highest confidence
```

**Key parameters:**
- `steps=50` → full quality; `steps=20` → faster but slightly worse
- `scale=3.0` → CFG scale; higher = more faithful to condition, less diverse
- Different seeds give different completions — run 4 and pick best via Gemini

---

## VRAM Management

```python
import gc
import torch

# Before loading pix2gestalt — ensure ~10 GB free
free_gb = torch.cuda.mem_get_info()[0] / 1024**3
print(f"Free VRAM: {free_gb:.1f} GB")
assert free_gb > 9.0, "Not enough VRAM for pix2gestalt — unload other models first"

# After completing inference:
# pix2gestalt loads models internally; you can't easily del them
# Force cleanup:
gc.collect()
torch.cuda.empty_cache()
gc.collect()
```

---

## What pix2gestalt Takes and Returns

**Input:**
1. `whole_image` — full composite RGB image (PIL Image) — provides context
2. `mask` — binary modal mask of the occluded object (numpy H×W bool) — marks what's visible

**Output:**
- RGBA PIL Image — the complete object on a transparent/white background
- The previously hidden region is synthesized as a natural continuation
- The shape/silhouette is the key output (boundary recovery)

---

## Limitations

1. **VRAM:** ~10 GB in fp16 — must unload SAM2 before loading
2. **Checkpoint size:** 15.5 GB — takes 10–15 min to download on Colab
3. **No direct text control:** Can't say "add a 4th lens" — use best-of-N seeds
4. **For content detail fixes:** Use SD inpainting on top of pix2gestalt output
5. **22–28 GB for full app** — the `run_pix2gestalt()` function alone is ~10 GB

---

## Fallback: SD Inpainting for Content Fixes

When Gemini validation identifies specific content issues (wrong lens count, seam), use targeted SD inpainting:

```python
from diffusers import StableDiffusionInpaintPipeline
import torch

# Load public mirror — no token required
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()   # fits in ~3.5 GB VRAM
pipe.enable_vae_slicing()

def targeted_inpaint(image_pil, region_mask_pil, prompt, negative_prompt="", strength=0.85):
    """
    image_pil: PIL Image (RGB) — the pix2gestalt output
    region_mask_pil: PIL Image (L mode) — white=inpaint, black=keep
    prompt: Gemini's suggested_inpainting_prompt
    """
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "artifacts, blurry, wrong details",
        image=image_pil.convert("RGB"),
        mask_image=region_mask_pil,
        strength=strength,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    return result

# IP-Adapter for reference-image conditioning (optional, public)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                     weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.6)
```

---

## Summary

| Property | Value |
|---|---|
| Source | Clone from GitHub (no pip package) |
| Checkpoint | 15.5 GB, public, no HF token |
| Download URL | `https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt` |
| VRAM needed | ~10 GB fp16 (standalone inference) |
| HF token | Not required |
| Inference API | `run_pix2gestalt()` in `pix2gestalt/inference.py` |
| Best practice | Run 4 seeds, pick best with Gemini validation |
| Fallback | SD inpainting (`stable-diffusion-v1-5/stable-diffusion-inpainting`) |
