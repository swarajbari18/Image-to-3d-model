# Image-to-3D Backend — Implementation Plan
> Synthesized from all research files — May 12, 2026

## File Layout
```
src/
  config.py           # env vars, paths, dtype
  model_manager.py    # VRAM load/unload context manager
  gemini_client.py    # Gemini API: parse, validate, dimension lookup
  segmentation.py     # SAM2 + Gemini feedback loop
  amodal_completion.py # pix2gestalt + SD inpainting fallback
  reconstruction.py   # Hunyuan3D-2mv + TRELLIS fallback
  mesh_processing.py  # cleanup + orient + scale + GLB export
  pipeline.py         # end-to-end orchestrator
.env.example
requirements.txt
```

## Key Decisions
| | Choice | Reason |
|---|---|---|
| Gemini SDK | `google-genai` v2.1+ | `google-generativeai` EOL Nov 2025 |
| Gemini model | `gemini-2.5-flash` | Stable, multimodal, 10x cheaper |
| SAM2 | Hiera-Large (Colab) | Best masks, fits 16 GB |
| Amodal | pix2gestalt 4 seeds | Correct silhouette over texture quality |
| 3D | Hunyuan3D-2mv (primary) | Native front+back multi-view |
| 3D fallback | TRELLIS | MIT, pip install, no custom CUDA |
| Mesh | trimesh + PyMeshLab + pymeshfix | Complementary tools |
| Secrets | Colab Secrets + .env fallback | Security + local dev |
| VRAM | Sequential load/unload (ModelManager) | 5 models, 16 GB total |

## VRAM Budget
```
SAM2 Large:           1.5 GB  → unload
pix2gestalt:         10.0 GB  → unload
SD inpainting:        3.5 GB  → unload (optional)
Hunyuan3D shape:      6.0 GB  → unload
Hunyuan3D texture:   10-14 GB → unload
Mesh cleanup:         0 GB (CPU)
Gemini API:           0 GB (cloud)
```

## .env Variables Needed
```
GEMINI_API_KEY          # from aistudio.google.com/apikey
GEMINI_MODEL            # default: gemini-2.5-flash
HF_TOKEN                # required ONLY for Hunyuan3D-2mv (gated)
MODEL_CACHE_DIR         # default: /content/models
HF_HOME                 # HF hub cache dir
SAM2_VARIANT            # tiny|small|base_plus|large
SAM2_CHECKPOINT_DIR
PIX2GESTALT_CKPT_DIR
PIX2GESTALT_EPOCH       # 000005 (default) or 000010
HUNYUAN3D_CACHE_DIR
SD_INPAINT_MODEL        # stable-diffusion-v1-5/stable-diffusion-inpainting
DEVICE                  # auto|cuda|cpu
TORCH_DTYPE             # fp16|fp32
OUTPUT_DIR
```

## Colab Install Order
```bash
pip install -q transformers diffusers accelerate safetensors huggingface_hub
pip install -q google-genai python-dotenv pydantic rich
pip install -q git+https://github.com/facebookresearch/sam2.git
# pix2gestalt — clone only, no pip package
git clone https://github.com/cvlab-columbia/pix2gestalt.git /content/pix2gestalt
pip install -q -r /content/pix2gestalt/requirements.txt
git clone https://github.com/CompVis/taming-transformers.git && pip install -q -e taming-transformers/
git clone https://github.com/openai/CLIP.git && pip install -q -e CLIP/
# Hunyuan3D — clone + compile CUDA
git clone https://github.com/Tencent/Hunyuan3D-2 /content/Hunyuan3D-2
pip install -q -r /content/Hunyuan3D-2/requirements.txt
cd /content/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && python3 setup.py install
cd /content/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && bash compile_mesh_painter.sh
# Mesh processing
pip install -q "trimesh[easy]" pymeshfix pymeshlab open3d xatlas rembg opencv-python-headless scipy scikit-image
```

## Model Downloads
```python
# SAM2 — Meta CDN, no token
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# pix2gestalt — Columbia server, no token (~15.5 GB)
wget -c https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt

# Hunyuan3D-2mv — HuggingFace, HF_TOKEN required (gated)
from huggingface_hub import snapshot_download, login
login(token=HF_TOKEN)
snapshot_download("tencent/Hunyuan3D-2mv", cache_dir=HUNYUAN3D_CACHE_DIR)

# SD Inpainting — public mirror, no token
# auto-downloaded by diffusers on first from_pretrained call
```

## Component Details

### config.py
- `_get_secret(key)`: tries Colab Secrets → .env → os.environ
- Sets DEVICE (cuda/cpu), TORCH_DTYPE (fp16/fp32)
- Creates output directories

### model_manager.py
- `ModelManager.use(name, loader_fn)` context manager
- On exit: `model.cpu()` → `del model` → `gc.collect()` → `torch.cuda.empty_cache()`
- `assert_vram(required_gb)` uses `torch.cuda.mem_get_info()` (accurate, not nvidia-smi)

### gemini_client.py
SDK: `from google import genai` — client = `genai.Client()`

Four methods:
1. `parse_composite(image_path)` → `CompositeParseResult` (Pydantic)
   - Bbox format: `[ymin, xmin, ymax, xmax]` 0–1000
   - response_schema + response_mime_type="application/json"
2. `validate_masks(overlay_image_path, view_labels)` → `MaskValidationResult`
3. `validate_amodal(completed_image_path, object_label)` → `AmodalValidationResult`
   - Two-call: first generate criteria, then validate against them
4. `lookup_dimensions(object_label)` → `ProductDimensions`
   - Two-call: grounded search first, schema extract second

### segmentation.py
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Coord conversion: Gemini [ymin,xmin,ymax,xmax]×1000 → SAM [xmin,ymin,xmax,ymax] px
def gemini_box_to_sam(box, img_h, img_w):
    ymin,xmin,ymax,xmax = box
    return np.array([
        int(xmin/1000*img_w), int(ymin/1000*img_h),
        int(xmax/1000*img_w), int(ymax/1000*img_h)
    ])

# set_image() once, predict() twice (front + back reuse encoding)
predictor.set_image(np.array(composite_rgb))
masks, scores, _ = predictor.predict(box=sam_box, multimask_output=True)
best_mask = masks[np.argmax(scores)]
```
Feedback loop: render overlay → Gemini validate → apply fixes (neg points, bbox adjust) → repeat ≤3×

### amodal_completion.py
```python
import sys; sys.path.insert(0, "/content/pix2gestalt")
from pix2gestalt.inference import run_pix2gestalt

for seed in [1,2,3,4]:
    result = run_pix2gestalt(
        ckpt_path=f"{PIX2GESTALT_CKPT_DIR}/epoch=000005.ckpt",
        whole_image=composite_pil, mask=back_modal_mask_np,
        seed=seed, steps=50, scale=3.0, device="cuda"
    )
# pick best via Gemini score; SD inpaint targeted regions if needed
```

### reconstruction.py
```python
import sys; sys.path.insert(0, "/content/Hunyuan3D-2")
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# Stage 1 shape (~6 GB), then unload
# Stage 2 texture (~10-14 GB), then unload
# TRELLIS fallback if CUDA compilation failed
```

### mesh_processing.py
Steps (in order):
1. `trimesh.load(glb, process=False)` → extract largest geometry
2. Remove floaters: `mesh.split()` → keep area > 1% of largest
3. pymeshfix: `tin.fix_connectivity()` + `tin.fill_small_boundaries()`
4. Taubin smooth: PyMeshLab `apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53)`
5. Isotropic remesh: PyMeshLab `meshing_isotropic_explicit_remeshing(targetlen=PercentageValue(1))`
6. Decimate: PyMeshLab `meshing_decimation_quadric_edge_collapse(targetfacenum=40000)`
7. Fix normals: `trimesh.repair.fix_normals(mesh)`
8. Re-UV: `xatlas.parametrize(vertices, faces)`
9. Canonical orient: `mesh.apply_obb()` + axis permute (longest→Y, middle→X, shortest→Z)
10. Metric scale: diagonal matrix `diag([w_mm/cx, h_mm/cy, d_mm/cz, 1])`
11. Export: `scene.export("phone_final.glb")`

Note: PyMeshLab cannot export GLB — use PLY/OBJ as intermediate, trimesh for final GLB.
UV mapping is invalidated by pymeshfix/decimate — always re-UV with xatlas.

## Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Hunyuan3D CUDA compile fails | TRELLIS fallback (MIT, pip-only) |
| pix2gestalt 15.5 GB slow download | Cache to Google Drive; copy local on session start |
| pix2gestalt VRAM >10 GB | assert_vram before load; reduce steps to 20 |
| Gemini loop not converging | Cap 3 iters; fall to Grounded-SAM (groundingdino-py) |
| Hunyuan3D texture OOM | octree_resolution=128; empty_cache between stages |
| Gemini grounding + schema | Two separate API calls (search, then extract) |
| Non-manifold mesh | pymeshfix one-liner |

## What's Achievable on 16 GB Colab
✅ Gemini scene parse + structured JSON
✅ SAM2 segmentation with feedback loop
✅ pix2gestalt amodal completion (4 seeds + Gemini pick)
✅ Metric scaling via Gemini web-search grounding
✅ Mesh cleanup → watertight manifold GLB
✅ phone_final.glb at 162.8 × 77.6 × 8.2 mm
⚠️ Hunyuan3D-2mv (depends on CUDA rasterizer compiling)
⚠️ pix2gestalt on 16 GB (tight — must unload everything else first)


## how to run in google collab

```
Cell: !git clone your_repo && cd Image-to-3d-model
Cell: !pip install ... (all deps)
Cell: from src.config import *
Cell: from src.gemini_client import GeminiClient
      result = GeminiClient().parse_composite("composite.jpg")
      display(result)           ← shows Gemini JSON + bboxes drawn on image

Cell: from src.segmentation import run_segmentation
      masks = run_segmentation(result)
      display(masks.overlay_image)   ← shows colored mask overlay

Cell: from src.amodal_completion import run_amodal
      completed = run_amodal(masks.back_modal)
      display(completed.best_candidate)  ← shows all 4 seeds side by side

Cell: from src.reconstruction import run_reconstruction
      glb_path = run_reconstruction(masks.front_rgba, completed.back_rgba)
      ← shows "saved to phone_raw.glb", VRAM stats

Cell: from src.mesh_processing import run_mesh_processing
      final = run_mesh_processing(glb_path, dims=completed.dimensions)
      display(final.stats_table)   ← shows triangle count, extents in mm
```