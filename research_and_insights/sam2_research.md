# SAM2 Segmentation Research

> Live log — written incrementally. Last updated: 2026-05-12.
> Context: Image-to-3D pipeline for converting Samsung S25 Ultra marketing composite
> (front + back side-by-side) into a textured 3D mesh.

---

## 1. What SAM / SAM2 Is — Core Architecture

**Segment Anything Model (SAM)** — Meta AI, April 2023.
**SAM 2 (Segment Anything Model 2)** — Meta AI, July 2024. Extends SAM to video + images.

### SAM Architecture (both SAM and SAM2 share this three-stage structure)

**Step 1 — Image Encoder** (runs ONCE per image, most expensive)
- SAM 1: heavyweight ViT (ViT-H = 636M params). SAM 2: Hiera (hierarchical ViT), much lighter.
- Produces a rich 1/16-resolution spatial feature map — a grid of vectors encoding edges, textures, depth cues, object boundaries across the full image.
- Deliberate design: separate this expensive step from prompting so you pay the cost once and re-use for many prompts.

**Step 2 — Prompt Encoder** (cheap, runs per-prompt)
- Takes your hint: bounding box, point(s), coarse mask, or free-form text (SAM2).
- Converts hints to lightweight embedding tokens: "the object is somewhere in this region."
- Cost is negligible relative to Step 1.

**Step 3 — Mask Decoder** (cheap, runs per-prompt)
- Cross-attends image features (Step 1) with spatial hint (Step 2).
- Returns **three candidate masks** with IoU confidence scores — pick the best one.
- `multimask_output=True` gives all three; useful for ambiguous prompts.

### SAM 2 Additions Over SAM 1

- **Memory module**: memory attention + memory encoder + memory bank. Enables tracking across video frames by conditioning on past frames.
- **Hiera backbone**: Hierarchical ViT — faster and lighter than SAM 1's plain ViT-H. Multiple size variants from Tiny to Large.
- **Streaming architecture**: for video, each frame is processed in order, memory bank updated after each prediction.
- **Images treated as single-frame video**: SAM 2 processes a still image as a video of length 1 — so image segmentation in SAM 2 uses the same code path as video.
- **Better image performance**: SAM 2 outperforms SAM 1 on image segmentation benchmarks despite being lighter, due to better pretraining (SA-V dataset) and architecture improvements.

---

## 2. SAM 2 Model Variants — VRAM and Speed

| Variant | Backbone | Params | VRAM (approx) | Speed | Fits 4 GB GPU? |
|---|---|---|---|---|---|
| SAM 2 Hiera-Tiny | Hiera-T | ~39 M | ~1.5 GB | Fastest | YES |
| SAM 2 Hiera-Small | Hiera-S | ~46 M | ~2 GB | Fast | YES |
| SAM 2 Hiera-Base+ | Hiera-B+ | ~80 M | ~3.5 GB | Medium | YES (tight) |
| SAM 2 Hiera-Large | Hiera-L | ~224 M | ~5.5–7 GB | Slower | NO (use Colab) |
| SAM 1 ViT-H | ViT-H | 636 M | ~7–8 GB | Slow | NO |
| SAM 1 ViT-B | ViT-B | 91 M | ~2.5 GB | Fast | YES |
| MobileSAM | TinyViT | ~10 M | <1 GB | Fastest | YES |

**Recommendation for this pipeline:**
- **Local 4 GB GPU**: SAM 2 Hiera-Tiny (best quality-per-VRAM tradeoff)
- **Colab T4 (15 GB)**: SAM 2 Hiera-Large (best masks)

---

## 3. Why SAM Corrects Gemini's Imprecise Bounding Boxes

The mask decoder is NOT doing a simple crop-and-threshold. It reasons over the FULL image feature map (produced in Step 1) while being guided by the box hint from Step 2.

When Gemini gives a loose box that includes drop shadow:
- The decoder sees: phone has crisp silhouette + consistent texture; shadow has soft gradient edge + different color signature.
- Output mask traces actual phone boundary, not the box boundary.

```
Gemini's loose box:          SAM's output mask:
┌─────────────────────┐      ┌─────────────────────┐
│  (too loose,        │      │                     │
│   includes shadow)  │      │   ████████████████  │
│   ┌──────────────┐  │  →   │   ████████████████  │
│   │ actual phone │  │      │   ████████████████  │
│   └──────────────┘  │      │   ████████████████  │
│   shadow here       │      │                     │
└─────────────────────┘      └─────────────────────┘
```

**Key insight**: Gemini only needs to be SEMANTICALLY correct (front vs. back, which occludes which). Coordinate precision doesn't matter — SAM corrects it. The only unrecoverable Gemini failure is semantic (wrong label, missed view, reflection labeled as 3rd view).

---

## 4. Modal vs. Amodal — The Critical Distinction

**Modal segmentation**: labels only VISIBLE pixels. SAM is purely modal.
**Amodal segmentation**: labels the full true extent including hidden parts.

For our pipeline:
- Front panel: no occlusion → SAM's modal mask = clean, complete, no holes.
- Back panel: 20% hidden under front panel → SAM's modal mask has a phone-shaped bite out of it at the occlusion boundary.

If you feed the modal back-panel mask directly to a 3D reconstruction model, the model sees a physically incomplete object and interprets the hole as concave geometry — a literal notch in the 3D mesh. This is why pix2gestalt (amodal completion) is necessary before 3D reconstruction.

---

## 5. SAM Prompt Types and When to Use Each

### Bounding Box Prompt (primary for our pipeline)
```python
predictor.set_image(image_rgb)
masks, scores, logits = predictor.predict(
    box=np.array([xmin, ymin, xmax, ymax]),
    multimask_output=True
)
best_mask = masks[np.argmax(scores)]
```

### Point Prompt (positive + negative points)
```python
masks, scores, logits = predictor.predict(
    point_coords=np.array([[cx, cy], [sx, sy]]),  # centroid + negative
    point_labels=np.array([1, 0]),                 # 1=positive, 0=negative
    multimask_output=True
)
```

### Combined Box + Points (most robust for our pipeline)
```python
masks, scores, logits = predictor.predict(
    box=np.array([xmin, ymin, xmax, ymax]),
    point_coords=np.array([[cx, cy]]),   # centroid positive point
    point_labels=np.array([1]),
    multimask_output=True
)
```

### Text Prompt (SAM 2 + Grounding DINO = Grounded-SAM)
- SAM 2 itself doesn't support text natively.
- Grounding DINO: text → bounding box; then SAM segments from that box.
- Text like "rear panel of smartphone" or "camera island" — bypass for when Gemini fails.

---

## 6. Phone-Specific Failure Modes and Fixes

### Glass/screen highlights → spurious mask notches
- **Fix**: `multimask_output=True`, pick largest mask by pixel count.

### Drop shadows bleeding into mask
- **Fix**: add negative SAM points inside shadow region (pixel coords of shadow center).

### Near-black phone on dark background (low contrast)
- **Fix**: combine box prompt with positive centroid point. Or use BiRefNet / Matte Anything for clean alpha-matte edges.

### Camera island getting included in back-panel mask inconsistently
- **Fix**: camera island IS part of the back panel — don't add negative points there.

### Front/back panels too close → single mask covers both
- **Fix**: ensure Gemini gives separate bounding boxes; if overlapping, use `multimask_output` and select by area/shape heuristic.

### Reflective chrome frame → boundary wander
- **Fix**: post-process with morphological operations (erode + dilate) on the raw mask.

---

## 7. The Gemini ↔ SAM Feedback Loop

This is the core validation loop in the pipeline (from preliminary research):

**After SAM produces masks:**
1. Render masks as colored overlays on the original composite.
2. Send back to Gemini — now answering an EASIER question: "does this colored region correctly cover the front panel?"

**Gemini checks:**
- Mask bleed (front mask bleeding into back at seam)
- Mask incompleteness (back mask missing visible corners)
- Shadow inclusion (soft gradient extension = shadow, not device)
- Occlusion boundary correctness
- Semantic content check (display surface = front; camera island = back — catches label swaps)
- Spurious third region (reflection artifact)

**Gemini returns structured JSON:**
```json
{
  "validation_passed": false,
  "issues": [
    {
      "mask": "back",
      "issue_type": "shadow_included",
      "description": "Back mask extends ~40px below phone bottom edge into gradient shadow",
      "suggested_fix": {
        "add_negative_points": [[412, 891], [380, 903]],
        "tighten_bbox_bottom": 847
      }
    }
  ]
}
```

**Loop termination:**
1. `validation_passed: true` → use these masks.
2. Max 3 iterations → use highest-IoU masks seen, flag as "auto-validated with minor issues."
3. Same issues twice with no improvement → escalate to fallback chain.

---

## 8. Fallback Chain When Gemini Semantics Fail

1. **Gemini bbox → SAM → Gemini validation loop** (primary path)
2. **Grounded-SAM** (Grounding DINO text → box → SAM): "rear panel of smartphone" bypasses Gemini's initial parse entirely.
3. **User click in browser UI**: one point click per view, SAM segments from that seed — turns failure into a feature.

All three paths produce identical modal mask output; everything downstream is identical.

---

## 9. Installation and Setup (SAM 2)

### From PyPI (recommended)
```bash
pip install sam-2
```

### From GitHub source
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### Checkpoint download
```bash
# Hiera-Tiny (recommended for local 4 GB GPU)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

# Hiera-Large (recommended for Colab T4)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Python usage pattern
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build model
sam2_model = build_sam2(
    config_file="sam2_hiera_t.yaml",  # or sam2_hiera_l.yaml
    ckpt_path="sam2.1_hiera_tiny.pt",
    device=device
)
predictor = SAM2ImagePredictor(sam2_model)

# Set image (runs encoder once — expensive)
import numpy as np
from PIL import Image
image = np.array(Image.open("composite.jpg").convert("RGB"))
predictor.set_image(image)

# Predict from bounding box
masks, scores, logits = predictor.predict(
    box=np.array([xmin, ymin, xmax, ymax]),
    multimask_output=True
)
best_mask = masks[np.argmax(scores)]  # shape: (H, W), dtype: bool
```

---

## 10. SAM 2 vs. SAM 1 — Practical Differences for Our Pipeline

| Property | SAM 1 ViT-H | SAM 2 Hiera-Large | SAM 2 Hiera-Tiny |
|---|---|---|---|
| Image segmentation quality | High | Higher | Good |
| VRAM | 7–8 GB | 5.5–7 GB | 1.5 GB |
| Speed (image) | ~0.15 s | ~0.12 s | ~0.04 s |
| API compatibility | Old API | New API | New API |
| Video support | No | Yes | Yes |
| Text prompt support | No (with Grounding DINO) | No (same) | No (same) |
| Installation | pip install segment-anything | pip install sam-2 | pip install sam-2 |

For our pipeline: **SAM 2 is the clear winner** — better quality, lower VRAM, faster.

---

## 11. Key Coordinate Conventions

### Gemini's coordinate convention
- Format: `[ymin, xmin, ymax, xmax]`, integers normalized to **0–1000**.
- Convert to pixel space:
```python
img_h, img_w = image.shape[:2]
box = [124, 38, 891, 512]   # Gemini output [ymin, xmin, ymax, xmax]
ymin = int(box[0] / 1000 * img_h)
xmin = int(box[1] / 1000 * img_w)
ymax = int(box[2] / 1000 * img_h)
xmax = int(box[3] / 1000 * img_w)
```

### SAM's coordinate convention
- Format: `[xmin, ymin, xmax, ymax]` — NOTE: x first, opposite of Gemini.
- Always convert Gemini → SAM by swapping axes.

```python
# Gemini [ymin, xmin, ymax, xmax] → SAM [xmin, ymin, xmax, ymax]
sam_box = np.array([xmin, ymin, xmax, ymax])
```

---

## 12. Colored Mask Overlay for Gemini Validation

```python
import numpy as np
from PIL import Image

def render_mask_overlay(composite_rgb, masks_dict, alpha=0.5):
    """
    masks_dict: {"front": binary_mask_array, "back": binary_mask_array}
    Returns PIL Image with colored overlays.
    """
    overlay = composite_rgb.copy().astype(float)
    colors = {
        "front": np.array([0, 120, 255]),   # blue
        "back":  np.array([255, 80, 0]),    # orange
    }
    for label, mask in masks_dict.items():
        color = colors[label]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )
    return Image.fromarray(overlay.astype(np.uint8))
```

---

## 13. Extracting RGBA Crops from SAM Masks

```python
def extract_rgba_crop(image_rgb, mask):
    """
    image_rgb: np.ndarray (H, W, 3)
    mask: np.ndarray (H, W) bool
    Returns: PIL Image RGBA
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = (mask * 255).astype(np.uint8)
    
    # Tight crop around mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped = rgba[rmin:rmax+1, cmin:cmax+1]
    return Image.fromarray(cropped, 'RGBA')
```

---

## 14. SAM 2.1 Updates (September 2024)

SAM 2.1 is the latest release (September 2024), with improved checkpoints over the original SAM 2 (July 2024):
- Better performance on small objects.
- Improved memory efficiency for video.
- Updated checkpoint naming: `sam2.1_hiera_*.pt` (vs. `sam2_hiera_*.pt` for original SAM 2).
- Same API — drop-in replacement.

**Always use SAM 2.1 checkpoints** (`sam2.1_*`), not original SAM 2 checkpoints.

---

## 15. Grounded-SAM 2 Integration (Fallback Path)

For when Gemini's spatial reasoning fails — use text prompt directly to drive segmentation.

```bash
pip install groundingdino-py
```

```python
from groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Step 1: Grounding DINO → bounding boxes from text
gd_model = load_model("groundingdino_swint_ogc.pth", ...)
boxes, logits, phrases = predict(
    model=gd_model,
    image=image_transformed,
    caption="rear panel of smartphone . front face of smartphone",
    box_threshold=0.35,
    text_threshold=0.25
)

# Step 2: SAM 2 → masks from those boxes
predictor.set_image(image_rgb)
for box in boxes:
    masks, scores, _ = predictor.predict(
        box=box.numpy(),
        multimask_output=True
    )
    best = masks[np.argmax(scores)]
    # use best mask...
```

---

## 16. SAM on the S25 Ultra Composite — Expected Behavior

Given the Samsung S25 Ultra marketing composite (front panel left/center, back panel right with upper-left occluded by front):

**Front panel mask (expected)**:
- Clean full mask covering entire front display surface.
- No holes — front panel is fully visible.
- Sharp boundary at bezel edges.
- Punch-hole camera included in mask (it's part of the front).

**Back panel mask (expected)**:
- Modal mask — covers only VISIBLE portion of back panel.
- Straight-edge cut at the occlusion boundary (where front panel overlaps).
- Camera island included.
- Missing ~20% in upper-left quadrant (the occluded portion).
- This modal mask has a "bite" out of it — input to pix2gestalt.

---

## 17. Performance Notes and Batching

### Image encoder caching
```python
# Call set_image() ONCE, then run multiple predict() calls for different prompts
predictor.set_image(image_rgb)

# Segment front panel
front_masks, front_scores, _ = predictor.predict(box=front_box, multimask_output=True)

# Segment back panel (reuses cached image encoding — fast)
back_masks, back_scores, _ = predictor.predict(box=back_box, multimask_output=True)
```

### Batch inference (SAM 2)
```python
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Process multiple images at once
with torch.inference_mode():
    predictor.set_image_batch([img1, img2])
    masks_batch, scores_batch, _ = predictor.predict_batch(
        box_batch=[box1, box2],
        multimask_output_batch=[True, True]
    )
```

---

## 18. Data Flow in the Pipeline

```
composite.jpg
  ↓ Gemini 2.5 Flash → JSON: {front_box, back_box, occlusion info}
  ↓ Coordinate conversion: Gemini [ymin,xmin,ymax,xmax]×1000 → SAM [xmin,ymin,xmax,ymax] px
  ↓ SAM 2 set_image(composite_rgb)  ← image encoder runs once
  ↓ SAM 2 predict(front_box) → front_mask (clean, no holes)
  ↓ SAM 2 predict(back_box)  → back_mask_modal (hole at occlusion boundary)
  ↓ render_mask_overlay(composite, {front: ..., back: ...})
  ↓ Gemini validation → JSON issues
  ↓ [if issues] inject corrected bbox / negative points → re-run SAM predict()
  ↓ [repeat up to 3×]
  ↓ validated front_mask.png + back_mask_modal.png
  → pix2gestalt (amodal completion of back)
```

---

## 19. References / Links

- SAM 2 paper: "SAM 2: Segment Anything in Images and Videos" (Ravi et al., 2024) https://arxiv.org/abs/2408.00714
- SAM 2 GitHub: https://github.com/facebookresearch/sam2
- SAM 2 checkpoints: https://dl.fbaipublicfiles.com/segment_anything_2/092824/
- Grounded-SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- SA-1B dataset: https://segment-anything.com/dataset/index.html
- SA-V dataset (video, SAM 2 training): https://ai.meta.com/datasets/segment-anything-video/

---

## 20. Web Research Findings (2026-05-12)

*[Section being populated from live web searches — see below]*
