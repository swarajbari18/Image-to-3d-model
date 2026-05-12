# From flat panels to a spinning phone: an 8-part pipeline blueprint

**This report gives you a fully buildable, one-week recipe for converting a Samsung S25 Ultra composite marketing image into a correct, browser-renderable 3D mesh — using only pretrained models on a 4 GB GPU plus free Colab, no fine-tuning.** The crux is replacing Naya AI's "extrude the whole image" failure with a sequenced pipeline: a VLM parses the composite into labeled views, SAM segments each one, an amodal completion model reconstructs the full occluded back view, a feed-forward 3D generator (Hunyuan3D-2mv is the headline choice) ingests both views, the mesh is rescaled to the published 162.8 × 77.6 × 8.2 mm dimensions, and Three.js renders it side-by-side with a deliberately broken "flat-panels" baseline.

The conceptual reason Naya's pipeline fails is simple: their system has no operator for *parsing* a composite. It treats RGB-as-geometry, so the front-and-back side-by-side image becomes two flat rectangles in the depth direction. **Every single component below exists to inject a missing prior** — that the image contains two views of one object, that an occluded silhouette has a true extent behind the occluder, that a phone has thickness, that thickness is exactly 8.2 mm. The biggest risks are reflective-surface artifacts in depth/3D models and hallucinated geometry during amodal completion; each component below has explicit fallbacks for these.

A key architectural decision running through the pipeline is **Gemini-in-the-loop validation**. Rather than treating Gemini's initial parse and SAM's segmentation as one-shot operations, we build automated feedback loops at two critical junctions: between Gemini and SAM (Gemini validates the masks and can correct them), and between Gemini and pix2gestalt (Gemini validates the amodal completion and triggers targeted inpainting to fix specific errors). This eliminates the need for a human-in-the-loop during the demo while making the self-correcting nature of the pipeline a feature in its own right.

---

## Component 1: how a VLM reads a composite as "two views of one phone"

A Vision-Language Model is mechanically a large language model with a *vision adapter* glued to its front. The image is sliced into 14×14 or 16×16 patches, each patch becomes a token via a small linear "patch-embedding" layer, a vision transformer makes those tokens context-aware, and a projector remaps them into the LLM's embedding space. **Crucially, there is no special detector head in Gemini 2.5 — bounding boxes are emitted as literal text tokens**, predicted one digit at a time, learned during post-training on detection-style prompts. This is why prompt format matters disproportionately: the model is pattern-matching on a tokenization distribution.

What tells a VLM "two views, one phone" rather than "two phones"? A combination of identity cues (same titanium color, same proportions, same brand wordmark), view-geometry cues (front shows glass+punch-hole+display, back shows camera island+logo — mutually exclusive on a single device), staging cues (mirrored placement, equal scale, uniform gradient backdrop typical of press shots), and the explicit occlusion T-junction where the back-view silhouette is clipped exactly along the front-view's boundary. Marketing composites have a learned stylistic signature; the model has seen millions of them.

**Gemini 2.5 is the right pick** because Google explicitly post-trained it for spatial output. The coordinate contract is fixed: `box_2d = [ymin, xmin, ymax, xmax]`, integers normalized to **0–1000** (as if the image were resized to 1000×1000). Reordering to `[xmin, ymin, ...]` collapses accuracy because the model was tokenized on the specific order. Set `response_mime_type="application/json"` and attach a Pydantic schema via `response_schema` to force valid output. Gemini 2.5 Flash is recommended for spatial tasks; 2.5 Pro for harder reasoning.

### What Gemini's output actually looks like

When you send the composite S25 Ultra image with the right prompt, you get back a JSON blob like this:

```json
{
  "is_composite": true,
  "views": [
    {
      "label": "front face of Samsung S25 Ultra",
      "view_type": "front",
      "box_2d": [124, 38, 891, 512],
      "z_order": 0,
      "visibility": "full",
      "occlusion": {
        "is_occluded": false,
        "occluded_by": [],
        "estimated_occluded_fraction": 0.0,
        "occlusion_type": null,
        "occluded_region": null
      }
    },
    {
      "label": "rear panel of Samsung S25 Ultra",
      "view_type": "back",
      "box_2d": [98, 489, 876, 962],
      "z_order": 1,
      "visibility": "partial",
      "occlusion": {
        "is_occluded": true,
        "occluded_by": ["front face of Samsung S25 Ultra"],
        "estimated_occluded_fraction": 0.20,
        "occlusion_type": "same_composite_overlap",
        "occluded_region": "upper_left"
      }
    }
  ]
}
```

The `occlusion` block is the **pipeline routing signal**. The field `occlusion.is_occluded` tells downstream code whether to run amodal completion at all. If it is `false` for a view, the SAM crop is used directly — no pix2gestalt, no inpainting, no validation loop for that view. If it is `true`, `estimated_occluded_fraction` determines which completion strategy to invoke (see Component 3 routing logic). This means for images where both views are fully visible — a clean front/back pair with no overlap — the entire amodal completion stage is skipped entirely, saving compute and removing a potential failure point.

To convert bounding box coordinates to real pixel coordinates:

```python
img_h, img_w = image.shape[:2]
box = [124, 38, 891, 512]   # Gemini output [ymin, xmin, ymax, xmax]
ymin = int(box[0] / 1000 * img_h)
xmin = int(box[1] / 1000 * img_w)
ymax = int(box[2] / 1000 * img_h)
xmax = int(box[3] / 1000 * img_w)
```

That pixel-space rectangle is what you hand to SAM as a bounding box prompt.

A working prompt asks for `views[]` with `label`, `box_2d`, `z_order` (0 = frontmost), and the full `occlusion` block. Explicitly instruct: *"Do not include drop shadows. Ignore reflections. Treat camera lenses as part of the object, not separate objects. For each view, estimate what fraction of its total area is hidden behind another object."*

### Gemini's failure modes and why they don't matter much

Drop shadows inflate boxes vertically — fix with the explicit instruction above. Glass reflections occasionally produce spurious tiny "reflection" boxes — filter by area (<1% of image). Color-matched dark-on-dark backgrounds make box edges wander 10–30 pixels.

**The critical insight: never trust Gemini's box edge for the final crop.** Gemini's job is to be *semantically correct* — to identify that there are two views, which is front and which is back, and which occludes which. Coordinate precision does not matter because SAM corrects it in the next step. The only failure mode that truly breaks things is Gemini being *semantically wrong* — confusing front for back, missing a view, or labeling a reflection as a third view. That is why fallbacks exist.

The output is a small JSON blob: front (z_order 0, no occluders) and back (z_order 1, occluded_by ["front"]), plus approximate pixel bboxes. That's the handoff to Component 2.

---

## Component 2: segmenting two views — and understanding what SAM actually does with imprecise coordinates

### What SAM is

**Segment Anything Model (SAM)** is Meta's promptable segmentation foundation model. The architecture has three parts that run in strict sequence:

**Step 1 — the image encoder** is a heavy ViT (~636M params for ViT-H) that runs *once* on the full image and produces a rich 1/16-resolution feature map — a grid of vectors where each vector describes "what does this region look like, in context of the whole image." This step is expensive and SAM's design deliberately separates it from prompting so you only pay this cost once. At this point SAM has encoded everything: object boundaries, texture changes, edges, depth cues — all of it compressed into the feature grid.

**Step 2 — the prompt encoder** takes your bounding box (or point, or coarse mask) and converts it into lightweight embedding tokens. These tokens carry one message: "the object you should find is somewhere in this spatial region." This is a very cheap operation — the heavy work already happened in Step 1.

**Step 3 — the mask decoder** takes the image features from Step 1 and the spatial hint from Step 2, cross-attends them, and produces a precise binary mask. It returns three candidate masks plus IoU confidence scores so you can pick the best one.

SAM was trained on **SA-1B** — 11 million licensed images with 1.1 billion auto-labeled masks — which is what gives it true zero-shot generalization to any object category.

### Why SAM corrects Gemini's imprecise coordinates

The decoder is not doing a simple crop-and-threshold. It is doing learned reasoning over the full image feature map, *guided* by the box. When Gemini gives a slightly loose bounding box that includes some background shadow, the decoder looks at the features inside that region and reasons: the phone has a crisp silhouette with consistent texture, the shadow has a soft gradient edge and a different color signature — these are clearly two different things. The output mask traces the actual phone boundary, not the box boundary.

Concretely:

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

The mask follows the actual phone silhouette, not the box edge. This is SAM doing exactly what it was designed for — the box is a *hint about where to look*, not a *definition of what the mask is*.

### What SAM cannot correct

SAM can fix imprecise *coordinates* but it cannot fix wrong *semantics*. Three failure cases:

**Wrong object entirely.** If Gemini's box for the "back panel" accidentally centers on a large drop shadow, SAM will faithfully segment the shadow — it has no idea the box was semantically wrong.

**Two objects inside one box.** If front and back are close together and Gemini returns a single box containing both, SAM will segment the dominant object (probably the front) and miss the back.

**The occlusion boundary.** Where the front panel sits on top of the back, SAM correctly labels the *visible* pixels of the back and excludes the *covered* pixels. It draws the mask boundary right at the occlusion edge — a straight line where the front panel cuts across the back. That visible-only mask is called a **modal mask**. The hole that remains is what Component 3 fills.

### Modal vs amodal — the key distinction

**Modal segmentation** labels only what is currently visible. SAM is purely modal. **Amodal segmentation** labels the full true extent of the object including what is hidden. These are fundamentally different tasks. SAM's back-view mask is a phone-shape with a phone-shaped bite taken out of it at the occlusion boundary — the 20% that was hidden under the front. If you crop that directly and feed it to a 3D reconstruction model, the model sees a physically incomplete object and interprets the hole as concave geometry, producing a literal phone-shaped notch in the mesh. This is why the amodal completion step is necessary.

### VRAM by SAM variant

| Variant | Params | VRAM | 4 GB GPU | Colab T4 |
|---|---|---|---|---|
| SAM ViT-H | 636 M | ~7–8 GB | ❌ | ✅ |
| SAM ViT-B | 91 M | ~2.5 GB | ✅ | ✅ |
| SAM 2 Hiera-Tiny | 39 M | ~1.5 GB | ✅ | ✅ |
| MobileSAM | 10 M | <1 GB | ✅ | ✅ |

**Recommended**: SAM 2 Hiera-Tiny locally; SAM ViT-H on Colab for maximum mask quality.

### Phone-specific failures and fixes

Glass-screen highlights can create spurious mask notches — use `multimask_output=True` and pick the largest mask. Drop shadows can bleed into the mask — add background "negative" points inside the shadow region. Near-black phone on charcoal background — combine the box prompt with a positive centroid point, or refine edges with **BiRefNet** or **Matte Anything** for clean alpha edges.

The output per view is a binary mask PNG plus an RGBA crop. The front view is clean with no holes. The back view has a hole where the front occludes it — that modal mask is the input to Component 3.

### The Gemini↔SAM automated feedback loop

Rather than treating Gemini's initial parse and SAM's segmentation as a one-shot handoff, we build an automated validation loop. After SAM produces masks, we render them as colored overlays on the original composite image and send that back to Gemini. Gemini is now answering a much easier question than before — not "where are the objects?" but "does this colored region correctly cover the front panel without bleeding into the background or the back panel?" This asymmetry is important: initial reasoning over an unseen image is hard; validating a proposed answer over a visual is easy.

**What Gemini checks in the validation step:**

- **Mask bleed** — does the front mask extend beyond the physical edge of the front panel into the back panel at the overlap seam?
- **Mask incompleteness** — does the back mask cover all *visible* portions of the rear panel including its corners and frame?
- **Shadow inclusion** — does either mask have a soft gradient extension that is clearly a drop shadow rather than the device surface?
- **Occlusion boundary correctness** — does the back mask trace the right boundary at the overlap, neither cutting into visible back-panel area nor including front-panel pixels?
- **Semantic content check** — does the front mask contain a display surface with a punch-hole camera? Does the back mask contain a camera island? This catches front/back label swaps.
- **Spurious third region** — are there more than two masked regions? If so, which is a reflection or shadow artifact to discard?

**What Gemini returns to drive re-prompting:**

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
    },
    {
      "mask": "back",
      "issue_type": "occlusion_boundary_aggressive",
      "description": "Back mask cuts into visible upper-left corner of back panel",
      "suggested_fix": {
        "adjust_bbox_left": 312
      }
    }
  ]
}
```

The code acts on this directly: add negative SAM points at those pixel coordinates, adjust the bbox, re-run SAM, re-render the overlay, re-validate. No human needed at any step.

**Termination conditions** — the loop needs three exit rules to avoid running forever:

1. Gemini returns `"validation_passed": true` → stop, use these masks.
2. Maximum iterations reached (3) → stop, use the highest-quality masks seen so far (tracked by SAM's IoU confidence scores across iterations), flag in the UI as "auto-validated with minor issues."
3. Same issues raised twice in a row with no mask improvement → feedback is not helping; escalate to the fallback chain below.

**An honest limit of this loop:** Gemini is validating its own upstream reasoning. If it fundamentally misunderstood the image in round one (e.g., mistook front for back), it may perpetuate that error in validation. The semantic content checks (display surface vs camera island) give it a chance to catch its own earlier mistake, but they are not guaranteed. This is why the fallback chain still exists.

### Fallback chain when Gemini's semantics fail

Since Gemini only provides spatial hints (which SAM corrects anyway), there are three ways to get SAM a valid prompt:

1. **Gemini bbox → SAM → Gemini validation loop** (primary path)
2. **Grounded-SAM** (Grounding DINO detects from text like "rear panel of smartphone" → box → SAM) — bypasses Gemini's initial parse entirely, useful when the initial semantic reasoning fails
3. **User click in the browser UI** — one point click per view, SAM segments from that seed — turns a failure mode into a feature

All three paths produce the same modal mask output. Everything downstream is identical regardless of which path was taken. Building the click-fallback into the demo UI is worth doing regardless — a Naya engineer will immediately ask "what if the VLM gets it wrong?" and a visible correction mechanism is a better answer than "it never fails."

---

## Component 3: amodal completion — reconstructing the full back view

### Why we chose amodal completion over inpainting

This is the most important architectural decision in the pipeline's front half. Both approaches fill the missing 20% of the back view, but they operate on fundamentally different conceptual levels.

**Inpainting** asks: *"fill in this hole with something plausible given the surrounding pixels."* It is a 2D texture continuation problem. The model looks at the visible edges and colors around the hole and tries to extend them smoothly. It has no understanding that the hole exists *because another physical object is sitting on top*. The prior it uses is "what texture would look consistent here" — not "what is the true shape of this partially hidden object." You can guide it with text prompts and reference images (more on this below), but at its core it is filling a gap, not understanding an occlusion.

**Amodal completion** asks a fundamentally different question: *"this object is partially hidden — what is its complete true shape including the hidden part?"* It treats occlusion as a physical concept, not a visual gap. **Pix2gestalt** (Columbia, CVPR 2024 Highlight) was trained specifically on (occluded object, complete object) pairs. When it sees a phone-shaped back panel disappearing behind another phone-shaped front panel at a straight edge, it has learned that the hidden region continues the silhouette of the back panel — not just the texture, but the full geometric extent of the object.

This distinction matters enormously for downstream 3D reconstruction. The 3D model's outer mesh shape is driven primarily by the *silhouette* — the boundary of the object in the image. Getting the silhouette wrong produces a fundamentally wrong shape. Inpainting might get the texture right in the hole but still produce a subtly wrong boundary if it doesn't understand that the back panel is a complete rectangle behind the front panel. Amodal completion gets the boundary right by design, because recovering the full extent of an occluded object is literally what it was trained to do.

**The tradeoff table:**

| Property | Inpainting (SD + IP-Adapter) | Amodal Completion (pix2gestalt) |
|---|---|---|
| Understands occlusion physically | ❌ No | ✅ Yes |
| Recovers correct silhouette boundary | ⚠️ Sometimes | ✅ By design |
| Texture quality in filled region | ✅ High (diffusion) | ⚠️ Moderate |
| Guidable with reference image | ✅ Via IP-Adapter | ❌ Not directly |
| Hallucination risk | Medium (wrong lens count) | Low (shape-focused) |
| VRAM needed | 4–8 GB | ~6–8 GB fp16 |
| Runs on 4 GB local | ✅ SD 1.5 with offload | ❌ (use Colab) |
| Runs on Colab T4 | ✅ | ✅ |

**Decision: use amodal completion (pix2gestalt) as the primary path.** The silhouette is more important than texture quality for 3D reconstruction, and the VRAM constraint is solved by running on Colab T4. Inpainting is retained as a fallback specifically for texture quality — if pix2gestalt's completed region looks tonally wrong, a light img2img harmonization pass with SD at low strength (~0.3 denoising) can fix texture without disturbing the correctly recovered shape.

### How pix2gestalt works

Pix2gestalt takes two inputs: the full RGB composite image and the modal mask of the occluded object (the back panel with the hole). It runs a conditional diffusion process — similar to SD 1.5's architecture but conditioned on both inputs — to produce the complete unoccluded object on a clean background. The key is that the conditioning signal includes not just the visible pixels but the *shape context* of what an occluded object behind a known occluder should look like. It outputs a complete RGBA of the back panel with the missing 20% synthesized as a natural continuation of the visible 80%.

VRAM: ~6–8 GB in fp16. Use `enable_model_cpu_offload()` on a 4 GB local GPU (slow but works); comfortable on Colab T4 without offloading.

### Inpainting as a concept — kept for reference and fallback

Even though we chose amodal completion as the primary path, understanding inpainting is useful because it is the fallback and because the texture harmonization pass uses it. Three paradigms:

**Classical patch-based** (Criminisi 2004, PatchMatch 2009 — the engine behind Photoshop Content-Aware Fill): walks the mask boundary and copies the best-matching small patch from elsewhere in the image. Fast, deterministic, and **incapable of inventing anything not already in the image**. Useless for the camera island.

**LaMa** (2021): specialist for large masks. Central innovation is *Fast Fourier Convolutions* — a layer that works in the frequency domain giving every position an image-wide receptive field from layer one. Unconditional (no text prompt) — smoothly extends surrounding texture but won't invent the right camera arrangement. Runs in <2 GB VRAM. Good safety fallback.

**Diffusion inpainting**: at each denoising step, the model receives the noisy latent of the masked region plus the clean re-noised latent of the unmasked region plus the mask itself. A fine-tuned inpainting UNet (SD 1.5 or SDXL) takes 5 extra input channels — 4 for the encoded masked image and 1 for the mask — trained on (real image, random mask) pairs. Current state of the art for texture quality.

If inpainting is used as a fallback, four levers guide it toward the correct S25 Ultra back:
1. **Prompt engineering**: "Back of Samsung Galaxy S25 Ultra, four circular rear camera lenses, titanium frame, polished glass back, product photo." Negative: "extra lenses, watermark, text artifacts."
2. **IP-Adapter** (~22M params): injects CLIP-Vision embeddings of a reference Samsung product photo through a separate cross-attention pathway — the single most powerful lever for preserving exact camera arrangement.
3. **ControlNet** (Canny): forces the inpaint to continue visible edges into the hole.
4. **Mask dilation** of ~8 pixels: the VAE's 8× downsampling creates latent contamination at tight mask edges; dilate and composite the fill back in pixel space only where masked.

**The hardest inpainting failure**: hallucinated lens count. The S25 Ultra has four lenses; getting three or five is immediately visible in the final 3D. IP-Adapter conditioning with an official reference photo is the mitigation. Pix2gestalt avoids this risk entirely because it completes shape, not inventing specific features.

### The Gemini↔pix2gestalt automated feedback loop

After pix2gestalt completes the back view, we do not pass it straight to the 3D model. We send the completed RGBA image back to Gemini for validation — the same principle as the Gemini↔SAM loop, but now Gemini is checking semantic content and visual quality rather than mask boundaries.

**What Gemini checks in this validation step — and how it generates the checks dynamically:**

This is a critical design principle: Gemini is not given a hardcoded checklist. Instead, a preparatory Gemini call fires before validation, asking: *"You identified this as the rear panel of a Samsung S25 Ultra. List 5-7 specific visual features that must be present and verifiable in a reconstruction of this view. Be concrete — name counts, positions, shapes, and expected relationships."* Gemini returns something like:

```json
{
  "object_label": "rear panel of Samsung S25 Ultra",
  "validation_criteria": [
    "exactly four circular camera lenses arranged vertically in upper-left quadrant",
    "titanium-colored rectangular frame with uniform width around perimeter",
    "Samsung wordmark centered horizontally in lower third of panel",
    "camera island raised slightly from main surface, darker material",
    "no visible seam or color discontinuity between original and synthesized regions",
    "glass back surface with consistent reflective sheen across full panel"
  ]
}
```

These self-generated criteria become the validation checklist for that specific object. For an alloy wheel the criteria would be "N spokes with rotational symmetry, uniform spoke width, center cap present" — generated by Gemini without any hardcoding. For a dress the criteria would be "neckline continues naturally, fabric pattern is consistent across seam, no abrupt texture change." **Zero object-specific knowledge is hardcoded anywhere in the pipeline.** The same code runs for any product.

**What Gemini returns:**

```json
{
  "validation_passed": false,
  "overall_score": 0.6,
  "issues": [
    {
      "issue_type": "wrong_lens_count",
      "severity": "critical",
      "description": "Three lenses visible, S25 Ultra has four. Missing lens in upper position.",
      "affected_region": [45, 38, 210, 180],
      "suggested_inpainting_prompt": "four circular camera lenses arranged vertically, top lens visible, titanium ring surround, dark glass"
    },
    {
      "issue_type": "seam_artifact",
      "severity": "minor",
      "description": "Faint horizontal line visible at y≈340 where synthesized region meets original.",
      "affected_region": [330, 0, 360, 512],
      "suggested_inpainting_prompt": "smooth titanium gray metal surface, uniform texture, no seam"
    }
  ],
  "passed_checks": ["frame_continuity", "logo_placement", "color_consistency"]
}
```

**Can pix2gestalt directly act on Gemini's natural language feedback?** No — and this is an important honest limit. Pix2gestalt is a diffusion model that takes image + mask as inputs and runs a stochastic sampling process. It has no chat interface and cannot receive an instruction like "add a fourth lens in the upper right." Its architecture simply does not support that.

**How the loop works despite this:**

The strategy is a two-stage response to Gemini's feedback, depending on what type of issue was found:

**For shape/boundary issues** (the synthesized region has the wrong silhouette extent, the seam is in the wrong place): re-run pix2gestalt with a different random seed. Diffusion models are stochastic — the same inputs with a different seed produce a different completion. Run 4 seeds in parallel and have Gemini score all of them, selecting the best. This is best-of-N sampling and it works well for shape-level variation.

**For content/detail issues** (wrong lens count, wrong camera arrangement, seam texture artifact): escalate to targeted SD inpainting. Gemini's validation JSON already contains `affected_region` (the bounding box of the problem area) and `suggested_inpainting_prompt` (the text describing what should be there). The code extracts these and runs a focused SD inpainting pass — only over the small problematic region — using the suggested prompt plus IP-Adapter conditioning with an official Samsung reference photo. Pix2gestalt handled the shape; IP-Adapter inpainting handles the content detail. This hybrid is the best of both.

**The full loop in pseudocode:**

```
pix2gestalt(modal_mask, composite_image, seeds=[1,2,3,4])
  → 4 candidate completed back images

Gemini.validate(candidates)
  → scores + issues per candidate

if best_score >= threshold:
    use best candidate → proceed

else:
    for each critical issue in best_candidate.issues:
        run SD_inpainting(
            image=best_candidate,
            mask=issue.affected_region,
            prompt=issue.suggested_inpainting_prompt,
            ip_adapter_reference=samsung_s25_back_reference.jpg
        )
    
    Gemini.validate(repaired_image)
    
    if passes: proceed
    else if iterations < max(2): loop again
    else: use best seen, flag in UI as "validated with minor issues"
```

**Why this architecture matters beyond the demo:** The Gemini validation loop over the amodal output is a pattern that extends to the entire pipeline. The same mechanism — render intermediate output, send to Gemini with a structured validation prompt, get machine-readable issues back, trigger targeted remediation — could validate the 3D mesh against the original image by re-rendering from the original camera angle. That future extension requires zero new infrastructure, just a new validation prompt. For a Naya audience, this self-correcting architecture with logged reasoning at each step is exactly what enterprise manufacturing customers need — silent failures that produce wrong cost estimates are far worse than slow pipelines with explicit quality gates.

---

## Component 4: monocular depth, and why it's *not* on the critical path

Depth is the distance along the camera's optical axis from the pinhole to the world point that projected onto a pixel. A depth map is a 2D array, same H×W as the input image, with a scalar per pixel. Monocular models output it in either "closer = brighter" (inverse depth / disparity, what MiDaS and Depth Anything produce) or "closer = darker" (true Z in meters, what depth sensors produce). **This convention bites everyone at least once** — Hugging Face Space cards typically colorize inverse depth.

How can a single image even contain depth? Through learned monocular cues: linear perspective (vanishing points), relative and familiar size, occlusion T-junctions, texture gradients (denser textures recede), shading and shape-from-shading, aerial perspective, contact shadows. A model trained on millions of (image, depth) pairs internalizes all of these jointly.

**Depth Anything V2** (NeurIPS 2024) is the current general-purpose SOTA — DINOv2 ViT backbone + DPT decoder, trained on 595K synthetic labeled + 62M real pseudo-labeled images. Four sizes from ViT-S (25M, ~1 GB) to ViT-G (1.3B). **Marigold** is a Stable-Diffusion-based depth model with extremely crisp edges but 10–50× slower (iterative denoising) and 6–8 GB VRAM. **Metric3D v2** outputs true millimeter depth via a canonical-camera-space trick. **DUSt3R/MASt3R** consume two or more images and output joint depth+pose+pointmaps — but for our front-vs-back pair this fails because the views share zero matchable surface content.

**What does Depth Anything V2 actually output?** Relative inverse depth, post-normalized to [0,1] for display. Values describe ordering up to an unknown affine transform `Z_pred = a·Z_true + b`. The reason models produce relative depth is dataset mixing: training on heterogeneous datasets (indoor scenes in meters, stereo movies in disparity pixels, synthetic) only works if the loss is invariant to per-dataset scale and shift. Converting relative → metric needs ground-truth anchors — given any source of true distance (e.g., known phone thickness 8.2 mm), fit two scalars by linear least squares.

Phone-specific failures are severe: flat textureless glass has no monocular cues so disparity drifts toward the mean; reflective surfaces cause the model to estimate the depth of the *reflected scene* rather than the screen surface; sharp bezel edges get smoothed into multi-pixel ramps; the camera island bump (~1–1.5 mm proud) almost never resolves correctly.

**Is depth estimation even necessary?** Honest answer: **not on the critical path, but useful for adjacent jobs.** Modern image-to-3D models like Hunyuan3D-2 do depth-like reasoning internally and don't ask for an explicit depth input. Explicit depth is useful for: (a) sanity-checking metric scale against the spec sheet via Metric3D, (b) validating the reconstructed mesh by re-rendering depth and comparing, (c) post-hoc refinement via short differentiable-rendering passes, and (d) building a fallback "2.5D extruded shell" from silhouette + depth + known dimensions if the generative model fails entirely. Run Depth Anything V2 in parallel, but don't bet the demo on it.

---

## Component 5: from front + back views to a coherent textured mesh

This is the heart of the pipeline, and the one place where the breadth of approaches matters.

### Why classical methods don't work here

Structure-from-Motion / photogrammetry (COLMAP) is the production solution when you have many overlapping photos. It runs SIFT feature detection, mutual-nearest-neighbor matching, RANSAC essential-matrix geometric verification, bundle adjustment of joint poses+points, and dense Multi-View Stereo. **It catastrophically fails for our case** because front-view and back-view share zero matchable surface points — no correspondences means no triangulation means no pose recovery. Even hand-supplied poses don't help because MVS still needs overlapping coverage to triangulate, and the *sides* of the phone (SIM tray, buttons, USB-C) appear in neither view.

This problem is genuinely ill-posed and demands a generative prior — knowledge that the world contains coherent 3D objects with typical thicknesses and that smartphones are thin slabs with rounded corners and camera bumps. That prior comes from a generative model trained on Objaverse (~800K assets) or Objaverse-XL (10M+).

### The conceptual landscape of generative 3D

**NeRF** (Neural Radiance Field): a small MLP that maps (x,y,z, view direction) → (RGB, density). Rendering is volumetric ray-marching with alpha compositing. Requires per-scene optimization with many calibrated views. Not directly usable here; conceptually important because it is the representation many other methods optimize into.

**3D Gaussian Splatting**: a scene as millions of explicit 3D Gaussian blobs, each with position, scale, rotation quaternion, opacity, and spherical-harmonic colors. Rendering is GPU rasterization — 100× faster than NeRF. Explicit and editable. Also requires many calibrated views in its native form.

**Score Distillation Sampling (DreamFusion)**: use a frozen 2D diffusion model as a 3D prior. Parametrize a NeRF, render it from random viewpoints, ask the 2D model to denoise the rendering, push the noise-prediction error as gradient back into the NeRF. **Slow (30 min – several hours), Janus-faced artifacts on hard surfaces, hard to dual-condition on front+back.** Not for a one-week demo.

**Feed-forward reconstruction models** — the practical winner. Train one large network on Objaverse-scale 3D data; at inference time, a single forward pass maps image(s) → 3D in 5–30 seconds. No per-scene optimization. The chronology: Zero-1-to-3 → LRM → TripoSR → InstantMesh → SF3D → **TRELLIS (Nov 2024)** → **Hunyuan3D-2 (Jan 2025)** → **Hunyuan3D-2mv (March 2025)** → Hunyuan3D-2.1.

### The current open-source leaders

| Model | Input | Output | Speed (A100) | VRAM | License | Multi-view native |
|---|---|---|---|---|---|---|
| TripoSR | 1 img | mesh | <0.5 s | ~6 GB | MIT | No |
| InstantMesh | 1 img | mesh | ~10 s | ~10–16 GB | Apache 2.0 | No |
| SF3D | 1 img | UV mesh + materials | ~3 s | ~7 GB | Stability Community | No |
| TRELLIS | 1 img / text | mesh / 3DGS / RF | ~10 s | 12–16 GB | MIT | No (TRELLIS.2 yes) |
| Hunyuan3D-2 | 1 img | textured mesh | ~60 s full | 6 GB shape / 16 GB full | Tencent Community | No |
| **Hunyuan3D-2mv** | **front/back/L/R** | **textured mesh** | **~30 s** | **~10 GB** | **Tencent Community** | **Yes** |
| SPAR3D | 1 img (+ pts) | UV mesh | 0.7 s | 7–10.5 GB | Stability Community | No |
| TripoSG | 1 img | mesh | ~10 s | ~10 GB | MIT | No |

**TRELLIS** is the quality benchmark. Its central abstraction is **Structured Latents (SLAT)** — a sparse 3D grid where each voxel holds a latent feature vector. A single SLAT can be decoded by three separate decoders into 3D Gaussians, a NeRF-style radiance field, or a triangle mesh — one latent, three output formats. VRAM ~12–16 GB; marginal on free Colab T4, comfortable on Pro. MIT-licensed.

**Hunyuan3D-2mv** is the right primary choice. Its two-stage split — Hunyuan3D-DiT for bare shape, Hunyuan3D-Paint for PBR texture — decouples geometry from appearance. The **2mv variant takes `{"front": img, "back": img, "left": img, "right": img}`** — exactly our input, with left/right omitted or None. Shape stage ~6 GB; full pipeline ~10–16 GB; fits Colab T4.

### What our two-view input means

Five strategies for fusing front + back, in order of viability:

1. **Native multi-view model (Hunyuan3D-2mv) — recommended.** Designed for exactly this case.
2. Run single-view twice and fuse — fusion without correspondences is hard; not realistic.
3. Front-only with hallucinated back — bad for phones because backs are distinctive.
4. Back-only — worse; featureless fronts invent nonsense.
5. Composite 2×1 grid into single-view model — unpredictable; worth one quick experiment as sanity check.

### What a phone specifically looks like after reconstruction

Phones are adversarial for Objaverse-trained models. Thinness (depth ~5–8% of height) is rare in the training distribution, so models systematically inflate phones 1.5–3× too thick. Camera bumps a few mm proud often merge into the body. Reflective glass and metal get baked as frozen highlights into the texture. Side details (SIM tray, buttons, USB-C) appear in neither input view and will be omitted or invented.

**What works**: rough silhouette, dominant color, aspect ratio, large features. **What goes wrong**: exact thickness, sides, camera-bump geometry, lens detail. For the demo, this is "good enough for a web preview" — the message is about the *parsing capability*, not photoreal fidelity.

---

## Component 6: metric scaling — from arbitrary units to real-world dimensions

Why do reconstruction models produce arbitrary-scale outputs? Because **single-image projection is scale-ambiguous**: a die-cast toy car at 30 cm and the real car at 5.4 m, framed identically, produce indistinguishable pixels. The model picks an arbitrary convention — usually a unit cube — and the units mean nothing.

**Metric priors** are any side-channel that fixes scale: known object dimensions, camera intrinsics + reference plane, calibrated stereo, or active depth sensor. We use known dimensions where available, and fall back gracefully when they are not.

Three places to inject dimensions:
- **Pre-reconstruction**: generative models don't expose this interface. Skip.
- **During reconstruction**: soft loss feasible for NeRF/3DGS optimization, not feed-forward. Skip for one week.
- **Post-processing — the answer.** Reconstruct in arbitrary units, orient canonically, then rescale.

The recipe in trimesh: center at origin, align principal axes via PCA / `apply_obb()`, sort axes so longest → Y (height), middle → X (width), shortest → Z (depth). Resolve PCA sign ambiguity with a heuristic (camera bump on +Z, smoother surface on −Z). Compute per-axis scale factors from the provided dimensions. Use `max(s)/min(s)` as a reconstruction quality metric — under 1.05: apply non-uniform scaling; larger: apply uniform scaling by geometric mean and accept some dimensional slack.

Non-uniform scaling distorts circular features (a lens becomes elliptical), so prefer uniform scaling when proportions are off. Log all three scale factors and surface them in the UI as a "proportional accuracy" score.

### How dimensions are obtained — four strategies in priority order

**Strategy 1 — Gemini web-search grounding (fully automatic).** When Gemini identifies an object in the initial parse (e.g., "rear panel of Samsung S25 Ultra"), a follow-up Gemini call with web search grounding fires automatically: *"What are the exact external dimensions in millimeters of the Samsung Galaxy S25 Ultra? Return JSON with width_mm, height_mm, depth_mm and your source."* For any manufactured product with a published spec sheet, Gemini finds the numbers without user input. This works for phones, laptops, cars, wheels, appliances, industrial parts.

```json
{
  "object": "Samsung Galaxy S25 Ultra",
  "width_mm": 77.6,
  "height_mm": 162.8,
  "depth_mm": 8.2,
  "source": "Samsung official specifications page"
}
```

**Strategy 2 — User-provided dimensions in the UI (precise manual override).** Three number input fields in the browser — Width, Height, Depth in mm — that the user can fill in or correct. When the user types values and presses Apply, the Three.js scene reacts instantly: the mesh rescales to exactly those dimensions in real time. No pipeline re-run needed — it is a pure Three.js `.scale` operation. This is the right tool for bespoke objects (a custom alloy wheel with a known 18-inch diameter), objects where Gemini's lookup returned wrong numbers, or any case where the user wants a specific known-good measurement.

```javascript
function scaleToMM(model, targetW, targetH, targetD) {
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    model.scale.set(
        targetW / size.x,
        targetH / size.y,
        targetD / size.z
    );
}
// Called live as user types — instant feedback
inputW.addEventListener('input', () =>
    scaleToMM(model, inputW.value, inputH.value, inputD.value));
```

**Strategy 3 — Uniform scale slider (rough sizing).** A single slider that scales all three axes together while preserving proportions. Useful when the user wants to make the model "roughly phone-sized" without knowing exact numbers. Displayed alongside the per-axis fields — when the user drags the slider it updates all three fields to the new uniform values.

**Strategy 4 — Skip scaling entirely.** For objects where dimensions are unknown and unimportant for the demo (a fashion item, an abstract shape), output the mesh in normalized units and omit the dimension overlay from the UI. The mesh is still correct in shape — just not anchored to real-world millimeters. The UI shows "dimensions unknown" rather than wrong numbers.

### UI implementation — dynamic scaling controls

The Three.js `.scale` property is a 3D vector multiplied into every vertex position during the GPU draw call — changing it is essentially free, with no mesh recomputation. This means all four strategies above update the viewport instantly without reloading the GLB.

The control panel sits below the "good" model viewport:

```
┌─────────────────────────────────────────┐
│  Dimensions                             │
│  W [  77.6 ] mm  H [ 162.8 ] mm        │
│  D [   8.2 ] mm  ← auto-filled by      │
│                    Gemini lookup        │
│  Uniform scale: ──●────────── 1.0×     │
│  [ Apply ]  [ Reset ]  [ Unknown ]     │
└─────────────────────────────────────────┘
```

Auto-fill from Gemini lookup is shown with a small ✓ and source citation. If Gemini's lookup failed, fields show empty with a placeholder "enter mm". The "Unknown" button clears the overlay and switches to normalized display. All changes reflect instantly in the spinning model — engineers can drag the slider to feel the scale, then type exact numbers to lock it.

**Camera intrinsics** — don't try to recover them from the marketing image. The known-dimensions route bypasses K entirely; it's not worth the complexity.

---

## Component 7: cleaning the mesh and shipping it as GLB

A 3D mesh is built from four primitives. **Vertices** are 3D points in a flat array. **Faces** are triangles — triples of vertex indices, keeping the surface connected with shared vertices stored once. **Normals** are unit vectors perpendicular to the surface, per-face (flat shading) or per-vertex (smooth shading — interpolated lighting makes coarse meshes look round). **UV coordinates** (u,v in [0,1]²) attach to each vertex and map surface points to texture image pixels. **PBR textures** are a stack: albedo, normal map, roughness, metallic, optional occlusion and emissive.

**Topology vocabulary**: *manifold* = every edge shared by exactly 1 or 2 faces; *watertight* = fully enclosed volume, no boundary edges; *genus 0* = no through-holes (a phone should be). Failures cascade: non-manifold edges crash slicers; unclosed meshes have undefined volume.

**Raw output from Hunyuan3D/TRELLIS/TripoSR** is usually 50K–300K triangles. Common problems: holes in unseen regions (bottom, USB-C port); noise on flat back glass; non-manifold edges at thin features; floater blobs near camera ring; baked-in lighting that fights new HDRI.

**Cleanup steps**, sequenced so each doesn't break the next:

1. Load with **trimesh**.
2. Drop disconnected components below ~1% of largest by surface area.
3. Run **pymeshfix** to enforce manifoldness and watertightness in one call.
4. **Taubin smoothing** (λ ≈ 0.5, μ ≈ −0.53, 5–10 iterations) — band-stop filter removing noise *without* shrinking volume, unlike pure Laplacian which collapses geometry.
5. **Isotropic remeshing** (PyMeshLab) to make triangle sizes uniform — marching-cubes output has long slivers in flat regions; uniformity makes decimation cleaner.
6. **Quadric Edge Collapse Decimation** (QSlim / Garland-Heckbert) to ~30–50K triangles. Each vertex accumulates a 4×4 quadric error matrix; edge collapses processed cheapest-first; planar regions collapse free, curved regions retain detail.
7. **Re-orient normals** consistently (trimesh `fix_normals`).
8. **Re-UV** via xatlas and **bake textures** from the high-res mesh — any geometry edit invalidates the original UV map.

Library choice: **trimesh** for I/O and basic ops; **Open3D** for point clouds and Poisson reconstruction; **PyMeshLab** for the comprehensive filter toolbox; **pymeshfix** as the single-purpose watertight hammer.

**Format choice — why GLB.** OBJ is text + separate MTL + separate texture files (awkward to ship). STL is geometry-only, no color. PLY supports vertex colors but no proper materials. **GLB / glTF 2.0** is the Khronos web standard: single binary file with geometry, PBR materials, optional Draco compression (60–90% smaller), KTX2 GPU-native textures. Three.js loads via `GLTFLoader`; `<model-viewer>` displays with one HTML tag. STEP and IGES are parametric CAD formats — fundamentally different, discussed below.

**Why mesh ≠ CAD, and why Naya AI cares.** A mesh is a *discrete approximation*: a rounded edge is hundreds of small triangles on a torus locus — the mesh doesn't know they form a "fillet" and there's no editable fillet radius parameter. A **parametric B-rep CAD** model is *exact and semantic*: NURBS surfaces trimmed along curves, organized in a feature tree ("extrude 8.2 mm, fillet four edges at 1.5 mm radius"). Every parameter is editable, every feature is named. CNC mills, injection molds, and manufacturing-cost estimators **require CAD** — they cannot consume meshes directly.

Mesh-to-CAD is hard because it's an under-determined reasoning problem: segment mesh into primitives, recover exact intersections, infer feature semantics (active research: Point2Cyl, ComplexGen, BrepGen, DeepCAD). None ships a robust general pipeline yet — **that gap is Naya's commercial wedge**.

Our demo's positioning is clean: we deliver a clean textured watertight **mesh** — emphatically not manufacturable CAD, that's Naya's job. What we prove is the *front of the pipeline*: the "geometrize the input" step that must happen before any mesh-to-CAD system can run. The pitch: *"Your B-rep extraction needs a clean watertight mesh. Today your customers build that by hand. We build it from a marketing image automatically."*

---

## Component 8: spinning models side-by-side in a browser

**WebGL** exposes OpenGL ES to JavaScript at a very low level — buffers, shaders, draw calls, no concept of "a mesh." Coding raw WebGL for a week is self-sabotage. **Three.js** is the right abstraction: a Scene of Object3D nodes, a Camera, Meshes of Geometry + Material, a WebGLRenderer that walks the tree.

For day-1 prototyping use Google's **`<model-viewer>`** web component: `<model-viewer src="phone.glb" auto-rotate camera-controls>` and you have a working demo in 30 seconds. Mid-week swap to bare Three.js for synchronized controls, wireframe toggles, dimension overlays, and explode views.

Core Three.js concepts: `PerspectiveCamera(fov=35)` (mimics product-photographer lensing), `WebGLRenderer({antialias: true})` with `ACESFilmicToneMapping` and `outputColorSpace = SRGBColorSpace`, `requestAnimationFrame` loop. Materials: `MeshStandardMaterial` for PBR; metalness ~0.9 + roughness ~0.3 for frame; metalness 0 + roughness 0.05 + `clearcoat 1` on `MeshPhysicalMaterial` for glass back. **HDRI environment maps** are essential — use `RGBELoader` with a 1k studio HDR from polyhaven.com; without them glass and metal look plasticky.

**Side-by-side comparison.** Two `<canvas>` elements with independent renderers, scenes, cameras, and `OrbitControls`. **Synchronize rotation** by listening to each `OrbitControls`'s `'change'` event and copying camera position, target, and quaternion to the other, gated by a `syncing` flag to prevent ping-pong loops. Set `autoRotate = true` on both with identical `autoRotateSpeed`.

**Python backend.** Don't build one if you can avoid it. Pre-bake `phone_bad.glb` and `phone_good.glb`, serve with `python -m http.server 8000`. The "bad" model is ~30 lines of Python: two PlaneGeometries from the composite image, offset by 8 mm in Z, welded edges, exported via trimesh. For a live pipeline demo: FastAPI with `StaticFiles` plus a `/process` POST endpoint; run the heavy Colab steps via pyngrok tunnel; the frontend `fetch()`es the ngrok URL.

### UI polish that actually convinces engineers

- **Pipeline strip** above viewports: input photo → Gemini initial parse (with occlusion routing signal) → SAM masks → Gemini validation (show iteration count and what was fixed) → amodal completion (or "skipped — no occlusion detected") → Gemini amodal validation with self-generated criteria → final mesh, with CSS hover-zoom thumbnails. Shows "real self-correcting pipeline, not a black box" — and the iteration count is itself compelling ("auto-corrected mask in 2 iterations").
- **Dynamic dimension controls** below the good-model viewport: per-axis MM input fields auto-filled by Gemini web-search grounding, uniform scale slider, Apply/Reset/Unknown buttons. Mesh rescales instantly in Three.js on every keystroke. The 8.2 mm depth label in the dimension overlay updates live as the user types.
- **Real-world dimension overlay**: bounding box edge lines with floating labels "162.8 mm", "77.6 mm", "**8.2 mm**" projected via `vector.project(camera)` each frame. The 8.2 mm label is the punchline — Naya's bad model reads ~0 mm here.
- **Wireframe toggle**: `model.traverse(o => o.isMesh && (o.material.wireframe = !o.material.wireframe))`. "Naya: 8 verts / 4 faces" vs "Ours: 48,212 verts / 96,108 faces."
- **Click-to-explode**: animate the bad model's two children flying apart in Z on click. The void between flat sheets becomes physically visible. Nothing is more memorable.
- **X-ray toggle**: `material.transparent = true; opacity = 0.35; depthWrite = false`. Camera lens cutouts visible through the front in our model; bad model is just two flat sheets.
- **Stage indicator** ticking through `Running VLM (3s)... SAM (2s)... Amodal completion (5s)... Hunyuan3D (30s)...` — even if pre-baked, fake the timing for narrative.
- **Aesthetic**: dark background `#0b0f14`, radial gradient behind each model, cyan for good / muted red for bad, Inter font, `toneMappingExposure ≈ 1.0`.

---

## End-to-end pipeline architecture for one week

**Stage-by-stage data flow:**

```
composite.jpg (input)
  ↓ Gemini 2.5 Flash — spatial JSON prompt (API, ~3 s)
  {views: [{label:front, box, z_order:0},
           {label:back,  box, z_order:1, occluded_by:[front]}]}
  ↓ SAM 2 Hiera-Tiny — box prompts → binary masks
  front_mask.png, back_mask_modal.png (back has hole at occlusion)
  ↓
  ┌─────────────────────────────────────────────────┐
  │  GEMINI↔SAM VALIDATION LOOP (max 3 iterations)  │
  │  Render masks as colored overlay on composite   │
  │  Gemini validates: bleed / completeness /       │
  │  shadow / occlusion boundary / label swap       │
  │  If issues: inject corrected bbox/neg-points    │
  │  back into SAM, re-segment, re-validate         │
  └─────────────────────────────────────────────────┘
  ↓ validated front_mask.png, back_mask_modal.png
  ↓ pix2gestalt × 4 seeds in parallel (Colab T4, ~5 s each)
  4 candidate back_complete_rgba images
  ↓
  ┌──────────────────────────────────────────────────────┐
  │  GEMINI↔AMODAL VALIDATION LOOP (max 2 iterations)   │
  │  Gemini scores all 4 candidates:                    │
  │  lens count / arrangement / frame continuity /      │
  │  logo placement / seam artifacts                    │
  │  If best passes threshold: use it                   │
  │  If all fail: run targeted SD inpainting with       │
  │  Gemini's affected_region + suggested_prompt        │
  │  + IP-Adapter Samsung reference, re-validate        │
  └──────────────────────────────────────────────────────┘
  ↓ back_complete_rgba.png (validated), front_rgba.png
  ↓ rembg / clean alpha; resize to 512×512 with padding
  front_input.png, back_input.png (clean RGBA, white background)
  ↓ Hunyuan3D-2mv (Colab T4, ~30 s shape + ~60 s texture)
  phone_raw.glb (~50–100K tris, PBR texture, arbitrary scale)
  ↓ trimesh + pymeshfix + PyMeshLab (local, CPU-only)
  phone_clean.glb (~30–50K tris, watertight, manifold)
  ↓ PCA orient + rescale: Gemini lookup → 162.8 × 77.6 × 8.2 mm
    (or user-provided dimensions; or normalized if unknown)
  phone_final.glb (metric scale, canonical orientation)
  ↓ serve statically alongside phone_bad.glb
  web/index.html — Three.js dual-canvas side-by-side
```

**Where each stage runs:**

| Stage | Location | Justification |
|---|---|---|
| Gemini VLM (initial parse) | API | Cloud, no GPU needed |
| SAM segmentation | 4 GB local | Hiera-Tiny fits; cheap |
| Gemini↔SAM validation loop | API + local | Gemini API calls; SAM re-runs locally |
| Pix2gestalt (×4 seeds) | Colab T4 | Needs ~6–8 GB; parallel seeds run sequentially |
| Gemini↔amodal validation loop | API + Colab | Gemini API calls; SD inpainting on T4 if needed |
| Hunyuan3D-2mv | Colab T4 | 4 GB cannot run any quality 3D model |
| Mesh cleanup | 4 GB local | CPU-only; trimesh+PyMeshLab |
| Three.js viewer | Browser | No GPU server needed |

**Biggest risks and mitigations:**

1. **Hunyuan3D environment fragility on Colab.** Custom CUDA rasterizers need compilation; community reports failed installs. *Mitigation*: have TRELLIS backup notebook and TripoSR local fallback ready on day 2. Budget half a day for environment debugging.

2. **Gemini↔SAM loop not converging.** Gemini raises the same issue repeatedly but SAM corrections don't resolve it — typically happens on very dark or low-contrast composites. *Mitigation*: cap at 3 iterations, fall through to Grounded-SAM with text prompts, then user-click as final fallback. Use best-IoU masks seen across all iterations regardless of which iteration they came from.

3. **Pix2gestalt hallucinating wrong camera-lens arrangement across all 4 seeds.** *Mitigation*: the Gemini validation loop catches this and triggers targeted SD inpainting with IP-Adapter Samsung reference conditioning on the specific affected region. Pix2gestalt remains responsible for shape/silhouette; SD inpainting is only called to fix identified content details. This hybrid avoids the main risk of pure inpainting (wrong boundary) while fixing the main risk of pure amodal completion (wrong content).

4. **Reflective surfaces baked into texture.** Glass and metal pull frozen specular highlights into albedo. *Mitigation*: SF3D or Hunyuan3D-2.1 if VRAM permits (they disentangle lighting). Otherwise slightly desaturate the texture and reduce HDR exposure in Three.js.

5. **Reconstruction inflates phone thickness 2–3×.** *Mitigation*: the metric rescale step fixes this automatically. Log per-axis scale factors as a "proportional accuracy" score in the UI — turns a failure into a transparency feature.

6. **Camera bump merged into back surface.** Sub-mm features below voxel resolution. *Mitigation*: label it in the UI with a wireframe arrow and note "model resolves to 64³ voxels; sub-mm features may be smoothed." Honesty is more impressive than fake perfection to engineers.

7. **Non-manifold mesh or holes.** *Mitigation*: pymeshfix in one line.

8. **Free Colab T4 OOM on texture stage.** *Mitigation*: `torch.cuda.empty_cache()` between shape and texture passes; reduce `octree_resolution`; upgrade to Colab Pro L4 for $10 if blocked.

9. **Live demo failure.** *Mitigation*: pre-bake both GLBs and serve statically; record a screen capture as backup; never live-demo without an offline fallback.

**One-week schedule:**

- **Day 1**: Gemini 2.5 prompt iteration on the actual S25 Ultra composite. SAM segmentation running locally. Verify the Gemini→SAM one-shot path works end-to-end. Generate the "bad" GLB (two PlaneGeometries) in 30 lines of Python.
- **Day 2**: Build the Gemini↔SAM validation loop — overlay rendering, validation prompt, structured JSON response parsing, corrected hint injection back into SAM. Test convergence on the actual composite. Colab notebook with Hunyuan3D-2mv installed and validated on a sample asset.
- **Day 3**: Pix2gestalt running on Colab with 4-seed parallel sampling. Build the Gemini↔amodal validation loop — scoring prompt, issue extraction, targeted SD inpainting with IP-Adapter for content fixes. Visually validate the completed back panel output.
- **Day 4**: Full end-to-end run through both feedback loops on the actual composite. Mesh cleanup. Metric rescaling.
- **Day 5**: Three.js dual-canvas viewer with synchronized OrbitControls and HDRI lighting.
- **Day 6**: UI polish — wireframe toggle, explode view, dimension overlay, pipeline strip showing loop iterations, vertex counter. Surface Gemini's validation reasoning in the pipeline strip (show what it checked and what it fixed).
- **Day 7**: Deploy to Vercel. Record screen capture backup. Dry-run the pitch.

---

## Conclusion: what this pipeline actually proves

Naya AI's pipeline fails on the composite not because their CAD-extraction is broken but because they have no parser between "pixels" and "geometry." **Every component above is a parser** — the VLM parses semantics, SAM parses visible regions precisely, amodal completion parses what's hidden behind occluders, the feed-forward 3D model parses 2D into 3D, and the metric rescaler parses arbitrary units into millimeters. Stacked, they constitute a missing front-of-pipeline that's now buildable in 2025 because Objaverse-scale 3D datasets and feed-forward models like Hunyuan3D-2mv exist.

The division of labor between components is clean and principled: **Gemini is right about semantics, SAM is right about pixels, pix2gestalt is right about hidden shape, and the feedback loops ensure each step meets a quality bar before the next one runs.** Each component does exactly one job it was purpose-built for, and none of them needs to be retrained.

Critically, the pipeline is **not rigidly bound to phones**. The only object-specific knowledge in the entire system is generated dynamically by Gemini at runtime — the validation criteria for amodal completion, the dimension lookup for metric scaling, and the inpainting prompts are all produced by Gemini reasoning about whatever object it identified in the initial parse. Swapping the input image from a Samsung S25 Ultra to an alloy wheel, a shoe, or an industrial bracket requires zero code changes. The four places that previously looked rigid (validation prompts, reference image, dimensions, the bad-model baseline) are all now either Gemini-generated at runtime, user-supplied via the UI, or gracefully skipped when not available.

The Gemini-in-the-loop validation architecture deserves emphasis as a standalone idea. By placing Gemini as a quality gate at two critical junctions — after segmentation and after amodal completion — the pipeline becomes self-correcting rather than silently wrong. The structured JSON responses from Gemini's validation prompts are machine-readable enough to drive targeted remediation automatically: bad mask boundaries trigger corrected SAM re-prompts; wrong lens counts trigger focused SD inpainting on exactly the affected region. This pattern — render intermediate output, validate with a VLM, extract structured issues, trigger remediation — extends naturally to validating the final 3D mesh against the original image, requiring no new infrastructure.

The deepest insight for an audience of design-to-CAD engineers is that the gap between "image" and "manufacturable CAD" is **two distinct hard problems**. Image → mesh is now largely solved by generative priors. Mesh → parametric B-rep is unsolved — and that is where Naya's wedge lives. By delivering a clean, dimensioned, watertight GLB produced by a self-validating pipeline, our demo hands them exactly the input their semantic-recovery models want. The pitch writes itself: same input, two outputs, one obviously correct, self-correcting at every step, no model training required.