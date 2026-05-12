# Image-to-3D Pipeline

> Converts a marketing composite image (front + back side-by-side) into a metric-scaled, browser-ready GLB mesh.

---

## Architecture

```
composite.jpg
│
├─ Stage 1 ─ Gemini 2.5 Flash ───────────── parse composite → bboxes + occlusion info
│
├─ Stage 2 ─ SAM2 Hiera-Large ──────────── segment front + back panels
│             └─ Gemini feedback loop        validate masks → fix → repeat ≤3×
│
├─ Stage 3 ─ Gemini (grounded search) ───── lookup real-world dimensions (mm)
│
├─ Stage 4 ─ pix2gestalt ────────────────── amodal-complete occluded back panel
│             └─ 4 seeds + Gemini scoring    pick best, SD inpaint if score < 7
│
├─ Stage 5 ─ Hunyuan3D-2mv ──────────────── shape + texture generation (multi-view)
│             └─ TRELLIS fallback            if CUDA rasterizer compilation fails
│
└─ Stage 6 ─ Mesh processing (CPU) ──────── clean → orient → scale → export GLB
```

---

## File Structure

```
Image-to-3d-model/
├── src/
│   ├── __init__.py           # Package docstring
│   ├── config.py             # All env-vars, path constants, dtype resolution
│   ├── model_manager.py      # Sequential VRAM load/unload context manager
│   ├── gemini_client.py      # Gemini 2.5 Flash — parse, validate, lookup
│   ├── segmentation.py       # SAM2 + Gemini feedback loop
│   ├── amodal_completion.py  # pix2gestalt + SD inpainting fallback
│   ├── reconstruction.py     # Hunyuan3D-2mv shape+texture + TRELLIS fallback
│   ├── mesh_processing.py    # 10-step mesh cleanup → GLB export
│   └── pipeline.py           # End-to-end orchestrator
├── research_and_insights/    # All research notes (do not modify)
├── COLAB_CELLS.py            # Copy-paste cells for Colab notebook
├── .env.example              # Template — copy to .env and fill in secrets
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Gemini SDK | `google-genai` v2.1+ | `google-generativeai` EOL Nov 2025 |
| Gemini model | `gemini-2.5-flash` | Stable, multimodal, 10× cheaper than Pro |
| SAM2 | Hiera-Large | Best masks, 1.5 GB (fits with room for other models) |
| Amodal | pix2gestalt × 4 seeds | Best silhouette recovery for occluded objects |
| 3D | Hunyuan3D-2mv | Native front+back multi-view input |
| 3D fallback | TRELLIS | MIT, pip install, no custom CUDA compilation |
| Mesh | trimesh + PyMeshLab + pymeshfix | Complementary: manifold fix + decimate + GLB export |
| Secrets | Colab Secrets → .env → os.environ | Security-first, with local dev fallback |
| VRAM | Sequential load/unload via ModelManager | 5 models, 16 GB total — never load 2 simultaneously |

---

## VRAM Budget (Colab T4 — 16 GB)

```
SAM2 Large:           1.5 GB  → unload before pix2gestalt
pix2gestalt:         10.0 GB  → unload before Hunyuan3D
SD inpainting:        3.5 GB  → unload (optional, only if amodal score < 7)
Hunyuan3D shape:      6.0 GB  → unload before texture stage
Hunyuan3D texture:   10-14 GB → unload after GLB export
Mesh cleanup:         0 GB    (CPU-only)
Gemini API:           0 GB    (cloud)
CUDA context:         ~0.3 GB (permanent overhead)
```

`ModelManager.use()` is the core primitive — load on entry, unload on exit, always.

---

## Environment Variables

Copy `.env.example` → `.env` and fill in:

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** From [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `HF_TOKEN` | — | Required for Hunyuan3D-2mv (gated). From [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept terms at [tencent/Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv). |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Override to `gemini-2.5-pro` for harder scenes |
| `SAM2_VARIANT` | `large` | `tiny` for local 4 GB GPU |
| `PIX2GESTALT_EPOCH` | `000005` | `000010` for synthetic-occlusion training data |
| `DEVICE` | `auto` | `auto` → cuda if available, else cpu |
| `TORCH_DTYPE` | `fp16` | `fp32` on CPU |
| `OUTPUT_DIR` | `/content/output` | All intermediate + final files saved here |

**For Colab:** use Colab Secrets instead of `.env` (secrets are encrypted, not shared when you share the notebook). `config._get_secret()` checks Colab Secrets first.

---

## How to Run in Google Colab

### Step 1 — Open Colab

Go to [colab.research.google.com](https://colab.research.google.com) → New Notebook.

Set runtime: **Runtime → Change runtime type → T4 GPU**.

### Step 2 — Add Secrets

Click the 🔑 key icon in the left sidebar → **Secrets** → Add:
- `GEMINI_API_KEY` — from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- `HF_TOKEN` — from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access)

Before using `HF_TOKEN`, accept the Hunyuan3D-2mv license at:
[https://huggingface.co/tencent/Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv)
(takes seconds, auto-approved)

### Step 3 — Upload Image

Upload your composite image to `/content/` using the Files panel (📁 icon → Upload).

### Step 4 — Run Cells

Open `COLAB_CELLS.py` and copy each cell block sequentially into Colab cells.  
Run them **one at a time**, in order:

| Cell | Stage | Time | VRAM |
|---|---|---|---|
| Cell 1 | Mount Google Drive | <1 min | 0 |
| Cell 2 | Set Colab Secrets | <1 min | 0 |
| Cell 3 | Install dependencies | 3-5 min | 0 |
| Cell 4 | Install pix2gestalt | 2-3 min | 0 |
| Cell 5 | Install Hunyuan3D | 5-10 min (CUDA compile) | 0 |
| Cell 6 | Download model weights | **15-30 min** (first run) / 5 min (Drive cache) | 0 |
| Cell 7 | Clone repo + set paths | <1 min | 0 |
| Cell 8 | Gemini parse | ~5 sec | 0 |
| Cell 9 | SAM2 segmentation | 1-3 min | 1.5 GB |
| Cell 10 | Dimension lookup | ~10 sec | 0 |
| Cell 11 | pix2gestalt amodal | **15-25 min** (4 seeds) | 10 GB |
| Cell 12 | Hunyuan3D reconstruction | **10-20 min** | 14 GB |
| Cell 13 | Mesh cleanup + export | 2-5 min (CPU) | 0 |

**Total: ~60-90 min on first run** (mostly model downloads + pix2gestalt).  
**Subsequent runs: ~30-40 min** (models cached in Google Drive).

### Alternative — Run Everything in One Cell

Use **Cell 14** to call `run_pipeline()` — this is equivalent to Cells 8-13.

---

## Model Download Notes

| Model | Size | Source | Token Required |
|---|---|---|---|
| SAM2 Hiera-Large | 224 MB | Meta CDN | No |
| pix2gestalt epoch=000005 | 15.5 GB | Columbia University | No |
| Hunyuan3D-2mv | ~8-10 GB | HuggingFace (gated) | **Yes** (`HF_TOKEN`) |
| SD Inpainting | ~3.5 GB | HuggingFace (public) | No |

The pipeline caches all models to Google Drive (`/content/drive/MyDrive/model_cache/`) on first download. Subsequent sessions copy from Drive (~5 min) instead of re-downloading (~30 min).

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Hunyuan3D CUDA compile fails | TRELLIS fallback (MIT, pip-only) |
| pix2gestalt 15.5 GB slow download | Drive cache; Cell 6 checks Drive first |
| pix2gestalt VRAM > 10 GB | `assert_vram_available` before load; try `steps=20` |
| Gemini loop not converging | Cap 3 iterations; proceed with best masks |
| Hunyuan3D texture OOM | Auto-retry with `octree_resolution=128` |
| Gemini grounding + schema conflict | Two separate calls (search, then extract) |
| Non-manifold mesh | pymeshfix one-liner |
| Decimation invalidates UVs | xatlas re-UV always runs after topology changes |

---

## What's Achievable on 16 GB Colab

| Feature | Status |
|---|---|
| Gemini scene parse + structured JSON | ✅ |
| SAM2 segmentation with feedback loop | ✅ |
| pix2gestalt amodal completion (4 seeds + Gemini pick) | ✅ (tight — unload everything else first) |
| Metric scaling via Gemini web-search grounding | ✅ |
| Mesh cleanup → watertight manifold GLB | ✅ |
| `phone_final.glb` at 162.8 × 77.6 × 8.2 mm (S25 Ultra) | ✅ |
| Hunyuan3D-2mv (depends on CUDA rasterizer compiling) | ⚠️ TRELLIS fallback if compile fails |
| pix2gestalt on 16 GB | ⚠️ Tight — must unload SAM2 first |

---

## Output Files

All intermediate and final files are saved to `OUTPUT_DIR` (default: `/content/output/`):

```
/content/output/
├── parse_result.jpg           ← Gemini bboxes drawn on composite
├── overlay_iter0.png          ← SAM mask overlay (iteration 0)
├── final_overlay.png          ← Final validated mask overlay
├── front_rgba.png             ← Front panel RGBA crop
├── back_modal_rgba.png        ← Back panel RGBA (occluded region missing)
├── amodal_seed1.png           ← pix2gestalt seed 1 completion
├── amodal_seed2.png           ← pix2gestalt seed 2 completion
├── amodal_seed3.png           ← pix2gestalt seed 3 completion
├── amodal_seed4.png           ← pix2gestalt seed 4 completion
├── amodal_candidates.png      ← All 4 candidates side-by-side
├── amodal_best.png            ← Best candidate (+ SD inpaint if applied)
├── back_amodal_rgba.png       ← Final amodal back panel RGBA
├── phone_raw.glb              ← Raw Hunyuan3D output (arbitrary scale)
└── phone_final.glb            ← ✅ FINAL: cleaned + oriented + metric-scaled GLB
```

---

## Implementation Notes

### Coordinate Conventions

Gemini returns bounding boxes as `[ymin, xmin, ymax, xmax]` normalized to 0-1000.  
SAM2 expects `[xmin, ymin, xmax, ymax]` in absolute pixels.  
`segmentation.gemini_box_to_sam()` converts between these.  **Do not swap this — Gemini's format is intentional and critical for model performance.**

### Why Two API Calls for Dimensions

Combining `tools=[google_search]` with `response_schema` in a single Gemini request is unreliable as of 2026 — the model sometimes ignores the schema when grounding is active.  `gemini_client.lookup_dimensions()` uses two calls: search (raw text) → extract (structured JSON).

### Why xatlas Re-UV After Every Topology Change

pymeshfix, isotropic remeshing, and decimation all renumber the vertex array.  The original UV coordinates map vertex index → texture coordinate — when the vertex array is rebuilt, these mappings are invalid.  xatlas generates a fresh UV atlas from scratch.  The original texture image (PIL) is preserved separately and re-attached.

### pix2gestalt VRAM

The full Gradio app uses 22-28 GB.  The `run_pix2gestalt()` function alone uses ~10 GB in fp16.  This is why all other models must be unloaded before Stage 4.

### Hunyuan3D Two-Stage Split

Shape generation (~6 GB) and texture generation (~10-14 GB) must run in separate ModelManager contexts.  Loading both simultaneously requires ~20 GB and will OOM on a 16 GB T4.  The `del shape_pipeline; gc.collect(); torch.cuda.empty_cache()` sequence between stages is mandatory.