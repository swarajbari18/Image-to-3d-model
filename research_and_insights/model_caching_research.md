# Model Downloading, Caching & Environment Setup Research
**Pipeline:** Image-to-3D (SAM2 → pix2gestalt → Hunyuan3D-2mv → SD Inpainting + Gemini API)
**Research Date:** May 12, 2026

---

## 1. HuggingFace Public Model Downloads — No Account Required?

### The Core Rule
`hf_hub_download()` and `from_pretrained()` work **without a token for public, non-gated models**. The HF library checks for a token in this priority order:
1. Explicit `token=` argument
2. `HF_TOKEN` environment variable
3. Google Colab `userdata.get("HF_TOKEN")` (auto-checked by huggingface_hub)
4. `~/.cache/huggingface/token` (saved via `huggingface-cli login`)
5. **No token (anonymous)** — works for public non-gated repos

For public models, anonymous access is fine. No account needed.

```python
# Works with zero credentials for PUBLIC, non-gated models:
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="facebook/sam2-hiera-large", filename="config.json")

# Also works:
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/sam2-hiera-large")  # no token needed
```

### Model-by-Model Gating Status

| Model | HuggingFace Repo | Gated? | Token Required? |
|---|---|---|---|
| SAM2 (Meta) | `facebook/sam2-hiera-large`, `facebook/sam2.1-hiera-large` | **No** | No |
| pix2gestalt (cvlab) | `cvlab/pix2gestalt-weights` | **No** | No |
| Hunyuan3D-2mv (Tencent) | `tencent/Hunyuan3D-2mv` | **YES** — `extra_gated_eu_disallowed: true`, Tencent community license | **Yes** — must accept terms + token |
| Hunyuan3D-2 (base) | `tencent/Hunyuan3D-2` | **YES** | Yes |
| Hunyuan3D-2.1 | `tencent/Hunyuan3D-2.1` | **YES** | Yes |
| SD Inpainting | `runwayml/stable-diffusion-inpainting` | Historically public; mirror exists at `stable-diffusion-v1-5/stable-diffusion-inpainting` | No for public mirror |
| IP-Adapter | `h94/IP-Adapter` | **No** | No |

### What Happens With `from_pretrained()` and No Token on a GATED Model

```
OSError: Repository tencent/Hunyuan3D-2mv is gated.
Access to the model is restricted. You have to be authenticated to access it.
Visit https://huggingface.co/tencent/Hunyuan3D-2mv and accept the conditions.
Then pass your token to the `token` argument or run `huggingface-cli login`.
```

### Downloading Gated Models Without an Account — Is It Possible in 2026?

**No legitimate workaround exists.** For gated models (like all Hunyuan3D-2 variants), you must:
1. Create a free HuggingFace account (free, no credit card)
2. Visit the model page and accept the license terms (auto-approved in seconds for Hunyuan3D)
3. Generate a read-access token at https://huggingface.co/settings/tokens
4. Use that token as `HF_TOKEN`

There are no bypass methods. Gating is enforced server-side on HF CDN. The `extra_gated_eu_disallowed` flag on Hunyuan3D-2mv additionally blocks EU IP addresses from even viewing terms, which may affect some Colab datacenter IPs — use a VPN or different region runtime if you encounter this.

---

## 2. Direct Download Alternatives (No HuggingFace)

### SAM2 — Meta CDN Direct Links

Meta hosts all SAM2 checkpoints on their own CDN at `dl.fbaipublicfiles.com`. **No account needed.**

```bash
# SAM 2.1 checkpoints (latest, September 2024 release — "092824" date code)
BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# Tiny (~38MB)
wget -P /content/models/sam2/ ${BASE}/sam2.1_hiera_tiny.pt

# Small (~46MB)
wget -P /content/models/sam2/ ${BASE}/sam2.1_hiera_small.pt

# Base+ (~80MB)
wget -P /content/models/sam2/ ${BASE}/sam2.1_hiera_base_plus.pt

# Large (~224MB) — RECOMMENDED for best accuracy
wget -P /content/models/sam2/ ${BASE}/sam2.1_hiera_large.pt

# Original SAM2 checkpoints (072824 — earlier, less accurate)
BASE_OLD="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
wget ${BASE_OLD}/sam2_hiera_large.pt
```

The official `checkpoints/download_ckpts.sh` script in the facebookresearch/sam2 repo uses exactly these URLs.

### pix2gestalt — Columbia University Direct Link

```bash
# Primary: Columbia University server (no auth)
wget -O /content/models/pix2gestalt/epoch=000005.ckpt \
  https://gestalt.cs.columbia.edu/assets/epoch=000005.ckpt

# Alternative: HuggingFace (public, no token needed)
# cvlab/pix2gestalt-weights is NOT gated
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    repo_id="cvlab/pix2gestalt-weights",
    filename="epoch=000005.ckpt",
    cache_dir="/content/models/pix2gestalt/"
)
```

Note: The checkpoint is ~15.5 GB. Allow 5–10 minutes on Colab.
The `epoch=000010.ckpt` variant (also ~15.5 GB) is better for synthetic occlusion but generalizes slightly worse.

### Hunyuan3D-2mv — No Non-HF Direct Download

There are no official direct download CDN URLs outside of HuggingFace for Hunyuan3D-2mv. The GitHub repo (Tencent-Hunyuan/Hunyuan3D-2) does not host model weights in its releases. You **must** use HuggingFace with a token.

```python
# After accepting terms on HF and setting HF_TOKEN:
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="tencent/Hunyuan3D-2mv",
    cache_dir="/content/models/hunyuan3d/",
    token=os.environ["HF_TOKEN"]
)
```

### Stable Diffusion Inpainting — Public Mirror Available

The original `runwayml/stable-diffusion-inpainting` is maintained but the official community mirror is cleaner:

```python
# Public mirror — no token required:
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
```

---

## 3. Colab Caching Strategy

### Session Duration

| Tier | Max Session | Idle Timeout |
|---|---|---|
| Free | ~12 hours | 90 minutes |
| Colab Pro | ~24 hours | 90 minutes |
| Colab Pro+ | ~24 hours (priority) | 90 minutes |

**Practical consequence:** On the free tier, you lose all downloaded models every ~12 hours (or sooner under load). On Pro/Pro+, you get at most 24 hours — still not permanent.

### Cache Directory Choice

**Recommendation: Use `/content/models/` as runtime cache, back it up to Google Drive.**

| Directory | Pros | Cons |
|---|---|---|
| `/content/models/` | Fast local NVMe, simple paths | Wiped every session |
| `/root/.cache/huggingface/` | Used automatically by `from_pretrained` with no config | Wiped every session, less visible |
| `/content/drive/MyDrive/hf_cache/` | Survives session resets | Slow I/O (Google Drive FUSE mount, ~10–30 MB/s vs ~500+ MB/s local), remount needed each session |

**Best pattern:** Download to Drive on first run, copy to local `/content/` at session start.

### Cache-Check-Before-Download Pattern

```python
import os
import shutil

GDRIVE_CACHE = "/content/drive/MyDrive/model_cache"
LOCAL_CACHE  = "/content/models"

def ensure_model(model_name: str, download_fn) -> str:
    """
    Check Google Drive cache → copy to local → return local path.
    If not in Drive, download to Drive then copy locally.
    """
    gdrive_path = os.path.join(GDRIVE_CACHE, model_name)
    local_path  = os.path.join(LOCAL_CACHE,  model_name)

    os.makedirs(local_path, exist_ok=True)

    if os.path.exists(gdrive_path) and os.listdir(gdrive_path):
        print(f"[cache] Found {model_name} in Google Drive, copying to local...")
        shutil.copytree(gdrive_path, local_path, dirs_exist_ok=True)
    else:
        print(f"[cache] {model_name} not in Drive, downloading...")
        os.makedirs(gdrive_path, exist_ok=True)
        download_fn(gdrive_path)
        shutil.copytree(gdrive_path, local_path, dirs_exist_ok=True)

    return local_path
```

### Google Drive Mount + Persistent Cache Setup

```python
# Cell 1: Mount Drive (run once per session, prompts for OAuth)
from google.colab import drive
drive.mount("/content/drive")

# Cell 2: Set HF cache to Drive so from_pretrained() auto-caches there
import os
os.environ["HF_HOME"]            = "/content/drive/MyDrive/hf_cache"
os.environ["HF_HUB_CACHE"]       = "/content/drive/MyDrive/hf_cache/hub"
os.environ["HF_DATASETS_CACHE"]  = "/content/drive/MyDrive/hf_cache/datasets"
# IMPORTANT: these must be set BEFORE importing transformers/diffusers/datasets

# Verify Drive is writable:
test_file = "/content/drive/MyDrive/hf_cache/.colab_test"
os.makedirs("/content/drive/MyDrive/hf_cache", exist_ok=True)
with open(test_file, "w") as f:
    f.write("ok")
print("Drive cache ready:", os.path.exists(test_file))
```

**Trade-off summary:** Google Drive I/O is 10–50x slower than local disk. For large models (15 GB pix2gestalt, ~8 GB Hunyuan3D-2mv), copying from Drive to local at session start takes 5–15 minutes but saves re-downloading (which takes 10–30 minutes on Colab's network). Use Drive as the persistent store, local disk as the working copy.

---

## 4. Environment Variables in Colab — Two Approaches

### Approach A: python-dotenv with `.env` file

**Setup:** Upload your `.env` file to Colab (Files panel → upload, or from Drive).

```python
# Install
# pip install python-dotenv

from dotenv import load_dotenv
import os

# Option 1: .env in working directory
load_dotenv()

# Option 2: explicit path (e.g., from Drive)
load_dotenv("/content/drive/MyDrive/secrets/.env")

# Access variables
gemini_key = os.getenv("GEMINI_API_KEY")
hf_token   = os.getenv("HF_TOKEN")
```

**Risks:**
- `.env` file is uploaded as a real file in the Colab filesystem — visible to anyone with session access
- If you accidentally share the notebook with "include outputs," paths to secrets may leak
- `.env` file must be re-uploaded every session (unless stored in Drive — but Drive is also accessible to you)

### Approach B: Colab Secrets (userdata)

**Setup:** In Colab, click the key icon (left sidebar) → "Secrets" → add key/value pairs. Secrets are tied to your Google account and are NOT shared when you share a notebook.

```python
from google.colab import userdata

gemini_key = userdata.get("GEMINI_API_KEY")
hf_token   = userdata.get("HF_TOKEN")

# Set as env vars so libraries (huggingface_hub, google-genai) pick them up automatically
import os
os.environ["GEMINI_API_KEY"] = gemini_key
os.environ["HF_TOKEN"]       = hf_token
```

**Pros:**
- Encrypted on Google's servers
- Not visible in shared notebooks
- huggingface_hub automatically reads `HF_TOKEN` from Colab secrets as of 2024+

### Recommended Approach: Graceful Fallback (Use BOTH)

```python
import os

def _get_secret(key: str, dotenv_path: str = None) -> str | None:
    """
    Try to get a secret from Colab Secrets, then .env, then environment.
    Returns None if unavailable in all sources.
    """
    # 1. Try Colab Secrets
    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return val
    except (ImportError, Exception):
        pass  # Not in Colab, or secret not set

    # 2. Try .env file
    try:
        from dotenv import load_dotenv
        env_path = dotenv_path or os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)  # don't override existing env vars
    except ImportError:
        pass

    # 3. Try environment variable (covers HF_TOKEN set by huggingface-cli login)
    return os.getenv(key)


# Usage at pipeline startup:
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
HF_TOKEN       = _get_secret("HF_TOKEN")

# Inject back into environment for libraries that read env vars directly
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# Validate
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found — Gemini features will be disabled")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found — gated model downloads will fail (Hunyuan3D-2mv)")
```

**Which is better?**

For this pipeline running on Colab: **Colab Secrets is better** for security. Use `.env` as the fallback for local development (Jupyter, VS Code). The graceful fallback pattern above handles both transparently.

---

## 5. Complete `.env.example`

```bash
# ============================================================
# Image-to-3D Pipeline — Environment Variables
# Copy this file to .env and fill in real values.
# NEVER commit .env to version control.
# For Colab: use Colab Secrets instead (see README).
# ============================================================

# --- Gemini API ---
# Required for scene understanding / captioning step
# Get from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini model to use (leave default unless you want to override)
GEMINI_MODEL=gemini-2.5-pro

# --- HuggingFace ---
# Required ONLY for Hunyuan3D-2mv (gated model).
# Get from: https://huggingface.co/settings/tokens
# Must have ACCEPTED TERMS at: https://huggingface.co/tencent/Hunyuan3D-2mv
# SAM2 and pix2gestalt do NOT require this.
HF_TOKEN=hf_your_token_here

# --- Model Cache Directories ---
# Where downloaded model weights are stored locally.
# Change to a Google Drive path for persistence across Colab sessions.
# Default (ephemeral Colab local disk):
MODEL_CACHE_DIR=/content/models
# Persistent Google Drive cache (set this if using Drive):
# MODEL_CACHE_DIR=/content/drive/MyDrive/model_cache

# HuggingFace hub cache — overrides default ~/.cache/huggingface/
# If using Drive, point to Drive:
# HF_HOME=/content/drive/MyDrive/hf_cache
HF_HOME=/content/models/hf_home

# --- SAM2 ---
# Which SAM2 variant to use: tiny | small | base_plus | large
# "large" = best accuracy, ~224MB; "small" = faster, ~46MB
SAM2_CHECKPOINT=sam2.1_hiera_large
SAM2_CHECKPOINT_DIR=/content/models/sam2

# --- pix2gestalt ---
# Which checkpoint epoch: 000005 (default, better zero-shot) or 000010
PIX2GESTALT_CHECKPOINT=epoch=000005.ckpt
PIX2GESTALT_CHECKPOINT_DIR=/content/models/pix2gestalt

# --- Hunyuan3D-2mv ---
HUNYUAN3D_CACHE_DIR=/content/models/hunyuan3d

# --- Stable Diffusion Inpainting ---
SD_INPAINT_MODEL=stable-diffusion-v1-5/stable-diffusion-inpainting
# Set to "true" to enable CPU offload (saves VRAM, ~20% slower)
SD_INPAINT_CPU_OFFLOAD=true

# --- Pipeline Behavior ---
# Device: "cuda", "cpu", or "auto"
DEVICE=auto
# Mixed precision: "fp16" or "bf16" or "fp32"
TORCH_DTYPE=fp16
# Maximum image size to process (longer edge in pixels)
MAX_IMAGE_SIZE=1024
```

---

## 6. SD Inpainting Without a HuggingFace Account

### Is `runwayml/stable-diffusion-inpainting` Gated?

The original `runwayml/stable-diffusion-inpainting` repo requires accepting CreativeML Open RAIL-M license terms (it may show a gating prompt depending on your IP and account status). However, the **community mirror** `stable-diffusion-v1-5/stable-diffusion-inpainting` is fully public with no gating and identical weights.

```python
from diffusers import StableDiffusionInpaintPipeline
import torch

# Use the public mirror — no token needed:
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = pipe.to("cuda")
```

### VRAM Usage on 16 GB with `enable_model_cpu_offload()`

| Configuration | VRAM Usage | Notes |
|---|---|---|
| Full pipeline on GPU, fp16 | ~5–6 GB | Fine for 16 GB |
| + VAE slicing | ~4–5 GB | Better for large images |
| + CPU offload | ~2.5–3.5 GB | Model components moved to CPU when not active |
| + Sequential CPU offload | ~1.5–2 GB | Slowest, moves every submodule |

```python
# Recommended for our use case (sharing 16 GB with other loaded models):
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()   # UNet stays on GPU during forward pass, others on CPU
pipe.enable_vae_slicing()         # Process VAE in slices for large images
pipe.enable_attention_slicing()   # Slice attention computation
```

### IP-Adapter Weights

`h94/IP-Adapter` is public and requires **no token**.

```python
# Load IP-Adapter for SD 1.5 inpainting:
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)
pipe.set_ip_adapter_scale(0.6)

# For better quality with ViT-H features:
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.safetensors"
)
```

---

## 7. GPU Memory Management Pattern

### The Core Problem

Five large models on 16 GB VRAM:

| Model | Estimated VRAM (fp16) |
|---|---|
| SAM2 large | ~1.5 GB |
| pix2gestalt | ~9–10 GB (SD-based, large UNet) |
| Hunyuan3D-2mv | ~8–10 GB |
| SD Inpainting | ~5–6 GB (without offload) |
| **Total if all loaded** | **~25–30 GB — does NOT fit** |

Strategy: Load → Infer → Unload → repeat. Only one large model active at a time.

### Does `torch.cuda.empty_cache()` + `gc.collect()` Actually Free Memory?

**Partially yes, with important caveats:**

- `del model` removes the Python reference but does **not** immediately free CUDA memory
- `gc.collect()` triggers Python's garbage collector, which frees Python-level objects
- `torch.cuda.empty_cache()` releases PyTorch's **cached (reserved but unused)** memory pool back to the OS/CUDA runtime
- **~254 MB minimum is permanently held** by the CUDA context for the life of the process (cannot be freed)
- `nvidia-smi` shows "reserved" memory which includes PyTorch's cache — may appear higher than actual active allocation

```python
import gc
import torch

def free_vram(model=None):
    """
    Properly unload a model and free GPU memory.
    Returns (allocated_before_MB, allocated_after_MB).
    """
    before = torch.cuda.memory_allocated() / 1024**2

    if model is not None:
        # Move to CPU first (helps with some model types)
        try:
            model.cpu()
        except Exception:
            pass
        del model

    gc.collect()
    torch.cuda.empty_cache()
    # Second collect after empty_cache clears some cyclic refs
    gc.collect()

    after = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"VRAM freed: {before - after:.1f} MB | "
          f"Allocated: {after:.1f} MB | Reserved: {reserved:.1f} MB")
    return before, after


def print_vram_stats(label=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved()  / 1024**2
    free      = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_reserved()) / 1024**2
    print(f"[{label}] Allocated: {allocated:.0f} MB | "
          f"Reserved: {reserved:.0f} MB | Free: {free:.0f} MB")
```

### Verifying VRAM is Freed

```python
# Method 1: PyTorch API
torch.cuda.memory_allocated()  # bytes currently allocated (active tensors)
torch.cuda.memory_reserved()   # bytes in PyTorch's cache (may not be released to OS)
torch.cuda.mem_get_info()      # (free_bytes, total_bytes) from CUDA driver — most accurate

# Method 2: nvidia-smi from Python
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
used, free, total = result.stdout.strip().split(", ")
print(f"nvidia-smi: {used} MB used / {total} MB total ({free} MB free)")

# Method 3: Inline in Colab
# !nvidia-smi
```

**Note:** `torch.cuda.mem_get_info()[0]` gives the most reliable "truly free" number matching the CUDA driver view. Use this to decide if you have enough headroom before loading the next model.

### Reusable ModelManager Class

```python
import gc
import torch
from contextlib import contextmanager
from typing import Callable, Any

class ModelManager:
    """
    Manages loading and unloading of large models to avoid VRAM overflow.
    Only one model is active at a time (configurable).
    """

    def __init__(self, device: str = "cuda", verbose: bool = True):
        self.device  = device
        self.verbose = verbose
        self._loaded: dict[str, Any] = {}

    def _log(self, msg: str):
        if self.verbose:
            allocated = torch.cuda.memory_allocated() / 1024**2
            print(f"[ModelManager] {msg} | VRAM: {allocated:.0f} MB")

    def load(self, name: str, loader_fn: Callable, **kwargs) -> Any:
        """Load a model if not already loaded. Returns the model."""
        if name in self._loaded:
            self._log(f"Using cached {name}")
            return self._loaded[name]

        self._log(f"Loading {name}...")
        model = loader_fn(**kwargs)
        self._loaded[name] = model
        self._log(f"Loaded {name}")
        return model

    def unload(self, name: str):
        """Unload a specific model and free VRAM."""
        if name not in self._loaded:
            return
        self._log(f"Unloading {name}...")
        model = self._loaded.pop(name)
        try:
            model.cpu()
        except Exception:
            pass
        del model
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        self._log(f"Unloaded {name}")

    def unload_all(self):
        """Unload every loaded model."""
        for name in list(self._loaded.keys()):
            self.unload(name)

    @contextmanager
    def use(self, name: str, loader_fn: Callable, **kwargs):
        """
        Context manager: load model, yield it, then unload.

        Usage:
            with manager.use("sam2", load_sam2_fn) as model:
                result = model.predict(image)
            # model is automatically unloaded here
        """
        model = self.load(name, loader_fn, **kwargs)
        try:
            yield model
        finally:
            self.unload(name)

    def free_vram_mb(self) -> float:
        """Return free VRAM in MB from CUDA driver perspective."""
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / 1024**2

    def assert_vram_available(self, required_mb: float):
        """Raise if insufficient VRAM is available."""
        available = self.free_vram_mb()
        if available < required_mb:
            raise MemoryError(
                f"Need {required_mb:.0f} MB VRAM but only {available:.0f} MB free. "
                "Call manager.unload_all() first."
            )


# --- Usage Example ---
manager = ModelManager(device="cuda")

# Load → use → auto-unload via context manager:
with manager.use("sam2", load_sam2, checkpoint="/content/models/sam2/sam2.1_hiera_large.pt") as sam2:
    masks = sam2.predict(image)

# At this point, sam2 is unloaded and VRAM is freed

manager.assert_vram_available(required_mb=9000)  # ensure 9 GB free before Hunyuan

with manager.use("hunyuan3d", load_hunyuan3d, model_dir="/content/models/hunyuan3d") as h3d:
    mesh = h3d.generate(image, masks)
```

### Sequential Pipeline Memory Budget (16 GB)

```
Step 1: SAM2 segmentation
  Load SAM2 large (~1.5 GB) → run → unload
  VRAM after: ~0.3 GB (CUDA context)

Step 2: pix2gestalt amodal completion
  Requires ~9-10 GB → assert_vram_available(9000) → load → run → unload
  VRAM after: ~0.3 GB

Step 3: Hunyuan3D-2mv generation
  Requires ~8-10 GB → assert_vram_available(8000) → load → run → unload
  VRAM after: ~0.3 GB

Step 4: SD Inpainting texture fix (optional)
  Requires ~5-6 GB with offload → load → run → unload
  VRAM after: ~0.3 GB

Gemini API: zero VRAM (cloud API call)
```

---

## 8. Complete `requirements.txt`

```
# ============================================================
# Image-to-3D Pipeline — Python Requirements
# Generated: May 2026
# Python: 3.10+
# CUDA: 12.1+ (for Colab T4/L4)
# ============================================================

# --- Core Python utilities ---
python-dotenv>=1.0.0
Pillow>=10.0.0
numpy>=1.24.0,<2.0.0      # numpy 2.x has breaking changes; pin <2.0 until all libs update
requests>=2.28.0
tqdm>=4.65.0

# --- PyTorch (install separately with correct CUDA version) ---
# On Colab, torch is pre-installed. For fresh env, use:
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Minimum required versions:
torch>=2.5.1
torchvision>=0.20.1

# --- Gemini API (replaces deprecated google-generativeai, EOL Nov 2025) ---
google-genai>=1.0.0

# --- HuggingFace ecosystem ---
huggingface_hub>=0.23.0
transformers>=4.45.0
accelerate>=0.30.0
safetensors>=0.4.0

# --- SAM2 ---
# Install from GitHub (pip package available but GitHub has latest):
#   pip install git+https://github.com/facebookresearch/sam2.git
# OR from PyPI:
sam2>=1.0.0
# SAM2 CUDA extension (optional, for speed):
#   SAM2_BUILD_CUDA=1 pip install git+https://github.com/facebookresearch/sam2.git

# --- Diffusion models (SD Inpainting + pix2gestalt backbone) ---
diffusers>=0.36.0
# pix2gestalt uses a modified SD pipeline; install from source:
#   pip install git+https://github.com/cvlab-columbia/pix2gestalt.git
# Or with omegaconf/ldm (pix2gestalt dependency):
omegaconf>=2.3.0
einops>=0.7.0

# --- Attention optimization (optional but recommended on Colab L4/A100) ---
xformers>=0.0.28           # requires matching torch version
# flash-attn: heavy CUDA build, only if needed
# flash-attn>=2.5.0        # uncomment if Ampere+ GPU and time to build

# --- 3D mesh processing ---
trimesh>=4.0.0
open3d>=0.18.0
pymeshlab>=2023.12

# --- Image processing ---
opencv-python-headless>=4.8.0   # headless = no GUI deps, better for Colab
scikit-image>=0.21.0
imageio>=2.31.0
matplotlib>=3.7.0

# --- Utilities ---
scipy>=1.10.0
pydantic>=2.0.0            # for config validation
PyYAML>=6.0
rich>=13.0.0               # pretty console output
```

### Installation Order for Colab

```bash
# In a Colab notebook, run these cells in order:

# Cell 1: Upgrade pip
!pip install -q --upgrade pip

# Cell 2: Install torch (pre-installed on Colab, skip if already correct version)
# !pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Cell 3: Core HuggingFace + diffusers
!pip install -q transformers diffusers accelerate safetensors huggingface_hub

# Cell 4: Gemini SDK (new unified SDK)
!pip install -q google-genai

# Cell 5: SAM2
!pip install -q git+https://github.com/facebookresearch/sam2.git

# Cell 6: pix2gestalt (installs SD dependencies + modified CLIP)
!pip install -q git+https://github.com/cvlab-columbia/pix2gestalt.git

# Cell 7: Mesh processing + image utilities
!pip install -q trimesh open3d pymeshlab opencv-python-headless scikit-image

# Cell 8: Optional performance
!pip install -q xformers python-dotenv rich

# Cell 9: Verify key imports
python3 -c "import torch; import transformers; import diffusers; import sam2; print('All OK')"
```

---

## 9. Key Findings Summary

1. **SAM2 is fully public** — download from Meta CDN (`dl.fbaipublicfiles.com`) with zero credentials. No HF account needed.

2. **pix2gestalt is fully public** — HF repo `cvlab/pix2gestalt-weights` has no gating. Can also download directly from Columbia servers. Weights are ~15.5 GB.

3. **Hunyuan3D-2mv IS gated** — requires HuggingFace account + token + license acceptance. No workaround. The acceptance is auto-approved (seconds). Store `HF_TOKEN` as a Colab Secret named `HF_TOKEN`.

4. **SD Inpainting** — use `stable-diffusion-v1-5/stable-diffusion-inpainting` (public mirror). Original `runwayml/` may require license agreement. IP-Adapter (`h94/IP-Adapter`) is fully public.

5. **`google-generativeai` is deprecated (EOL Nov 2025)** — use `google-genai` package instead.

6. **Colab sessions last max 12–24 hours** — must re-download models each time unless using Google Drive as persistent cache. Drive I/O is slow (~10–30 MB/s); copy to local disk at session start for fast inference.

7. **GPU memory management** — `del model; gc.collect(); torch.cuda.empty_cache()` is the correct sequence. ~254 MB CUDA context overhead is permanent per process. Use `torch.cuda.mem_get_info()` for accurate free VRAM measurement (not `nvidia-smi` which shows reserved cache too).

8. **Colab Secrets > python-dotenv** for cloud security. Use graceful fallback that tries Colab Secrets first, then `.env`, then environment variables — works in both Colab and local dev.

9. **pix2gestalt 2026 alternatives** exist (tuning-free inpainting-based methods, Amodal-Depth-Anything from ICCV 2025) with better VRAM efficiency (+4.8x) and mIoU (+5.3%). Consider evaluating these if pix2gestalt's 15.5 GB checkpoint is too heavy for your Colab tier.

---

## Sources

- [HuggingFace Hub Documentation — Security Tokens](https://huggingface.co/docs/hub/en/security-tokens)
- [HuggingFace Hub Documentation — Gated Models](https://huggingface.co/docs/hub/en/models-gated)
- [HuggingFace Hub Environment Variables](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)
- [facebookresearch/sam2 GitHub](https://github.com/facebookresearch/sam2)
- [sam2/checkpoints/download_ckpts.sh](https://github.com/facebookresearch/sam2/blob/main/checkpoints/download_ckpts.sh)
- [tencent/Hunyuan3D-2mv — HuggingFace](https://huggingface.co/tencent/Hunyuan3D-2mv)
- [Tencent-Hunyuan/Hunyuan3D-2 GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [cvlab/pix2gestalt-weights — HuggingFace](https://huggingface.co/cvlab/pix2gestalt-weights)
- [cvlab-columbia/pix2gestalt GitHub](https://github.com/cvlab-columbia/pix2gestalt)
- [stable-diffusion-v1-5/stable-diffusion-inpainting — HuggingFace](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
- [h94/IP-Adapter — HuggingFace](https://huggingface.co/h94/IP-Adapter)
- [HuggingFace Diffusers — Reduce Memory Usage](https://huggingface.co/docs/diffusers/optimization/memory)
- [Google Gen AI SDK — PyPI](https://pypi.org/project/google-genai/)
- [Migrate to Google GenAI SDK](https://ai.google.dev/gemini-api/docs/migrate)
- [google-generativeai Deprecated](https://github.com/google-gemini/deprecated-generative-ai-python)
- [Google Colab FAQ — Session Limits](https://research.google.com/colaboratory/faq.html)
- [HuggingFace in Google Colab — Cache Setup](https://www.appsloveworld.com/google-colaboratory/5/setting-huggingface-cache-in-google-colab-notebook-to-google-drive)
- [Colab Secrets Guide](https://labs.thinktecture.com/secrets-in-google-colab-the-new-way-to-protect-api-keys/)
- [HuggingFace Login on Colab](https://dev.to/0xkoji/huggingface-login-on-google-colab-54ek)
- [How to Release HuggingFace Models from VRAM](https://mjunya.com/en/posts/2025-01-27-hf-torch-clear-memory/)
- [PyTorch — torch.cuda.empty_cache()](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html)
- [Tuning-Free Amodal Segmentation via Inpainting Models](https://arxiv.org/html/2503.18947v1)
- [Amodal-Depth-Anything (ICCV 2025)](https://github.com/zhyever/Amodal-Depth-Anything)
- [trimesh PyPI](https://pypi.org/project/trimesh/)
- [pymeshlab PyPI](https://pypi.org/project/pymeshlab/)
- [SAM2 on HuggingFace Transformers](https://huggingface.co/docs/transformers/model_doc/sam2)
- [flash-attn PyPI](https://pypi.org/project/flash-attn/)
