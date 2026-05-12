"""
config.py — Central configuration for the Image-to-3D pipeline.
================================================================
Architecture:
  - _get_secret() resolves values from three sources in priority order:
      1. Google Colab Secrets  (via google.colab.userdata — encrypted, not shared)
      2. .env file             (via python-dotenv — for local dev)
      3. os.environ            (fallback / CI)
  - All path constants default to /content/* (Colab) but are fully
    overridable via env-vars, making local runs identical in code path.
  - TORCH_DTYPE and DEVICE are resolved once and shared across all modules.

Usage:
    from src.config import cfg
    print(cfg.GEMINI_API_KEY)
    print(cfg.DEVICE)
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Secret resolution
# ---------------------------------------------------------------------------

def _get_secret(key: str, dotenv_path: Optional[str] = None) -> Optional[str]:
    """
    Try to retrieve *key* from three sources in order:

    1. Google Colab Secrets  — encrypted, account-bound, not visible in shared notebooks.
    2. .env file             — loaded via python-dotenv; fallback for local dev.
    3. os.environ            — covers env-vars set by shell / CI / huggingface-cli login.

    Returns the value as a string, or None if unavailable in all sources.
    """
    # --- 1. Colab Secrets ---
    try:
        from google.colab import userdata  # type: ignore[import]
        val = userdata.get(key)
        if val:
            return val
    except Exception:
        pass  # Not running in Colab, or the secret was not set

    # --- 2. .env file ---
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        env_path = dotenv_path or os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)  # never overwrite already-set vars
    except ImportError:
        pass

    # --- 3. Environment variable ---
    return os.getenv(key)


# ---------------------------------------------------------------------------
# VRAM helpers (used at module level to set TORCH_DTYPE)
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    raw = _get_secret("DEVICE") or "auto"
    if raw == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return raw


def _resolve_dtype(device: str) -> torch.dtype:
    """
    Resolve the torch dtype to use for model inference.

    fp16 is used on CUDA (halves VRAM), fp32 on CPU (no fp16 support).
    Override via TORCH_DTYPE env-var ('fp16' | 'bf16' | 'fp32').
    """
    raw = _get_secret("TORCH_DTYPE") or ("fp16" if device == "cuda" else "fp32")
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    return mapping.get(raw, torch.float16)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All tuneable parameters for the Image-to-3D pipeline.

    Attributes are grouped by pipeline stage.  Every attribute has a default
    that works out-of-the-box on a 16 GB Colab T4 session.
    """

    # --- Gemini ---
    GEMINI_API_KEY: Optional[str] = field(default_factory=lambda: _get_secret("GEMINI_API_KEY"))
    GEMINI_MODEL: str = field(
        default_factory=lambda: _get_secret("GEMINI_MODEL") or "gemini-2.5-flash"
    )

    # --- HuggingFace ---
    HF_TOKEN: Optional[str] = field(default_factory=lambda: _get_secret("HF_TOKEN"))

    # --- Compute ---
    DEVICE: str = field(default_factory=_resolve_device)
    TORCH_DTYPE: torch.dtype = field(default=None)  # filled in __post_init__

    # --- Paths ---
    MODEL_CACHE_DIR: Path = field(
        default_factory=lambda: Path(_get_secret("MODEL_CACHE_DIR") or "/content/models")
    )
    HF_HOME: Path = field(
        default_factory=lambda: Path(_get_secret("HF_HOME") or "/content/models/hf_home")
    )
    OUTPUT_DIR: Path = field(
        default_factory=lambda: Path(_get_secret("OUTPUT_DIR") or "/content/output")
    )

    # --- SAM2 ---
    SAM2_VARIANT: str = field(
        default_factory=lambda: _get_secret("SAM2_VARIANT") or "large"
    )
    SAM2_CHECKPOINT_DIR: Path = field(
        default_factory=lambda: Path(
            _get_secret("SAM2_CHECKPOINT_DIR") or "/content/models/sam2"
        )
    )

    # --- pix2gestalt ---
    PIX2GESTALT_REPO_DIR: Path = field(
        default_factory=lambda: Path("/content/pix2gestalt")
    )
    PIX2GESTALT_CKPT_DIR: Path = field(
        default_factory=lambda: Path(
            _get_secret("PIX2GESTALT_CKPT_DIR") or "/content/models/pix2gestalt"
        )
    )
    PIX2GESTALT_EPOCH: str = field(
        default_factory=lambda: _get_secret("PIX2GESTALT_EPOCH") or "000005"
    )

    # --- Hunyuan3D ---
    HUNYUAN3D_REPO_DIR: Path = field(
        default_factory=lambda: Path("/content/Hunyuan3D-2")
    )
    HUNYUAN3D_CACHE_DIR: Path = field(
        default_factory=lambda: Path(
            _get_secret("HUNYUAN3D_CACHE_DIR") or "/content/models/hunyuan3d"
        )
    )

    # --- SD Inpainting ---
    SD_INPAINT_MODEL: str = field(
        default_factory=lambda: (
            _get_secret("SD_INPAINT_MODEL")
            or "stable-diffusion-v1-5/stable-diffusion-inpainting"
        )
    )

    def __post_init__(self) -> None:
        """Resolve derived fields and inject env-vars used by HF libraries."""
        # Dtype depends on DEVICE — resolve after DEVICE is set
        if self.TORCH_DTYPE is None:
            self.TORCH_DTYPE = _resolve_dtype(self.DEVICE)

        # Create output directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.SAM2_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.PIX2GESTALT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.HUNYUAN3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Inject HF env-vars BEFORE any transformers/diffusers import
        if self.HF_TOKEN:
            os.environ["HF_TOKEN"] = self.HF_TOKEN
        if self.HF_HOME:
            os.environ["HF_HOME"] = str(self.HF_HOME)
        if self.GEMINI_API_KEY:
            os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY

        self._validate()

    def _validate(self) -> None:
        """Print warnings for missing optional secrets."""
        if not self.GEMINI_API_KEY:
            print("[config] WARNING: GEMINI_API_KEY not set — Gemini calls will fail.")
        if not self.HF_TOKEN:
            print("[config] WARNING: HF_TOKEN not set — Hunyuan3D-2mv download will fail.")

    @property
    def sam2_checkpoint_path(self) -> Path:
        """Full path to the SAM2 .pt checkpoint file."""
        return self.SAM2_CHECKPOINT_DIR / f"sam2.1_hiera_{self.SAM2_VARIANT}.pt"

    @property
    def pix2gestalt_ckpt_path(self) -> Path:
        """Full path to the pix2gestalt .ckpt file."""
        return self.PIX2GESTALT_CKPT_DIR / f"epoch={self.PIX2GESTALT_EPOCH}.ckpt"

    @property
    def sam2_config_name(self) -> str:
        """Map variant → SAM2 YAML config filename."""
        variant_map = {
            "tiny": "sam2_hiera_t.yaml",
            "small": "sam2_hiera_s.yaml",
            "base_plus": "sam2_hiera_b+.yaml",
            "large": "sam2_hiera_l.yaml",
        }
        return variant_map.get(self.SAM2_VARIANT, "sam2_hiera_l.yaml")


# ---------------------------------------------------------------------------
# Module-level singleton — import `cfg` everywhere
# ---------------------------------------------------------------------------

cfg = PipelineConfig()
