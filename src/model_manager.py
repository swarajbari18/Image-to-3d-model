"""
model_manager.py — Sequential VRAM load/unload context manager.
================================================================
Architecture:
  Only one large model lives on GPU at a time.  The pattern is:

      with manager.use("sam2", loader_fn) as model:
          results = model.predict(...)
      # ← model is automatically moved to CPU, deleted, and VRAM freed

  This mirrors the "Sequential Pipeline Memory Budget" from model_caching_research.md:
      SAM2 (~1.5 GB) → unload → SynergyAmodal (~10 GB) → unload → Hunyuan3D (~10-14 GB)

  assert_vram_available() uses torch.cuda.mem_get_info() (CUDA driver),
  NOT torch.cuda.memory_allocated() (PyTorch pool).  The driver view is
  accurate; nvidia-smi shows the PyTorch cache and over-reports usage.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Callable, Generator

import torch


class ModelManager:
    """
    Manages sequential loading and unloading of large models to stay within
    the 16 GB VRAM budget of a Colab T4.

    Only one model is held on the GPU at a time.  Calling `use()` automatically
    frees VRAM after the with-block exits.

    Attributes:
        device:  Target device string ('cuda' or 'cpu').
        verbose: If True, print VRAM stats on every load/unload.
    """

    # Approximate VRAM (MB) each model requires — used for pre-flight checks.
    VRAM_REQUIREMENTS: dict[str, float] = {
        "sam2": 1_600,
        "synergyamodal": 10_000,
        "sd_inpaint": 5_000,
        "hunyuan3d_shape": 6_500,
        "hunyuan3d_texture": 6_500,
    }

    def __init__(self, device: str = "cuda", verbose: bool = True) -> None:
        self.device = device
        self.verbose = verbose
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print message with current VRAM allocation if verbose is True."""
        if not self.verbose:
            return
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
            print(
                f"[ModelManager] {msg} | "
                f"Alloc={alloc_mb:.0f} MB  Reserved={reserved_mb:.0f} MB"
            )
        else:
            print(f"[ModelManager] {msg} | (CPU mode — no VRAM stats)")

    def _free_memory(self, model: Any) -> None:
        """
        Move *model* to CPU, delete it, and flush PyTorch's CUDA allocator.

        Steps:
          1. model.cpu()     — offload parameters so GPU tensors are released
          2. del model       — drop Python reference → gc can reclaim
          3. gc.collect()    — run CPython cycle collector
          4. empty_cache()   — return reserved-but-idle pages to CUDA runtime
          5. gc.collect()    — second pass (some cycles only visible after step 4)
        """
        try:
            model.cpu()
        except Exception:
            pass  # some model objects don't have .cpu() — ignore
        del model
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def free_vram_mb(self) -> float:
        """
        Return free VRAM in megabytes as reported by the CUDA driver.

        This is the most accurate measurement.  PyTorch's memory_allocated()
        only reports its own pool; the CUDA driver view reflects all allocations
        including those from CUDA libraries (cuDNN, cuBLAS, etc.).
        """
        if not torch.cuda.is_available():
            return float("inf")  # CPU mode — no VRAM constraint
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / 1024 ** 2

    def assert_vram_available(self, required_mb: float) -> None:
        """
        Raise MemoryError if fewer than *required_mb* MB are free.

        Call this before every load() to give a clear error message instead
        of a cryptic CUDA OOM inside a model's forward pass.
        """
        available = self.free_vram_mb()
        if available < required_mb:
            raise MemoryError(
                f"Insufficient VRAM — need {required_mb:.0f} MB but only "
                f"{available:.0f} MB free.  Call manager.unload_all() first."
            )

    def load(self, name: str, loader_fn: Callable[..., Any], **kwargs: Any) -> Any:
        """
        Load model *name* using *loader_fn* if not already cached.

        Args:
            name:      Unique identifier for this model (e.g. 'sam2').
            loader_fn: Zero-argument callable that returns the loaded model.
                       Receives any extra **kwargs.
            **kwargs:  Forwarded verbatim to loader_fn.

        Returns:
            The loaded (and VRAM-resident) model object.
        """
        if name in self._loaded:
            self._log(f"Cache hit — reusing '{name}'")
            return self._loaded[name]

        # Pre-flight VRAM check
        required = self.VRAM_REQUIREMENTS.get(name, 0)
        if required:
            self.assert_vram_available(required)

        self._log(f"Loading '{name}'...")
        model = loader_fn(**kwargs)
        self._loaded[name] = model
        self._log(f"'{name}' loaded")
        return model

    def unload(self, name: str) -> None:
        """Unload *name* and free its VRAM."""
        if name not in self._loaded:
            return
        self._log(f"Unloading '{name}'...")
        model = self._loaded.pop(name)
        self._free_memory(model)
        self._log(f"'{name}' unloaded")

    def unload_all(self) -> None:
        """Unload every currently-loaded model."""
        for name in list(self._loaded.keys()):
            self.unload(name)

    @contextmanager
    def use(
        self, name: str, loader_fn: Callable[..., Any], **kwargs: Any
    ) -> Generator[Any, None, None]:
        """
        Context manager: load → yield → auto-unload.

        Usage::

            with manager.use("sam2", build_sam2_fn, checkpoint=path) as sam2:
                masks = sam2.predict(image)
            # GPU memory freed here automatically

        The model is always unloaded in the finally-block, even if the
        with-body raises an exception.
        """
        model = self.load(name, loader_fn, **kwargs)
        try:
            yield model
        finally:
            self.unload(name)

    def print_vram_stats(self, label: str = "") -> None:
        """Print a formatted VRAM status line."""
        if not torch.cuda.is_available():
            print(f"[{label}] CPU mode — no VRAM")
            return
        alloc = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        free_driver = self.free_vram_mb()
        print(
            f"[VRAM{' ' + label if label else ''}] "
            f"Alloc={alloc:.0f} MB  Reserved={reserved:.0f} MB  "
            f"Free(driver)={free_driver:.0f} MB  Total={total:.0f} MB"
        )


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------

manager = ModelManager()
