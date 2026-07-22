"""Irodori-TTS package: text-conditioned RF diffusion over DACVAE latents."""

# Consolidate the torch.compile / Triton on-disk caches into one directory
# instead of the split upstream defaults (/tmp/torchinductor_* and
# ~/.triton/cache), so warmed kernels survive reboots and are easy to find and
# wipe. Uses the conventional user cache dir (XDG_CACHE_HOME or ~/.cache). Must
# run before torch/triton import; an explicit env value still wins.
import os as _os
from pathlib import Path as _Path

_cache_root = _Path(_os.environ.get("XDG_CACHE_HOME") or _Path.home() / ".cache") / "irodori-tts"
_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_cache_root / "inductor"))
_os.environ.setdefault("TRITON_CACHE_DIR", str(_cache_root / "triton"))
del _os, _Path, _cache_root

from .config import ModelConfig, SamplingConfig, TrainConfig
from .lora import LORA_TARGET_PRESETS
from .model import TextToLatentRFDiT
from .tokenizer import ByteTokenizer, PretrainedTextTokenizer

__all__ = [
    "ByteTokenizer",
    "LORA_TARGET_PRESETS",
    "ModelConfig",
    "PretrainedTextTokenizer",
    "SamplingConfig",
    "TextToLatentRFDiT",
    "TrainConfig",
]
