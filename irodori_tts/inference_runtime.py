from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import secrets
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .config import ModelConfig
from .duration import build_duration_features
from .lora import (
    LORA_ADAPTER_CONFIG_NAME,
    checkpoint_state_uses_lora,
    is_lora_adapter_dir,
    load_lora_adapter,
)
from .model import TextToLatentRFDiT
from .rf import (
    _end_pool_recording_after_failed_capture,
    _get_warmup_stream,
    sample_euler_rf_cfg,
)
from .speaker_inversion import (
    load_speaker_inversion_payload,
    speaker_inversion_batch_tensors,
)
from .text_normalization import normalize_text
from .tokenizer import PretrainedTextTokenizer
from .watermark import SilentCipherWatermarker


def _is_mps_available() -> bool:
    backends = getattr(torch, "backends", None)
    if backends is None or not hasattr(backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def _is_xpu_available() -> bool:
    try:
        return bool(torch.xpu.is_available())
    except AttributeError:
        return False


def resolve_runtime_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return resolved
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")
        return resolved
    if resolved.type == "mps":
        if resolved.index is not None:
            raise ValueError("MPS device index is not supported. Use 'mps'.")
        if not _is_mps_available():
            raise ValueError("MPS device requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if resolved.type == "xpu":
        if resolved.index is not None:
            raise ValueError("XPU device index is not supported. Use 'xpu'.")
        if not _is_xpu_available():
            raise ValueError("XPU device requested but torch.xpu.is_available() is False.")
        return torch.device("xpu")
    raise ValueError(
        f"Unsupported inference device={resolved!s}. Expected one of: cpu, cuda, mps, xpu."
    )


def list_available_runtime_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if _is_mps_available():
        devices.append("mps")
    if _is_xpu_available():
        devices.append("xpu")
    devices.append("cpu")
    return devices


def default_runtime_device() -> str:
    return list_available_runtime_devices()[0]


def list_available_runtime_precisions(device: str | torch.device) -> list[str]:
    resolved = resolve_runtime_device(device)
    if resolved.type in ("cuda", "xpu"):
        return ["fp32", "bf16"]
    return ["fp32"]


_FAST_MATH_CONFIGURED = False


def _enable_cuda_fast_math() -> None:
    """
    Allow TF32 tensor-core execution for fp32 matmuls/convolutions and enable
    cuDNN autotuning. Gives a large speedup on Ampere+ GPUs with negligible
    numerical impact. Opt out with IRODORI_DISABLE_TF32=1.
    """
    global _FAST_MATH_CONFIGURED
    if _FAST_MATH_CONFIGURED:
        return
    if os.environ.get("IRODORI_DISABLE_TF32", "0").strip() == "1":
        _FAST_MATH_CONFIGURED = True
        return
    # TF32 for matmuls only. Empirically, forcing TF32/cudnn-benchmark on the
    # codec's convolutional decoder can select much slower conv algorithms, so
    # convolutions are left at default fp32 behavior.
    #
    # Use the legacy allow_tf32 flag rather than the >=2.9 fp32_precision API:
    # torch 2.10 raises on mixing the two, and inductor's pad_mm pass still
    # READS the legacy flag for fp32 matmuls, so setting the new API breaks
    # torch.compile of fp32 models ("mix of the legacy and new APIs" error).
    # Fall back to the new API if a future release removes the legacy setter.
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    _FAST_MATH_CONFIGURED = True


def _text_bucket_lengths() -> list[int]:
    """
    Text-length buckets (in tokens). Short texts are padded to the smallest
    bucket that fits instead of the checkpoint's full max_text_len, shrinking
    the per-step cross-attention over masked padding keys. Comma-separated in
    IRODORI_TEXT_BUCKETS; empty or "0" disables bucketing (always pad to
    max_text_len, the upstream behavior).
    """
    raw = os.environ.get("IRODORI_TEXT_BUCKETS", "64").strip()
    if raw in {"", "0"}:
        return []
    lengths = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"IRODORI_TEXT_BUCKETS entries must be > 0, got {part!r}")
        lengths.append(value)
    return sorted(set(lengths))


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()
    elif device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and hasattr(xpu, "synchronize"):
            xpu.synchronize()


def _sync_devices(*devices: torch.device) -> None:
    seen: set[tuple[str, int | None]] = set()
    for device in devices:
        key = (device.type, device.index)
        if key in seen:
            continue
        _sync_device(device)
        seen.add(key)


def _measure_start(device: torch.device, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter()


def _measure_end(device: torch.device, t0: float, *extra_devices: torch.device) -> float:
    _sync_devices(device, *extra_devices)
    return time.perf_counter() - t0


def _coerce_latent_shape(latent: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if latent.ndim == 3 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.ndim != 2:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
    if latent.shape[1] == latent_dim:
        return latent
    if latent.shape[0] == latent_dim:
        return latent.transpose(0, 1).contiguous()
    raise ValueError(
        f"Could not infer latent layout for shape={tuple(latent.shape)} and latent_dim={latent_dim}"
    )


def find_flattening_point(
    latent: torch.Tensor,
    target_value: float = 0.0,
    window_size: int = 20,
    std_threshold: float = 0.05,
    mean_threshold: float = 0.1,
) -> int:
    """
    Echo-style heuristic: find first index where a trailing window becomes near-flat and near-zero.

    Args:
      latent: (T, D) latent sequence.
    Returns:
      Flattening index in [0, T].
    """
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape (T, D), got {tuple(latent.shape)}")
    total_steps = int(latent.shape[0])
    if total_steps <= 0 or window_size <= 0:
        return total_steps

    pad = torch.zeros(
        (window_size, latent.shape[1]),
        device=latent.device,
        dtype=latent.dtype,
    )
    padded = torch.cat([latent, pad], dim=0)
    # Vectorized over all trailing windows (equivalent to the original per-window
    # loop, but with a single device sync instead of one per window).
    num_windows = padded.shape[0] - window_size
    windows = padded.unfold(0, window_size, 1)[:num_windows]  # (N, D, W)
    flat = windows.reshape(num_windows, -1)
    window_std = flat.std(dim=1, unbiased=False)
    window_mean = flat.mean(dim=1)
    hits = (window_std < std_threshold) & (torch.abs(window_mean - target_value) < mean_threshold)
    first_hit = int(torch.argmax(hits.to(torch.int8)).item()) if bool(hits.any().item()) else -1
    if first_hit >= 0:
        return first_hit
    return total_steps


@dataclass(frozen=True)
class RuntimeKey:
    checkpoint: str
    model_device: str
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    model_precision: str = "fp32"
    codec_device: str = "cpu"
    codec_precision: str = "fp32"
    codec_deterministic_encode: bool = True
    codec_deterministic_decode: bool = True
    compile_model: bool = False
    compile_dynamic: bool = False


@dataclass
class SamplingRequest:
    text: str
    caption: str | None = None
    ref_wav: str | None = None
    ref_latent: str | None = None
    ref_embed: str | None = None
    no_ref: bool = False
    # In-memory alternative to ref_wav: a float waveform of shape (samples,)
    # or (channels, samples). Requires ref_audio_sample_rate and is processed
    # identically to ref_wav (max_ref_seconds trim, loudness normalize, codec
    # encode). Cannot be combined with ref_wav/ref_latent/ref_embed.
    ref_audio: torch.Tensor | None = None
    ref_audio_sample_rate: int | None = None
    ref_normalize_db: float | None = -16.0
    ref_ensure_max: bool = True
    num_candidates: int = 1
    decode_mode: str = "sequential"
    seconds: float | None = None
    duration_scale: float = 1.0
    min_seconds: float = 0.5
    max_seconds: float = 30.0
    max_ref_seconds: float | None = 30.0
    max_text_len: int | None = None
    max_caption_len: int | None = None
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    context_kv_cache: bool = True
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = None
    speaker_kv_max_layers: int | None = None
    speaker_uncond_mode: str = "mask"
    seed: int | None = None
    t_schedule_mode: str = "linear"
    sway_coeff: float = -1.0
    trim_tail: bool = True
    tail_window_size: int = 20
    tail_std_threshold: float = 0.05
    tail_mean_threshold: float = 0.1
    lora_adapter: str | None = None
    lora_hot_swap: bool = False
    apply_watermark: bool = True


@dataclass
class SamplingResult:
    audio: torch.Tensor
    audios: list[torch.Tensor]
    sample_rate: int
    stage_timings: list[tuple[str, float]]
    total_to_decode: float
    used_seed: int
    messages: list[str]
    # Per-candidate seeds: candidate i was drawn with used_seed + i, so any
    # candidate can be regenerated alone with num_candidates=1 and its seed.
    # Optional so existing constructors keep working; synthesize always fills it.
    used_seeds: list[int] | None = None


def _maybe_compile_inference_model(
    model: TextToLatentRFDiT,
    *,
    enabled: bool,
    dynamic: bool,
) -> TextToLatentRFDiT:
    # IRODORI_COMPILE=1 opts into torch.compile without changing the runtime
    # key (requires a working Triton toolchain: Linux/WSL2 + a C compiler).
    # Compiled kernels are replayed inside the CUDA step graphs, combining
    # inductor's kernel fusion with graph replay's zero launch overhead.
    if os.environ.get("IRODORI_COMPILE", "0").strip() == "1":
        enabled = True
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_model=True requires torch.compile (PyTorch 2+).")
    # dynamic=None (auto) compiles the first-seen shape statically and switches
    # to dynamic-shape kernels on the second, so the many latent-length buckets
    # do not each pay a full compile.
    compile_kwargs = {"dynamic": True if dynamic else None}
    model.encode_conditions = torch.compile(model.encode_conditions, **compile_kwargs)
    model.build_context_kv_cache = torch.compile(model.build_context_kv_cache, **compile_kwargs)
    model.forward_with_encoded_conditions = torch.compile(
        model.forward_with_encoded_conditions,
        **compile_kwargs,
    )
    model._irodori_compiled = True
    print("[runtime] torch.compile enabled for inference model", flush=True)
    return model


def _move_inference_module(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    module.to(device=device)
    with torch.no_grad():
        for param in module.parameters():
            if param.is_floating_point() and param.dtype != dtype:
                param.data = param.data.to(device=device, dtype=dtype)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device=device, dtype=dtype)
        for child in module.modules():
            for name, buffer in child._buffers.items():
                if buffer is None:
                    continue
                if buffer.is_floating_point() and buffer.dtype != dtype:
                    child._buffers[name] = buffer.to(device=device, dtype=dtype)
                elif buffer.device != device:
                    child._buffers[name] = buffer.to(device=device)
    return module


def resolve_runtime_dtype(*, precision: str, device: torch.device) -> torch.dtype:
    mode = str(precision).strip().lower()
    if mode == "fp32":
        return torch.float32
    if mode == "bf16":
        if device.type not in ("cuda", "xpu"):
            raise ValueError("precision='bf16' currently requires CUDA or XPU device.")
        return torch.bfloat16
    raise ValueError(f"Unsupported precision={precision!r}. Expected one of: fp32, bf16.")


def resolve_cfg_scales(
    *,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_caption: float,
    cfg_scale_speaker: float,
    cfg_scale: float | None,
    use_caption_condition: bool = True,
    use_speaker_condition: bool = True,
) -> tuple[float, float, float, list[str]]:
    """Normalize/validate CFG scales for guidance mode."""
    messages: list[str] = []
    text_val = float(cfg_scale_text)
    caption_val = float(cfg_scale_caption)
    speaker_val = float(cfg_scale_speaker)

    if cfg_scale is not None:
        text_val = float(cfg_scale)
        caption_val = float(cfg_scale)
        speaker_val = float(cfg_scale)
    if not use_speaker_condition:
        if speaker_val > 0.0:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint or request; "
                "ignoring cfg_scale_speaker."
            )
        speaker_val = 0.0

    mode = str(cfg_guidance_mode).strip().lower()
    enabled_vals = [value for value in (text_val, speaker_val) if value > 0.0]
    if use_caption_condition and caption_val > 0.0:
        enabled_vals.append(caption_val)
    if mode == "joint" and enabled_vals and (max(enabled_vals) - min(enabled_vals) > 1e-6):
        raise ValueError(
            "cfg_guidance_mode='joint' requires equal enabled cfg_scale_text/cfg_scale_caption/cfg_scale_speaker, "
            "or set cfg_scale."
        )

    return text_val, caption_val, speaker_val, messages


def _load_torch_checkpoint_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


_CONFIG_META_KEY = "config_json"
_INFERENCE_CONFIG_KEYS = {
    "max_text_len",
    "max_caption_len",
    "fixed_target_latent_steps",
}


def _load_checkpoint_from_pt(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    ckpt = _load_torch_checkpoint_payload(path)
    model_state = ckpt.get("model")
    model_cfg = ckpt.get("model_config")
    train_cfg = ckpt.get("train_config")

    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint missing model weights dictionary: {path}")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Checkpoint missing model_config dictionary: {path}")
    if train_cfg is not None and not isinstance(train_cfg, dict):
        raise ValueError(f"Checkpoint train_config must be a dictionary when present: {path}")

    if checkpoint_state_uses_lora(model_state):
        raise ValueError(
            f"LoRA checkpoints must be loaded from adapter directories or merged safetensors: {path}"
        )
    return model_state, model_cfg, _extract_inference_train_config(train_cfg)


def _parse_json_mapping(
    raw: str | None,
    *,
    field: str,
    path: Path,
    required: bool = False,
) -> dict | None:
    if raw is None:
        if required:
            raise ValueError(f"Missing required metadata field '{field}' in checkpoint: {path}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata for checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object: {path}")
    return payload


def _extract_inference_train_config(raw: dict | None) -> dict | None:
    if raw is None:
        return None

    inference_cfg: dict[str, int] = {}
    for key in _INFERENCE_CONFIG_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, int):
            raise ValueError(f"Inference config key '{key}' must be int, got {type(value)!r}.")
        inference_cfg[key] = int(value)

    return inference_cfg or None


def _split_flat_checkpoint_config(path: Path, flat_config: dict) -> tuple[dict, dict | None]:
    model_cfg: dict[str, object] = {}
    inference_cfg: dict[str, int] = {}
    for key, value in flat_config.items():
        if key in _INFERENCE_CONFIG_KEYS:
            if not isinstance(value, int):
                raise ValueError(
                    f"Inference config key '{key}' must be int in checkpoint metadata: {path}"
                )
            inference_cfg[key] = int(value)
            continue
        model_cfg[key] = value
    return model_cfg, (inference_cfg or None)


def _load_checkpoint_from_safetensors(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    model_state = load_safetensors_file(str(path), device="cpu")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"Safetensors checkpoint has no model weights: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    flat_config = _parse_json_mapping(
        metadata.get(_CONFIG_META_KEY),
        field=_CONFIG_META_KEY,
        path=path,
        required=True,
    )
    model_cfg, inference_cfg = _split_flat_checkpoint_config(path=path, flat_config=flat_config)
    return model_state, model_cfg, inference_cfg


def _load_checkpoint_for_inference(
    path: Path,
) -> tuple[dict[str, torch.Tensor], dict, dict | None]:
    if path.suffix.lower() == ".safetensors":
        return _load_checkpoint_from_safetensors(path)
    return _load_checkpoint_from_pt(path)


class InferenceRuntime:
    def __init__(
        self,
        *,
        key: RuntimeKey,
        model_cfg: ModelConfig,
        train_cfg: dict | None,
        model: TextToLatentRFDiT,
        tokenizer: PretrainedTextTokenizer,
        caption_tokenizer: PretrainedTextTokenizer | None,
        codec: DACVAECodec,
        default_text_max_len: int,
        default_caption_max_len: int,
    ) -> None:
        self.key = key
        self.model_device = resolve_runtime_device(key.model_device)
        self.codec_device = resolve_runtime_device(key.codec_device)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.model = model
        self.tokenizer = tokenizer
        self.caption_tokenizer = caption_tokenizer
        self.codec = codec
        self.default_text_max_len = default_text_max_len
        self.default_caption_max_len = default_caption_max_len
        self.watermarker = SilentCipherWatermarker(device=str(self.codec_device))
        self._infer_lock = threading.Lock()
        self._model_dtype = next(self.model.parameters()).dtype
        self._lora_adapter_names: dict[str, str] = {}
        self._merged_lora_path: str | None = None
        self._cuda_graph_cache: dict = {}
        self._last_lora_token: str | None = None

    @classmethod
    def from_key(cls, key: RuntimeKey) -> InferenceRuntime:
        model_device = resolve_runtime_device(key.model_device)
        codec_device = resolve_runtime_device(key.codec_device)
        if model_device.type == "cuda" or codec_device.type == "cuda":
            _enable_cuda_fast_math()
        model_dtype = resolve_runtime_dtype(
            precision=key.model_precision,
            device=model_device,
        )
        codec_dtype = resolve_runtime_dtype(
            precision=key.codec_precision,
            device=codec_device,
        )

        model_state, model_cfg_dict, train_cfg = _load_checkpoint_for_inference(
            Path(key.checkpoint)
        )
        model_cfg = ModelConfig(**model_cfg_dict)

        model = TextToLatentRFDiT(model_cfg).to(model_device)
        model.load_state_dict(model_state)
        model = _move_inference_module(model, device=model_device, dtype=model_dtype)
        model.eval()
        model = _maybe_compile_inference_model(
            model,
            enabled=bool(key.compile_model),
            dynamic=bool(key.compile_dynamic),
        )

        tokenizer = PretrainedTextTokenizer.from_pretrained(
            repo_id=model_cfg.text_tokenizer_repo,
            add_bos=bool(model_cfg.text_add_bos),
            local_files_only=False,
        )
        if tokenizer.vocab_size != model_cfg.text_vocab_size:
            raise ValueError(
                f"text_vocab_size mismatch: checkpoint text_vocab_size={model_cfg.text_vocab_size} but tokenizer "
                f"({model_cfg.text_tokenizer_repo}) vocab_size={tokenizer.vocab_size}."
            )
        caption_tokenizer = None
        if model_cfg.use_caption_condition:
            caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
                repo_id=model_cfg.caption_tokenizer_repo_resolved,
                add_bos=model_cfg.caption_add_bos_resolved,
                local_files_only=False,
            )
            if caption_tokenizer.vocab_size != model_cfg.caption_vocab_size_resolved:
                raise ValueError(
                    f"caption_vocab_size mismatch: checkpoint caption_vocab_size={model_cfg.caption_vocab_size_resolved} but tokenizer ({model_cfg.caption_tokenizer_repo_resolved}) "
                    f"vocab_size={caption_tokenizer.vocab_size}."
                )

        default_text_max_len = 256
        default_caption_max_len = default_text_max_len
        if isinstance(train_cfg, dict):
            ckpt_text_max_len = train_cfg.get("max_text_len")
            if isinstance(ckpt_text_max_len, int) and ckpt_text_max_len > 0:
                default_text_max_len = int(ckpt_text_max_len)
            ckpt_caption_max_len = train_cfg.get("max_caption_len")
            if isinstance(ckpt_caption_max_len, int) and ckpt_caption_max_len > 0:
                default_caption_max_len = int(ckpt_caption_max_len)
            else:
                default_caption_max_len = default_text_max_len

        codec = DACVAECodec.load(
            repo_id=key.codec_repo,
            device=str(codec_device),
            dtype=codec_dtype,
            deterministic_encode=bool(key.codec_deterministic_encode),
            deterministic_decode=bool(key.codec_deterministic_decode),
        )
        if model_cfg.latent_dim != codec.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: checkpoint latent_dim={model_cfg.latent_dim} but codec latent_dim={codec.latent_dim}. "
                "Use a compatible codec/checkpoint pair."
            )

        # Build the lazily-grown RoPE caches now (normal mode, before any
        # compile or CUDA graph capture) so they never rebuild inside compiled
        # or captured code. Sized to the default max_seconds bound (30 s);
        # longer manual requests just rebuild lazily as before.
        hop_length = int(codec.model.hop_length)
        max_frames = max(1, math.ceil(30.0 * codec.sample_rate / hop_length))
        graph_bucket = max(1, int(os.environ.get("IRODORI_CUDA_GRAPH_BUCKET", "16")))
        max_patched = math.ceil(max_frames / model_cfg.latent_patch_size)
        max_padded = ((max_patched + graph_bucket - 1) // graph_bucket) * graph_bucket
        speaker_len = None
        if model_cfg.use_speaker_condition_resolved:
            speaker_len = (
                math.ceil(max_frames / max(1, int(model_cfg.speaker_patch_size))) + 1
            )
        model.prewarm_rope_caches(
            text_len=default_text_max_len,
            latent_len=max_padded,
            speaker_len=speaker_len,
        )

        return cls(
            key=key,
            model_cfg=model_cfg,
            train_cfg=train_cfg if isinstance(train_cfg, dict) else None,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            default_text_max_len=default_text_max_len,
            default_caption_max_len=default_caption_max_len,
        )

    def _resolve_lora_adapter_path(self, adapter_path: str | None) -> str | None:
        if adapter_path is None:
            return None
        raw = str(adapter_path).strip()
        if raw.lower() in {"", "none", "null", "off", "disable", "disabled", "base"}:
            return None

        path = Path(raw).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"LoRA adapter directory not found: {path}")
        if not is_lora_adapter_dir(path):
            raise ValueError(
                f"LoRA adapter directory must contain adapter_config.json and adapter weights: {path}"
            )
        return str(path.resolve())

    @staticmethod
    def _adapter_name_for_path(path: str) -> str:
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
        return f"runtime_{digest}"

    def _prepare_lora_for_request(
        self,
        adapter_path: str | None,
        *,
        messages: list[str],
        stage_timings: list[tuple[str, float]],
        log_fn: Callable[[str], None],
        hot_swap: bool = False,
    ) -> Any:
        should_time = adapter_path is not None and str(adapter_path).strip() != ""
        t0 = _measure_start(self.model_device) if should_time else None
        try:
            return self._prepare_lora_for_request_inner(
                adapter_path,
                messages=messages,
                log_fn=log_fn,
                hot_swap=hot_swap,
            )
        finally:
            if t0 is not None:
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("prepare_lora", stage_sec))
                log_fn(f"[runtime] prepare_lora: {stage_sec * 1000.0:.1f} ms")

    def _lora_hot_swap_safe(self, resolved_path: str | None) -> tuple[bool, str]:
        """
        Check whether cached CUDA graphs can survive a LoRA switch.

        Graphs reference weight *storage*, not values. merge/unmerge writes
        deltas into the same storage, so graphs stay valid as long as no module
        used inside the graph is replaced wholesale (modules_to_save) and the
        previous adapter was actually merged.
        """
        if os.environ.get("IRODORI_DISABLE_LORA_MERGE", "0").strip() == "1":
            return False, "LoRA merge is disabled (IRODORI_DISABLE_LORA_MERGE=1)"
        if (
            self._last_lora_token not in (None, "__base__")
            and self._merged_lora_path != self._last_lora_token
        ):
            return False, "previous adapter was not merged into base weights"
        if resolved_path is None:
            return True, ""
        try:
            with open(Path(resolved_path) / LORA_ADAPTER_CONFIG_NAME, encoding="utf-8") as fh:
                adapter_cfg = json.load(fh)
        except Exception as exc:
            return False, f"could not read adapter config ({exc!r})"
        modules_to_save = adapter_cfg.get("modules_to_save") or []
        unsafe_modules = [m for m in modules_to_save if m != "duration_predictor"]
        if unsafe_modules:
            return False, (
                f"modules_to_save={unsafe_modules} replaces module storage used by graphs"
            )
        if adapter_cfg.get("use_dora"):
            return False, "DoRA adapters are not supported for hot swap"
        return True, ""

    def _prepare_lora_for_request_inner(
        self,
        adapter_path: str | None,
        *,
        messages: list[str],
        log_fn: Callable[[str], None],
        hot_swap: bool = False,
    ) -> Any:
        resolved_path = self._resolve_lora_adapter_path(adapter_path)
        lora_token = resolved_path if resolved_path is not None else "__base__"
        if lora_token != self._last_lora_token:
            # The duration/condition graph bakes in modules the sampler graphs
            # never touch (duration_predictor, which hot-swap explicitly allows
            # in modules_to_save), so it cannot survive any adapter switch.
            # Recapture is cheap and happens on the next request/prewarm.
            self._cuda_graph_cache.pop("__duration__", None)
            cached_graphs = len([k for k in self._cuda_graph_cache if not isinstance(k, str)])
            preserved = False
            if hot_swap and cached_graphs > 0:
                safe, reason = self._lora_hot_swap_safe(resolved_path)
                if safe:
                    preserved = True
                    msg = (
                        f"info: LoRA hot-swap: keeping {cached_graphs} cached CUDA "
                        "graphs across adapter switch (weights are swapped in place; "
                        "tiny fp rounding drift can accumulate per swap)."
                    )
                    messages.append(msg)
                    log_fn(msg)
                else:
                    msg = f"info: LoRA hot-swap unavailable ({reason}); dropping cached graphs."
                    messages.append(msg)
                    log_fn(msg)
            if not preserved:
                # Captured CUDA graphs bake in the module structure/weights active
                # at capture time; invalidate them when the LoRA state changes.
                self._cuda_graph_cache.clear()
                if cached_graphs > 0:
                    msg = (
                        f"warning: LoRA state changed ({self._last_lora_token} -> {lora_token}); "
                        f"dropped {cached_graphs} cached CUDA graphs. Prewarm again with the same "
                        "LoRA Adapter Directory, or enable LoRA Hot-Swap to keep them."
                    )
                    messages.append(msg)
                    log_fn(msg)
            self._last_lora_token = lora_token
        if resolved_path is None:
            self._unmerge_lora_if_needed(log_fn=log_fn)
            disable_adapter = getattr(self.model, "disable_adapter", None)
            if callable(disable_adapter):
                msg = "info: dynamic LoRA disabled for this request; using base model."
                messages.append(msg)
                log_fn(msg)
                return disable_adapter()
            return nullcontext()

        if getattr(self.model, "_irodori_compiled", False):
            msg = (
                "info: torch.compile is active; the LoRA structure change triggers a "
                "one-time recompile during the next warmup (prewarm absorbs it)."
            )
            messages.append(msg)
            log_fn(msg)

        if self._merged_lora_path == resolved_path:
            msg = f"info: using merged LoRA adapter (cached): {resolved_path}"
            messages.append(msg)
            log_fn(msg)
            return nullcontext()

        self._unmerge_lora_if_needed(log_fn=log_fn)

        adapter_name = self._lora_adapter_names.get(resolved_path)
        if adapter_name is None:
            adapter_name = self._adapter_name_for_path(resolved_path)
            msg = f"info: loading LoRA adapter: {resolved_path}"
            messages.append(msg)
            log_fn(msg)
        else:
            msg = f"info: using cached LoRA adapter: {resolved_path}"
            messages.append(msg)
            log_fn(msg)

        self.model = load_lora_adapter(
            self.model,
            resolved_path,
            is_trainable=False,
            adapter_name=adapter_name,
            torch_device=str(self.model_device),
        )
        self._lora_adapter_names[resolved_path] = adapter_name
        self.model = _move_inference_module(
            self.model,
            device=self.model_device,
            dtype=self._model_dtype,
        )
        self.model.eval()
        self._merge_lora_for_speed(resolved_path, messages=messages, log_fn=log_fn)
        return nullcontext()

    def _merge_lora_for_speed(
        self,
        resolved_path: str,
        *,
        messages: list[str],
        log_fn: Callable[[str], None],
    ) -> None:
        """
        Merge the active LoRA adapter into base weights so sampling avoids the
        extra low-rank matmuls on every forward pass. Mathematically equivalent
        (up to float rounding) to running with an unmerged adapter.
        """
        if os.environ.get("IRODORI_DISABLE_LORA_MERGE", "0").strip() == "1":
            return
        merge_adapter = getattr(self.model, "merge_adapter", None)
        if not callable(merge_adapter):
            return
        try:
            merge_adapter()
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"warning: LoRA merge failed ({exc!r}); using unmerged adapter."
            messages.append(msg)
            log_fn(msg)
            return
        self._merged_lora_path = resolved_path
        msg = "info: merged LoRA adapter into base weights for faster inference."
        messages.append(msg)
        log_fn(msg)

    def _unmerge_lora_if_needed(self, *, log_fn: Callable[[str], None]) -> None:
        if self._merged_lora_path is None:
            return
        unmerge_adapter = getattr(self.model, "unmerge_adapter", None)
        if callable(unmerge_adapter):
            unmerge_adapter()
            log_fn(f"[runtime] unmerged LoRA adapter: {self._merged_lora_path}")
        self._merged_lora_path = None

    def _load_reference_latent(
        self,
        *,
        req: SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        runtime_dtype = next(self.model.parameters()).dtype
        if not self.model_cfg.use_speaker_condition_resolved:
            if req.ref_wav is not None or req.ref_latent is not None or req.ref_audio is not None:
                messages.append(
                    "info: speaker conditioning is disabled for this checkpoint; ignoring reference input."
                )
            return None, None
        if req.no_ref:
            ref_len = max(1, int(self.model_cfg.speaker_patch_size))
            ref_latent_patched = torch.zeros(
                (
                    batch_size,
                    ref_len,
                    self.model_cfg.latent_dim * self.model_cfg.latent_patch_size,
                ),
                device=self.model_device,
                dtype=runtime_dtype,
            )
            ref_mask = torch.zeros(
                (batch_size, ref_len), dtype=torch.bool, device=self.model_device
            )
            return ref_latent_patched, ref_mask

        if req.ref_wav is None and req.ref_latent is None and req.ref_audio is None:
            raise ValueError("Specify ref_wav, ref_latent or ref_audio, or set no_ref=True.")
        if req.ref_audio is not None and (req.ref_wav is not None or req.ref_latent is not None):
            raise ValueError("ref_audio cannot be combined with ref_wav/ref_latent.")

        max_ref_latent_steps = None
        if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
            max_ref_latent_steps = max(
                1,
                math.ceil(
                    float(req.max_ref_seconds)
                    * float(self.codec.sample_rate)
                    / float(int(self.codec.model.hop_length))
                ),
            )

        if req.ref_latent is not None:
            latent_raw = torch.load(req.ref_latent, map_location="cpu", weights_only=True)
            ref_latent = _coerce_latent_shape(
                latent_raw, latent_dim=self.model_cfg.latent_dim
            ).unsqueeze(0)
            ref_latent = ref_latent.to(dtype=runtime_dtype)
        else:
            if req.ref_audio is not None:
                wav, sr = _coerce_reference_audio(req.ref_audio, req.ref_audio_sample_rate)
            else:
                wav, sr = _load_audio(req.ref_wav)
            if req.max_ref_seconds is not None and req.max_ref_seconds > 0:
                max_ref_samples = max(1, int(float(req.max_ref_seconds) * float(sr)))
                if wav.shape[1] > max_ref_samples:
                    messages.append(
                        f"warning: reference audio exceeds max_ref_seconds ({req.max_ref_seconds}s). "
                        f"Trimming from {float(wav.shape[1]) / float(sr):.2f}s to {float(max_ref_samples) / float(sr):.2f}s."
                    )
                    wav = wav[:, :max_ref_samples]
            if req.ref_normalize_db is not None:
                messages.append(
                    f"info: reference loudness normalize enabled (target_db={float(req.ref_normalize_db):.2f}, includes peak safety scaling)."
                )
            elif req.ref_ensure_max:
                messages.append("info: reference peak safety scaling enabled (ensure_max=True).")
            ref_latent = self.codec.encode_waveform(
                wav.unsqueeze(0),
                sample_rate=int(sr),
                normalize_db=req.ref_normalize_db,
                ensure_max=bool(req.ref_ensure_max),
            ).cpu()

        if max_ref_latent_steps is not None and ref_latent.shape[1] > max_ref_latent_steps:
            messages.append(
                f"warning: reference latent steps ({ref_latent.shape[1]}) exceed max_ref_seconds bound ({max_ref_latent_steps} steps). "
                "Trimming reference latent."
            )
            ref_latent = ref_latent[:, :max_ref_latent_steps]

        ref_latent_patched = patchify_latent(ref_latent, self.model_cfg.latent_patch_size).to(
            device=self.model_device,
            dtype=runtime_dtype,
        )
        if ref_latent_patched.shape[1] == 0:
            raise ValueError(
                "Reference latent length became zero after patchify. Use longer reference audio."
            )
        if batch_size > 1:
            ref_latent_patched = ref_latent_patched.repeat(batch_size, 1, 1)
        ref_mask = torch.ones(
            (batch_size, ref_latent_patched.shape[1]),
            dtype=torch.bool,
            device=self.model_device,
        )
        return ref_latent_patched, ref_mask

    def _load_speaker_embedding_condition(
        self,
        *,
        req: SamplingRequest,
        batch_size: int,
        messages: list[str],
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if req.ref_embed is None:
            return None, None
        if not self.model_cfg.use_speaker_condition_resolved:
            messages.append(
                "info: speaker conditioning is disabled for this checkpoint; ignoring speaker embedding."
            )
            return None, None
        if (
            req.ref_wav is not None
            or req.ref_latent is not None
            or req.ref_audio is not None
            or req.no_ref
        ):
            raise ValueError(
                "ref_embed/--ref-embed cannot be combined with ref_wav/ref_latent/ref_audio/"
                "no_ref. Use exactly one speaker conditioning source."
            )

        runtime_dtype = next(self.model.parameters()).dtype
        speaker_embedding = load_speaker_inversion_payload(req.ref_embed)["speaker_embedding"]
        state, mask = speaker_inversion_batch_tensors(
            speaker_embedding,
            batch_size=batch_size,
            device=self.model_device,
            dtype=runtime_dtype,
        )
        messages.append(
            "info: using speaker inversion embedding "
            f"tokens={state.shape[1]} uncond_mode={req.speaker_uncond_mode}."
        )
        return state, mask

    def _encode_and_predict_duration(
        self,
        *,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        ref_latent: torch.Tensor | None,
        ref_mask: torch.Tensor | None,
        caption_ids: torch.Tensor | None,
        caption_mask: torch.Tensor | None,
        speaker_state_override: torch.Tensor | None,
        speaker_mask_override: torch.Tensor | None,
        speaker_uncond_mode: str,
        duration_features: torch.Tensor,
        has_speaker: torch.Tensor,
        has_caption: torch.Tensor | None,
        graph_eligible: bool,
        log_fn: Callable[[str], None],
    ) -> tuple[tuple, torch.Tensor]:
        """
        Run encode_conditions + predict_duration_log_frames, replaying them as
        a single CUDA graph when the request shape allows it.

        The eager pair launches hundreds of tiny kernels, so on Windows (no
        Triton/torch.compile) the stage is dominated by CPU launch overhead,
        just like the sampling loop. Shapes are fixed for no-reference requests
        (text padded to its bucket length or max_text_len, the no_ref speaker
        placeholder, aux features), so one graph per shape signature suffices.
        Reference-audio requests have per-request shapes and stay eager.

        Returns the encoded-conditions tuple and the predicted log-frames. On
        the graph path these are the graph's static output tensors: they are
        valid until the next call of this method and must be consumed within
        the current request (synthesize holds the inference lock throughout).
        """

        def _run_eager() -> tuple[tuple, torch.Tensor]:
            encoded = self.model.encode_conditions(
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state_override,
                speaker_mask_override=speaker_mask_override,
                speaker_uncond_mode=speaker_uncond_mode,
            )
            pred = self.model.predict_duration_log_frames(
                text_state=encoded[0],
                text_mask=encoded[1],
                speaker_state=encoded[2],
                speaker_mask=encoded[3],
                caption_state=encoded[4],
                caption_mask=encoded[5],
                duration_features=duration_features,
                has_speaker=has_speaker,
                has_caption=has_caption,
            )
            return encoded, pred

        graph_supported = (
            graph_eligible
            and speaker_state_override is None
            and self.model_device.type == "cuda"
            and os.environ.get("IRODORI_DISABLE_CUDA_GRAPH", "0").strip() != "1"
            and os.environ.get("IRODORI_DISABLE_DURATION_GRAPH", "0").strip() != "1"
            and not self._cuda_graph_cache.get("__duration_disabled__", False)
        )
        if not graph_supported:
            return _run_eager()

        request_inputs = [
            text_ids,
            text_mask,
            ref_latent,
            ref_mask,
            caption_ids,
            caption_mask,
            duration_features,
            has_speaker,
            has_caption,
        ]
        presence = tuple(value is not None for value in request_inputs)
        flat_inputs = [value for value in request_inputs if value is not None]
        sig = (
            presence,
            tuple((tuple(t.shape), str(t.dtype)) for t in flat_inputs),
            str(speaker_uncond_mode),
        )
        bundles = self._cuda_graph_cache.setdefault("__duration__", {})
        entry = bundles.get(sig)
        if entry is not None:
            for static_buf, value in zip(entry["inputs"], flat_inputs, strict=True):
                static_buf.copy_(value)
            entry["graph"].replay()
            return entry["encoded"], entry["pred"]

        static_inputs = [value.clone() for value in flat_inputs]
        buffer_iter = iter(static_inputs)
        (
            s_text_ids,
            s_text_mask,
            s_ref_latent,
            s_ref_mask,
            s_caption_ids,
            s_caption_mask,
            s_features,
            s_has_speaker,
            s_has_caption,
        ) = [next(buffer_iter) if flag else None for flag in presence]

        def _forward() -> tuple[tuple, torch.Tensor]:
            encoded = self.model.encode_conditions(
                text_input_ids=s_text_ids,
                text_mask=s_text_mask,
                ref_latent=s_ref_latent,
                ref_mask=s_ref_mask,
                caption_input_ids=s_caption_ids,
                caption_mask=s_caption_mask,
                speaker_state_override=None,
                speaker_mask_override=None,
                speaker_uncond_mode=speaker_uncond_mode,
            )
            pred = self.model.predict_duration_log_frames(
                text_state=encoded[0],
                text_mask=encoded[1],
                speaker_state=encoded[2],
                speaker_mask=encoded[3],
                caption_state=encoded[4],
                caption_mask=encoded[5],
                duration_features=s_features,
                has_speaker=s_has_speaker,
                has_caption=s_has_caption,
            )
            return encoded, pred

        try:
            # Warmup on the shared side stream initializes cuBLAS/cuDNN and
            # produces reference outputs the captured replay is checked against.
            warm_stream = _get_warmup_stream()
            warm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm_stream):
                warm_encoded, warm_pred = _forward()
                warm_flat = [t.clone() for t in [*warm_encoded, warm_pred] if t is not None]
            torch.cuda.current_stream().wait_stream(warm_stream)
            torch.cuda.synchronize(self.model_device)

            pool = self._cuda_graph_cache.get("__pool__")
            if pool is None:
                pool = torch.cuda.graph_pool_handle()
                self._cuda_graph_cache["__pool__"] = pool
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out_encoded, out_pred = _forward()

            # Validate one replay against the eager warmup outputs.
            graph.replay()
            torch.cuda.synchronize(self.model_device)
            out_flat = [t for t in [*out_encoded, out_pred] if t is not None]
            if len(out_flat) != len(warm_flat) or not all(
                torch.equal(out, warm) for out, warm in zip(out_flat, warm_flat, strict=True)
            ):
                raise RuntimeError(
                    "duration CUDA graph replay mismatch against eager warmup output."
                )
        except Exception as exc:
            self._cuda_graph_cache["__duration_disabled__"] = True
            _end_pool_recording_after_failed_capture(
                self.model_device, self._cuda_graph_cache.get("__pool__")
            )
            log_fn(f"[runtime] duration CUDA graph disabled: {exc!r}")
            return _run_eager()

        bundles[sig] = {
            "graph": graph,
            "inputs": static_inputs,
            "encoded": out_encoded,
            "pred": out_pred,
        }
        log_fn(
            f"[runtime] captured duration/condition CUDA graph (entries: {len(bundles)})"
        )
        return out_encoded, out_pred

    def synthesize(
        self,
        req: SamplingRequest,
        *,
        log_fn: Callable[[str], None] | None = None,
    ) -> SamplingResult:
        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        messages: list[str] = []
        _log(
            (
                "[runtime] start synthesize "
                "model_device={} model_precision={} codec_device={} codec_precision={} "
                "silentcipher_watermark={} mode={} seconds={} steps={} seed={} candidates={} decode_mode={}"
            ).format(
                self.key.model_device,
                self.key.model_precision,
                self.key.codec_device,
                self.key.codec_precision,
                self.watermarker.ready,
                req.cfg_guidance_mode,
                req.seconds,
                req.num_steps,
                "random" if req.seed is None else int(req.seed),
                req.num_candidates,
                req.decode_mode,
            )
        )

        manual_seconds = None if req.seconds is None else float(req.seconds)
        if manual_seconds is not None and manual_seconds <= 0:
            raise ValueError(f"seconds must be > 0 when provided, got {req.seconds}")
        duration_scale = float(req.duration_scale)
        if duration_scale <= 0:
            raise ValueError(f"duration_scale must be > 0, got {duration_scale}")
        min_seconds = float(req.min_seconds)
        max_seconds = float(req.max_seconds)
        if min_seconds <= 0:
            raise ValueError(f"min_seconds must be > 0, got {min_seconds}")
        if max_seconds < min_seconds:
            raise ValueError(
                f"max_seconds must be >= min_seconds, got min={min_seconds} max={max_seconds}"
            )
        num_candidates = int(req.num_candidates)
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be > 0, got {num_candidates}")
        decode_mode = str(req.decode_mode).strip().lower()
        if decode_mode not in {"sequential", "batch"}:
            raise ValueError(
                f"Unsupported decode_mode={req.decode_mode!r}. Expected one of: sequential, batch."
            )

        raw_text = str(req.text)
        normalized_text = normalize_text(raw_text).strip()
        if normalized_text == "":
            raise ValueError("text became empty after normalization.")

        text_max_len = (
            self.default_text_max_len if req.max_text_len is None else int(req.max_text_len)
        )
        if text_max_len <= 0:
            raise ValueError(f"max_text_len must be > 0, got {text_max_len}")
        caption_max_len = (
            self.default_caption_max_len
            if req.max_caption_len is None
            else int(req.max_caption_len)
        )
        if self.model_cfg.use_caption_condition and caption_max_len <= 0:
            raise ValueError(f"max_caption_len must be > 0, got {caption_max_len}")
        has_caption_text = bool(
            self.model_cfg.use_caption_condition
            and req.caption is not None
            and str(req.caption).strip() != ""
        )

        truncation_factor = None if req.truncation_factor is None else float(req.truncation_factor)
        rescale_k = None if req.rescale_k is None else float(req.rescale_k)
        rescale_sigma = None if req.rescale_sigma is None else float(req.rescale_sigma)
        if truncation_factor is not None and truncation_factor <= 0:
            raise ValueError(f"truncation_factor must be > 0, got {truncation_factor}")
        if (rescale_k is None) != (rescale_sigma is None):
            raise ValueError("rescale_k and rescale_sigma must be set together.")
        if rescale_k is not None and rescale_k <= 0:
            raise ValueError(f"rescale_k must be > 0, got {rescale_k}")
        if rescale_sigma is not None and rescale_sigma <= 0:
            raise ValueError(f"rescale_sigma must be > 0, got {rescale_sigma}")

        speaker_kv_scale = None if req.speaker_kv_scale is None else float(req.speaker_kv_scale)
        speaker_kv_min_t = None
        speaker_kv_max_layers = (
            None if req.speaker_kv_max_layers is None else int(req.speaker_kv_max_layers)
        )
        use_speaker_for_request = bool(
            self.model_cfg.use_speaker_condition_resolved and not req.no_ref
        )
        if speaker_kv_scale is not None:
            if not use_speaker_for_request:
                messages.append(
                    "info: speaker conditioning is disabled for this request; ignoring speaker_kv_scale."
                )
                speaker_kv_scale = None
            else:
                if speaker_kv_scale <= 0:
                    raise ValueError(f"speaker_kv_scale must be > 0, got {speaker_kv_scale}")
                speaker_kv_min_t = (
                    0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)
                )
                if not (0.0 <= speaker_kv_min_t <= 1.0):
                    raise ValueError(f"speaker_kv_min_t must be in [0, 1], got {speaker_kv_min_t}")
                if speaker_kv_max_layers is not None and speaker_kv_max_layers < 0:
                    raise ValueError(
                        f"speaker_kv_max_layers must be >= 0 when specified, got {speaker_kv_max_layers}"
                    )

        cfg_mode = str(req.cfg_guidance_mode).strip().lower()
        if cfg_mode not in {"independent", "joint", "alternating"}:
            raise ValueError(
                f"Unsupported cfg_guidance_mode={req.cfg_guidance_mode!r}. "
                "Expected one of: independent, joint, alternating."
            )

        cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, scale_messages = resolve_cfg_scales(
            cfg_guidance_mode=cfg_mode,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_scale=req.cfg_scale,
            use_caption_condition=has_caption_text,
            use_speaker_condition=use_speaker_for_request,
        )
        messages.extend(scale_messages)
        for msg in scale_messages:
            _log(msg)

        stage_timings: list[tuple[str, float]] = []
        if req.seed is None:
            used_seed = int(secrets.randbits(63))
            msg = f"info: seed not specified; using random seed {used_seed}."
            messages.append(msg)
            _log(msg)
        else:
            used_seed = int(req.seed)
            _log(f"[runtime] using seed: {used_seed}")
        used_seeds = [used_seed + index for index in range(num_candidates)]
        if num_candidates > 1:
            msg = (
                f"info: candidate seeds {used_seeds[0]}..{used_seeds[-1]} "
                "(candidate i uses seed + i and can be regenerated alone with "
                "num_candidates=1)."
            )
            messages.append(msg)
            _log(msg)
        post_load_t0 = _measure_start(self.model_device, self.codec_device)

        with (
            self._infer_lock,
            self._prepare_lora_for_request(
                req.lora_adapter,
                messages=messages,
                stage_timings=stage_timings,
                log_fn=_log,
                hot_swap=bool(req.lora_hot_swap),
            ),
            torch.inference_mode(),
        ):
            t0 = _measure_start(self.model_device)
            text_ids, text_mask = self.tokenizer.batch_encode(
                [normalized_text] * num_candidates,
                max_length=text_max_len,
            )
            # Text-length bucketing: pad short texts to the smallest bucket
            # instead of the full max_text_len. Masked padding keys are
            # mathematically inert but still cost cross-attention compute
            # every step. Tokenization is right-padded, so slicing the padded
            # batch equals re-encoding at the bucket length. Note that
            # duration features keep the original max_text_len normalization
            # (text_max_len below is unchanged), so predicted durations are
            # unaffected by the bucket choice. Explicit max_text_len requests
            # bypass bucketing.
            if req.max_text_len is None:
                token_count = int(text_mask.sum(dim=1).max())
                for bucket_len in _text_bucket_lengths():
                    if token_count <= bucket_len < text_ids.shape[1]:
                        text_ids = text_ids[:, :bucket_len].contiguous()
                        text_mask = text_mask[:, :bucket_len].contiguous()
                        break
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("tokenize_text", stage_sec))
            _log(f"[runtime] tokenize_text: {stage_sec * 1000.0:.1f} ms")
            text_ids = text_ids.to(self.model_device)
            text_mask = text_mask.to(self.model_device)
            caption_ids = None
            caption_mask = None
            if self.model_cfg.use_caption_condition:
                if self.caption_tokenizer is None:
                    raise RuntimeError(
                        "Caption conditioning is enabled but caption tokenizer is not loaded."
                    )
                caption_text = "" if req.caption is None else str(req.caption).strip()
                caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                    [caption_text] * num_candidates,
                    max_length=caption_max_len,
                )
                if caption_text == "":
                    caption_mask.zero_()
                caption_ids = caption_ids.to(self.model_device)
                caption_mask = caption_mask.to(self.model_device)

            t0 = _measure_start(self.model_device, self.codec_device)
            msg_count_before_ref = len(messages)
            (
                speaker_state_override,
                speaker_mask_override,
            ) = self._load_speaker_embedding_condition(
                req=req,
                batch_size=num_candidates,
                messages=messages,
            )
            if speaker_state_override is None:
                ref_latent, ref_mask = self._load_reference_latent(
                    req=req,
                    batch_size=num_candidates,
                    messages=messages,
                )
            else:
                ref_latent, ref_mask = None, None
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("prepare_reference", stage_sec))
            for msg in messages[msg_count_before_ref:]:
                _log(msg)
            _log(f"[runtime] prepare_reference: {stage_sec * 1000.0:.1f} ms")

            encoded_conditions = None
            hop_length = int(self.codec.model.hop_length)
            if manual_seconds is not None:
                clamped_seconds = min(max_seconds, max(min_seconds, manual_seconds))
                if clamped_seconds != manual_seconds:
                    duration_msg = (
                        f"warning: manual duration {manual_seconds:.3f}s was clamped to "
                        f"{clamped_seconds:.3f}s."
                    )
                    messages.append(duration_msg)
                    _log(duration_msg)
                target_samples = max(1, int(clamped_seconds * self.codec.sample_rate))
                latent_steps = math.ceil(target_samples / hop_length)
                duration_msg = f"info: using manual duration {clamped_seconds:.3f}s."
                messages.append(duration_msg)
                _log(duration_msg)
            elif self.model_cfg.use_duration_predictor:
                t0 = _measure_start(self.model_device)
                has_speaker_duration = torch.zeros(
                    (num_candidates,), dtype=torch.bool, device=self.model_device
                )
                if speaker_mask_override is not None:
                    has_speaker_duration = speaker_mask_override.any(dim=1)
                elif self.model_cfg.use_speaker_condition_resolved and ref_mask is not None:
                    has_speaker_duration = ref_mask.any(dim=1)
                duration_features = build_duration_features(
                    [normalized_text] * num_candidates,
                    token_counts=text_mask.sum(dim=1),
                    max_text_len=text_max_len,
                    has_speaker=has_speaker_duration,
                ).to(self.model_device)
                has_caption_duration = (
                    torch.full(
                        (num_candidates,),
                        has_caption_text,
                        dtype=torch.bool,
                        device=self.model_device,
                    )
                    if self.model_cfg.use_caption_condition
                    else None
                )
                # Reference-audio requests have per-request conditioning shapes;
                # only fixed-shape (no-reference) requests take the graph path.
                encoded_conditions, pred_log_frames = self._encode_and_predict_duration(
                    text_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_ids=caption_ids,
                    caption_mask=caption_mask,
                    speaker_state_override=speaker_state_override,
                    speaker_mask_override=speaker_mask_override,
                    speaker_uncond_mode=req.speaker_uncond_mode,
                    duration_features=duration_features,
                    has_speaker=has_speaker_duration,
                    has_caption=has_caption_duration,
                    graph_eligible=bool(req.no_ref) or ref_latent is None,
                    log_fn=_log,
                )
                pred_frames = torch.expm1(pred_log_frames).float().mean().item()
                scaled_frames = pred_frames * duration_scale
                min_frames = max(1, math.ceil(min_seconds * self.codec.sample_rate / hop_length))
                max_frames = max(1, math.floor(max_seconds * self.codec.sample_rate / hop_length))
                latent_steps = int(round(scaled_frames))
                latent_steps = max(min_frames, min(max_frames, latent_steps))
                target_samples = int(latent_steps * hop_length)
                stage_sec = _measure_end(self.model_device, t0)
                stage_timings.append(("predict_duration", stage_sec))
                msg = (
                    f"info: predicted duration frames={pred_frames:.1f}, "
                    f"scale={duration_scale:.3f}, using_frames={latent_steps} "
                    f"({target_samples / float(self.codec.sample_rate):.3f}s)."
                )
                messages.append(msg)
                _log(msg)
                _log(f"[runtime] predict_duration: {stage_sec * 1000.0:.1f} ms")
            else:
                fallback_seconds = 30.0
                target_samples = int(fallback_seconds * self.codec.sample_rate)
                latent_steps = math.ceil(target_samples / hop_length)
                msg = "info: checkpoint has no duration predictor; falling back to 30.000s."
                messages.append(msg)
                _log(msg)
            patched_steps = math.ceil(latent_steps / self.model_cfg.latent_patch_size)

            if isinstance(self.train_cfg, dict):
                fixed_steps = self.train_cfg.get("fixed_target_latent_steps")
                if isinstance(fixed_steps, int) and fixed_steps > 0 and latent_steps > fixed_steps:
                    msg = (
                        f"warning: requested latent length ({latent_steps}) exceeds fixed_target_latent_steps ({fixed_steps}) "
                        "used in training. Long-tail stability may degrade."
                    )
                    messages.append(msg)
                    _log(msg)

            t0 = _measure_start(self.model_device)
            z_patched = sample_euler_rf_cfg(
                model=self.model,
                text_input_ids=text_ids,
                text_mask=text_mask,
                ref_latent=ref_latent,
                ref_mask=ref_mask,
                sequence_length=patched_steps,
                caption_input_ids=caption_ids,
                caption_mask=caption_mask,
                speaker_state_override=speaker_state_override,
                speaker_mask_override=speaker_mask_override,
                speaker_uncond_mode=req.speaker_uncond_mode,
                num_steps=int(req.num_steps),
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_guidance_mode=cfg_mode,
                cfg_min_t=float(req.cfg_min_t),
                cfg_max_t=float(req.cfg_max_t),
                seed=used_seed,
                truncation_factor=truncation_factor,
                rescale_k=rescale_k,
                rescale_sigma=rescale_sigma,
                use_context_kv_cache=bool(req.context_kv_cache),
                speaker_kv_scale=speaker_kv_scale,
                speaker_kv_max_layers=speaker_kv_max_layers,
                speaker_kv_min_t=speaker_kv_min_t,
                t_schedule_mode=str(req.t_schedule_mode),
                sway_coeff=float(req.sway_coeff),
                cuda_graph_cache=self._cuda_graph_cache,
                encoded_conditions=encoded_conditions,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("sample_rf", stage_sec))
            _log(f"[runtime] sample_rf: {stage_sec * 1000.0:.1f} ms")

            t0 = _measure_start(self.model_device)
            z = unpatchify_latent(
                z_patched,
                patch_size=self.model_cfg.latent_patch_size,
                latent_dim=self.model_cfg.latent_dim,
            )
            stage_sec = _measure_end(self.model_device, t0)
            stage_timings.append(("unpatchify_latent", stage_sec))
            _log(f"[runtime] unpatchify_latent: {stage_sec * 1000.0:.1f} ms")
            z = z[:, :latent_steps]

            t0 = _measure_start(self.model_device, self.codec_device)
            trimmed_audios: list[torch.Tensor] = []
            if decode_mode == "batch":
                audio_batch = self.codec.decode_latent(z).cpu()
                for i in range(num_candidates):
                    audio_i = audio_batch[i]
                    max_samples = target_samples
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            else:
                for i in range(num_candidates):
                    audio_i = self.codec.decode_latent(z[i : i + 1]).cpu()[0]
                    max_samples = target_samples
                    if bool(req.trim_tail):
                        flattening_point = find_flattening_point(
                            z[i],
                            window_size=max(1, int(req.tail_window_size)),
                            std_threshold=float(req.tail_std_threshold),
                            mean_threshold=float(req.tail_mean_threshold),
                        )
                        flattening_samples = int(
                            flattening_point * int(self.codec.model.hop_length)
                        )
                        if flattening_samples > 0:
                            max_samples = min(max_samples, flattening_samples)
                    trimmed_audios.append(audio_i[:, :max_samples])
            stage_sec = _measure_end(self.model_device, t0, self.codec_device)
            stage_timings.append(("decode_latent", stage_sec))
            _log(f"[runtime] decode_latent ({decode_mode}): {stage_sec * 1000.0:.1f} ms")

            if not bool(req.apply_watermark):
                msg = "info: SilentCipher watermark disabled by request."
                messages.append(msg)
                _log(msg)
            elif self.watermarker.ready:
                t0 = _measure_start(self.codec_device)
                trimmed_audios = self.watermarker.encode_batch(
                    trimmed_audios,
                    sample_rate=int(self.codec.sample_rate),
                )
                stage_sec = _measure_end(self.codec_device, t0)
                stage_timings.append(("silentcipher_watermark", stage_sec))
                _log(f"[runtime] silentcipher_watermark: {stage_sec * 1000.0:.1f} ms")
            else:
                msg = (
                    "warning: SilentCipher watermark is unavailable; generated audio was not "
                    "watermarked."
                )
                messages.append(msg)
                _log(msg)

            total_to_decode = _measure_end(self.model_device, post_load_t0, self.codec_device)
            _log(f"[runtime] total_to_decode: {total_to_decode:.3f} s")

        _log("[runtime] done synthesize")
        return SamplingResult(
            audio=trimmed_audios[0],
            audios=trimmed_audios,
            sample_rate=int(self.codec.sample_rate),
            stage_timings=stage_timings,
            total_to_decode=total_to_decode,
            used_seed=used_seed,
            messages=messages,
            used_seeds=used_seeds,
        )

    def prewarm_cuda_graphs(
        self,
        *,
        lora_adapter: str | None = None,
        lora_hot_swap: bool = False,
        max_seconds: float = 15.0,
        num_steps: int = 40,
        num_candidates: int = 1,
        cfg_guidance_mode: str = "independent",
        cfg_scale_text: float = 3.0,
        cfg_scale_caption: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        cfg_scale: float | None = None,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        context_kv_cache: bool = True,
        t_schedule_mode: str = "linear",
        sway_coeff: float = -1.0,
        log_fn: Callable[[str], None] | None = None,
    ) -> str:
        """
        Capture CUDA graphs for every latent-length bucket up to max_seconds so
        the first real generation of any duration takes the fast replay path.
        Graphs are captured for the no-reference (LoRA / base voice) request
        shape; requests with reference audio still capture on first use.
        """

        def _log(msg: str) -> None:
            if log_fn is not None:
                log_fn(msg)

        if self.model_device.type != "cuda":
            return "prewarm skipped: model device is not CUDA."
        if os.environ.get("IRODORI_DISABLE_CUDA_GRAPH", "0").strip() == "1":
            return "prewarm skipped: IRODORI_DISABLE_CUDA_GRAPH=1."
        if float(max_seconds) <= 0:
            return "prewarm skipped: max_seconds must be > 0."

        messages: list[str] = []
        stage_timings: list[tuple[str, float]] = []
        bucket = max(1, int(os.environ.get("IRODORI_CUDA_GRAPH_BUCKET", "16")))
        hop_length = int(self.codec.model.hop_length)
        max_frames = max(1, math.ceil(float(max_seconds) * self.codec.sample_rate / hop_length))
        max_patched = math.ceil(max_frames / self.model_cfg.latent_patch_size)
        max_padded = ((max_patched + bucket - 1) // bucket) * bucket
        lengths = list(range(bucket, max_padded + 1, bucket))
        # One sampler-graph entry is captured per (text bucket, latent bucket)
        # pair; the largest text length (max_text_len, i.e. no bucketing) is
        # always covered so long texts stay on the fast path.
        text_lengths = sorted(
            {
                int(self.default_text_max_len),
                *[
                    b
                    for b in _text_bucket_lengths()
                    if b < int(self.default_text_max_len)
                ],
            },
            reverse=True,
        )
        max_entries = max(1, int(os.environ.get("IRODORI_CUDA_GRAPH_CACHE", "64")))
        per_text_budget = max(1, max_entries // len(text_lengths))
        if len(lengths) > per_text_budget:
            # Keep the shortest buckets: short utterances are by far the most
            # common, so they must stay covered when the cache cannot hold all.
            lengths = lengths[:per_text_budget]
            messages.append(
                f"warning: prewarm truncated to {per_text_budget} latent buckets per text bucket "
                f"(covers up to {lengths[-1] * self.model_cfg.latent_patch_size * hop_length / self.codec.sample_rate:.1f}s); "
                "raise IRODORI_CUDA_GRAPH_CACHE or IRODORI_CUDA_GRAPH_BUCKET to cover more."
            )
        # Capture the largest bucket first so the shared memory pool is sized
        # once and smaller graphs reuse its blocks.
        lengths = sorted(lengths, reverse=True)

        t_start = time.perf_counter()
        with (
            self._infer_lock,
            self._prepare_lora_for_request(
                lora_adapter,
                messages=messages,
                stage_timings=stage_timings,
                log_fn=_log,
                hot_swap=bool(lora_hot_swap),
            ),
            torch.inference_mode(),
        ):
            text_ids_full, text_mask_full = self.tokenizer.batch_encode(
                ["こんにちは。"] * int(num_candidates),
                max_length=self.default_text_max_len,
            )
            caption_ids = None
            caption_mask = None
            if self.model_cfg.use_caption_condition and self.caption_tokenizer is not None:
                caption_ids, caption_mask = self.caption_tokenizer.batch_encode(
                    [""] * int(num_candidates),
                    max_length=self.default_caption_max_len,
                )
                caption_mask.zero_()
                caption_ids = caption_ids.to(self.model_device)
                caption_mask = caption_mask.to(self.model_device)
            ref_latent, ref_mask = self._load_reference_latent(
                req=SamplingRequest(text="prewarm", no_ref=True),
                batch_size=int(num_candidates),
                messages=messages,
            )
            scale_text, scale_caption, scale_speaker, _ = resolve_cfg_scales(
                cfg_guidance_mode=cfg_guidance_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_caption=cfg_scale_caption,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_scale=cfg_scale,
                use_caption_condition=False,
                use_speaker_condition=False,
            )
            prewarm_has_speaker = torch.zeros(
                (int(num_candidates),), dtype=torch.bool, device=self.model_device
            )
            if ref_mask is not None:
                prewarm_has_speaker = ref_mask.any(dim=1)
            prewarm_features = build_duration_features(
                ["こんにちは。"] * int(num_candidates),
                token_counts=text_mask_full.sum(dim=1),
                max_text_len=self.default_text_max_len,
                has_speaker=prewarm_has_speaker,
            ).to(self.model_device)
            prewarm_has_caption = (
                torch.zeros((int(num_candidates),), dtype=torch.bool, device=self.model_device)
                if self.model_cfg.use_caption_condition
                else None
            )
            for text_len in text_lengths:
                # Tokenization is right-padded, so slicing the full-length
                # batch equals re-encoding at the bucket length.
                text_ids = text_ids_full[:, :text_len].contiguous().to(self.model_device)
                text_mask = text_mask_full[:, :text_len].contiguous().to(self.model_device)
                if self.model_cfg.use_duration_predictor:
                    # Capture the duration/condition graph too, so the first
                    # real request skips its one-time capture cost. Shapes
                    # (not values) key the graph, so the placeholder text is
                    # irrelevant.
                    encoded_conditions, _ = self._encode_and_predict_duration(
                        text_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_ids=caption_ids,
                        caption_mask=caption_mask,
                        speaker_state_override=None,
                        speaker_mask_override=None,
                        speaker_uncond_mode="mask",
                        duration_features=prewarm_features,
                        has_speaker=prewarm_has_speaker,
                        has_caption=prewarm_has_caption,
                        graph_eligible=True,
                        log_fn=_log,
                    )
                else:
                    encoded_conditions = self.model.encode_conditions(
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                    )
                for seq_len in lengths:
                    sample_euler_rf_cfg(
                        model=self.model,
                        text_input_ids=text_ids,
                        text_mask=text_mask,
                        ref_latent=ref_latent,
                        ref_mask=ref_mask,
                        sequence_length=int(seq_len),
                        caption_input_ids=caption_ids,
                        caption_mask=caption_mask,
                        num_steps=int(num_steps),
                        cfg_scale_text=scale_text,
                        cfg_scale_caption=scale_caption,
                        cfg_scale_speaker=scale_speaker,
                        cfg_guidance_mode=str(cfg_guidance_mode),
                        cfg_min_t=float(cfg_min_t),
                        cfg_max_t=float(cfg_max_t),
                        seed=0,
                        use_context_kv_cache=bool(context_kv_cache),
                        t_schedule_mode=str(t_schedule_mode),
                        sway_coeff=float(sway_coeff),
                        cuda_graph_cache=self._cuda_graph_cache,
                        encoded_conditions=encoded_conditions,
                    )
                    _log(
                        f"[runtime] prewarm: captured graph for {seq_len} patched steps "
                        f"(text {text_len})"
                    )
                    if self._cuda_graph_cache.get("__disabled__", False):
                        return (
                            "prewarm aborted: CUDA graph capture failed; running in eager mode."
                        )

        # Release cached (non-graph) allocator blocks accumulated by the warmup
        # runs; graph private pools are owned by the graphs and stay intact.
        if self.model_device.type == "cuda":
            torch.cuda.empty_cache()

        # Warm the decode/watermark stages last (after empty_cache, so their
        # working-set allocations survive into the first real request). The
        # first decode/watermark calls otherwise pay one-time cuDNN plan and
        # allocator-growth costs of ~40 ms each. Sized to the prewarmed
        # duration range so shorter real requests fit within the warmed blocks.
        with self._infer_lock, torch.inference_mode():
            if self.codec.device.type == "cuda":
                dummy_latent = torch.zeros(
                    (1, int(max_frames), self.codec.latent_dim),
                    device=self.codec.device,
                    dtype=torch.float32,
                )
                self.codec.decode_latent(dummy_latent)
                _log("[runtime] prewarm: warmed codec decoder")

            if self.watermarker.ready:
                # Deterministic dummy signal so global RNG state stays
                # untouched.
                try:
                    sample_rate = int(self.codec.sample_rate)
                    t = torch.arange(int(max_frames) * hop_length, dtype=torch.float32)
                    dummy_audio = (0.1 * torch.sin(2.0 * math.pi * 440.0 * t / sample_rate))[
                        None, :
                    ]
                    self.watermarker.encode_batch([dummy_audio], sample_rate=sample_rate)
                    _log("[runtime] prewarm: warmed watermarker")
                except Exception as exc:  # pragma: no cover - defensive
                    _log(f"[runtime] prewarm: watermark warmup skipped ({exc!r})")

        elapsed = time.perf_counter() - t_start
        graph_count = len([k for k in self._cuda_graph_cache if not isinstance(k, str)])
        lora_note = self._last_lora_token if self._last_lora_token is not None else "__base__"
        summary = (
            f"prewarmed {len(lengths)} length buckets x {len(text_lengths)} text buckets "
            f"(<= {float(max_seconds):.1f}s) in {elapsed:.1f}s; graphs cached: {graph_count}\n"
            f"lora: {lora_note}\n"
            "note: graphs are dropped if the LoRA Adapter Directory changes, "
            "unless LoRA Hot-Swap is enabled."
        )
        for msg in messages:
            _log(msg)
        _log(f"[runtime] {summary}")
        return summary

    def unload(self) -> None:
        self._cuda_graph_cache.clear()
        del self.model
        del self.tokenizer
        del self.codec
        gc.collect()
        for device in (self.model_device, self.codec_device):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                mps = getattr(torch, "mps", None)
                if mps is not None and hasattr(mps, "empty_cache"):
                    mps.empty_cache()
            elif device.type == "xpu":
                xpu = getattr(torch, "xpu", None)
                if xpu is not None and hasattr(xpu, "empty_cache"):
                    xpu.empty_cache()


_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE_KEY: RuntimeKey | None = None
_RUNTIME_CACHE_VALUE: InferenceRuntime | None = None


def get_cached_runtime(key: RuntimeKey) -> tuple[InferenceRuntime, bool]:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        if _RUNTIME_CACHE_VALUE is not None and _RUNTIME_CACHE_KEY == key:
            return _RUNTIME_CACHE_VALUE, False

        old_runtime = _RUNTIME_CACHE_VALUE
        runtime = InferenceRuntime.from_key(key)
        _RUNTIME_CACHE_KEY = key
        _RUNTIME_CACHE_VALUE = runtime

    if old_runtime is not None:
        old_runtime.unload()

    return runtime, True


def clear_cached_runtime() -> None:
    global _RUNTIME_CACHE_KEY, _RUNTIME_CACHE_VALUE
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE_VALUE
        _RUNTIME_CACHE_KEY = None
        _RUNTIME_CACHE_VALUE = None

    if runtime is not None:
        runtime.unload()


def _coerce_reference_audio(
    audio: torch.Tensor,
    sample_rate: int | None,
) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, torch.Tensor):
        raise ValueError("ref_audio must be a torch.Tensor waveform.")
    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError("ref_audio requires a positive ref_audio_sample_rate.")
    wav = audio.detach().to(device="cpu", dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.ndim != 2:
        raise ValueError(
            "ref_audio must have shape (samples,) or (channels, samples), "
            f"got {tuple(audio.shape)}"
        )
    if wav.shape[1] == 0:
        raise ValueError("ref_audio must contain at least one sample.")
    return wav.contiguous(), int(sample_rate)


def _load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except RuntimeError:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.T
        return wav, sr


def save_wav(path: str | Path, audio: torch.Tensor, sample_rate: int) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
    try:
        torchaudio.save(str(out_path), audio_cpu, sample_rate)
    except RuntimeError:
        import soundfile as sf

        audio_np = audio_cpu.squeeze(0).numpy() if audio_cpu.shape[0] == 1 else audio_cpu.T.numpy()
        sf.write(str(out_path), audio_np, sample_rate)
    return out_path
