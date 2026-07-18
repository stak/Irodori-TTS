"""Measure VRAM cost per sampler-graph entry and shared-condition-buffer size.

Used to estimate the VRAM increase of text-length bucketing (graph entries
multiply by the number of text buckets; __cond__ buffers get one signature
per text bucket).

Run from the repo root:
    .venv/Scripts/python.exe tools/probe_vram.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

LORA = str(Path("outputs") / "lora_shiduka" / "checkpoint_final")
TEXT = "いつまで待たせるつもり？ 早くしてよね。"


def _device_used_mib() -> float:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024**2)


def _reserved_mib() -> float:
    return torch.cuda.memory_reserved() / (1024**2)


def _tensor_mib(tensors) -> float:
    return sum(t.numel() * t.element_size() for t in tensors) / (1024**2)


def main() -> None:
    from gradio_app import _resolve_checkpoint_path

    key = RuntimeKey(
        checkpoint=_resolve_checkpoint_path("Aratako/Irodori-TTS-500M-v3"),
        model_device="cuda",
        codec_device="cuda",
    )
    runtime = InferenceRuntime.from_key(key)
    cfg = runtime.model_cfg
    print(
        f"[vram] model cfg: hidden={getattr(cfg, 'hidden_dim', '?')} "
        f"blocks={getattr(cfg, 'num_blocks', getattr(cfg, 'n_blocks', '?'))} "
        f"text_dim={getattr(cfg, 'text_dim', '?')} "
        f"latent_patch={cfg.latent_patch_size} text_max_len={runtime.default_text_max_len}"
    )

    base = _device_used_mib()
    print(f"[vram] before prewarm: used={base:.0f} MiB reserved={_reserved_mib():.0f} MiB")
    runtime.prewarm_cuda_graphs(lora_adapter=LORA, max_seconds=15.0)
    after_prewarm = _device_used_mib()
    entries = len([k for k in runtime._cuda_graph_cache if not isinstance(k, str)])
    print(
        f"[vram] after prewarm ({entries} entries): used={after_prewarm:.0f} MiB "
        f"(delta {after_prewarm - base:+.0f}) reserved={_reserved_mib():.0f} MiB"
    )

    # Shared condition buffers (one signature today: text always padded to 256).
    cond = runtime._cuda_graph_cache.get("__cond__", {})
    for sig, flat in cond.items():
        print(f"[vram] __cond__ signature: {len(flat)} tensors, {_tensor_mib(flat):.1f} MiB")
    dur = runtime._cuda_graph_cache.get("__duration__", {})
    for sig, entry in dur.items():
        tensors = [*entry["inputs"], *[t for t in entry["encoded"] if t is not None], entry["pred"]]
        print(f"[vram] __duration__ entry: {_tensor_mib(tensors):.1f} MiB")

    # Marginal cost per additional sampler-graph entry: capture the same
    # shapes again under different CFG scales (same node count, pool reused).
    req_base = dict(
        text=TEXT,
        no_ref=True,
        seed=1,
        num_steps=40,
        lora_adapter=LORA,
    )
    runtime.synthesize(SamplingRequest(**req_base))  # steady-state allocations
    marginals = []
    prev = _device_used_mib()
    for i, scale in enumerate([3.01, 3.02, 3.03, 3.04, 3.05, 3.06], start=1):
        runtime.synthesize(SamplingRequest(**req_base, cfg_scale_text=scale))
        now = _device_used_mib()
        marginals.append(now - prev)
        print(f"[vram] extra capture #{i} (cfg={scale}): +{now - prev:.1f} MiB")
        prev = now
    entries2 = len([k for k in runtime._cuda_graph_cache if not isinstance(k, str)])
    print(
        f"[vram] entries now: {entries2}; per-entry marginal: "
        f"median {statistics.median(marginals):.1f} MiB, mean {statistics.mean(marginals):.1f} MiB"
    )


if __name__ == "__main__":
    main()
