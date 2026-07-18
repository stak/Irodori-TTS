"""Iteration probe for torch.compile x our manual CUDA graphs.

Runs ONE variant: compile (default mode, dynamic=None) + our graphs,
base model, bf16, with recompile diagnostics. Run with
TORCH_LOGS=recompiles to see guard-failure reasons on stderr.

Usage (WSL):
    TORCH_LOGS=recompiles /home/stak/venvs/irodori-tts/bin/python tools/probe_compile3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

TEXT = "いつまで待たせるつもり？ 早くしてよね。"
WARM_RUNS = 4


def main() -> None:
    from gradio_app import _resolve_checkpoint_path

    ckpt = _resolve_checkpoint_path("Aratako/Irodori-TTS-500M-v3")
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ckpt,
            model_device="cuda",
            model_precision="bf16",
            codec_device="cuda",
        )
    )
    runtime.model.encode_conditions = torch.compile(runtime.model.encode_conditions)
    runtime.model.build_context_kv_cache = torch.compile(runtime.model.build_context_kv_cache)
    runtime.model.forward_with_encoded_conditions = torch.compile(
        runtime.model.forward_with_encoded_conditions
    )

    stats: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for run in range(WARM_RUNS + 1):
        result = runtime.synthesize(
            SamplingRequest(text=TEXT, no_ref=True, seed=1, num_steps=40),
            log_fn=lambda m: print(m, flush=True) if run == 0 else None,
        )
        if run == 0:
            print(f"[c3] first request {time.perf_counter() - t0:.1f}s", flush=True)
            continue
        for name, sec in result.stage_timings:
            stats.setdefault(name, []).append(sec * 1000.0)
    med = {k: sorted(v)[len(v) // 2] for k, v in stats.items()}
    print(
        f"[c3] compile+graphs: sample_rf {med.get('sample_rf', 0):6.1f} ms  "
        f"predict_duration {med.get('predict_duration', 0):5.1f} ms  "
        f"decode {med.get('decode_latent', 0):5.1f} ms",
        flush=True,
    )
    from torch._dynamo.utils import counters

    print(f"[c3] dynamo counters: {dict(counters['stats'])}", flush=True)


if __name__ == "__main__":
    main()
