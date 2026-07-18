"""Decoupled torch.compile speed probe: compile variants with our CUDA graphs
disabled, so compile interplay problems don't mask the fusion signal.

Variants (base model, bf16, fixed text/seed):
  1. eager + our graphs   (current production path, control)
  2. eager, no graphs     (launch-overhead reference on this platform)
  3. compile default mode, no graphs
  4. compile reduce-overhead (inductor-managed cudagraphs), no graphs
  5. compile default mode + our graphs (known interplay issue; kept to observe)

Usage (WSL):
    /home/stak/venvs/irodori-tts/bin/python tools/probe_compile2.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

TEXT = "いつまで待たせるつもり？ 早くしてよね。"
WARM_RUNS = 4
PRECISION = "bf16"

VARIANTS = [
    ("eager + our graphs", False, None, True),
    ("eager, no graphs", False, None, False),
    ("compile default, no graphs", True, None, False),
    ("compile reduce-overhead, no graphs", True, "reduce-overhead", False),
    ("compile default + our graphs", True, None, True),
]


def main() -> None:
    from gradio_app import _resolve_checkpoint_path

    ckpt = _resolve_checkpoint_path("Aratako/Irodori-TTS-500M-v3")

    for label, use_compile, mode, use_graphs in VARIANTS:
        os.environ["IRODORI_DISABLE_CUDA_GRAPH"] = "0" if use_graphs else "1"
        try:
            torch._dynamo.reset()
            runtime = InferenceRuntime.from_key(
                RuntimeKey(
                    checkpoint=ckpt,
                    model_device="cuda",
                    model_precision=PRECISION,
                    codec_device="cuda",
                )
            )
            if use_compile:
                kwargs = {"dynamic": False}
                if mode is not None:
                    kwargs["mode"] = mode
                runtime.model.encode_conditions = torch.compile(
                    runtime.model.encode_conditions, **kwargs
                )
                runtime.model.build_context_kv_cache = torch.compile(
                    runtime.model.build_context_kv_cache, **kwargs
                )
                runtime.model.forward_with_encoded_conditions = torch.compile(
                    runtime.model.forward_with_encoded_conditions, **kwargs
                )

            stats: dict[str, list[float]] = {}
            t0 = time.perf_counter()
            for run in range(WARM_RUNS + 1):
                result = runtime.synthesize(
                    SamplingRequest(text=TEXT, no_ref=True, seed=1, num_steps=40)
                )
                if run == 0:
                    print(
                        f"[c2] {label}: first request {time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
                    continue
                for name, sec in result.stage_timings:
                    stats.setdefault(name, []).append(sec * 1000.0)
            med = {k: sorted(v)[len(v) // 2] for k, v in stats.items()}
            print(
                f"[c2] {label}: sample_rf {med.get('sample_rf', 0):6.1f} ms  "
                f"predict_duration {med.get('predict_duration', 0):5.1f} ms",
                flush=True,
            )
            del runtime
            torch.cuda.empty_cache()
        except Exception as exc:
            msg = repr(exc).replace("\n", " ")[:300]
            print(f"[c2] {label}: FAILED {msg}", flush=True)


if __name__ == "__main__":
    main()
