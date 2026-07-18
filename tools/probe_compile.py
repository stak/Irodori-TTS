"""Measure sample_rf with and without torch.compile (compile_model flag).

Runs the base model (no LoRA: dynamic LoRA is incompatible with
compile_model=True; merged-LoRA compute is identical, so the speed signal
carries over). Works on Windows and Linux/WSL.

Usage (WSL):
    UV_PROJECT_ENVIRONMENT=~/venvs/irodori-tts uv run --no-sync \
        python tools/probe_compile.py [--precision bf16] [--skip-eager]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

TEXT = "いつまで待たせるつもり？ 早くしてよね。"
WARM_RUNS = 3


def _measure(runtime: InferenceRuntime, label: str) -> None:
    stats: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    for run in range(WARM_RUNS + 1):
        result = runtime.synthesize(
            SamplingRequest(text=TEXT, no_ref=True, seed=1, num_steps=40)
        )
        if run == 0:
            print(f"[compile] {label}: first request took {time.perf_counter() - t0:.1f}s")
            continue
        for name, sec in result.stage_timings:
            stats.setdefault(name, []).append(sec * 1000.0)
    med = {k: sorted(v)[len(v) // 2] for k, v in stats.items()}
    print(
        f"[compile] {label}: sample_rf {med.get('sample_rf', 0):6.1f} ms  "
        f"predict_duration {med.get('predict_duration', 0):5.1f} ms  "
        f"decode {med.get('decode_latent', 0):5.1f} ms  "
        f"(platform={sys.platform})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--skip-eager", action="store_true")
    parser.add_argument("--dynamic", action="store_true", help="compile with dynamic=True")
    args = parser.parse_args()

    from gradio_app import _resolve_checkpoint_path

    ckpt = _resolve_checkpoint_path("Aratako/Irodori-TTS-500M-v3")

    variants = [] if args.skip_eager else [("no-compile", False)]
    variants.append(("compile", True))
    for label, compile_model in variants:
        try:
            runtime = InferenceRuntime.from_key(
                RuntimeKey(
                    checkpoint=ckpt,
                    model_device="cuda",
                    model_precision=args.precision,
                    codec_device="cuda",
                    compile_model=compile_model,
                    compile_dynamic=bool(args.dynamic),
                )
            )
            _measure(runtime, f"{label}/{args.precision}")
            del runtime
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[compile] {label}/{args.precision}: FAILED {exc!r}")


if __name__ == "__main__":
    main()
