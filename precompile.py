#!/usr/bin/env python3
"""
Populate the torch.compile on-disk cache for a production shape grid.

With IRODORI_COMPILE=1, the DiT forward is compiled per input-shape signature
(candidate count x text bucket x latent bucket x CFG configuration). Inductor
caches compiled kernels on disk, but a shape that was never compiled on this
machine stalls its first real request for seconds to minutes. Run this script
once per machine (and after torch/driver upgrades, precision changes, or a
different adapter structure) so production requests never hit an on-demand
compile.

Run it with the same environment you serve with: the same IRODORI_* variables,
the same precision flags, and IRODORI_COMPILE=1 (requires a Triton toolchain,
i.e. Linux/WSL2). Without IRODORI_COMPILE the script still exercises the shape
grid, but produces no compile cache.

If production uses LoRA adapters, pass --lora-adapter so compilation sees the
PEFT-wrapped module structure; any adapter with the production adapter config
works (a zero-weight adapter keeps outputs identical to the base model).
"""

from __future__ import annotations

import argparse
import math
import os
import time

from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
)


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.checkpoint is not None:
        return str(args.checkpoint)
    repo_id = str(args.hf_checkpoint).strip()
    if repo_id == "":
        raise ValueError("hf_checkpoint must be non-empty.")
    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    print(f"[checkpoint] hf://{repo_id} -> {path}", flush=True)
    return str(path)


def _parse_candidate_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise argparse.ArgumentTypeError("candidate counts must be >= 1")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("expected at least one candidate count")
    return sorted(set(values))


# Short text stays inside the smallest text bucket; the long text exceeds it
# and is padded to the checkpoint's max_text_len, covering the second text
# shape. Content is irrelevant: only tokenized shapes key compiles/graphs.
SHORT_TEXT = "こんにちは、今日はいい天気ですね。"
LONG_TEXT = "".join(
    [
        "これは事前コンパイル用の長いテキストです。",
        "内容そのものに意味はなく、トークン数がテキストバケットの境界を超えることだけが目的です。",
        "同じ文をもう一度繰り返して、確実に最大テキスト長の形状に到達させます。",
        "これは事前コンパイル用の長いテキストです。",
        "内容そのものに意味はなく、トークン数がテキストバケットの境界を超えることだけが目的です。",
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompile the torch.compile cache for a production shape grid.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        default=None,
        help="Local model checkpoint path (.pt or .safetensors).",
    )
    checkpoint_group.add_argument(
        "--hf-checkpoint",
        default=None,
        help="Hugging Face model repo id to download model.safetensors from.",
    )
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help=(
            "Optional PEFT LoRA adapter directory. Load one whenever production "
            "serves LoRA requests, so compilation sees the PEFT-wrapped module "
            "structure."
        ),
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Upper bound of the audio duration range to cover (default: 15).",
    )
    parser.add_argument(
        "--num-candidates",
        type=_parse_candidate_list,
        default=[1],
        help=(
            "Comma-separated candidate counts to cover (default: 1). Add every "
            "n value production will request, e.g. '1,4'."
        ),
    )
    parser.add_argument(
        "--cfg-scale-text",
        type=float,
        default=3.0,
        help=(
            "Text CFG scale. Only whether it is zero matters here: it toggles "
            "the CFG term and therefore the compiled batch shape."
        ),
    )
    parser.add_argument(
        "--cfg-scale-speaker",
        type=float,
        default=5.0,
        help="Speaker CFG scale; only zero/non-zero affects the shape grid.",
    )
    parser.add_argument(
        "--model-device",
        default=default_runtime_device(),
        help="Model inference device (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--model-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Model precision for weights/compute.",
    )
    parser.add_argument(
        "--codec-device",
        default=default_runtime_device(),
        help="Codec device (e.g. cuda, cpu).",
    )
    parser.add_argument(
        "--codec-precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Codec precision for weights/compute.",
    )
    args = parser.parse_args()

    import torch

    compile_enabled = os.environ.get("IRODORI_COMPILE", "0").strip() == "1"
    if not compile_enabled:
        print(
            "[precompile] warning: IRODORI_COMPILE is not set to 1 - the shape "
            "grid will be exercised but no compile cache is produced.",
            flush=True,
        )

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=_resolve_checkpoint_path(args),
            model_device=str(args.model_device),
            model_precision=str(args.model_precision),
            codec_device=str(args.codec_device),
            codec_precision=str(args.codec_precision),
        )
    )

    hop_length = int(runtime.codec.model.hop_length)
    sample_rate = int(runtime.codec.sample_rate)
    patch_size = int(runtime.model_cfg.latent_patch_size)
    bucket = max(1, int(os.environ.get("IRODORI_CUDA_GRAPH_BUCKET", "16")))
    max_frames = max(1, math.ceil(float(args.max_seconds) * sample_rate / hop_length))
    max_patched = math.ceil(max_frames / patch_size)
    max_padded = ((max_patched + bucket - 1) // bucket) * bucket
    padded_lengths = list(range(bucket, max_padded + 1, bucket))

    def synthesize(text: str, *, num_candidates: int, seconds: float | None) -> None:
        runtime.synthesize(
            SamplingRequest(
                text=text,
                no_ref=True,
                num_candidates=num_candidates,
                # num_steps does not affect compiled or captured shapes, so a
                # single Euler step per shape keeps the sweep fast.
                num_steps=1,
                seed=0,
                seconds=seconds,
                # Clamp the duration-predicted request into the sweep range so
                # it does not compile/capture a shape beyond max_seconds.
                max_seconds=float(args.max_seconds),
                cfg_scale_text=float(args.cfg_scale_text),
                cfg_scale_speaker=float(args.cfg_scale_speaker),
                lora_adapter=args.lora_adapter,
            ),
            log_fn=None,
        )

    t_start = time.perf_counter()
    covered = 0
    for num_candidates in args.num_candidates:
        for label, text in (("short", SHORT_TEXT), ("long", LONG_TEXT)):
            # One duration-predicted request per (n, text bucket) exercises the
            # exact production path for condition encoding + duration
            # prediction (and its CUDA graph capture).
            t0 = time.perf_counter()
            synthesize(text, num_candidates=num_candidates, seconds=None)
            covered += 1
            print(
                f"[precompile] n={num_candidates} text={label} duration=auto: "
                f"{time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            for padded in padded_lengths:
                # A seconds value just below the padded frame count lands the
                # request exactly in this latent bucket.
                seconds = (padded * patch_size - 0.5) * hop_length / sample_rate
                t0 = time.perf_counter()
                synthesize(text, num_candidates=num_candidates, seconds=seconds)
                covered += 1
                print(
                    f"[precompile] n={num_candidates} text={label} "
                    f"latent={padded}: {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_start
    device_name = (
        torch.cuda.get_device_name(0)
        if str(args.model_device).startswith("cuda") and torch.cuda.is_available()
        else str(args.model_device)
    )
    print(
        "\n".join(
            [
                f"[precompile] done: {covered} shapes in {elapsed:.1f}s",
                f"[precompile] grid: n={args.num_candidates} x text buckets (short/long) "
                f"x {len(padded_lengths)} latent buckets (<= {float(args.max_seconds):.1f}s, "
                f"bucket={bucket}) + duration-auto per text bucket",
                f"[precompile] environment: torch={torch.__version__} "
                f"cuda={torch.version.cuda} device={device_name} "
                f"model_precision={args.model_precision} "
                f"codec_precision={args.codec_precision} "
                f"IRODORI_COMPILE={'1' if compile_enabled else '0'} "
                f"lora_adapter={args.lora_adapter or 'none'}",
                "[precompile] the compile cache is keyed by this environment; rerun "
                "after torch/driver upgrades or precision/adapter-structure changes.",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
