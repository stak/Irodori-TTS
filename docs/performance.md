# Inference Performance Fork Notes

This fork adds a set of inference-speed optimizations on top of upstream
Irodori-TTS. On the reference setup (RTX 40-series, Windows, fp32, 40 steps,
~4 s utterance, LoRA adapter, watermark enabled), steady-state
`total_to_decode` in the Gradio app drops from **~1.0 s to ~0.45-0.5 s**, and
further if the GPU is prevented from downclocking between requests (see
[Latency variance](#latency-variance-gpu-clocks)).

All optimizations are inference-only. Training code paths are untouched.

## Summary of Changes

| Change | Devices | Output impact | Opt-out |
|--------|---------|---------------|---------|
| Cache timestep-embedding frequency table | all | exact (bit-identical) | - |
| Vectorized `find_flattening_point` (tail trim) | all | exact | - |
| Reuse encoded conditions from duration prediction | all | exact | - |
| Remove per-step GPU syncs in the Euler loop | all | exact | - |
| TF32 tensor-core matmul for fp32 | CUDA (Ampere+) | ~41 dB SNR vs upstream (inaudible) | `IRODORI_DISABLE_TF32=1` |
| LoRA adapter merged into base weights per request | all | fp rounding only | `IRODORI_DISABLE_LORA_MERGE=1` |
| CUDA per-step graph replay of the sampling loop | CUDA only | exact vs eager; see bucketing note | `IRODORI_DISABLE_CUDA_GRAPH=1` |
| CUDA graph replay of condition encoding + duration prediction | CUDA only | exact (bit-identical vs eager) | `IRODORI_DISABLE_DURATION_GRAPH=1` |
| Latent-length bucketing (graphs shared across lengths) | CUDA only | ~47 dB SNR when padding occurs (inaudible, deterministic) | `IRODORI_CUDA_GRAPH_BUCKET=1` |
| Graph prewarm (runtime API + Gradio button) | CUDA only | - | (button; opt-in) |
| LoRA hot-swap keeping cached graphs | CUDA only | tiny fp drift per swap | (checkbox; default off) |
| Gradio: SilentCipher watermark toggle | - | none (watermark on by default) | - |
| Gradio: offline-first HF checkpoint resolution | - | none | see limitations |

## Prerequisites and Assumptions

The headline speedup comes from **CUDA Graphs**, which assumes:

- **An NVIDIA GPU with CUDA** (`Model Device = cuda`). On CPU/MPS/XPU the
  graph path deactivates itself and behavior matches upstream (the exact
  optimizations above still apply).
- **TF32 requires Ampere or newer** (RTX 30/40/50, A100, ...). On older GPUs
  the flag is ignored by PyTorch.
- The graph fast path engages only for `cfg_guidance_mode=independent`
  (the default) without `rescale_k/rescale_sigma` and without
  `speaker_kv_scale`. Other configurations silently fall back to eager
  execution, i.e. upstream behavior plus the exact optimizations.
- Graph capture is validated against eager execution at capture time; any
  capture failure disables graphs for the session and falls back to eager.

The optimization targets the **launch-overhead-bound** regime: this model's
per-step compute is small, so eager sampling time is dominated by CPU-side
kernel launches (especially on Windows, where Triton/`torch.compile` is not
available for CUDA). CUDA graph replay removes that overhead.

## Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `IRODORI_DISABLE_TF32` | `0` | `1` disables TF32 matmul (exact fp32, slower) |
| `IRODORI_DISABLE_LORA_MERGE` | `0` | `1` keeps LoRA adapters unmerged (upstream behavior) |
| `IRODORI_DISABLE_CUDA_GRAPH` | `0` | `1` disables graph capture/replay entirely |
| `IRODORI_DISABLE_DURATION_GRAPH` | `0` | `1` keeps condition encoding + duration prediction eager (sampler graphs unaffected) |
| `IRODORI_CUDA_GRAPH_BUCKET` | `16` | Latent-length bucket size in patched steps; `1` disables padding |
| `IRODORI_CUDA_GRAPH_CACHE` | `64` | Max cached graph entries (per-entry VRAM is small; pool and condition buffers are shared) |

Setting `IRODORI_DISABLE_TF32=1`, `IRODORI_DISABLE_LORA_MERGE=1` and
`IRODORI_DISABLE_CUDA_GRAPH=1` together reproduces upstream outputs
**bit-identically** (the remaining optimizations are exact).

## Recommended Gradio Workflow

1. Load Model.
2. Fill in the LoRA Adapter Directory (if any) **before** prewarming.
3. Press **Prewarm Graphs** (Prewarm Max Seconds bounds the covered duration
   range; 15 s ≈ 24 buckets ≈ ~25-40 s one-time cost, ~0.5-1 GiB VRAM).
4. Generate. Any text whose predicted duration falls inside the prewarmed
   range takes the fast path from the first request.

Notes:

- Graphs are keyed by tensor shapes and CFG scales. `Num Steps`,
  `Time Schedule`, `Sway Coeff`, `CFG Min/Max t` and `Seed` can be changed
  freely without re-capture. Changing `CFG Scale Text/Speaker` values,
  `Num Candidates`, precision/device, or supplying reference audio triggers a
  fresh capture per length bucket (one-time, ~1 s each).
- The `predict_duration` stage (condition encoding + duration head) is also
  replayed as one CUDA graph (~14 ms -> ~3-4 ms per request on the reference
  setup). It is captured during Prewarm (or lazily on the first eligible
  request, ~0.1 s one-time) and validated bit-exact against eager at capture.
  Only fixed-shape requests take this path: reference-audio and
  speaker-embedding requests, and non-default `max_text_len`, run the
  unchanged eager code. Unlike the sampler graphs, this graph is dropped on
  *every* LoRA switch — hot-swap explicitly permits
  `modules_to_save=duration_predictor`, which replaces module storage the
  duration graph captured — and recaptures on the next request.
- Changing the LoRA Adapter Directory drops all cached graphs unless
  **LoRA Hot-Swap** is enabled. Hot-swap swaps adapter weights in place
  (unmerge old, merge new) so graphs survive; a tiny floating-point drift can
  accumulate per swap. It is refused automatically for adapters whose
  `modules_to_save` replaces modules used inside the graph (anything other
  than `duration_predictor`) and for DoRA adapters.

## Latency Variance (GPU Clocks)

After these optimizations the sampler is short enough that consumer-GPU
power management dominates run-to-run variance: after a few idle seconds the
GPU drops to a low-power state and the next request runs partly at low
clocks (e.g. ~150 ms vs ~50 ms `sample_rf` for back-to-back requests). For
consistent latency, lock clocks (`nvidia-smi -lgc <min>,<max>`, requires
admin) or set the NVIDIA Control Panel power mode to "Prefer maximum
performance".

## Output Differences vs Upstream

Same seed remains deterministic in all configurations. Numerical differences
from upstream, all far below audibility:

- **TF32 + LoRA merge**: measured ~41 dB SNR against upstream output for the
  same seed.
- **Length bucketing**: when a request's latent length is padded to the
  bucket boundary, attention runs over a longer (masked) sequence; the
  masked math is equivalent but kernel reduction order changes, giving
  ~47 dB SNR vs an unpadded run. Results are still deterministic per length.
  With `IRODORI_CUDA_GRAPH_BUCKET=1` (no padding), graph replay is
  bit-identical to eager execution.
- **LoRA hot-swap** (opt-in): each in-place unmerge/merge accumulates
  rounding on the order of the fp32 epsilon.

## Known Limitations / Behavior Changes

- **One-shot CLI (`infer.py`) pays capture cost**: the graph path captures on
  the first request, which is pure overhead for a single-generation process
  (~+0.5-1 s). Set `IRODORI_DISABLE_CUDA_GRAPH=1` for one-shot CLI runs.
- **VRAM**: cached graphs add roughly 0.5-1 GiB after a 15 s prewarm
  (shared memory pool + shared condition buffers + driver-side graph
  executables). Upstream uses correspondingly less.
- **Offline-first checkpoint resolution**: the Gradio app now resolves
  HF-hub checkpoints from the local cache first. If the upstream repo
  publishes a newer `model.safetensors`, it is **not** picked up
  automatically while a cached copy exists; clear the HF cache entry (or
  download manually) to update.
- **Gradio API surface**: `_run_generation` and the prewarm handler take
  additional inputs (hot-swap flag, watermark flag, prewarm seconds), so
  `gradio_client` calls written against upstream need their argument lists
  updated.
- **Reference-audio requests and per-request `max_text_len` overrides**
  produce different conditioning shapes per request, so they re-capture per
  shape instead of hitting prewarmed graphs (prewarm covers the
  no-reference/LoRA path with the checkpoint's default text length).
- Upstream functionality is otherwise preserved. Configurations not eligible
  for the graph path (`joint`/`alternating` CFG, temporal rescale,
  speaker-KV scaling, non-CUDA devices) run through the unchanged eager
  path; VoiceDesign/caption checkpoints, v2 checkpoints, manual duration and
  multiple candidates are supported on both paths.
