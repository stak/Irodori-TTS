# Inference Performance Fork Notes

This fork adds a set of inference-speed optimizations on top of upstream
Irodori-TTS. On the reference setup (RTX 40-series, 40 steps, ~4 s utterance,
LoRA adapter, watermark enabled, locked GPU clocks), steady-state
`total_to_decode` in the Gradio app drops from **~1.0 s to ~0.25 s**
(Windows-native, default fp32 settings). Opting into bf16 model precision
brings it to **~0.20 s**, and additionally enabling torch.compile on
Linux/WSL2 to **~0.14-0.15 s** (see [Precision Choice](#precision-choice) and
[torch.compile](#torchcompile-linux--wsl2-opt-in)). Without locked clocks,
GPU power management adds run-to-run variance (see
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
| fp16 codec decode (decoder-only, deterministic algorithms) | CUDA only | ~58-60 dB SNR vs fp32 (inaudible, deterministic) | `IRODORI_DISABLE_FP16_DECODE=1` |
| Text-length bucketing (short texts padded to 64 instead of max_text_len) | all | ~41 dB SNR for short texts (inaudible, deterministic) | `IRODORI_TEXT_BUCKETS=0` |
| torch.compile-fused kernels replayed inside the CUDA step graphs (opt-in) | CUDA + Triton (Linux/WSL2) | different sampling trajectory (judge by ear); deterministic | `IRODORI_COMPILE=1` to enable |
| Prewarm also warms codec decoder + watermarker | CUDA only | none | (part of Prewarm) |
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
available for CUDA). CUDA graph replay removes that overhead. On Linux/WSL2,
torch.compile can additionally be layered on top of the graphs to shrink the
kernels themselves (see [torch.compile](#torchcompile-linux--wsl2-opt-in)).

## Environment Variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `IRODORI_DISABLE_TF32` | `0` | `1` disables TF32 matmul (exact fp32, slower) |
| `IRODORI_DISABLE_LORA_MERGE` | `0` | `1` keeps LoRA adapters unmerged (upstream behavior) |
| `IRODORI_DISABLE_CUDA_GRAPH` | `0` | `1` disables graph capture/replay entirely |
| `IRODORI_DISABLE_DURATION_GRAPH` | `0` | `1` keeps condition encoding + duration prediction eager (sampler graphs unaffected) |
| `IRODORI_DISABLE_FP16_DECODE` | `0` | `1` keeps the codec decoder in fp32 (exact decode, ~2x slower) |
| `IRODORI_TEXT_BUCKETS` | `64` | Comma-separated text-length buckets (tokens); short texts are padded to the smallest fitting bucket instead of max_text_len. `0` or empty disables (upstream padding) |
| `IRODORI_COMPILE` | `0` | `1` runs the DiT through torch.compile and replays the fused kernels inside the CUDA step graphs. Requires a working Triton toolchain (Linux/WSL2 + C compiler); no effect on outputs' determinism, but the sampling trajectory changes |
| `IRODORI_CUDA_GRAPH_BUCKET` | `16` | Latent-length bucket size in patched steps; `1` disables padding |
| `IRODORI_CUDA_GRAPH_CACHE` | `64` | Max cached graph entries (per-entry VRAM is small; pool and condition buffers are shared) |

Setting `IRODORI_DISABLE_TF32=1`, `IRODORI_DISABLE_LORA_MERGE=1`,
`IRODORI_DISABLE_CUDA_GRAPH=1`, `IRODORI_DISABLE_FP16_DECODE=1` and
`IRODORI_TEXT_BUCKETS=0` together reproduces upstream outputs
**bit-identically** (the remaining optimizations are exact;
`IRODORI_DISABLE_CUDA_GRAPH=1` also disables the duration graph).

## Precision Choice

- **Model Precision = bf16** is worth trying: sample_rf drops a further ~23%
  (measured 216 -> 169 ms; total_to_decode ~242 -> ~196 ms on the reference
  setup) and same-seed outputs remain bit-reproducible. Unlike the SNR-graded
  changes above, bf16 sampling follows a *different trajectory* — outputs are
  effectively a different take of the same text (predicted duration can shift
  by a frame), so judge it by ear, not SNR. LoRA merge rounding also lands in
  bf16. Keep fp32 when exact fp32 fidelity or fp32-tagged reproducibility
  matters.
- **Codec Precision should stay fp32**: with the default fp16 decode path the
  decoder already runs at half precision speed (~10 ms) with ~60 dB SNR and
  an fp32 encoder. Selecting bf16/fp16 codec precision is not faster and only
  lowers quality (bf16 decode measured ~42 dB), and it additionally moves the
  reference-audio *encoder* off fp32.

## torch.compile (Linux / WSL2, opt-in)

With `IRODORI_COMPILE=1`, `encode_conditions`, `build_context_kv_cache` and
`forward_with_encoded_conditions` run through torch.compile, and the fused
Triton kernels are replayed inside the existing CUDA step graphs: inductor's
kernel fusion stacks with graph replay's zero launch overhead. Measured on
the reference setup (bf16 model, WSL2): sample_rf ~169 -> ~105-117 ms,
total_to_decode ~196 -> ~135-153 ms. Neither piece wins alone: compiled
kernels without graph replay were *slower* than the eager graphs (fusion does
not pay for the per-step launch overhead), and graphs without compile leave
the fusion gains on the table.

Requirements and behavior:

- Needs a working Triton toolchain: Linux or WSL2, CUDA torch build, and a C
  compiler (`apt-get install build-essential`). On Windows-native (no
  Triton) leave it off; the flag only changes how kernels are generated, so
  everything else in this document applies unchanged.
- Compilation happens during Prewarm / the first request's warmup, never
  inside a capture (capture-time recompiles invalidate the capture; the
  runtime pre-builds the RoPE caches and avoids trace-unstable code paths to
  guarantee this). Cold-cache Prewarm takes ~3.5 min; inductor's on-disk
  cache cuts later processes to ~1 min.
- Outputs change (fused kernels = different float schedule), like switching
  model precision: same seed remains deterministic - verified bit-identical
  across independent processes - but it is a different take than the
  non-compiled output. Judge by ear.
- LoRA loading/switching changes module structure and triggers a one-time
  recompile absorbed by the next warmup/prewarm; hot-swap keeps structures
  and does not recompile.

## Recommended Gradio Workflow

1. Load Model.
2. Fill in the LoRA Adapter Directory (if any) **before** prewarming.
3. Press **Prewarm Graphs** (Prewarm Max Seconds bounds the covered duration
   range; 15 s ≈ 24 latent buckets × 2 text buckets = 48 graphs ≈ ~50 s
   one-time cost; each graph entry holds ~16 MiB of driver-side memory, so
   this is roughly 1-1.5 GiB of VRAM). Prewarm also runs one dummy codec
   decode and watermark pass so the first real request skips their one-time
   setup costs.
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

Same seed remains deterministic in all configurations. The default-on changes
below cause numerical differences from upstream that are all far below
audibility. (Opt-in bf16 model precision and `IRODORI_COMPILE` instead change
the sampling trajectory outright — a different take of the same text — and
are covered in their own sections above.)

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
- **Text-length bucketing** (default: bucket 64): texts of at most 64 tokens
  are padded to 64 instead of the checkpoint's max_text_len (256), removing
  ~190 masked keys from every cross-attention step (~-16 ms sample_rf, ~-9 ms
  of it vs no bucketing at all for typical short texts). The masked math is
  equivalent; kernel reduction order changes, measured ~41 dB SNR end to end
  for short texts. Deterministic per text length. Duration features keep the
  max_text_len normalization, so predicted durations are unaffected by the
  bucket choice (identical predicted frames in verification); in rare
  rounding-boundary cases the integer frame count could still differ by one.
  Long texts (and requests with an explicit `max_text_len`) are unaffected.
- **fp16 codec decode** (default on CUDA): the DACVAE decode-only modules
  (`quantizer.out_proj` + decoder) run in fp16 while the encoder stays fp32,
  so reference-audio encoding is unchanged. Measured ~60 dB SNR vs fp32
  decode (end-to-end ~58 dB after watermarking) — numerically cleaner than
  the TF32 matmul change above. cuDNN is forced to deterministic conv
  algorithms for reduced-precision decode calls (half-precision autoselect
  is otherwise nondeterministic run-to-run, and the deterministic algorithms
  are not measurably slower here); this also makes explicitly selected
  fp16/bf16 codec precision reproducible, which it previously was not.

## Rejected decode optimizations (measured)

For future reference, these were evaluated for `decode_latent` and rejected —
the DACVAE decoder is **GPU-compute-bound**, not launch-bound:

- **CUDA graph replay** of the decoder: 19.1 -> 18.3 ms (~4%); not worth the
  captured-shape management. Padding the latent to shared bucket lengths also
  perturbs the *entire* output (~68-72 dB SNR, global — the decoder is not
  shift-local end to end), not just the trimmed tail.
- **`cudnn.benchmark` scoped to the decoder**: warm time unchanged (19.1 ms)
  and each previously unseen latent length pays a ~1 s autotune stall. This
  reconfirms the earlier regression that led to matmul-only TF32.
- **Conv TF32 without benchmark**: cuDNN picks the same kernels; output and
  timing are bit-for-bit unchanged.

## Known Limitations / Behavior Changes

- **One-shot CLI (`infer.py`) pays capture cost**: the graph path captures on
  the first request, which is pure overhead for a single-generation process
  (~+0.5-1 s). Set `IRODORI_DISABLE_CUDA_GRAPH=1` for one-shot CLI runs, and
  keep `IRODORI_COMPILE` off there (a cold compile adds minutes).
- **VRAM**: cached graphs add roughly 1-1.5 GiB after a 15 s prewarm
  (24 latent buckets x 2 text buckets; ~16 MiB of driver-side graph memory
  per entry, plus the shared memory pool and per-text-bucket condition
  buffers). `IRODORI_TEXT_BUCKETS=0` roughly halves this. Upstream uses
  correspondingly less.
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
  no-reference/LoRA path at each configured text bucket length and the
  checkpoint's default text length).
- Upstream functionality is otherwise preserved. Configurations not eligible
  for the graph path (`joint`/`alternating` CFG, temporal rescale,
  speaker-KV scaling, non-CUDA devices) run through the unchanged eager
  path; VoiceDesign/caption checkpoints, v2 checkpoints, manual duration and
  multiple candidates are supported on both paths.
