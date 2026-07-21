# Inference Performance Fork Notes

This fork adds a set of inference-speed optimizations on top of upstream
Irodori-TTS. With `IRODORI_PERF_PROFILE=recommended` on the reference setup
(RTX 40-series, 40 steps, ~4 s utterance, LoRA adapter, watermark enabled,
locked GPU clocks), steady-state `total_to_decode` in the Gradio app drops
from **~1.0 s to ~0.25 s** (Windows-native, fp32 settings). Opting into bf16
model precision brings it to **~0.20 s**, and additionally enabling
torch.compile on Linux/WSL2 to **~0.14-0.15 s** (see
[Precision Choice](#precision-choice) and
[torch.compile](#torchcompile-linux--wsl2-opt-in)). Without locked clocks,
GPU power management adds run-to-run variance (see
[Latency variance](#latency-variance-gpu-clocks)).

All optimizations are inference-only. Training code paths are untouched.

## Performance Profile

The optimizations whose outputs differ from upstream are grouped behind one
switch:

| `IRODORI_PERF_PROFILE` | Meaning |
|---|---|
| `upstream` (default) | TF32, LoRA merge, CUDA graphs, fp16 decode and text bucketing are all **off**. Outputs are **bit-identical to unmodified upstream Irodori-TTS** (for `num_candidates > 1` the fork's per-candidate seed derivation applies regardless of profile; see the summary table). |
| `recommended` | The fork's tuned set: all five on. This is the configuration every measurement in this document refers to. |

The individual variables (`IRODORI_DISABLE_*`, `IRODORI_TEXT_BUCKETS`)
override the profile in both directions — e.g. `IRODORI_DISABLE_TF32=1`
forces TF32 off under `recommended`, and `IRODORI_DISABLE_TF32=0` forces it
on under `upstream`. Optimizations that are exact (bit-identical, first four
rows below) are always on. `IRODORI_COMPILE` is a separate opt-in because it
needs a Triton toolchain.

## Summary of Changes

| Change | Devices | Output impact | Gate |
|--------|---------|---------------|------|
| Cache timestep-embedding frequency table | all | exact (bit-identical) | always on |
| Vectorized `find_flattening_point` (tail trim) | all | exact | always on |
| Reuse encoded conditions from duration prediction | all | exact | always on |
| Remove per-step GPU syncs in the Euler loop | all | exact | always on |
| TF32 tensor-core matmul for fp32 | CUDA (Ampere+) | ~41 dB SNR vs upstream (inaudible) | profile; `IRODORI_DISABLE_TF32` |
| LoRA adapter merged into base weights per request | all | fp rounding only | profile; `IRODORI_DISABLE_LORA_MERGE` |
| CUDA per-step graph replay of the sampling loop | CUDA only | exact vs eager; see bucketing note | profile; `IRODORI_DISABLE_CUDA_GRAPH` |
| CUDA graph replay of condition encoding + duration prediction | CUDA only | exact (bit-identical vs eager) | profile; `IRODORI_DISABLE_DURATION_GRAPH` |
| fp16 codec decode (decoder-only, deterministic algorithms) | CUDA only | ~58-60 dB SNR vs fp32 (inaudible, deterministic) | profile; `IRODORI_DISABLE_FP16_DECODE` |
| Text-length bucketing (short texts padded to 64 instead of max_text_len) | all | ~41 dB SNR for short texts (inaudible, deterministic) | profile; `IRODORI_TEXT_BUCKETS` |
| torch.compile-fused kernels replayed inside the CUDA step graphs (opt-in) | CUDA + Triton (Linux/WSL2) | different sampling trajectory (judge by ear); deterministic | `IRODORI_COMPILE=1` to enable |
| Latent-length bucketing (graphs shared across lengths) | CUDA only | ~47 dB SNR when padding occurs (inaudible, deterministic) | `IRODORI_CUDA_GRAPH_BUCKET=1` disables padding |
| LoRA hot-swap keeping cached graphs | CUDA only | tiny fp drift per swap | (checkbox; default off) |
| Per-candidate seeds for `num_candidates > 1` (candidate i draws from seed + i) | all | different multi-candidate outputs per seed vs upstream; single-candidate unchanged | always on (semantic API change, reported via `used_seeds`) |
| Gradio: SilentCipher watermark toggle | - | none (watermark on by default) | - |
| Gradio: offline-first HF checkpoint resolution | - | none | `IRODORI_OFFLINE_FIRST_CHECKPOINT=1` to enable |

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
| `IRODORI_PERF_PROFILE` | `upstream` | `recommended` enables the five profile-gated optimizations below |
| `IRODORI_DISABLE_TF32` | (profile) | `1` disables TF32 matmul, `0` forces it on |
| `IRODORI_DISABLE_LORA_MERGE` | (profile) | `1` keeps LoRA adapters unmerged (upstream behavior), `0` forces merging |
| `IRODORI_DISABLE_CUDA_GRAPH` | (profile) | `1` disables graph capture/replay entirely, `0` forces it on |
| `IRODORI_DISABLE_DURATION_GRAPH` | (profile) | `1` keeps condition encoding + duration prediction eager (sampler graphs unaffected), `0` forces the graph on |
| `IRODORI_DISABLE_FP16_DECODE` | (profile) | `1` keeps the codec decoder in fp32 (exact decode, ~2x slower), `0` forces fp16 decode |
| `IRODORI_TEXT_BUCKETS` | (profile: `64` / `0`) | Comma-separated text-length buckets (tokens); short texts are padded to the smallest fitting bucket instead of max_text_len. `0` or empty disables (upstream padding) |
| `IRODORI_COMPILE` | `0` | `1` runs the DiT through torch.compile and replays the fused kernels inside the CUDA step graphs. Requires a working Triton toolchain (Linux/WSL2 + C compiler); no effect on outputs' determinism, but the sampling trajectory changes |
| `IRODORI_CUDA_GRAPH_BUCKET` | `16` | Latent-length bucket size in patched steps; `1` disables padding |
| `IRODORI_CUDA_GRAPH_CACHE` | `64` | Max cached graph entries (per-entry VRAM is small; pool and condition buffers are shared) |
| `IRODORI_OFFLINE_FIRST_CHECKPOINT` | `0` | `1` makes the Gradio app resolve HF checkpoints from the local cache first (skips the network round-trip, but does not pick up new uploads) |

The default (`upstream` profile with nothing else set) reproduces upstream
outputs **bit-identically**; the remaining always-on optimizations are exact.

## Recommended Configurations

| Scenario | Settings |
|---|---|
| Upstream parity, output comparison, one-shot `infer.py` runs | defaults (nothing set) |
| Windows interactive / batch production | `IRODORI_PERF_PROFILE=recommended`; bf16 model + fp32 codec precision recommended |
| Linux/WSL2 maximum speed | `recommended` + `IRODORI_COMPILE=1`; run `precompile.py` once beforehand (see torch.compile section) |
| Rotating LoRA adapters | any of the above + `lora_hot_swap=true` (non-DoRA adapters, `modules_to_save` at most `duration_predictor`) |
| Speaker-inversion or reference-audio conditioning | as above; SI shapes are stable and reuse graphs, each new reference-audio length captures once (~1 s) |
| Best-of-N (`num_candidates > 1`) | as above; each candidate count is its own graph/compile shape, so fix one operating `n` where possible |

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
- Compilation happens during the first request's warmup for each new shape,
  never inside a capture (capture-time recompiles invalidate the capture; the
  runtime pre-builds the RoPE caches and avoids trace-unstable code paths to
  guarantee this). A cold inductor cache costs ~3.5 min of compiles across
  the full shape range; the on-disk cache cuts later processes to ~1 min.
- Outputs change (fused kernels = different float schedule), like switching
  model precision: same seed remains deterministic - verified bit-identical
  across independent processes - but it is a different take than the
  non-compiled output. Judge by ear.
- LoRA behavior is three-staged (measured, RTX 4090/WSL2): the **first load
  of each adapter path** in a process triggers a recompile (~40 s with a cold
  disk cache, ~9 s once cached) — hot-swap does not avoid this, because
  loading registers a new named adapter; **swapping among already-loaded
  adapters** with hot-swap recompiles nothing (~0.1 s); a **plain switch**
  (hot-swap off) additionally drops all captured graphs.

### Recompilation and recapture triggers

torch.compile and CUDA graph capture are separate layers with different
invalidation rules and lifetimes:

| Trigger | torch.compile | CUDA graphs |
|---|---|---|
| New shape (candidate count, text bucket) | recompile — cached on disk across processes | capture (~1 s), per process |
| New latent length | dynamic after the second distinct length (free) | capture per length bucket |
| CFG scale values | no effect (combine runs outside compiled code) | new capture (values are baked in) |
| CFG term on/off (a scale set to 0) | recompile (batch shape changes) | new capture |
| First load of an adapter path | recompile (~40 s cold / ~9 s disk-cached) | survives with hot-swap, dropped otherwise |
| Swap among loaded adapters (hot-swap) | none | none |
| `num_steps`, schedule, sway, `cfg_min/max_t`, seed | none | none |
| Process restart | re-trace per trace unit (~8-12 s each; kernels come from the disk cache) | all captures redone lazily |

Practical consequences:

- **Run `precompile.py` once per machine** with the production settings
  (same `IRODORI_*` variables, `IRODORI_COMPILE=1`, every production
  `--num-candidates` value and every production `--lora-adapter`). This
  moves the cold-compile minutes to a chosen point in time; without it, each
  uncached shape stalls its first real request.
- **Per-process warmup cannot be cached away**: each process re-traces once
  per trace unit and per loaded adapter (~9-12 s each). For adapter
  rotation, request each adapter once right after startup (with hot-swap
  enabled); all later switches are then instant.
- Reference-audio conditioning makes shapes per-request (each reference
  length is a new shape); speaker-inversion embeddings have fixed token
  counts and stay on the cached path.

## Graph Capture Lifecycle

Graphs are captured lazily: the first request at a new shape (length bucket ×
text bucket × candidate count × CFG configuration) pays a one-time capture
cost (~1 s per shape; a few seconds for `num_candidates > 1`), and every
following request with that shape replays the captured graph. The first
request of a session additionally pays the one-time codec-decoder and
watermarker setup (~40 ms each). There is no pre-capture step; in steady
state the set of graphs converges to exactly the shapes the workload uses
(each entry holds ~16 MiB of driver-side memory).

Notes:

- Graphs are keyed by tensor shapes and CFG scales. `Num Steps`,
  `Time Schedule`, `Sway Coeff`, `CFG Min/Max t` and `Seed` can be changed
  freely without re-capture. Changing `CFG Scale Text/Speaker` values,
  `Num Candidates`, precision/device, or supplying reference audio triggers a
  fresh capture per length bucket (one-time, ~1 s each).
- The `predict_duration` stage (condition encoding + duration head) is also
  replayed as one CUDA graph (~14 ms -> ~3-4 ms per request on the reference
  setup). It is captured lazily on the first eligible request (~0.1 s
  one-time) and validated bit-exact against eager at capture.
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

Same seed remains deterministic in all configurations. The
recommended-profile changes below cause numerical differences from upstream
that are all far below audibility. (Opt-in bf16 model precision and
`IRODORI_COMPILE` instead change the sampling trajectory outright — a
different take of the same text — and are covered in their own sections
above.)

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
- **Text-length bucketing** (recommended profile: bucket 64): texts of at most 64 tokens
  are padded to 64 instead of the checkpoint's max_text_len (256), removing
  ~190 masked keys from every cross-attention step (~-16 ms sample_rf, ~-9 ms
  of it vs no bucketing at all for typical short texts). The masked math is
  equivalent; kernel reduction order changes, measured ~41 dB SNR end to end
  for short texts. Deterministic per text length. Duration features keep the
  max_text_len normalization, so predicted durations are unaffected by the
  bucket choice (identical predicted frames in verification); in rare
  rounding-boundary cases the integer frame count could still differ by one.
  Long texts (and requests with an explicit `max_text_len`) are unaffected.
- **fp16 codec decode** (recommended profile, CUDA): the DACVAE decode-only modules
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

- **One-shot CLI (`infer.py`) pays capture cost under the recommended
  profile**: the graph path captures on the first request, which is pure
  overhead for a single-generation process (~+0.5-1 s). Keep the default
  `upstream` profile (or set `IRODORI_DISABLE_CUDA_GRAPH=1`) for one-shot
  CLI runs, and keep `IRODORI_COMPILE` off there.
- **VRAM**: cached graphs add roughly 1-1.5 GiB once a workload has touched
  the full 15 s duration range (24 latent buckets x 2 text buckets; ~16 MiB
  of driver-side graph memory per entry, plus the shared memory pool and
  per-text-bucket condition buffers). Workloads that use fewer shapes hold
  proportionally fewer graphs. `IRODORI_TEXT_BUCKETS=0` roughly halves this.
  Upstream uses correspondingly less.
- **Offline-first checkpoint resolution** (opt-in,
  `IRODORI_OFFLINE_FIRST_CHECKPOINT=1`): the Gradio app then resolves HF-hub
  checkpoints from the local cache first, skipping a network round-trip per
  click. While a cached copy exists, a newer `model.safetensors` published
  upstream is **not** picked up automatically; clear the HF cache entry (or
  download manually) to update.
- **Gradio API surface**: `_run_generation` takes additional inputs
  (hot-swap flag, watermark flag), so `gradio_client` calls written against
  upstream need their argument lists updated.
- **Reference-audio requests and per-request `max_text_len` overrides**
  produce different conditioning shapes per request, so each new reference
  length captures its own graphs on first use. Speaker-embedding (SI) and
  no-reference requests have stable conditioning shapes and reuse their
  graphs across requests.
- Upstream functionality is otherwise preserved. Configurations not eligible
  for the graph path (`joint`/`alternating` CFG, temporal rescale,
  speaker-KV scaling, non-CUDA devices) run through the unchanged eager
  path; VoiceDesign/caption checkpoints, v2 checkpoints, manual duration and
  multiple candidates are supported on both paths.
