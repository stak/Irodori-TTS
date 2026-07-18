# Performance verification harness and probe scripts (archive)

Working assets from the 2026-07 inference-optimization effort that produced
the numbers recorded in `docs/performance.md`. Kept on this archive branch —
not part of `main` — for reproducing measurements or continuing the tuning
work. To use them from a `main` checkout:

```
git checkout archive/perf-probes -- tools/
```

All scripts assume the repo root as the working directory and the standard
verification request (text 「いつまで待たせるつもり？ 早くしてよね。」,
Seed 1, LoRA `outputs/lora_shiduka/checkpoint_final`, 40 steps) unless noted.
Lock GPU clocks (`nvidia-smi -lgc`) before timing anything.

## The harness

- **`verify_api.py`** — end-to-end verification via the Gradio API
  (spawns/attaches to a server, Load -> Prewarm -> Generate xN, parses the
  per-stage timings, saves wavs + `summary.json` under `verify_runs/<tag>`).
  `compare <tagA> <tagB>` bit-compares wavs and reports SNR on differences.
  `--env KEY=VALUE` passes `IRODORI_*` opt-outs to the spawned server;
  `--model-precision bf16`, `--attach`, `--port` as needed. This is how every
  bit-exactness and SNR claim in `docs/performance.md` was checked.
- **`start_wsl_server.sh`** — starts the Gradio server inside WSL and waits
  for readiness (server startup over /mnt/c is slow; pair with
  `verify_api.py --attach`). Contains user-specific absolute paths; adjust.

## One-shot probes (findings already recorded in docs/performance.md)

- `probe_decode.py` — decode_latent options: CUDA-graph replay timing,
  latent-padding bleed SNR, decoder-scoped cudnn.benchmark. Result: decoder
  is compute-bound; graph/benchmark rejected.
- `probe_decode_precision.py` — fp16/bf16 decode speed + SNR vs fp32.
- `probe_decode_det.py` — fp16 decode determinism (cudnn.deterministic) and
  conv-TF32-without-benchmark check.
- `probe_vram.py` — VRAM cost per sampler-graph entry (~16 MiB) and shared
  condition-buffer sizes; basis of the text-bucketing VRAM estimate.
- `probe_textlen.py` — sample_rf vs padded text length via the per-request
  max_text_len override; basis of the {64,256} bucket decision.
- `probe_model_precision.py` — sample_rf across model_precision fp32/bf16.
- `probe_compile.py` / `probe_compile2.py` / `probe_compile3.py` —
  torch.compile experiments (WSL2): compile-only vs graphs-only vs combined,
  recompile diagnostics (`TORCH_LOGS=recompiles`). Basis of the
  IRODORI_COMPILE integration; probe_compile3 is the useful one for checking
  that steady-state requests trigger zero recompiles after changes to the
  compiled forward paths.
