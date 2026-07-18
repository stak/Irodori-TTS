"""Probe fp16/bf16 DACVAE decode speed and quality vs fp32.

Run from the repo root:
    .venv/Scripts/python.exe tools/probe_decode_precision.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from irodori_tts.codec import DACVAECodec
from irodori_tts.inference_runtime import _enable_cuda_fast_math

REPO_ROOT = Path(__file__).resolve().parent.parent
WAV_PATH = REPO_ROOT / "verify_runs" / "api_baseline" / "req01_cand01.wav"
TIMING_ITERS = 30


def _sync() -> None:
    torch.cuda.synchronize()


def _time_ms(fn, iters: int = TIMING_ITERS) -> list[float]:
    times = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    noise = (ref - test).double()
    noise_power = float((noise**2).sum())
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * torch.log10((ref.double() ** 2).sum() / noise_power).item()


def main() -> None:
    _enable_cuda_fast_math()
    import torchaudio

    wav, sr = torchaudio.load(str(WAV_PATH))

    codec32 = DACVAECodec.load(device="cuda")
    z = codec32.encode_waveform(wav.unsqueeze(0), sample_rate=int(sr))  # fp32 latent
    with torch.inference_mode():
        ref = codec32.decode_latent(z).float()
        times = _time_ms(lambda: codec32.decode_latent(z))
        print(f"[probe] fp32 decode : median {statistics.median(times):6.2f} ms")

        for dtype, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            codec = DACVAECodec.load(device="cuda", dtype=dtype)
            out = codec.decode_latent(z).float()
            n = min(out.shape[-1], ref.shape[-1])
            snr = _snr_db(ref[..., :n], out[..., :n])
            outs = [codec.decode_latent(z) for _ in range(3)]
            det = all(torch.equal(outs[0], o) for o in outs[1:])
            times = _time_ms(lambda: codec.decode_latent(z))
            print(
                f"[probe] {name} decode : median {statistics.median(times):6.2f} ms  "
                f"SNR vs fp32 {snr:6.1f} dB  repeat-deterministic={det}"
            )
            del codec
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
