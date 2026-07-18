"""Probe fp16 decode determinism options and conv-TF32 (without benchmark).

Run from the repo root:
    .venv/Scripts/python.exe tools/probe_decode_det.py
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
    z = codec32.encode_waveform(wav.unsqueeze(0), sample_rate=int(sr))

    with torch.inference_mode():
        ref32 = codec32.decode_latent(z).float()

        # ---- conv TF32 without benchmark, fp32 weights ----
        try:
            try:
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            except Exception:
                torch.backends.cudnn.allow_tf32 = True
            out = codec32.decode_latent(z)
            times = _time_ms(lambda: codec32.decode_latent(z))
            print(
                f"[probe] fp32 + conv TF32 : median {statistics.median(times):6.2f} ms  "
                f"SNR vs fp32 {_snr_db(ref32, out.float()):6.1f} dB"
            )
        finally:
            try:
                torch.backends.cudnn.conv.fp32_precision = "ieee"
            except Exception:
                torch.backends.cudnn.allow_tf32 = False

        # ---- fp16 non-determinism magnitude ----
        codec16 = DACVAECodec.load(device="cuda", dtype=torch.float16)
        a = codec16.decode_latent(z).float()
        b = codec16.decode_latent(z).float()
        n = min(a.shape[-1], b.shape[-1])
        print(
            f"[probe] fp16 repeat-vs-repeat: equal={torch.equal(a, b)} "
            f"SNR {_snr_db(a[..., :n], b[..., :n]):6.1f} dB  "
            f"max|diff| {(a - b).abs().max().item():.3e}"
        )

        # ---- fp16 + cudnn.deterministic ----
        try:
            torch.backends.cudnn.deterministic = True
            outs = [codec16.decode_latent(z) for _ in range(5)]
            det = all(torch.equal(outs[0], o) for o in outs[1:])
            times = _time_ms(lambda: codec16.decode_latent(z))
            print(
                f"[probe] fp16 + cudnn.deterministic: median {statistics.median(times):6.2f} ms  "
                f"repeat-identical={det}  SNR vs fp32 {_snr_db(ref32[..., :n], outs[0].float()[..., :n]):6.1f} dB"
            )
        finally:
            torch.backends.cudnn.deterministic = False

        # ---- which module is non-deterministic? decoder vs wm passthrough ----
        z_t = z.transpose(1, 2).contiguous().to(codec16.device, dtype=codec16.dtype)
        dec = codec16.model.decoder
        # decode() internals may differ; probe the decoder module directly.
        x1 = dec(z_t)
        x2 = dec(z_t)
        print(f"[probe] fp16 decoder-module-only repeat equal: {torch.equal(x1, x2)}")


if __name__ == "__main__":
    main()
