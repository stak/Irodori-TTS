"""Pre-implementation probes for decode_latent speedup options.

Measures, on the real DACVAE decoder with a real utterance latent:
  1. eager decode timing (current behavior)
  2. CUDA-graph replay timing at fixed shape (speed ceiling) + bitwise check
  3. padding bleed: decode(pad(z)) vs decode(z) on the kept region
     (zero padding and replicate padding, several bucket sizes)
  4. decoder-scoped cudnn.benchmark: autotune cost, warm timing, determinism

Run from the repo root:
    .venv/Scripts/python.exe tools/probe_decode.py
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
DEVICE = "cuda"
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


def _fmt(times: list[float]) -> str:
    return (
        f"median {statistics.median(times):7.2f} ms  "
        f"min {min(times):7.2f}  max {max(times):7.2f}"
    )


def _snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    noise = (ref - test).double()
    noise_power = float((noise**2).sum())
    if noise_power == 0.0:
        return float("inf")
    return 10.0 * torch.log10((ref.double() ** 2).sum() / noise_power).item()


def main() -> None:
    _enable_cuda_fast_math()
    torch.manual_seed(0)

    import torchaudio

    wav, sr = torchaudio.load(str(WAV_PATH))
    print(f"[probe] input wav: {WAV_PATH.name} sr={sr} samples={wav.shape[1]}")

    codec = DACVAECodec.load(device=DEVICE)
    hop = int(codec.model.hop_length)
    print(f"[probe] codec sample_rate={codec.sample_rate} hop={hop} dtype={codec.dtype}")

    z = codec.encode_waveform(wav.unsqueeze(0), sample_rate=int(sr))  # (1, T, D)
    T = z.shape[1]
    print(f"[probe] latent shape={tuple(z.shape)} ({T} frames = {T * hop / codec.sample_rate:.2f}s)")

    with torch.inference_mode():
        # ---------- 1. eager timing ----------
        ref = codec.decode_latent(z)  # warm up lazy init / algo selection
        ref = codec.decode_latent(z)
        keep = T * hop
        print(f"[probe] decode output samples={ref.shape[-1]} keep={keep}")
        times = _time_ms(lambda: codec.decode_latent(z))
        print(f"[probe] 1. eager decode        : {_fmt(times)}")

        # Repeat-determinism of the current eager path (sanity).
        again = codec.decode_latent(z)
        print(f"[probe]    eager repeat bitwise-identical: {torch.equal(ref, again)}")

        # ---------- 2. CUDA graph replay at fixed shape ----------
        z_static = z.clone().transpose(1, 2).contiguous().to(codec.device, dtype=codec.dtype)
        graph_ok = True
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                warm = codec.model.decode(z_static).clone()
            torch.cuda.current_stream().wait_stream(side)
            _sync()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_static = codec.model.decode(z_static)
            graph.replay()
            _sync()
            print(f"[probe] 2. graph replay bitwise vs eager warmup: {torch.equal(out_static, warm)}")
            print(f"[probe]    graph replay vs current eager ref   : {torch.equal(out_static, ref)}")
            times = _time_ms(lambda: graph.replay())
            print(f"[probe]    graph replay          : {_fmt(times)}")
        except Exception as exc:
            graph_ok = False
            print(f"[probe] 2. graph capture FAILED: {exc!r}")

        # ---------- 3. padding bleed ----------
        ref_keep = ref[..., :keep]
        for bucket in (8, 16, 32, 64):
            padded_T = ((T + bucket - 1) // bucket) * bucket
            pad = padded_T - T
            if pad == 0:
                padded_T += bucket
                pad = bucket
            for mode in ("zero", "replicate"):
                if mode == "zero":
                    tail = torch.zeros((1, pad, z.shape[2]), device=z.device, dtype=z.dtype)
                else:
                    tail = z[:, -1:, :].expand(1, pad, z.shape[2])
                z_pad = torch.cat([z, tail], dim=1)
                out_pad = codec.decode_latent(z_pad)[..., :keep]
                snr = _snr_db(ref_keep, out_pad)
                diff = (ref_keep - out_pad).abs().squeeze()
                nz = torch.nonzero(diff > 0)
                if nz.numel() == 0:
                    bleed_desc = "no difference at all"
                else:
                    first = int(nz[0])
                    bleed = keep - first
                    bleed_desc = (
                        f"bleed={bleed} samples ({bleed / hop:.1f} latent frames, "
                        f"{bleed / codec.sample_rate * 1000:.1f} ms), max|diff|={diff.max():.3e}"
                    )
                print(
                    f"[probe] 3. pad {mode:9s} T={T}->{padded_T} (+{pad:2d}): "
                    f"SNR {snr:6.1f} dB, {bleed_desc}"
                )

        # ---------- 4. decoder-scoped cudnn.benchmark ----------
        try:
            torch.backends.cudnn.benchmark = True
            _sync()
            t0 = time.perf_counter()
            bench_out = codec.decode_latent(z)
            _sync()
            autotune_ms = (time.perf_counter() - t0) * 1000.0
            times = _time_ms(lambda: codec.decode_latent(z))
            print(f"[probe] 4. cudnn.benchmark first call (autotune): {autotune_ms:.1f} ms")
            print(f"[probe]    cudnn.benchmark warm  : {_fmt(times)}")
            outs = [codec.decode_latent(z) for _ in range(5)]
            deterministic = all(torch.equal(outs[0], o) for o in outs[1:])
            print(f"[probe]    benchmark repeat bitwise-identical: {deterministic}")
            print(f"[probe]    benchmark vs default algo bitwise : {torch.equal(bench_out, ref)}")
            if not torch.equal(bench_out, ref):
                print(f"[probe]    benchmark vs default SNR: {_snr_db(ref, bench_out):.1f} dB")
            # New-shape autotune cost (what a not-yet-seen length would pay).
            z_new = torch.cat([z, z[:, -8:, :]], dim=1)
            _sync()
            t0 = time.perf_counter()
            codec.decode_latent(z_new)
            _sync()
            print(
                f"[probe]    benchmark new-shape first call: {(time.perf_counter() - t0) * 1000.0:.1f} ms"
            )
        finally:
            torch.backends.cudnn.benchmark = False

        # ---------- 5. graph at bucketed (padded) shape, replay timing ----------
        if graph_ok:
            bucket = 32
            padded_T = ((T + bucket - 1) // bucket) * bucket
            z_pad_static = torch.zeros((1, codec.latent_dim, padded_T), device=codec.device, dtype=codec.dtype)
            z_pad_static[:, :, :T] = z.transpose(1, 2)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                codec.model.decode(z_pad_static)
            torch.cuda.current_stream().wait_stream(side)
            _sync()
            graph_pad = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_pad):
                codec.model.decode(z_pad_static)
            times = _time_ms(lambda: graph_pad.replay())
            print(f"[probe] 5. graph replay at padded T={padded_T}: {_fmt(times)}")


if __name__ == "__main__":
    main()
