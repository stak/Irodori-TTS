"""Measure sample_rf with model_precision fp32 / bf16 / fp16 (existing UI option).

Run from the repo root:
    .venv/Scripts/python.exe tools/probe_model_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

LORA = str(Path("outputs") / "lora_shiduka" / "checkpoint_final")
TEXT = "いつまで待たせるつもり？ 早くしてよね。"
WARM_RUNS = 3


def main() -> None:
    from gradio_app import _resolve_checkpoint_path

    ckpt = _resolve_checkpoint_path("Aratako/Irodori-TTS-500M-v3")
    ref = None
    for precision in ("fp32", "bf16", "fp16"):
        try:
            runtime = InferenceRuntime.from_key(
                RuntimeKey(
                    checkpoint=ckpt,
                    model_device="cuda",
                    model_precision=precision,
                    codec_device="cuda",
                )
            )
            stats: dict[str, list[float]] = {}
            audio = None
            for run in range(WARM_RUNS + 1):
                result = runtime.synthesize(
                    SamplingRequest(
                        text=TEXT, no_ref=True, seed=1, num_steps=40, lora_adapter=LORA
                    )
                )
                audio = result.audios[0].float()
                if run == 0:
                    continue
                for name, sec in result.stage_timings:
                    stats.setdefault(name, []).append(sec * 1000.0)
            med = {k: sorted(v)[len(v) // 2] for k, v in stats.items()}
            snr = ""
            if precision == "fp32":
                ref = audio
            elif ref is not None and audio is not None:
                n = min(ref.shape[-1], audio.shape[-1])
                noise = (ref[..., :n] - audio[..., :n]).double()
                p = float((noise**2).sum())
                snr_db = float("inf") if p == 0 else 10.0 * torch.log10(
                    (ref[..., :n].double() ** 2).sum() / p
                ).item()
                snr = f"  SNR vs fp32 {snr_db:5.1f} dB (len {ref.shape[-1]} vs {audio.shape[-1]})"
            print(
                f"[prec] {precision}: sample_rf {med.get('sample_rf', 0):6.1f} ms  "
                f"predict_duration {med.get('predict_duration', 0):4.1f} ms  "
                f"decode {med.get('decode_latent', 0):5.1f} ms{snr}"
            )
            del runtime
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[prec] {precision}: FAILED {exc!r}")


if __name__ == "__main__":
    main()
