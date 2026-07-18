"""Measure how padded text length (bucketing candidates) affects sampling speed.

Uses the existing per-request max_text_len override: each value produces the
same conditioning shapes a text-bucketed implementation would use, so warm
sample_rf timings here ARE the post-bucketing timings.

Run from the repo root:
    .venv/Scripts/python.exe tools/probe_textlen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

LORA = str(Path("outputs") / "lora_shiduka" / "checkpoint_final")
TEXT = "いつまで待たせるつもり？ 早くしてよね。"
WARM_RUNS = 3


def main() -> None:
    from gradio_app import _resolve_checkpoint_path

    key = RuntimeKey(
        checkpoint=_resolve_checkpoint_path("Aratako/Irodori-TTS-500M-v3"),
        model_device="cuda",
        codec_device="cuda",
    )
    runtime = InferenceRuntime.from_key(key)
    ids, mask = runtime.tokenizer.batch_encode([TEXT], max_length=256)
    print(f"[textlen] text token count: {int(mask.sum())}")

    for mtl in (256, 128, 64, 32):
        stats: dict[str, list[float]] = {}
        for run in range(WARM_RUNS + 1):  # first run captures graphs
            result = runtime.synthesize(
                SamplingRequest(
                    text=TEXT,
                    no_ref=True,
                    seed=1,
                    num_steps=40,
                    max_text_len=mtl,
                    lora_adapter=LORA,
                )
            )
            if run == 0:
                continue
            for name, sec in result.stage_timings:
                stats.setdefault(name, []).append(sec * 1000.0)
        parts = {
            name: sorted(vals)[len(vals) // 2]
            for name, vals in stats.items()
        }
        total = sum(parts.values())
        print(
            f"[textlen] max_text_len={mtl:3d}: "
            f"sample_rf {parts.get('sample_rf', 0):6.1f} ms  "
            f"predict_duration {parts.get('predict_duration', 0):5.1f} ms  "
            f"decode {parts.get('decode_latent', 0):5.1f} ms  "
            f"total-ish {total:6.1f} ms"
        )


if __name__ == "__main__":
    main()
