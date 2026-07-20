#!/usr/bin/env python3
"""Blend Speaker Inversion embeddings into a single .speaker.safetensors file.

Interpolating between two (or more) learned speaker embeddings rides the base
model's speaker-conditioning manifold, so the result is typically a natural
intermediate voice. Two blend modes are provided:

- ``lerp`` (default): weighted mean of the token tensors (weights normalized
  to sum to 1). Requires every source to have the same token count. Assumes
  token slot i of one embedding corresponds to slot i of the other, which is
  not guaranteed for independently trained runs -- it usually works, but if
  the blended voice sounds muddy, try ``concat``.
- ``concat``: concatenates the token sequences (no slot-correspondence
  assumption; token counts may differ). Each source's tokens are scaled by
  ``normalized_weight * num_sources`` so equal weights reproduce a plain
  concatenation. Experimental: scaling interacts nonlinearly with attention.

The output is a standard Speaker Inversion checkpoint usable with
``infer.py --ref-embed`` or the server's per-request field, and blend
provenance is written next to it as ``<output>.blend.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from irodori_tts.speaker_inversion import (
    SPEAKER_EMBEDDING_KEY,
    load_speaker_inversion_payload,
    save_speaker_inversion_safetensors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend Speaker Inversion embeddings (.speaker.safetensors)."
    )
    parser.add_argument(
        "--embed",
        action="append",
        required=True,
        help="Source .speaker.safetensors file. Repeat for each source (>= 2).",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        required=True,
        help="Blend weight per source, in --embed order (e.g. 0.5 0.5).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path (must end with .speaker.safetensors).",
    )
    parser.add_argument(
        "--mode",
        choices=("lerp", "concat"),
        default="lerp",
        help="Blend mode (default: lerp).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def blend_embeddings(
    embeddings: list[torch.Tensor], weights: list[float], mode: str
) -> torch.Tensor:
    total = sum(weights)
    normalized = [w / total for w in weights]

    dims = {int(e.shape[1]) for e in embeddings}
    if len(dims) != 1:
        raise ValueError(f"Speaker dims differ across sources: {sorted(dims)}")

    if mode == "lerp":
        token_counts = {int(e.shape[0]) for e in embeddings}
        if len(token_counts) != 1:
            raise ValueError(
                "lerp requires identical token counts across sources, got "
                f"{sorted(token_counts)}. Retrain with matching "
                "speaker_inversion_tokens, or use --mode concat."
            )
        out = torch.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, normalized, strict=True):
            out += emb * weight
        return out

    scaled = [
        emb * (weight * len(embeddings)) for emb, weight in zip(embeddings, normalized, strict=True)
    ]
    return torch.cat(scaled, dim=0)


def main() -> None:
    args = parse_args()
    sources = [Path(p).expanduser() for p in args.embed]
    weights = [float(w) for w in args.weights]
    output = Path(args.output).expanduser()

    if len(sources) < 2:
        raise ValueError("Need at least two --embed files to blend.")
    if len(weights) != len(sources):
        raise ValueError(
            f"--weights count ({len(weights)}) must match --embed count ({len(sources)})."
        )
    if any(w <= 0.0 for w in weights):
        raise ValueError("All --weights must be > 0.")
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {output} (use --force)")

    embeddings = [load_speaker_inversion_payload(path)[SPEAKER_EMBEDDING_KEY] for path in sources]
    blended = blend_embeddings(embeddings, weights, args.mode)
    save_speaker_inversion_safetensors(output, {SPEAKER_EMBEDDING_KEY: blended})

    reloaded = load_speaker_inversion_payload(output)[SPEAKER_EMBEDDING_KEY]
    if not torch.allclose(reloaded, blended.float(), atol=1e-6):
        raise RuntimeError(f"Reloaded blend does not match computed tensor: {output}")

    provenance = {
        "mode": args.mode,
        "sources": [
            {"embed": str(path), "weight": float(weight)}
            for path, weight in zip(sources, weights, strict=True)
        ],
        "normalized_weights": [float(w / sum(weights)) for w in weights],
        "tokens": int(blended.shape[0]),
        "dim": int(blended.shape[1]),
    }
    sidecar = output.with_name(output.name + ".blend.json")
    sidecar.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Saved blended embedding: {output}")
    print(f"mode={args.mode} tokens={blended.shape[0]} dim={blended.shape[1]}")
    print(f"provenance: {sidecar}")


if __name__ == "__main__":
    main()
