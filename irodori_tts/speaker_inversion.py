from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

SPEAKER_INVERSION_UNCOND_MODES = {"mask", "noise"}
SPEAKER_INVERSION_SAFETENSORS_SUFFIX = ".speaker.safetensors"
SPEAKER_EMBEDDING_KEY = "speaker_embedding"
SPEAKER_INVERSION_BLEND_MODES = ("lerp", "concat")


def normalize_speaker_embedding_tensor(
    tensor: torch.Tensor,
    *,
    speaker_dim: int,
    field_name: str = SPEAKER_EMBEDDING_KEY,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{field_name} must have shape (tokens, dim), got {tuple(tensor.shape)}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError(f"{field_name} must contain at least one token.")
    if int(tensor.shape[1]) != int(speaker_dim):
        raise ValueError(
            f"{field_name} dim mismatch: expected {int(speaker_dim)}, got {int(tensor.shape[1])}"
        )

    return tensor.detach().float().contiguous()


def is_speaker_inversion_safetensors_path(path: str | Path) -> bool:
    return Path(path).name.endswith(SPEAKER_INVERSION_SAFETENSORS_SUFFIX)


class SpeakerInversionEmbedding(nn.Module):
    """Learned speaker/style tokens that bypass the reference latent speaker encoder."""

    def __init__(
        self,
        *,
        num_tokens: int,
        speaker_dim: int,
        init_std: float,
        init_embedding: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        num_tokens = int(num_tokens)
        speaker_dim = int(speaker_dim)
        init_std = float(init_std)
        if num_tokens <= 0:
            raise ValueError(f"speaker inversion tokens must be > 0, got {num_tokens}")
        if speaker_dim <= 0:
            raise ValueError(f"speaker_dim must be > 0, got {speaker_dim}")
        if init_std < 0:
            raise ValueError(f"speaker inversion init_std must be >= 0, got {init_std}")

        if init_embedding is None:
            embedding = torch.randn(num_tokens, speaker_dim, dtype=torch.float32) * init_std
        else:
            embedding = normalize_speaker_embedding_tensor(
                init_embedding,
                speaker_dim=speaker_dim,
                field_name=SPEAKER_EMBEDDING_KEY,
            )
            if int(embedding.shape[0]) != num_tokens:
                raise ValueError(
                    "speaker inversion init embedding token mismatch: "
                    f"expected {num_tokens}, got {int(embedding.shape[0])}"
                )
        self.embedding = nn.Parameter(embedding)

    @property
    def num_tokens(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def speaker_dim(self) -> int:
        return int(self.embedding.shape[1])

    def forward(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.embedding.to(device=device, dtype=dtype)[None, :, :].expand(
            int(batch_size),
            -1,
            -1,
        )
        mask = torch.ones((int(batch_size), self.num_tokens), dtype=torch.bool, device=device)
        return state, mask


def _extract_embedding_payload(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(raw, dict):
        raise ValueError(
            f"Speaker inversion file must contain a tensor dictionary, got {type(raw)!r}."
        )

    if SPEAKER_EMBEDDING_KEY in raw:
        embedding = raw[SPEAKER_EMBEDDING_KEY]
        if not isinstance(embedding, torch.Tensor):
            raise ValueError(
                f"Speaker inversion '{SPEAKER_EMBEDDING_KEY}' must be a tensor, "
                f"got {type(embedding)!r}."
            )
        return {SPEAKER_EMBEDDING_KEY: embedding}

    raise ValueError(f"Speaker inversion file is missing '{SPEAKER_EMBEDDING_KEY}'.")


def normalize_speaker_inversion_payload(
    raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    payload = _extract_embedding_payload(raw)
    embedding = payload[SPEAKER_EMBEDDING_KEY]

    out: dict[str, torch.Tensor] = {
        SPEAKER_EMBEDDING_KEY: embedding,
    }

    return out


def load_speaker_inversion_payload(
    path: str | Path,
) -> dict[str, torch.Tensor]:
    source = Path(path).expanduser()
    if not is_speaker_inversion_safetensors_path(source):
        raise ValueError(
            "Speaker Inversion embeddings must use the "
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX!r} suffix: {source}"
        )
    raw = load_safetensors_file(source, device="cpu")

    out = normalize_speaker_inversion_payload(raw)
    return out


def save_speaker_inversion_safetensors(
    path: str | Path,
    payload: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> None:
    target = Path(path)
    if not is_speaker_inversion_safetensors_path(target):
        raise ValueError(
            "Speaker Inversion safetensors output must use the "
            f"{SPEAKER_INVERSION_SAFETENSORS_SUFFIX!r} suffix: {target}"
        )
    normalized = normalize_speaker_inversion_payload(payload)
    tensors = {
        SPEAKER_EMBEDDING_KEY: normalized[SPEAKER_EMBEDDING_KEY].to(dtype=dtype),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_file(tensors, str(target), metadata={})


def blend_speaker_embeddings(
    embeddings: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    mode: str = "lerp",
) -> torch.Tensor:
    """Blend Speaker Inversion embeddings into a single (tokens, dim) tensor.

    Interpolating between learned speaker embeddings rides the base model's
    speaker-conditioning manifold, so the result is typically a natural
    intermediate voice.

    - ``lerp``: weighted mean of the token tensors (weights normalized to sum
      to 1). Requires every source to have the same token count. Assumes token
      slot i of one embedding corresponds to slot i of the other, which is not
      guaranteed for independently trained runs -- it usually works, but if
      the blended voice sounds muddy, try ``concat``.
    - ``concat``: concatenates the token sequences (no slot-correspondence
      assumption; token counts may differ). Each source's tokens are scaled by
      ``normalized_weight * num_sources`` so equal weights reproduce a plain
      concatenation. Experimental: scaling interacts nonlinearly with
      attention, and the grown token count changes conditioning shapes.
    """
    if mode not in SPEAKER_INVERSION_BLEND_MODES:
        raise ValueError(
            f"Unknown blend mode {mode!r}. Use one of: {', '.join(SPEAKER_INVERSION_BLEND_MODES)}."
        )
    embeddings = list(embeddings)
    weights = [float(w) for w in weights]
    if not embeddings:
        raise ValueError("Need at least one embedding to blend.")
    if len(weights) != len(embeddings):
        raise ValueError(
            f"weights count ({len(weights)}) must match embeddings count ({len(embeddings)})."
        )
    if any(w <= 0.0 for w in weights):
        raise ValueError("All blend weights must be > 0.")
    for index, embedding in enumerate(embeddings):
        if embedding.ndim != 2 or int(embedding.shape[0]) <= 0:
            raise ValueError(
                f"Blend source {index} must have shape (tokens, dim), "
                f"got {tuple(embedding.shape)}."
            )

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
                "speaker_inversion_tokens, or use the concat blend mode."
            )
        out = torch.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, normalized, strict=True):
            out += emb * weight
        return out

    scaled = [
        emb * (weight * len(embeddings)) for emb, weight in zip(embeddings, normalized, strict=True)
    ]
    return torch.cat(scaled, dim=0)


def speaker_inversion_batch_tensors(
    speaker_embedding: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = speaker_embedding.to(device=device, dtype=dtype)
    state = embedding[None, :, :].expand(int(batch_size), -1, -1)
    mask = torch.ones((int(batch_size), embedding.shape[0]), dtype=torch.bool, device=device)
    return state, mask


def speaker_inversion_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    module = getattr(model, "speaker_inversion", None)
    if not isinstance(module, SpeakerInversionEmbedding):
        raise ValueError("Model does not have an enabled SpeakerInversionEmbedding module.")

    return {
        SPEAKER_EMBEDDING_KEY: module.embedding.detach().cpu().float().clone(),
    }


def save_speaker_inversion_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = speaker_inversion_state_dict(model)
    save_speaker_inversion_safetensors(path, state)
