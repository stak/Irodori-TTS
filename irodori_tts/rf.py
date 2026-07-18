from __future__ import annotations

import math
import os

import torch

from .model import TextToLatentRFDiT
from .speaker_inversion import SPEAKER_INVERSION_UNCOND_MODES


def _make_rng(seed: int, device: torch.device) -> tuple[torch.Generator, torch.device]:
    # MPS generators are not available on some PyTorch builds; use CPU generator as fallback.
    try:
        return torch.Generator(device=device).manual_seed(seed), device
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(seed), torch.device("cpu")


def sample_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * std + mean
    t = torch.sigmoid(z)
    return t.clamp(min=t_min, max=t_max)


def sample_stratified_logit_normal_t(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    t_min: float = 1e-3,
    t_max: float = 0.999,
) -> torch.Tensor:
    """
    Stratified sampling for logit-normal timesteps.

    u ~ stratified U(0, 1), z = mean + std * Phi^{-1}(u), t = sigmoid(z)
    """
    if batch_size <= 0:
        return torch.empty((0,), device=device)
    u = (
        torch.arange(batch_size, device=device, dtype=torch.float32)
        + torch.rand(batch_size, device=device)
    ) / float(batch_size)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    z = torch.erfinv(2.0 * u - 1.0) * (2.0**0.5)
    z = z * std + mean
    t = torch.sigmoid(z)
    # Randomize assignment order so dataset ordering does not correlate with t bins.
    t = t[torch.randperm(batch_size, device=device)]
    return t.clamp(min=t_min, max=t_max)


def rf_interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Straight line interpolation: x_t = (1-t) x0 + t z.
    return (1.0 - t[:, None, None]) * x0 + t[:, None, None] * noise


def rf_velocity_target(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    # For x_t = (1-t) x0 + t z, velocity is d/dt x_t = z - x0.
    return noise - x0


def rf_predict_x0(x_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x_t = x0 + t * v  =>  x0 = x_t - t * v
    return x_t - t[:, None, None] * v_pred


def temporal_score_rescale(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: float | torch.Tensor,
    rescale_k: float,
    rescale_sigma: float,
) -> torch.Tensor:
    """
    Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
    """
    t_value = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    if t_value >= 1.0:
        return v_pred
    one_minus_t = 1.0 - t_value
    snr = (one_minus_t * one_minus_t) / (t_value * t_value)
    sigma_sq = float(rescale_sigma) * float(rescale_sigma)
    ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / float(rescale_k) + 1.0)
    return (ratio * (one_minus_t * v_pred + x_t) - x_t) / one_minus_t


def scale_speaker_kv_cache(
    context_kv_cache: list[tuple[torch.Tensor, ...]],
    scale: float,
    max_layers: int | None = None,
) -> None:
    """
    In-place scaling of speaker K/V tensors in precomputed context cache.
    """
    if max_layers is None:
        n_layers = len(context_kv_cache)
    else:
        n_layers = max(0, min(int(max_layers), len(context_kv_cache)))
    for i in range(n_layers):
        layer_kv = context_kv_cache[i]
        if len(layer_kv) < 4:
            raise ValueError(
                f"Expected at least 4 tensors in context KV cache entry, got {len(layer_kv)}"
            )
        k_speaker = layer_kv[2]
        v_speaker = layer_kv[3]
        k_speaker.mul_(scale)
        v_speaker.mul_(scale)


_WARMUP_STREAM: torch.cuda.Stream | None = None


def _get_warmup_stream() -> torch.cuda.Stream:
    """
    Reuse one side stream for all pre-capture warmup runs. Creating a fresh
    stream per capture makes the caching allocator strand each warmup's
    working set on a dead stream (blocks are only reused on the stream they
    were allocated on), which inflates VRAM by hundreds of MB per captured
    graph.
    """
    global _WARMUP_STREAM
    if _WARMUP_STREAM is None:
        _WARMUP_STREAM = torch.cuda.Stream()
    return _WARMUP_STREAM


@torch.inference_mode()
def sample_euler_rf_cfg(
    model: TextToLatentRFDiT,
    text_input_ids: torch.Tensor,
    text_mask: torch.Tensor,
    ref_latent: torch.Tensor | None,
    ref_mask: torch.Tensor | None,
    sequence_length: int,
    caption_input_ids: torch.Tensor | None = None,
    caption_mask: torch.Tensor | None = None,
    speaker_state_override: torch.Tensor | None = None,
    speaker_mask_override: torch.Tensor | None = None,
    speaker_uncond_mode: str = "mask",
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_caption: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_guidance_mode: str = "independent",
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    seed: int = 0,
    cfg_scale: float | None = None,
    truncation_factor: float | None = None,
    rescale_k: float | None = None,
    rescale_sigma: float | None = None,
    use_context_kv_cache: bool = True,
    speaker_kv_scale: float | None = None,
    speaker_kv_max_layers: int | None = None,
    speaker_kv_min_t: float | None = None,
    t_schedule_mode: str = "linear",
    sway_coeff: float = -1.0,
    cuda_graph_cache: dict | None = None,
    encoded_conditions: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    | None = None,
) -> torch.Tensor:
    """
    Euler sampling over RF ODE with text/reference/caption conditioning CFG.

    Args:
      encoded_conditions: Optional precomputed output of model.encode_conditions with
        identical conditioning arguments; avoids re-encoding when the caller already
        encoded conditions (e.g. for duration prediction).

    Returns:
      latent sequence in patched space, shape (B, sequence_length, patched_latent_dim)
    """
    device = model.device
    dtype = model.dtype
    batch_size = text_input_ids.shape[0]
    latent_dim = model.cfg.patched_latent_dim

    rng, rng_device = _make_rng(seed=seed, device=device)
    x_t = torch.randn(
        (batch_size, sequence_length, latent_dim), device=rng_device, dtype=dtype, generator=rng
    )
    if rng_device != device:
        x_t = x_t.to(device=device)
    if truncation_factor is not None:
        x_t = x_t * float(truncation_factor)

    if cfg_scale is not None:
        # Backward compatibility for old single-scale caller.
        cfg_scale_text = float(cfg_scale)
        cfg_scale_caption = float(cfg_scale)
        cfg_scale_speaker = float(cfg_scale)
    if not model.cfg.use_speaker_condition_resolved:
        cfg_scale_speaker = 0.0
        speaker_kv_scale = None
    speaker_uncond_mode = str(speaker_uncond_mode).strip().lower()
    if speaker_uncond_mode not in SPEAKER_INVERSION_UNCOND_MODES:
        raise ValueError(
            f"speaker_uncond_mode must be one of {sorted(SPEAKER_INVERSION_UNCOND_MODES)}, "
            f"got {speaker_uncond_mode!r}"
        )

    cfg_guidance_mode = str(cfg_guidance_mode).strip().lower()
    if cfg_guidance_mode not in {"independent", "joint", "alternating"}:
        raise ValueError(
            f"Unsupported cfg_guidance_mode={cfg_guidance_mode!r}. "
            "Expected one of: independent, joint, alternating."
        )

    init_scale = 0.999
    t_schedule_mode_norm = str(t_schedule_mode).strip().lower()
    sway_coeff_value = float(sway_coeff)
    if not math.isfinite(sway_coeff_value):
        raise ValueError(f"sway_coeff must be finite, got {sway_coeff!r}.")
    if t_schedule_mode_norm == "linear":
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    elif t_schedule_mode_norm == "sway":
        # F5-TTS-style Sway Sampling. Negative sway_coeff densifies the noise
        # side of the schedule (early steps); positive densifies the data side.
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        u = u + sway_coeff_value * (torch.cos(0.5 * math.pi * u) + u - 1.0)
        u = u.clamp(0.0, 1.0)
    else:
        raise ValueError(
            f"Unsupported t_schedule_mode={t_schedule_mode!r}. Expected 'linear' or 'sway'."
        )
    t_schedule = (1.0 - u) * init_scale
    if not bool(torch.all(t_schedule[:-1] > t_schedule[1:]).item()):
        raise ValueError("t_schedule must be strictly decreasing; adjust num_steps or sway_coeff.")
    # Materialize schedule as Python floats once, so the sampling loop below does not
    # trigger a device sync per step via Tensor.item().
    t_values = [float(v) for v in t_schedule.tolist()]
    use_independent_cfg = cfg_guidance_mode == "independent"
    use_joint_cfg = cfg_guidance_mode == "joint"
    use_alternating_cfg = cfg_guidance_mode == "alternating"

    if encoded_conditions is None:
        encoded_conditions = model.encode_conditions(
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_input_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state_override,
            speaker_mask_override=speaker_mask_override,
            speaker_uncond_mode=speaker_uncond_mode,
        )
    (
        text_state_cond,
        text_mask_cond,
        speaker_state_cond,
        speaker_mask_cond,
        caption_state_cond,
        caption_mask_cond,
    ) = encoded_conditions
    text_state_uncond = torch.zeros_like(text_state_cond)
    text_mask_uncond = torch.zeros_like(text_mask_cond)
    speaker_state_uncond = None
    speaker_mask_uncond = None
    if model.cfg.use_speaker_condition_resolved:
        if speaker_state_cond is None or speaker_mask_cond is None:
            raise RuntimeError(
                "Speaker conditioning is enabled but encoded speaker state is missing."
            )
        if speaker_uncond_mode == "noise":
            speaker_noise = torch.randn(
                speaker_state_cond.shape,
                device=rng_device,
                dtype=speaker_state_cond.dtype,
                generator=rng,
            )
            if rng_device != device:
                speaker_noise = speaker_noise.to(device=device)
            speaker_state_uncond = speaker_noise * speaker_state_cond.std().clamp_min(1e-6)
            speaker_mask_uncond = torch.ones_like(speaker_mask_cond)
        else:
            speaker_state_uncond = torch.zeros_like(speaker_state_cond)
            speaker_mask_uncond = torch.zeros_like(speaker_mask_cond)
    caption_state_uncond = None
    caption_mask_uncond = None
    if model.cfg.use_caption_condition:
        if caption_state_cond is None or caption_mask_cond is None:
            raise RuntimeError(
                "Caption conditioning is enabled but encoded caption state is missing."
            )
        caption_state_uncond = torch.zeros_like(caption_state_cond)
        caption_mask_uncond = torch.zeros_like(caption_mask_cond)

    has_text_cfg = cfg_scale_text > 0
    has_caption_cfg = (
        model.cfg.use_caption_condition
        and cfg_scale_caption > 0
        and caption_mask_cond is not None
        and bool(caption_mask_cond.any().item())
    )
    has_speaker_cfg = cfg_scale_speaker > 0

    def _bundle(
        *,
        text_state: torch.Tensor,
        text_mask_val: torch.Tensor,
        speaker_state: torch.Tensor | None,
        speaker_mask_val: torch.Tensor | None,
        caption_state: torch.Tensor | None,
        caption_mask_val: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        return (
            text_state,
            text_mask_val,
            speaker_state,
            speaker_mask_val,
            caption_state,
            caption_mask_val,
        )

    cond_bundle = _bundle(
        text_state=text_state_cond,
        text_mask_val=text_mask_cond,
        speaker_state=speaker_state_cond,
        speaker_mask_val=speaker_mask_cond,
        caption_state=caption_state_cond,
        caption_mask_val=caption_mask_cond,
    )
    enabled_cfg_names: list[str] = []
    cfg_scales: dict[str, float] = {}
    if has_text_cfg:
        enabled_cfg_names.append("text")
        cfg_scales["text"] = float(cfg_scale_text)
    if has_speaker_cfg:
        enabled_cfg_names.append("speaker")
        cfg_scales["speaker"] = float(cfg_scale_speaker)
    if has_caption_cfg:
        enabled_cfg_names.append("caption")
        cfg_scales["caption"] = float(cfg_scale_caption)

    independent_bundles = [cond_bundle]
    independent_names = ["cond"]
    if use_independent_cfg:
        for name in enabled_cfg_names:
            independent_names.append(name)
            independent_bundles.append(
                _bundle(
                    text_state=text_state_uncond if name == "text" else text_state_cond,
                    text_mask_val=text_mask_uncond if name == "text" else text_mask_cond,
                    speaker_state=(
                        speaker_state_uncond if name == "speaker" else speaker_state_cond
                    ),
                    speaker_mask_val=(
                        speaker_mask_uncond if name == "speaker" else speaker_mask_cond
                    ),
                    caption_state=(
                        caption_state_uncond if name == "caption" else caption_state_cond
                    ),
                    caption_mask_val=(
                        caption_mask_uncond if name == "caption" else caption_mask_cond
                    ),
                )
            )
    cfg_batch_mult = len(independent_bundles)

    def _cat_optional_tensors(values: list[torch.Tensor | None]) -> torch.Tensor | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        if len(present) != len(values):
            raise ValueError("Cannot concatenate optional condition tensors with mixed presence.")
        return torch.cat(present, dim=0)

    independent_text_state = torch.cat([bundle[0] for bundle in independent_bundles], dim=0)
    independent_text_mask = torch.cat([bundle[1] for bundle in independent_bundles], dim=0)
    independent_speaker_state = _cat_optional_tensors([bundle[2] for bundle in independent_bundles])
    independent_speaker_mask = _cat_optional_tensors([bundle[3] for bundle in independent_bundles])
    independent_caption_state = _cat_optional_tensors([bundle[4] for bundle in independent_bundles])
    independent_caption_mask = _cat_optional_tensors([bundle[5] for bundle in independent_bundles])

    joint_uncond_bundle = _bundle(
        text_state=text_state_uncond,
        text_mask_val=text_mask_uncond,
        speaker_state=speaker_state_uncond,
        speaker_mask_val=speaker_mask_uncond,
        caption_state=caption_state_uncond,
        caption_mask_val=caption_mask_uncond,
    )

    alternating_bundles: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ] = {
        "text": _bundle(
            text_state=text_state_uncond,
            text_mask_val=text_mask_uncond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        ),
        "caption": _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_cond,
            speaker_mask_val=speaker_mask_cond,
            caption_state=caption_state_uncond,
            caption_mask_val=caption_mask_uncond,
        ),
    }
    if has_speaker_cfg:
        alternating_bundles["speaker"] = _bundle(
            text_state=text_state_cond,
            text_mask_val=text_mask_cond,
            speaker_state=speaker_state_uncond,
            speaker_mask_val=speaker_mask_uncond,
            caption_state=caption_state_cond,
            caption_mask_val=caption_mask_cond,
        )

    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))

    context_kv_cond = None
    context_kv_cfg = None
    context_kv_joint_uncond = None
    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
    if effective_use_context_kv_cache:
        context_kv_cond = model.build_context_kv_cache(
            text_state=text_state_cond,
            speaker_state=speaker_state_cond,
            caption_state=caption_state_cond,
        )
        if use_independent_cfg and cfg_batch_mult > 1:
            context_kv_cfg = model.build_context_kv_cache(
                text_state=independent_text_state,
                speaker_state=independent_speaker_state,
                caption_state=independent_caption_state,
            )
        elif use_joint_cfg:
            if enabled_cfg_names:
                context_kv_joint_uncond = model.build_context_kv_cache(
                    text_state=joint_uncond_bundle[0],
                    speaker_state=joint_uncond_bundle[2],
                    caption_state=joint_uncond_bundle[4],
                )
        elif use_alternating_cfg:
            for name in enabled_cfg_names:
                bundle = alternating_bundles[name]
                context_kv_alternating[name] = model.build_context_kv_cache(
                    text_state=bundle[0],
                    speaker_state=bundle[2],
                    caption_state=bundle[4],
                )
    if speaker_kv_scale is not None:
        scale_speaker_kv_cache(
            context_kv_cache=context_kv_cond,
            scale=float(speaker_kv_scale),
            max_layers=speaker_kv_max_layers,
        )
        if context_kv_cfg is not None:
            scale_speaker_kv_cache(
                context_kv_cache=context_kv_cfg,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
        for cache in context_kv_alternating.values():
            scale_speaker_kv_cache(
                context_kv_cache=cache,
                scale=float(speaker_kv_scale),
                max_layers=speaker_kv_max_layers,
            )
    speaker_kv_active = speaker_kv_scale is not None

    # Optional padding mask for CUDA-graph length bucketing (set below). Masked
    # (padded) latent positions are excluded from attention keys, so outputs at
    # valid positions are bit-identical to an unpadded run.
    latent_mask: torch.Tensor | None = None

    def _run_sampling_loop(x_start: torch.Tensor) -> torch.Tensor:
        nonlocal speaker_kv_active
        if latent_mask is not None and cfg_batch_mult > 1:
            latent_mask_rep = torch.cat([latent_mask] * cfg_batch_mult, dim=0)
        else:
            latent_mask_rep = latent_mask
        x_t = x_start
        for i in range(num_steps):
            t = t_values[i]
            t_next = t_values[i + 1]
            tt = torch.full((batch_size,), t, device=device, dtype=dtype)

            use_cfg = bool(enabled_cfg_names) and (cfg_min_t <= t <= cfg_max_t)
            if use_cfg and use_independent_cfg:
                x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
                tt_cfg = tt.repeat(cfg_batch_mult)
                v_out = model.forward_with_encoded_conditions(
                    x_t=x_t_cfg,
                    t=tt_cfg,
                    text_state=independent_text_state,
                    text_mask=independent_text_mask,
                    speaker_state=independent_speaker_state,
                    speaker_mask=independent_speaker_mask,
                    caption_state=independent_caption_state,
                    caption_mask=independent_caption_mask,
                    latent_mask=latent_mask_rep,
                    context_kv_cache=context_kv_cfg,
                )
                chunks = v_out.chunk(cfg_batch_mult, dim=0)
                v = chunks[0]
                for name, chunk in zip(independent_names[1:], chunks[1:], strict=True):
                    v = v + cfg_scales[name] * (chunks[0] - chunk)
            elif use_cfg:
                v_cond = model.forward_with_encoded_conditions(
                    x_t=x_t.to(dtype),
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    caption_state=caption_state_cond,
                    caption_mask=caption_mask_cond,
                    latent_mask=latent_mask,
                    context_kv_cache=context_kv_cond,
                )
                if use_joint_cfg:
                    if len(enabled_cfg_names) > 1:
                        joint_scales = [cfg_scales[name] for name in enabled_cfg_names]
                        if max(joint_scales) - min(joint_scales) > 1e-6:
                            raise ValueError(
                                "cfg_guidance_mode='joint' expects equal enabled guidance scales; "
                                "set matching text/speaker/caption scales or use --cfg-scale."
                            )
                    joint_scale = cfg_scales[enabled_cfg_names[0]]
                    v_uncond_joint = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=joint_uncond_bundle[0],
                        text_mask=joint_uncond_bundle[1],
                        speaker_state=joint_uncond_bundle[2],
                        speaker_mask=joint_uncond_bundle[3],
                        caption_state=joint_uncond_bundle[4],
                        caption_mask=joint_uncond_bundle[5],
                        latent_mask=latent_mask,
                        context_kv_cache=context_kv_joint_uncond,
                    )
                    v = v_cond + joint_scale * (v_cond - v_uncond_joint)
                elif use_alternating_cfg:
                    alt_name = enabled_cfg_names[i % len(enabled_cfg_names)]
                    alt_bundle = alternating_bundles[alt_name]
                    v_uncond_alt = model.forward_with_encoded_conditions(
                        x_t=x_t.to(dtype),
                        t=tt,
                        text_state=alt_bundle[0],
                        text_mask=alt_bundle[1],
                        speaker_state=alt_bundle[2],
                        speaker_mask=alt_bundle[3],
                        caption_state=alt_bundle[4],
                        caption_mask=alt_bundle[5],
                        latent_mask=latent_mask,
                        context_kv_cache=context_kv_alternating.get(alt_name),
                    )
                    v = v_cond + cfg_scales[alt_name] * (v_cond - v_uncond_alt)
                else:
                    raise RuntimeError(f"Unexpected cfg_guidance_mode: {cfg_guidance_mode}")
            else:
                v = model.forward_with_encoded_conditions(
                    x_t=x_t.to(dtype),
                    t=tt,
                    text_state=text_state_cond,
                    text_mask=text_mask_cond,
                    speaker_state=speaker_state_cond,
                    speaker_mask=speaker_mask_cond,
                    caption_state=caption_state_cond,
                    caption_mask=caption_mask_cond,
                    latent_mask=latent_mask,
                    context_kv_cache=context_kv_cond,
                )

            if rescale_k is not None and rescale_sigma is not None:
                v = temporal_score_rescale(
                    v_pred=v,
                    x_t=x_t,
                    t=t,
                    rescale_k=float(rescale_k),
                    rescale_sigma=float(rescale_sigma),
                )

            if (
                speaker_kv_active
                and speaker_kv_min_t is not None
                and (t_next < speaker_kv_min_t)
                and (t >= speaker_kv_min_t)
            ):
                inv_scale = 1.0 / float(speaker_kv_scale)
                scale_speaker_kv_cache(
                    context_kv_cache=context_kv_cond,
                    scale=inv_scale,
                    max_layers=speaker_kv_max_layers,
                )
                if context_kv_cfg is not None:
                    scale_speaker_kv_cache(
                        context_kv_cache=context_kv_cfg,
                        scale=inv_scale,
                        max_layers=speaker_kv_max_layers,
                    )
                for cache in context_kv_alternating.values():
                    scale_speaker_kv_cache(
                        context_kv_cache=cache,
                        scale=inv_scale,
                        max_layers=speaker_kv_max_layers,
                    )
                speaker_kv_active = False

            x_t = x_t + v * (t_next - t)

        return x_t

    # ---- CUDA Graph fast path -------------------------------------------------
    # On Windows (no Triton/torch.compile) the sampling loop is dominated by
    # per-kernel CPU launch overhead. Capturing the whole Euler loop as a CUDA
    # graph and replaying it removes that overhead. The replayed kernels are
    # bit-identical to eager execution, so output audio does not change.
    graph_supported = (
        cuda_graph_cache is not None
        and device.type == "cuda"
        and cfg_guidance_mode == "independent"
        and rescale_k is None
        and speaker_kv_scale is None
        and os.environ.get("IRODORI_DISABLE_CUDA_GRAPH", "0").strip() != "1"
        and not cuda_graph_cache.get("__disabled__", False)
    )
    if not graph_supported:
        return _run_sampling_loop(x_t)

    # Length bucketing: pad the latent sequence up to a multiple of
    # IRODORI_CUDA_GRAPH_BUCKET patched steps and mask the padded tail, so texts
    # with similar predicted durations share one captured graph instead of each
    # unique length triggering a fresh (slow) capture. Masked positions cannot
    # influence valid positions, and the valid-region noise is unchanged, so
    # results are bit-identical to an unpadded run.
    bucket = max(1, int(os.environ.get("IRODORI_CUDA_GRAPH_BUCKET", "16")))
    valid_length = int(sequence_length)
    padded_length = ((valid_length + bucket - 1) // bucket) * bucket
    if padded_length != valid_length:
        pad_tail = torch.zeros(
            (batch_size, padded_length - valid_length, latent_dim),
            device=device,
            dtype=x_t.dtype,
        )
        x_t = torch.cat([x_t, pad_tail], dim=1)
    latent_mask = torch.zeros((batch_size, padded_length), dtype=torch.bool, device=device)
    latent_mask[:, :valid_length] = True

    def _flatten_kv(kv: list[tuple[torch.Tensor, ...]] | None) -> list[torch.Tensor]:
        if kv is None:
            return []
        return [tensor for layer in kv for tensor in layer]

    # Condition tensors (text/speaker/caption states and projected KV caches) do
    # not depend on the latent length, so a single shared set of static buffers
    # serves every captured graph. VRAM therefore does not grow with the number
    # of cached graphs; each graph only owns its (tiny) x_t/latent_mask buffers.
    named_cond = [
        independent_text_state,
        independent_text_mask,
        independent_speaker_state,
        independent_speaker_mask,
        independent_caption_state,
        independent_caption_mask,
        text_state_cond,
        text_mask_cond,
        speaker_state_cond,
        speaker_mask_cond,
        caption_state_cond,
        caption_mask_cond,
    ]
    presence = tuple(value is not None for value in named_cond)
    flat_named = [value for value in named_cond if value is not None]
    kv_cfg_arity = None if context_kv_cfg is None else tuple(len(layer) for layer in context_kv_cfg)
    kv_cond_arity = (
        None if context_kv_cond is None else tuple(len(layer) for layer in context_kv_cond)
    )
    cond_flat = [*flat_named, *_flatten_kv(context_kv_cfg), *_flatten_kv(context_kv_cond)]
    cond_sig = (
        presence,
        kv_cfg_arity,
        kv_cond_arity,
        tuple((tuple(tensor.shape), str(tensor.dtype)) for tensor in cond_flat),
    )

    shared_bundles = cuda_graph_cache.setdefault("__cond__", {})
    shared_flat = shared_bundles.get(cond_sig)
    if shared_flat is None:
        # Adopt this request's tensors as the shared static buffers.
        shared_bundles[cond_sig] = cond_flat
        shared_flat = cond_flat
    else:
        # Copy this request's condition values into the shared buffers and
        # rebind the loop variables so the (re)captured/replayed graph reads
        # from the shared storage.
        for static_buf, new_value in zip(shared_flat, cond_flat, strict=True):
            static_buf.copy_(new_value)
        buffer_iter = iter(shared_flat)
        rebound = [next(buffer_iter) if flag else None for flag in presence]
        (
            independent_text_state,
            independent_text_mask,
            independent_speaker_state,
            independent_speaker_mask,
            independent_caption_state,
            independent_caption_mask,
            text_state_cond,
            text_mask_cond,
            speaker_state_cond,
            speaker_mask_cond,
            caption_state_cond,
            caption_mask_cond,
        ) = rebound
        pos = len(flat_named)
        if kv_cfg_arity is not None:
            rebuilt: list[tuple[torch.Tensor, ...]] = []
            for arity in kv_cfg_arity:
                rebuilt.append(tuple(shared_flat[pos : pos + arity]))
                pos += arity
            context_kv_cfg = rebuilt
        if kv_cond_arity is not None:
            rebuilt = []
            for arity in kv_cond_arity:
                rebuilt.append(tuple(shared_flat[pos : pos + arity]))
                pos += arity
            context_kv_cond = rebuilt

    # Note: num_steps, cfg_min_t/max_t, t-schedule and sway are deliberately NOT
    # part of the key. Per-step graphs receive t/dt via input buffers, and the
    # schedule only determines the Python-side replay sequence, so cached graphs
    # remain valid when those sampling parameters change. Only the CFG scales
    # (baked into the guidance combine) and tensor shapes identify a graph.
    graph_key = (
        int(batch_size),
        int(padded_length),
        str(dtype),
        tuple(independent_names),
        tuple(sorted(cfg_scales.items())),
        bool(effective_use_context_kv_cache),
        cond_sig,
    )

    # Per-STEP CUDA graphs: capturing one sampling step (instead of the whole
    # num_steps loop) keeps the driver-side cudaGraphExec memory ~num_steps
    # times smaller while still removing almost all kernel-launch overhead.
    # The Python loop between replays only issues two tiny fill kernels.
    step_is_cfg = [
        bool(enabled_cfg_names) and (cfg_min_t <= t_values[i] <= cfg_max_t)
        for i in range(num_steps)
    ]

    def _replay_entry(entry: dict) -> torch.Tensor:
        tt_buf = entry["tt"]
        dt_buf = entry["dt"]
        graph_cfg = entry["graph_cfg"]
        graph_plain = entry["graph_plain"]
        for i in range(num_steps):
            tt_buf.fill_(t_values[i])
            dt_buf.fill_(t_values[i + 1] - t_values[i])
            if step_is_cfg[i]:
                graph_cfg.replay()
            else:
                graph_plain.replay()
        return entry["x"]

    entry = cuda_graph_cache.get(graph_key)
    if entry is not None:
        entry["x"].copy_(x_t)
        entry["mask"].copy_(latent_mask)
        out = _replay_entry(entry)
        return out[:, :valid_length].clone()

    x_initial = x_t.clone()
    warm_out: torch.Tensor | None = None
    try:
        tt_static = torch.empty((batch_size,), device=device, dtype=dtype)
        dt_static = torch.empty((), device=device, dtype=dtype)

        def _step_cfg() -> None:
            x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
            tt_cfg = tt_static.repeat(cfg_batch_mult)
            latent_mask_rep = torch.cat([latent_mask] * cfg_batch_mult, dim=0)
            v_out = model.forward_with_encoded_conditions(
                x_t=x_t_cfg,
                t=tt_cfg,
                text_state=independent_text_state,
                text_mask=independent_text_mask,
                speaker_state=independent_speaker_state,
                speaker_mask=independent_speaker_mask,
                caption_state=independent_caption_state,
                caption_mask=independent_caption_mask,
                latent_mask=latent_mask_rep,
                context_kv_cache=context_kv_cfg,
            )
            chunks = v_out.chunk(cfg_batch_mult, dim=0)
            v = chunks[0]
            for name, chunk in zip(independent_names[1:], chunks[1:], strict=True):
                v = v + cfg_scales[name] * (chunks[0] - chunk)
            x_t.add_(v * dt_static)

        def _step_plain() -> None:
            v = model.forward_with_encoded_conditions(
                x_t=x_t.to(dtype),
                t=tt_static,
                text_state=text_state_cond,
                text_mask=text_mask_cond,
                speaker_state=speaker_state_cond,
                speaker_mask=speaker_mask_cond,
                caption_state=caption_state_cond,
                caption_mask=caption_mask_cond,
                latent_mask=latent_mask,
                context_kv_cache=context_kv_cond,
            )
            x_t.add_(v * dt_static)

        def _run_step_loop() -> torch.Tensor:
            for i in range(num_steps):
                tt_static.fill_(t_values[i])
                dt_static.fill_(t_values[i + 1] - t_values[i])
                if step_is_cfg[i]:
                    _step_cfg()
                else:
                    _step_plain()
            return x_t

        # Both step variants are captured regardless of whether the current
        # schedule uses them, so the cached entry stays valid when num_steps,
        # the t-schedule, or cfg_min_t/max_t change later.
        capture_cfg_variant = bool(enabled_cfg_names)

        # Warmup on a shared side stream (also produces a correct result we can
        # fall back to if graph capture is unsupported on this platform).
        warm_stream = _get_warmup_stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            warm_out = _run_step_loop().clone()
            # Ensure cuBLAS/cuDNN are initialized for BOTH variants before
            # capture, even if the current schedule exercises only one kind.
            if capture_cfg_variant:
                _step_cfg()
            _step_plain()
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize(device)

        # All graphs share one memory pool: intermediates of different graphs
        # reuse the same VRAM (replays are serialized by the runtime lock, and
        # outputs are cloned immediately after each replay sequence).
        pool = cuda_graph_cache.get("__pool__")
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            cuda_graph_cache["__pool__"] = pool
        graph_cfg = None
        if capture_cfg_variant:
            graph_cfg = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph_cfg, pool=pool):
                _step_cfg()
        graph_plain = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_plain, pool=pool):
            _step_plain()

        entry = {
            "x": x_t,
            "mask": latent_mask,
            "tt": tt_static,
            "dt": dt_static,
            "graph_cfg": graph_cfg,
            "graph_plain": graph_plain,
        }
        # Validate one full replay sequence against the eager warmup result.
        x_t.copy_(x_initial)
        replay_out = _replay_entry(entry)
        torch.cuda.synchronize(device)
        if not torch.equal(replay_out, warm_out):
            raise RuntimeError("CUDA graph replay mismatch against eager warmup output.")
    except Exception as exc:
        cuda_graph_cache["__disabled__"] = True
        print(f"[rf] CUDA graph capture disabled: {exc!r}", flush=True)
        if warm_out is not None:
            return warm_out[:, :valid_length].clone()
        x_t.copy_(x_initial)
        return _run_sampling_loop(x_t)[:, :valid_length].clone()

    max_entries = max(1, int(os.environ.get("IRODORI_CUDA_GRAPH_CACHE", "64")))
    graph_keys = [key for key in cuda_graph_cache if not isinstance(key, str)]
    while len(graph_keys) >= max_entries:
        cuda_graph_cache.pop(graph_keys.pop(0), None)
    cuda_graph_cache[graph_key] = entry
    print(
        f"[rf] captured CUDA step graphs: padded_length={padded_length} "
        f"(entries cached: {len(graph_keys) + 1})",
        flush=True,
    )
    return replay_out[:, :valid_length].clone()
