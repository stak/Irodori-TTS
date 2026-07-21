"""
Central resolution of the performance-fork feature toggles.

The fork's default-on optimizations that change outputs relative to upstream
Irodori-TTS are grouped behind a single profile switch:

- ``IRODORI_PERF_PROFILE=upstream`` (default): every toggle off. Outputs are
  bit-identical to unmodified upstream Irodori-TTS.
- ``IRODORI_PERF_PROFILE=recommended``: the fork's tuned set - TF32 matmul,
  per-request LoRA merge, CUDA graph replay (sampler and duration graphs),
  fp16 codec decode, and text-length bucketing (bucket 64).

The individual environment variables always take precedence over the profile,
in both directions: ``IRODORI_DISABLE_TF32=1`` forces TF32 off under
``recommended``, and ``IRODORI_DISABLE_TF32=0`` forces it on under
``upstream``. ``IRODORI_TEXT_BUCKETS`` overrides the bucket list the same way
(``0`` or empty-after-set disables bucketing).

``IRODORI_COMPILE`` (torch.compile) is a separate opt-in - it requires a
Triton toolchain - and is intentionally not part of any profile.
"""

from __future__ import annotations

import os

PROFILE_ENV = "IRODORI_PERF_PROFILE"
PROFILES = ("upstream", "recommended")


def profile_name() -> str:
    raw = os.environ.get(PROFILE_ENV, "").strip().lower()
    if raw == "":
        return "upstream"
    if raw not in PROFILES:
        raise ValueError(
            f"{PROFILE_ENV} must be one of {sorted(PROFILES)}, got {raw!r}"
        )
    return raw


def optimization_enabled(disable_env: str) -> bool:
    """
    Resolve one IRODORI_DISABLE_* toggle: an explicitly set variable wins,
    otherwise the optimization is on exactly under the recommended profile.
    """
    raw = os.environ.get(disable_env, "").strip()
    if raw != "":
        return raw != "1"
    return profile_name() == "recommended"


def text_buckets_raw() -> str:
    """
    The effective IRODORI_TEXT_BUCKETS value: the variable itself when set,
    otherwise the profile default ("64" recommended / "0" upstream).
    """
    if "IRODORI_TEXT_BUCKETS" in os.environ:
        return os.environ["IRODORI_TEXT_BUCKETS"].strip()
    return "64" if profile_name() == "recommended" else "0"
