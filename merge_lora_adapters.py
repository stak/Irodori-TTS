#!/usr/bin/env python3
"""Merge multiple LoRA adapters into a single adapter directory (offline).

LoRA weights are combined with PEFT's ``add_weighted_adapter``. Fully saved
modules (``modules_to_save``, e.g. the duration predictor) are not low-rank
deltas and cannot be combined by PEFT, so they are replaced with a weighted
average of the source copies using weights normalized to sum to 1.

The merged directory has the same layout as a training adapter checkpoint and
can be passed directly to ``infer.py --lora-adapter`` or the server's
``irodori.lora_adapter`` request field. Base model weights are never needed:
both the LoRA combination and the full-module average operate on adapter
tensors only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from irodori_tts.config import ModelConfig
from irodori_tts.lora import (
    LORA_ADAPTER_CONFIG_NAME,
    LORA_ADAPTER_STATE_NAMES,
    LORA_METADATA_NAME,
    is_lora_adapter_dir,
    load_lora_adapter,
)
from irodori_tts.model import TextToLatentRFDiT

MERGED_ADAPTER_NAME = "merged"
RUN_CONFIG_NAME = "config.json"
SUPPORTED_COMBINATION_TYPES = ("cat", "linear", "svd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter checkpoints into a single adapter directory."
    )
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="LoRA adapter checkpoint directory. Repeat for each source adapter (>= 2).",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        required=True,
        help="Blend weight per adapter, in --adapter order (e.g. 0.7 0.3).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination directory for the merged adapter.",
    )
    parser.add_argument(
        "--combination-type",
        choices=SUPPORTED_COMBINATION_TYPES,
        default="cat",
        help=(
            "PEFT combination method. 'cat' concatenates ranks and is an exact "
            "weighted sum of the deltas (merged rank = sum of source ranks). "
            "'linear' keeps the source rank but introduces cross-term error. "
            "'svd' re-decomposes the summed delta at --svd-rank. Default: cat."
        ),
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=None,
        help="Target rank for --combination-type svd (default: PEFT's choice).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip reloading the merged adapter for verification.",
    )
    return parser.parse_args()


def _read_run_config(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / RUN_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Adapter checkpoint is missing {RUN_CONFIG_NAME} (needed for the model "
            f"architecture): {adapter_dir}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"Adapter run config must contain a model section: {config_path}")
    return payload


def _read_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    payload = json.loads((adapter_dir / LORA_ADAPTER_CONFIG_NAME).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Adapter config root must be a mapping: {adapter_dir}")
    return payload


def _read_adapter_metadata(adapter_dir: Path) -> dict[str, Any] | None:
    metadata_path = adapter_dir / LORA_METADATA_NAME
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_adapter_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    for name in LORA_ADAPTER_STATE_NAMES:
        path = adapter_dir / name
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            return load_file(str(path))
        return torch.load(path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in: {adapter_dir}")


def _validate_inputs(adapter_dirs: list[Path], weights: list[float], combination_type: str) -> None:
    if len(adapter_dirs) < 2:
        raise ValueError("Need at least two --adapter directories to merge.")
    if len(weights) != len(adapter_dirs):
        raise ValueError(
            f"--weights count ({len(weights)}) must match --adapter count ({len(adapter_dirs)})."
        )
    if any(w <= 0.0 for w in weights):
        raise ValueError("All --weights must be > 0.")
    for path in adapter_dirs:
        if not is_lora_adapter_dir(path):
            raise ValueError(f"Not a LoRA adapter checkpoint directory: {path}")

    configs = [_read_adapter_config(path) for path in adapter_dirs]
    reference = configs[0]
    for path, config in zip(adapter_dirs[1:], configs[1:], strict=False):
        if config.get("modules_to_save") != reference.get("modules_to_save"):
            raise ValueError(
                "Adapters disagree on modules_to_save; their full-module copies "
                f"cannot be averaged: {adapter_dirs[0]} vs {path}"
            )
        if config.get("target_modules") != reference.get("target_modules"):
            print(
                f"warning: target_modules differ between {adapter_dirs[0]} and {path}; "
                "PEFT merges over the union of targets."
            )
    for path, config in zip(adapter_dirs, configs, strict=False):
        if config.get("use_dora"):
            raise ValueError(f"DoRA adapters are not supported for merging: {path}")
        ranks = {config.get("r")} | set((config.get("rank_pattern") or {}).values())
        if combination_type == "linear" and ranks != {reference.get("r")}:
            raise ValueError(
                "combination_type=linear requires equal ranks across adapters; "
                "use cat or svd instead."
            )


def _average_full_modules(
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
    adapter_dirs: list[Path],
) -> dict[str, torch.Tensor]:
    """Weighted average of the non-LoRA (modules_to_save) tensors.

    Weights are normalized to sum to 1 so the result is always an
    interpolation of the source copies, independent of the raw weight scale
    used for the LoRA deltas.
    """
    key_sets = [frozenset(k for k in state if ".lora_" not in k) for state in states]
    for path, keys in zip(adapter_dirs[1:], key_sets[1:], strict=False):
        if keys != key_sets[0]:
            diff = sorted(keys ^ key_sets[0])
            raise ValueError(
                f"Adapters disagree on full-module tensors ({adapter_dirs[0]} vs {path}): "
                f"{diff[:8]}"
            )

    total = sum(weights)
    normalized = [w / total for w in weights]
    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(key_sets[0]):
        shapes = {tuple(state[key].shape) for state in states}
        if len(shapes) != 1:
            raise ValueError(f"Shape mismatch for full-module tensor {key}: {shapes}")
        acc = torch.zeros_like(states[0][key], dtype=torch.float32)
        for state, weight in zip(states, normalized, strict=False):
            acc += state[key].to(torch.float32) * weight
        averaged[key] = acc.contiguous()
    return averaged


def _build_base_model(adapter_dir: Path) -> TextToLatentRFDiT:
    run_config = _read_run_config(adapter_dir)
    model_cfg = ModelConfig(**run_config["model"])
    # Base weights are irrelevant here: merging only reads/writes adapter
    # tensors, so a randomly initialized skeleton is sufficient.
    return TextToLatentRFDiT(model_cfg)


def _stage_lora_only_adapters(
    tmp_root: Path,
    adapter_dirs: list[Path],
    states: list[dict[str, torch.Tensor]],
) -> list[Path]:
    """Write copies of the source adapters stripped down to their LoRA tensors.

    PEFT refuses ``add_weighted_adapter`` when the adapters share
    ``modules_to_save`` targets, so those tensors are removed here and merged
    separately (weighted average) at the file level.
    """
    staged_dirs: list[Path] = []
    for index, (path, state) in enumerate(zip(adapter_dirs, states, strict=False)):
        staged = tmp_root / f"src{index}"
        staged.mkdir()
        config = _read_adapter_config(path)
        config["modules_to_save"] = None
        (staged / LORA_ADAPTER_CONFIG_NAME).write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        save_file(
            {k: v.contiguous() for k, v in state.items() if ".lora_" in k},
            str(staged / "adapter_model.safetensors"),
        )
        staged_dirs.append(staged)
    return staged_dirs


def _merge_lora_weights(
    model: TextToLatentRFDiT,
    adapter_dirs: list[Path],
    states: list[dict[str, torch.Tensor]],
    weights: list[float],
    *,
    combination_type: str,
    svd_rank: int | None,
    tmp_root: Path,
) -> Any:
    staged_dirs = _stage_lora_only_adapters(tmp_root, adapter_dirs, states)
    peft_model = None
    adapter_names = [f"src{i}" for i in range(len(staged_dirs))]
    for name, path in zip(adapter_names, staged_dirs, strict=False):
        peft_model = load_lora_adapter(
            peft_model if peft_model is not None else model,
            path,
            is_trainable=False,
            adapter_name=name,
            torch_device="cpu",
        )

    kwargs: dict[str, Any] = {}
    if combination_type == "svd" and svd_rank is not None:
        kwargs["svd_rank"] = int(svd_rank)
    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=list(weights),
        adapter_name=MERGED_ADAPTER_NAME,
        combination_type=combination_type,
        **kwargs,
    )
    return peft_model


def _write_merged_dir(
    peft_model: Any,
    *,
    output_dir: Path,
    averaged_full: dict[str, torch.Tensor],
    adapter_dirs: list[Path],
    weights: list[float],
    combination_type: str,
    force: bool,
    tmp_root: Path,
) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output_dir} (use --force)")
        shutil.rmtree(output_dir)

    peft_model.save_pretrained(str(tmp_root), selected_adapters=[MERGED_ADAPTER_NAME])
    staged = tmp_root / MERGED_ADAPTER_NAME
    if not staged.is_dir():
        raise RuntimeError(f"PEFT did not write the merged adapter under: {staged}")

    state_path = staged / "adapter_model.safetensors"
    # Clone so no tensor keeps the source file memory-mapped while it is
    # overwritten below (Windows refuses to write to a mapped file).
    merged_state = {k: v.clone() for k, v in load_file(str(state_path)).items() if ".lora_" in k}
    # The staged sources were stripped of modules_to_save, so the merged
    # adapter carries none; add the weighted average of the source copies and
    # restore the config field so inference loads them again.
    merged_state.update(averaged_full)
    save_file({k: v.contiguous() for k, v in merged_state.items()}, str(state_path))

    merged_config = json.loads((staged / LORA_ADAPTER_CONFIG_NAME).read_text(encoding="utf-8"))
    merged_config["modules_to_save"] = _read_adapter_config(adapter_dirs[0]).get("modules_to_save")
    merged_config["inference_mode"] = True
    (staged / LORA_ADAPTER_CONFIG_NAME).write_text(
        json.dumps(merged_config, indent=2) + "\n", encoding="utf-8"
    )

    shutil.copyfile(adapter_dirs[0] / RUN_CONFIG_NAME, staged / RUN_CONFIG_NAME)
    _write_merge_metadata(
        staged,
        adapter_dirs=adapter_dirs,
        weights=weights,
        combination_type=combination_type,
    )
    shutil.move(str(staged), str(output_dir))


def _write_merge_metadata(
    staged: Path,
    *,
    adapter_dirs: list[Path],
    weights: list[float],
    combination_type: str,
) -> None:
    base_inits = [(_read_adapter_metadata(path) or {}).get("base_init") for path in adapter_dirs]
    reference = base_inits[0]
    for path, base_init in zip(adapter_dirs[1:], base_inits[1:], strict=False):
        if base_init != reference:
            print(
                f"warning: base_init metadata differs between {adapter_dirs[0]} and {path}; "
                "merging adapters trained on different base checkpoints is unsupported."
            )
            break

    total = sum(weights)
    payload: dict[str, Any] = {
        "merge": {
            "combination_type": combination_type,
            "sources": [
                {"adapter": str(path), "weight": float(weight)}
                for path, weight in zip(adapter_dirs, weights, strict=False)
            ],
            "full_module_weights": [float(w / total) for w in weights],
        }
    }
    if reference is not None:
        payload["base_init"] = reference
    (staged / LORA_METADATA_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _verify_merged_dir(output_dir: Path, averaged_full: dict[str, torch.Tensor]) -> None:
    if not is_lora_adapter_dir(output_dir):
        raise RuntimeError(f"Merged output is not a loadable adapter directory: {output_dir}")

    model = _build_base_model(output_dir)
    peft_model = load_lora_adapter(
        model, output_dir, is_trainable=False, adapter_name="default", torch_device="cpu"
    )
    state = peft_model.state_dict()
    module_names = _read_adapter_config(output_dir).get("modules_to_save") or []
    for key, expected in averaged_full.items():
        live_key = key
        for module_name in module_names:
            marker = f"{module_name}."
            if marker in live_key:
                live_key = live_key.replace(marker, f"{module_name}.modules_to_save.default.", 1)
                break
        live = state.get(live_key)
        if live is None:
            raise RuntimeError(f"Merged adapter did not load full-module tensor for: {key}")
        if not torch.allclose(live.to(torch.float32), expected, atol=1e-6):
            raise RuntimeError(f"Loaded full-module tensor differs from average: {key}")
    print(
        "Verified: merged adapter reloads and full-module tensors match "
        f"({len(averaged_full)} tensors)."
    )


def main() -> None:
    args = parse_args()
    adapter_dirs = [Path(p).expanduser() for p in args.adapter]
    weights = [float(w) for w in args.weights]
    output_dir = Path(args.output_dir).expanduser()

    _validate_inputs(adapter_dirs, weights, args.combination_type)
    if abs(sum(weights) - 1.0) > 1e-6:
        print(
            f"note: weights sum to {sum(weights):g} (not 1). LoRA deltas use the raw "
            "weights; full modules (duration predictor) always use weights normalized "
            "to sum to 1."
        )

    states = [_load_adapter_state(path) for path in adapter_dirs]
    averaged_full = _average_full_modules(states, weights, adapter_dirs)

    print(f"Building model skeleton from: {adapter_dirs[0] / RUN_CONFIG_NAME}")
    model = _build_base_model(adapter_dirs[0])
    print(f"Merging {len(adapter_dirs)} adapters (combination_type={args.combination_type}) ...")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output_dir.parent)) as tmp:
        tmp_root = Path(tmp)
        peft_model = _merge_lora_weights(
            model,
            adapter_dirs,
            states,
            weights,
            combination_type=args.combination_type,
            svd_rank=args.svd_rank,
            tmp_root=tmp_root,
        )
        _write_merged_dir(
            peft_model,
            output_dir=output_dir,
            averaged_full=averaged_full,
            adapter_dirs=adapter_dirs,
            weights=weights,
            combination_type=args.combination_type,
            force=bool(args.force),
            tmp_root=tmp_root,
        )

    merged_config = _read_adapter_config(output_dir)
    print(f"Saved merged adapter: {output_dir}")
    print(f"Merged rank: r={merged_config.get('r')} lora_alpha={merged_config.get('lora_alpha')}")
    if not args.skip_verify:
        _verify_merged_dir(output_dir, averaged_full)


if __name__ == "__main__":
    main()
