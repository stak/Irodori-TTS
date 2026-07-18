"""Automated inference verification via the Gradio API.

Replaces the manual browser workflow (Load Model -> Prewarm Graphs -> Generate)
with a scripted run against the auto-generated Gradio endpoints
(/_load_model, /_prewarm_graphs, /_run_generation).

Typical usage (from the repo root):

    # Spawn a fresh server, prewarm, run 4 generations, save results under a tag
    uv run --no-sync python tools/verify_api.py run --tag baseline

    # Opt-out env vars are passed to the spawned server with --env
    uv run --no-sync python tools/verify_api.py run --tag no_graph \
        --env IRODORI_DISABLE_CUDA_GRAPH=1

    # Attach to an already-running server instead of spawning one
    uv run --no-sync python tools/verify_api.py run --attach --port 7861

    # Bit-compare / SNR between two tagged runs
    uv run --no-sync python tools/verify_api.py compare baseline no_graph

Timings are parsed from the Timing textbox output of /_run_generation and
written to <out-dir>/<tag>/summary.json together with copies of the wavs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

DEFAULT_TEXT = "いつまで待たせるつもり？ 早くしてよね。"
DEFAULT_SEED = "1"
DEFAULT_LORA = str(Path("outputs") / "lora_shiduka" / "checkpoint_final")

TIMING_LINE_RE = re.compile(r"^\[timing\] (?P<name>[^:]+): (?P<value>[-\d.]+) (?P<unit>ms|s)$")
SAVED_LINE_RE = re.compile(r"^saved\[\d+\]: (?P<path>.+)$")
SEED_LINE_RE = re.compile(r"^seed_used: (?P<seed>-?\d+)$")


def _log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def _server_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/"


def _is_server_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(_server_url(port) + "config", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _spawn_server(port: int, env_overrides: dict[str, str], log_path: Path) -> subprocess.Popen:
    if _is_server_up(port):
        raise RuntimeError(
            f"Port {port} already serves a Gradio app. Use --attach to reuse it, "
            "or pick another --port (a stale server may be running old code)."
        )
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    import os

    env = dict(os.environ)
    env.update(env_overrides)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [python, "gradio_app.py", "--server-name", "127.0.0.1", "--server-port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _log(f"spawned server pid={proc.pid} port={port} log={log_path}")
    return proc


def _wait_for_server(port: int, proc: subprocess.Popen | None, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with code {proc.returncode}; check the server log."
            )
        if _is_server_up(port):
            _log("server is up")
            return
        time.sleep(1.0)
    raise TimeoutError(f"Server did not become ready within {timeout_sec:.0f}s.")


def _resolve_endpoint(client, keyword: str) -> str:
    info = client.view_api(return_format="dict", print_info=False)
    names = list(info.get("named_endpoints", {}).keys())
    matches = [name for name in names if keyword in name]
    if not matches:
        raise RuntimeError(f"No API endpoint matching '{keyword}'. Available: {names}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous endpoint for '{keyword}': {matches}")
    return matches[0]


def _parse_timings(timing_text: str) -> dict[str, float]:
    """Parse the Timing textbox into {stage: milliseconds} (+ total_to_decode in ms)."""
    timings: dict[str, float] = {}
    for line in timing_text.splitlines():
        match = TIMING_LINE_RE.match(line.strip())
        if not match:
            continue
        value = float(match.group("value"))
        if match.group("unit") == "s":
            value *= 1000.0
        timings[match.group("name").strip()] = value
    return timings


def _parse_detail(detail_text: str) -> tuple[list[str], int | None]:
    wav_paths: list[str] = []
    seed_used: int | None = None
    for line in detail_text.splitlines():
        line = line.strip()
        saved = SAVED_LINE_RE.match(line)
        if saved:
            wav_paths.append(saved.group("path"))
        seed = SEED_LINE_RE.match(line)
        if seed:
            seed_used = int(seed.group("seed"))
    return wav_paths, seed_used


def cmd_run(args: argparse.Namespace) -> int:
    env_overrides: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env_overrides[key] = value

    out_dir = Path(args.out_dir)
    tag_dir = out_dir / args.tag if args.tag else None
    if tag_dir is not None and tag_dir.exists():
        raise SystemExit(f"Tag directory already exists: {tag_dir} (pick a new --tag or delete it)")

    proc: subprocess.Popen | None = None
    try:
        if not args.attach:
            log_path = out_dir / f"server_{args.tag or 'untagged'}_{args.port}.log"
            proc = _spawn_server(args.port, env_overrides, log_path)
        elif env_overrides:
            _log("warning: --env has no effect with --attach (server already running)")
        _wait_for_server(args.port, proc, timeout_sec=args.server_timeout)

        from gradio_client import Client

        client = Client(_server_url(args.port), verbose=False)
        ep_load = _resolve_endpoint(client, "load_model")
        ep_prewarm = _resolve_endpoint(client, "prewarm_graphs")
        ep_generate = _resolve_endpoint(client, "run_generation")

        t0 = time.monotonic()
        status = client.predict(model_precision=args.model_precision, api_name=ep_load)
        _log(f"load_model ({time.monotonic() - t0:.1f}s):\n{status}")

        if args.prewarm_seconds > 0:
            t0 = time.monotonic()
            status = client.predict(
                model_precision=args.model_precision,
                prewarm_seconds=float(args.prewarm_seconds),
                lora_adapter_raw=args.lora,
                api_name=ep_prewarm,
            )
            _log(f"prewarm ({time.monotonic() - t0:.1f}s):\n{status}")
        else:
            _log("prewarm skipped (--prewarm-seconds 0)")

        requests_data: list[dict] = []
        for req_idx in range(1, args.requests + 1):
            t0 = time.monotonic()
            result = client.predict(
                model_precision=args.model_precision,
                text=args.text,
                uploaded_audio=None,
                uploaded_speaker_embedding=None,
                seed_raw=args.seed,
                lora_adapter_raw=args.lora,
                api_name=ep_generate,
            )
            wall_sec = time.monotonic() - t0
            detail_text, timing_text = result[-2], result[-1]
            timings = _parse_timings(timing_text)
            wav_paths, seed_used = _parse_detail(detail_text)
            requests_data.append(
                {
                    "request": req_idx,
                    "wall_sec": wall_sec,
                    "seed_used": seed_used,
                    "timings_ms": timings,
                    "wav_paths": wav_paths,
                }
            )
            total_ms = timings.get("total_to_decode")
            _log(
                f"generate #{req_idx}: total_to_decode="
                f"{'?' if total_ms is None else f'{total_ms:.1f}ms'} wall={wall_sec:.2f}s"
            )
            if args.request_interval > 0 and req_idx < args.requests:
                time.sleep(args.request_interval)

        _print_summary(requests_data)

        if tag_dir is not None:
            _save_tagged_run(tag_dir, args, env_overrides, requests_data)
        return 0
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            _log("server stopped")
        elif proc is not None:
            _log(f"server left running (pid={proc.pid}, port={args.port})")


def _print_summary(requests_data: list[dict]) -> None:
    if not requests_data:
        return
    stages: list[str] = []
    for req in requests_data:
        for name in req["timings_ms"]:
            if name not in stages:
                stages.append(name)
    # First request may pay capture cost; report it separately from warm medians.
    warm = requests_data[1:] if len(requests_data) > 1 else requests_data
    _log("---- per-stage timings (ms) ----")
    header = f"{'stage':<28}" + "".join(f"req{r['request']:>2} ".rjust(10) for r in requests_data)
    header += f"{'warm-median':>12}"
    _log(header)
    for stage in stages:
        cells = ""
        for req in requests_data:
            value = req["timings_ms"].get(stage)
            cells += (f"{value:.1f}" if value is not None else "-").rjust(10)
        warm_values = [r["timings_ms"][stage] for r in warm if stage in r["timings_ms"]]
        median = f"{statistics.median(warm_values):.1f}" if warm_values else "-"
        _log(f"{stage:<28}{cells}{median:>12}")


def _save_tagged_run(
    tag_dir: Path,
    args: argparse.Namespace,
    env_overrides: dict[str, str],
    requests_data: list[dict],
) -> None:
    tag_dir.mkdir(parents=True, exist_ok=True)
    for req in requests_data:
        copied = []
        for cand_idx, src in enumerate(req["wav_paths"], start=1):
            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = REPO_ROOT / src_path
            dst = tag_dir / f"req{req['request']:02d}_cand{cand_idx:02d}.wav"
            shutil.copy2(src_path, dst)
            copied.append(dst.name)
        req["copied_wavs"] = copied
    summary = {
        "tag": args.tag,
        "text": args.text,
        "seed": args.seed,
        "model_precision": args.model_precision,
        "lora": args.lora,
        "prewarm_seconds": args.prewarm_seconds,
        "env_overrides": env_overrides,
        "requests": requests_data,
    }
    summary_path = tag_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"saved run to {tag_dir}")


def cmd_compare(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    dir_a, dir_b = out_dir / args.tag_a, out_dir / args.tag_b
    for d in (dir_a, dir_b):
        if not d.is_dir():
            raise SystemExit(f"Tag directory not found: {d}")
    wavs_a = sorted(dir_a.glob("*.wav"))
    if not wavs_a:
        raise SystemExit(f"No wavs under {dir_a}")
    all_bit_identical = True
    for wav_a in wavs_a:
        wav_b = dir_b / wav_a.name
        if not wav_b.exists():
            _log(f"{wav_a.name}: MISSING in {args.tag_b}")
            all_bit_identical = False
            continue
        if wav_a.read_bytes() == wav_b.read_bytes():
            _log(f"{wav_a.name}: bit-identical")
            continue
        all_bit_identical = False
        snr = _compute_snr(wav_a, wav_b)
        _log(f"{wav_a.name}: DIFFERS ({snr})")
    _log(
        f"result: {'all bit-identical' if all_bit_identical else 'differences found'} "
        f"({args.tag_a} vs {args.tag_b})"
    )
    return 0 if all_bit_identical else 1


def _compute_snr(ref_path: Path, test_path: Path) -> str:
    try:
        import numpy as np
        import soundfile as sf

        ref, sr_ref = sf.read(str(ref_path), dtype="float64")
        test, sr_test = sf.read(str(test_path), dtype="float64")
        if sr_ref != sr_test:
            return f"sample rate mismatch: {sr_ref} vs {sr_test}"
        note = ""
        if len(ref) != len(test):
            note = f", length mismatch {len(ref)} vs {len(test)} (SNR over overlap)"
            n = min(len(ref), len(test))
            ref, test = ref[:n], test[:n]
        noise = ref - test
        noise_power = float(np.sum(noise**2))
        if noise_power == 0.0:
            return f"samples identical{note}"
        signal_power = float(np.sum(ref**2))
        snr_db = 10.0 * np.log10(signal_power / noise_power)
        return f"SNR {snr_db:.1f} dB{note}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"SNR unavailable: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run load -> prewarm -> generate xN and report timings.")
    run.add_argument("--port", type=int, default=7861)
    run.add_argument("--attach", action="store_true", help="Reuse a running server on --port.")
    run.add_argument("--keep-server", action="store_true", help="Leave the spawned server running.")
    run.add_argument("--server-timeout", type=float, default=180.0)
    run.add_argument("--requests", type=int, default=4)
    run.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Seconds to sleep between generate calls (GPU downclock simulation).",
    )
    run.add_argument("--prewarm-seconds", type=float, default=15.0, help="0 skips prewarm.")
    run.add_argument("--text", default=DEFAULT_TEXT)
    run.add_argument("--seed", default=DEFAULT_SEED)
    run.add_argument("--model-precision", default="fp32", choices=["fp32", "bf16"])
    run.add_argument("--lora", default=DEFAULT_LORA)
    run.add_argument("--tag", default=None, help="Save wavs + summary.json under <out-dir>/<tag>.")
    run.add_argument("--out-dir", default=str(REPO_ROOT / "verify_runs"))
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Env var for the spawned server (repeatable), e.g. IRODORI_DISABLE_CUDA_GRAPH=1.",
    )
    run.set_defaults(func=cmd_run)

    comp = sub.add_parser("compare", help="Bit-compare / SNR between two tagged runs.")
    comp.add_argument("tag_a")
    comp.add_argument("tag_b")
    comp.add_argument("--out-dir", default=str(REPO_ROOT / "verify_runs"))
    comp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
