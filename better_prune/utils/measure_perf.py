from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "better_prune" / "models"

if __package__ in (None, ""):
    # Script executed directly; ensure project root is importable.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model_runner import MemoryMetrics, NativeModelRunner  # type: ignore
else:
    from ..model_runner import MemoryMetrics, NativeModelRunner


def parse_batch_sizes(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError("Batch sizes must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("At least one batch size must be provided.")
    return values


def build_runner(args: argparse.Namespace) -> NativeModelRunner:
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        trust_remote_code=not args.no_trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, object] = {
        "trust_remote_code": not args.no_trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if args.torch_dtype != "auto":
        try:
            model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown torch dtype: {args.torch_dtype}") from exc
    else:
        model_kwargs["torch_dtype"] = "auto"

    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        **model_kwargs,
    )
    if(args.load_state != ""):
        print("Attempting to load model ")
        loaded= torch.load(args.load_state)
        model.load_state_dict(loaded["model"], strict=False)
    model.compile()
    runner = NativeModelRunner(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        is_chat_ft=not args.disable_chat_template,
        generate_kwargs={},
    )
    runner.generate_kwargs.setdefault("do_sample", False)
    runner.generate_kwargs.setdefault("use_cache", True)
    return runner


def expand_prompts(prompts: Sequence[str], target_size: int) -> List[str]:
    if len(prompts) >= target_size:
        return list(prompts[:target_size])
    expanded: List[str] = []
    while len(expanded) < target_size:
        expanded.extend(prompts)
    return expanded[:target_size]

def set_measure_ctx() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def measure_for_batch_size(
    runner: NativeModelRunner,
    prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    latency_runs: int,
    throughput_runs: int,
    batch_runs: int,
) -> Dict[str, object]:
    batch_prompts = expand_prompts(prompts, batch_size)
    latency_prompt = batch_prompts[0]

    latency_ms = runner.measure_latency(latency_prompt, max_new_tokens=max_new_tokens, runs=latency_runs)
    throughput = runner.measure_throughput(batch_prompts, max_new_tokens=max_new_tokens, runs=throughput_runs)
    batch_time = runner.measure_batch_inference_time(batch_prompts, max_new_tokens=max_new_tokens, runs=batch_runs)
    memory = runner.measure_memory_usage(batch_prompts, max_new_tokens=max_new_tokens)

    return {
        "batch_size": batch_size,
        "latency_ms": latency_ms,
        "throughput_tokens_per_second": throughput,
        "batch_inference_time_ms": batch_time,
        "memory": memory,
    }


def format_metrics_row(metrics: Dict[str, object]) -> str:
    memory: MemoryMetrics | None = metrics["memory"]  # type: ignore[assignment]
    peak_mem = None
    current_mem = None
    device_type = "-"
    if isinstance(memory, MemoryMetrics):
        peak_mem = memory.peak_device_memory_mb or memory.peak_cpu_memory_mb
        current_mem = memory.current_device_memory_mb or memory.current_cpu_memory_mb
        device_type = memory.device_type

    def fmt(value: float | None) -> str:
        return f"{value:10.2f}" if value is not None else f"{'n/a':>10}"

    return (
        f"{metrics['batch_size']:>6d}"
        f"{metrics['latency_ms']:>14.2f}"
        f"{metrics['throughput_tokens_per_second']:>18.2f}"
        f"{metrics['batch_inference_time_ms']:>18.2f}"
        f"{fmt(peak_mem)}"
        f"{fmt(current_mem)}"
        f"{device_type:>10}"
    )


def dump_results_json(results: Iterable[Dict[str, object]], output_path: Path) -> None:
    serialisable: List[Dict[str, object]] = []
    for item in results:
        row = dict(item)
        memory = row.get("memory")
        if isinstance(memory, MemoryMetrics):
            row["memory"] = asdict(memory)
        serialisable.append(row)
    output_path.write_text(json.dumps(serialisable, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure NativeModelRunner performance across batch sizes.")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model identifier or local path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("1,2,4,8"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--throughput-runs", type=int, default=10)
    parser.add_argument("--batch-runs", type=int, default=5)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory to cache downloaded models (default: ~/.cache/better_prune/models).",
    )
    parser.add_argument("--load-state", type=str, default="", help="Optional path to load model state from")
    parser.add_argument("--device-map", type=str, default=None, help="Optional device map hint passed to transformers.")
    parser.add_argument("--torch-dtype", type=str, default="auto", help="torch dtype to load weights with (e.g. float16).")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trusting remote code when loading.")
    parser.add_argument("--disable-chat-template", action="store_true", help="Skip applying chat templates to prompts.")
    parser.add_argument("--json-output", type=str, default=None, help="Optional path to dump metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    torch.set_float32_matmul_precision('high')
    torch._logging.set_logs(graph_code=False)
    args = parse_args()
    runner = build_runner(args)
    base_prompts = list(runner.default_prompts())
    if not base_prompts:
        raise ValueError("Model runner must provide at least one default prompt to measure performance.")

    print(
        f"{'batch':>6} {'latency_ms':>14} {'throughput_tps':>18} {'batch_time_ms':>18} "
        f"{'peak_mem_mb':>10} {'curr_mem_mb':>10} {'device':>10}"
    )
    measurements: List[Dict[str, object]] = []
    for batch_size in args.batch_sizes:
        metrics = measure_for_batch_size(
            runner=runner,
            prompts=base_prompts,
            batch_size=batch_size,
            max_new_tokens=args.max_new_tokens,
            latency_runs=args.latency_runs,
            throughput_runs=args.throughput_runs,
            batch_runs=args.batch_runs,
        )
        measurements.append(metrics)
        print(format_metrics_row(metrics))

    if args.json_output:
        output_path = Path(args.json_output)
        dump_results_json(measurements, output_path)
        print(f"\nSaved detailed metrics to {output_path}")


if __name__ == "__main__":
    main()
