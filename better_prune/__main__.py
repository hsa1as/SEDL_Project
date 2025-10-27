import argparse
from typing import Callable, Dict, List, Sequence

import torch

from better_prune.benchmarks import BenchmarkSuite
from better_prune.benchmarks.tasks import ARCTask, BoolQTask, GSM8KTask, MMLUTask
from better_prune.model_runner import get_model
from better_prune.pruner import calibrate
from better_prune.pruner.obc import add_actprune, disable_act_sparsity
from better_prune.utils.data_utils import get_loaders
from better_prune.utils.measure_perf import (
    format_metrics_row,
    measure_for_batch_size,
    set_measure_ctx,
)

SEQLEN = 2048
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DEFAULT_DATASET = "c4"
DEFAULT_CALIBRATION_SAMPLES = 128
DEFAULT_BATCH_SIZES = (64, 128, 256)
DEFAULT_MAX_TOKENS_PERF = 32
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TASKS = ("boolq","arc")
LATENCY_RUNS = 5
THROUGHPUT_RUNS = 10
BATCH_RUNS = 5

TaskFactory = Callable[[], object]

TASK_FACTORIES: Dict[str, TaskFactory] = {
    "gsm8k": lambda: GSM8KTask(limit=100, num_fewshot=8, batch_size=8),
    "boolq": lambda: BoolQTask(limit=100, num_fewshot=8, batch_size=10),
    "arc": lambda: ARCTask(variant="arc_challenge", limit=100, num_fewshot=25, batch_size=10),
    "mmlu": lambda: MMLUTask(subjects=("math",), limit=100, num_fewshot=5, batch_size=10),
}


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BetterPrune calibration, performance, and benchmark workflows.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Model identifier to load.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_FACTORIES.keys()),
        default=list(DEFAULT_TASKS),
        help="Benchmark tasks to run.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        metavar="N",
        help="Batch sizes to measure throughput and latency for.",
    )
    parser.add_argument(
        "--max-tokens-perf",
        type=parse_positive_int,
        default=DEFAULT_MAX_TOKENS_PERF,
        help="Maximum new tokens to generate during performance measurements.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=parse_positive_int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum new tokens to generate during accuracy benchmarks.",
    )
    parser.add_argument(
        "--calibration-dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset name for calibration data loading.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=parse_positive_int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help="Number of samples to retrieve for calibration.",
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Enable evaluation mode when building the calibration data loader.",
    )
    parser.add_argument(
        "--no-measure-base-perf",
        action="store_true",
        help="Disable measurement of performance metrics before pruning.",
    )
    parser.add_argument(
        "--no-run-base-benchmarks",
        action="store_true",
        help="Don't run accuracy benchmarks before pruning.",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip the calibration step before measuring the pruned model.",
    )
    parser.add_argument(
        "--skip-pruned-perf",
        action="store_true",
        help="Skip performance measurements after pruning.",
    )
    parser.add_argument(
        "--skip-pruned-benchmarks",
        action="store_true",
        help="Skip accuracy benchmarks after pruning.",
    )
    parser.add_argument("--save-path", type=str, default="", help="Optional path to save the pruned model state.")
    return parser.parse_args()


def resolve_tasks(task_names: Sequence[str]) -> List[object]:
    return [TASK_FACTORIES[name]() for name in task_names]


def measure_performance(model_runner, batch_sizes: Sequence[int], max_new_tokens: int) -> None:
    prompts = list(model_runner.default_prompts())
    if not prompts:
        raise ValueError("Model runner must provide at least one default prompt to measure performance.")

    set_measure_ctx()
    print(f"{'batch':>6} {'latency_ms':>14} {'throughput_tps':>18} {'batch_time_ms':>18} "
          f"{'peak_mem_mb':>10} {'curr_mem_mb':>10} {'device':>10}")
    for batch in batch_sizes:
        metrics = measure_for_batch_size(
            runner=model_runner,
            prompts=prompts,
            batch_size=batch,
            max_new_tokens=max_new_tokens,
            latency_runs=LATENCY_RUNS,
            throughput_runs=THROUGHPUT_RUNS,
            batch_runs=BATCH_RUNS,
        )
        print(format_metrics_row(metrics))


def run_benchmarks(model_runner, tasks: Sequence[object], max_new_tokens: int) -> None:
    suite = BenchmarkSuite(runner=model_runner, tasks=list(tasks), max_new_tokens=max_new_tokens)
    report = suite.run(collect_performance=False, verbose=True, sample_print_limit=5)
    print(report)


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    model = get_model(args.model_id, seqlen=SEQLEN)
    model.model.eval()

    with torch.no_grad():
        add_actprune(model.model)
        disable_act_sparsity(model.model)

        if (not args.no_measure_base_perf) or (not args.no_run_base_benchmarks):
            model.model.to(model.device)
            if not args.no_measure_base_perf:
                print("Measuring base model performance")
                measure_performance(model, args.batch_sizes, args.max_tokens_perf)
            if not args.no_run_base_benchmarks:
                print("Running base model benchmarks")
                run_benchmarks(model, tasks, args.max_new_tokens)

        if not args.skip_calibration:
            loader = get_loaders(
                args.calibration_dataset,
                args.calibration_samples,
                model=args.model_id,
                seqlen=SEQLEN,
                eval_mode=args.eval_mode,
            )
            calibrate.calibrate_model(model, loader)

        disable_act_sparsity(model.model)
        model.model.to(model.device)
        model.model.compile()

        print("Pruned model statistics")
        if not args.skip_pruned_perf:
            print("Measuring pruned model performance")
            measure_performance(model, args.batch_sizes, args.max_tokens_perf)

        if not args.skip_pruned_benchmarks:
            print("Running pruned model benchmarks")
            run_benchmarks(model, tasks, args.max_new_tokens)
        
        if args.save_path != "":
            print(f"Saving pruned model state to {args.save_path}")
            torch.save({"model": model.model.state_dict()}, args.save_path)


if __name__ == "__main__":
    main()
