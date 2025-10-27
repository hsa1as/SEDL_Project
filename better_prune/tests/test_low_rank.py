from __future__ import annotations

from pathlib import Path
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model_runner import NativeModelRunner
from ..pruner.low_rank import find_and_replace_linears_with_lowrank
from ..benchmarks.suite import BenchmarkSuite
from ..benchmarks.tasks import ARCTask, BoolQTask, GSM8KTask, MMLUTask  # noqa: F401
from ..utils.measure_perf import *  # noqa: F403


MODEL_CACHE = Path.home() / ".cache" / "better_prune" / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# Allow overriding the exact checkpoint if a specific Llama-2 8B variant is available locally.
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
PROMPTS = [
    "The quick brown fox jumps over the lazy ",
    "The early bird gets the ",
]
TEMPLATE_PROMPTS = [
    "What is top-p sampling? Top-p sampling is ",
]


def load_low_rank_runner() -> NativeModelRunner:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype="auto",
        device_map="cuda",
    )
    replacements = find_and_replace_linears_with_lowrank(model)
    model.compile()
    assert replacements, "Expected at least one nn.Linear layer to be replaced with LowRankLinear."
    return NativeModelRunner(model=model, tokenizer=tokenizer, device="cuda", is_chat_ft=False)


def test_low_rank_model_generates_outputs() -> None:
    runner = load_low_rank_runner()
    try:
        outputs = runner.generate(PROMPTS, max_new_tokens=64)
        print(outputs)
        assert len(outputs) == len(PROMPTS)
        assert all(isinstance(text, str) and len(text) > 0 for text in outputs)

        template_outputs = runner.generate(TEMPLATE_PROMPTS, max_new_tokens=64)
        print(template_outputs)
        assert len(template_outputs) == len(TEMPLATE_PROMPTS)
        assert all(isinstance(text, str) and len(text) > 0 for text in template_outputs)
    finally:
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def test_low_rank_model_benchmarks() -> None:
    with torch.no_grad():
        runner = load_low_rank_runner()
        gc.collect()
        torch.cuda.empty_cache()
        tasks = [
        #BoolQTask(limit=100, num_fewshot=3, batch_size=10),
        #ARCTask(variant="arc_challenge", limit=100, num_fewshot=25, batch_size=10),
        GSM8KTask(limit=100, num_fewshot=8, batch_size=8),
        ]
        suite = BenchmarkSuite(runner=runner, tasks=tasks, max_new_tokens=64)
        report = suite.run(collect_performance=False, verbose=True, sample_print_limit=5)

        for task_result in report.task_results:
            print(f"[Test] Task={task_result.task_name} accuracy={task_result.accuracy:.4f}")

def test_low_rank_model_perf() -> None:
    torch.set_float32_matmul_precision('high')
    torch._logging.set_logs(graph_code=False)
    with torch.no_grad():
        runner = load_low_rank_runner()
        gc.collect()
        torch.cuda.empty_cache()
        base_prompts = list(runner.default_prompts())
        for batch_size in [1,2,4,8,16,32,64,128,256]:
            try:
                metrics = measure_for_batch_size(runner=runner,  # noqa: F405
                    prompts=base_prompts,
                    batch_size=batch_size,
                    max_new_tokens=32,
                    latency_runs=5,
                    throughput_runs=10,
                    batch_runs=5,
                    )
                print(format_metrics_row(metrics))  # noqa: F405
            except Exception as e:
                print(f"Failed run for bsz {batch_size} with error {e}")


