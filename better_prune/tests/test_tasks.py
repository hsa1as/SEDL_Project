from __future__ import annotations

import torch
from pathlib import Path

import pytest
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoModelForCausalLM, AutoTokenizer

pytest.importorskip("lm_eval")

from ..benchmarks.suite import BenchmarkSuite
from ..benchmarks.tasks import ARCTask, BoolQTask, GSM8KTask, MMLUTask
from ..model_runner import NativeModelRunner
from ..utils.consts import DEFAULT_MODEL_ID

MODEL_CACHE = Path.home() / ".cache" / "better_prune" / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
#MODEL_ID = "openai/gpt-oss-20b"
#MODEL_ID = "meta-llama/Llama-3.1-8B"
#MODEL_ID ="meta-llama/Meta-Llama-3-8B"
MODEL_ID = DEFAULT_MODEL_ID

def load_test_runner() -> NativeModelRunner:
    """Load a medium-scale causal LM for testing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        dtype="auto",
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    return NativeModelRunner(model=model, tokenizer=tokenizer, device="cuda", is_chat_ft=False)


def test_native_model_runner_generate_outputs_text() -> None:
    """Validate native runner generates outputs."""
    runner = load_test_runner()
    outputs = runner.generate(["The quick brown fox jumps over the lazy ", "The early bird gets the "], max_new_tokens=100)
    print("Model outputs: ", outputs)
    assert len(outputs) == 2
    assert all(isinstance(text, str) and len(text) > 0 for text in outputs)
    del runner
    torch.cuda.empty_cache()

def test_native_model_runner_generate_template() -> None:
    """Validate native runner generates outputs with template."""
    runner = load_test_runner()
    outputs = runner.generate(["What is top-p sampling?"], max_new_tokens=100)
    print("Model outputs: ", outputs)
    assert len(outputs) == 1
    assert all(isinstance(text, str) and len(text) > 0 for text in outputs)
    del runner
    torch.cuda.empty_cache()


def test_benchmark_suite_runs_standard_tasks_real_data() -> None:
    """Run suite across standard tasks with downloaded datasets."""
    runner = load_test_runner()
    tasks = [
        #GSM8KTask(limit=100, num_fewshot=8, batch_size=8),
        BoolQTask(limit=100, num_fewshot=3, batch_size=10),
        ARCTask(variant="arc_challenge", limit=100, num_fewshot=25, batch_size=10),
       # MMLUTask(subjects=("math",), limit=100, num_fewshot=5, batch_size=10),
    ]

    suite = BenchmarkSuite(runner=runner, tasks=tasks, max_new_tokens=64)
    try:
        report = suite.run(collect_performance=False, verbose=True, sample_print_limit=5)
    except (RequestsConnectionError, OSError, RuntimeError) as exc:
        pytest.skip(f"Dataset download unavailable: {exc}")

    for task_result in report.task_results:
        print(f"[Test] Task={task_result.task_name} accuracy={task_result.accuracy:.4f}")

    assert len(report.task_results) == len(tasks)
    assert all(result.num_samples > 0 for result in report.task_results)
    del runner
    torch.cuda.empty_cache()
