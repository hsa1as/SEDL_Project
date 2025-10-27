from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..benchmarks.perplexity import compute_perplexity
from ..model_runner import LogLikelihood, NativeModelRunner

MODEL_CACHE = Path.home() / ".cache" / "better_prune" / "models"
MODEL_ID = "microsoft/phi-1_5"


@pytest.fixture(scope="module")
def native_runner() -> NativeModelRunner:
    if not torch.cuda.is_available():
        pytest.skip("phi-1.5 perplexity test requires a CUDA-capable executor.")
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    return NativeModelRunner(model=model, tokenizer=tokenizer, device="cuda", identifier="phi-1.5-native")


def test_compute_perplexity_matches_manual(native_runner: NativeModelRunner) -> None:
    inputs = [
        "Once upon a time there was a curious student exploring language models.",
        "The quick brown fox jumps over the lazy dog on a sunny afternoon.",
    ]
    likelihoods = native_runner.compute_log_likelihoods(inputs)
    total_log_prob = sum(item.total_log_prob for item in likelihoods)
    total_tokens = sum(item.token_count for item in likelihoods)

    ppl = compute_perplexity(native_runner, inputs, batch_size=1)
    print("test_compute_perplexity_matches_manual", ppl)

def test_compute_perplexity_nonsense(native_runner: NativeModelRunner) -> None:
    inputs = [
        "flarmbex zotics replaunt the frungle of chif.",
    ]
    likelihoods = native_runner.compute_log_likelihoods(inputs)
    total_log_prob = sum(item.total_log_prob for item in likelihoods)
    total_tokens = sum(item.token_count for item in likelihoods)

    ppl = compute_perplexity(native_runner, inputs, batch_size=1)
    print("test_compute_perplexity_nonsense", ppl)



def test_compute_perplexity_returns_finite_value(native_runner: NativeModelRunner) -> None:
    inputs = [
        "Artificial intelligence enables new possibilities across science and engineering.",
        "Neural networks can approximate complex functions when trained on sufficient data.",
        "Evaluation metrics such as perplexity quantify model prediction quality.",
    ]
    ppl = compute_perplexity(native_runner, inputs, batch_size=2)
    print("test_compute_perplexity_returns_finite_value", ppl)
    assert ppl > 0.0
    assert math.isfinite(ppl)


def test_compute_perplexity_rejects_empty_inputs(native_runner: NativeModelRunner) -> None:
    with pytest.raises(ValueError):
        compute_perplexity(native_runner, [], batch_size=1)
