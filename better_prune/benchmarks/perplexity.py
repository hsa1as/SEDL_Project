from __future__ import annotations

import math
from typing import Sequence

from ..model_runner import LogLikelihood, ModelRunner


def compute_perplexity(runner: ModelRunner, inputs: Sequence[str], batch_size: int = 4) -> float:
    """Calculate perplexity for a collection of input texts using the provided runner."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not inputs:
        raise ValueError("inputs must not be empty.")

    total_log_prob = 0.0
    total_tokens = 0

    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size]
        likelihoods = runner.compute_log_likelihoods(batch)
        for likelihood in likelihoods:
            total_log_prob += likelihood.total_log_prob
            total_tokens += likelihood.token_count

    if total_tokens <= 0:
        raise ValueError("Total token count is zero; cannot compute perplexity.")

    average_log_prob = total_log_prob / total_tokens
    return math.exp(-average_log_prob)
