from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

try:  # pragma: no cover - optional dependency flag
    import lm_eval  # type: ignore

    LM_EVAL_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    LM_EVAL_AVAILABLE = False


@dataclass(frozen=True)
class TaskConfig:
    tasks: Sequence[str]
    limit: int | None
    num_fewshot: int
    batch_size: int | str
    generation_kwargs: Dict[str, int]


class BenchmarkTask:
    def __init__(self, name: str, config: TaskConfig) -> None:
        self.name = name
        self.config = config

    def harness_task_names(self) -> Sequence[str]:
        return self.config.tasks

    def aggregate_metrics(self, metrics_list: Sequence[Dict]) -> float:
        return 0.0


class GSM8KTask(BenchmarkTask):
    def __init__(self, *, limit: int | None = 1000, num_fewshot: int = 8, batch_size: int = 1) -> None:
        config = TaskConfig(
            tasks=("gsm8k",),
            limit=limit,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            generation_kwargs={"max_gen_toks": 256},
        )
        super().__init__("gsm8k", config)

    def aggregate_metrics(self, metrics_list: Sequence[Dict]) -> float:
        if not metrics_list:
            return 0.0
        key = "exact_match,flexible-extract"
        total = sum(float(metric[key]) for metric in metrics_list)
        return total / len(metrics_list)


class BoolQTask(BenchmarkTask):
    def __init__(self, *, limit: int | None = 1000, num_fewshot: int = 3, batch_size: int = 2) -> None:
        config = TaskConfig(
            tasks=("boolq",),
            limit=limit,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            generation_kwargs={},
        )
        super().__init__("boolq", config)
    def aggregate_metrics(self, metrics_list: Sequence[Dict]) -> float:
        if not metrics_list:
            return 0.0
        key = "acc,none"
        total = sum(float(metric[key]) for metric in metrics_list)
        return total / len(metrics_list)


class ARCTask(BenchmarkTask):
    def __init__(
        self,
        *,
        variant: str = "arc_challenge",
        limit: int | None = 1000,
        num_fewshot: int = 25,
        batch_size: int = 2,
    ) -> None:
        config = TaskConfig(
            tasks=(variant,),
            limit=limit,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            generation_kwargs={},
        )
        super().__init__(variant, config)

    def aggregate_metrics(self, metrics_list: Sequence[Dict]) -> float:
        if not metrics_list:
            return 0.0
        key = "acc,none"
        total = sum(float(metric[key]) for metric in metrics_list)
        return total / len(metrics_list)


class MMLUTask(BenchmarkTask):
    DEFAULT_SUBJECTS: Sequence[str] = (
        "math",
        "biology",
        "chemistry",
        "physics",
    )

    def __init__(
        self,
        *,
        subjects: Sequence[str] | None = None,
        limit: int | None = 1000,
        num_fewshot: int = 5,
        batch_size: int = 2,
    ) -> None:
        selected = tuple(subjects) if subjects is not None else tuple(self.DEFAULT_SUBJECTS)
        if not selected:
            raise ValueError("MMLUTask requires at least one subject.")
        harness_tasks = tuple(f"mmlu_pro_plus_{subject}" for subject in selected)
        config = TaskConfig(
            tasks=harness_tasks,
            limit=limit,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            generation_kwargs={},
        )
        super().__init__("mmlu", config)

    def aggregate_metrics(self, metrics_list: Sequence[Dict]) -> float:
        if not metrics_list:
            return 0.0
        key = "exact_match,custom-extract"
        total = sum(float(metric[key]) for metric in metrics_list)
        return total / len(metrics_list)


__all__ = [
    "BenchmarkTask",
    "TaskConfig",
    "GSM8KTask",
    "BoolQTask",
    "ARCTask",
    "MMLUTask",
    "LM_EVAL_AVAILABLE",
]
