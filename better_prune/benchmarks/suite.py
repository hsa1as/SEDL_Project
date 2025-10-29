from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

#try:  # pragma: no cover - optional dependency
from lm_eval import simple_evaluate  # type: ignore

LM_EVAL_AVAILABLE = True
#except ImportError:  # pragma: no cover - handled at runtime
#    LM_EVAL_AVAILABLE = False

from ..model_runner import ModelRunner, PerformanceMetrics
from .tasks import BenchmarkTask


@dataclass
class TaskResult:
    task_name: str
    accuracy: float
    num_samples: int


@dataclass
class BenchmarkReport:
    task_results: List[TaskResult]
    performance: PerformanceMetrics | None = None


class BenchmarkSuite:
    def __init__(self, runner: ModelRunner, tasks: Sequence[BenchmarkTask], max_new_tokens: int = 128) -> None:
        """Initialize a suite that evaluates a runner on multiple tasks."""
        self.runner = runner
        self.tasks = list(tasks)
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        collect_performance: bool = True,
        verbose: bool = False,
        sample_print_limit: int | None = None,
    ) -> BenchmarkReport:
        """Run all benchmark tasks and optionally gather performance metrics."""
        if not LM_EVAL_AVAILABLE:
            raise ImportError("lm_eval is required to run the benchmark suite. Install eleuther-ai/lm-eval-harness.")

        task_results: List[TaskResult] = []
        harness_model_cache: Dict[int, Any] = {}

        for task in self.tasks:
            batch_size = task.config.batch_size
            harness_model = harness_model_cache.get(batch_size)
            if harness_model is None:
                harness_model = self.runner.get_lm_eval_model(batch_size=batch_size)
                harness_model_cache[batch_size] = harness_model
            result = self._evaluate_task(
                harness_model,
                task,
                verbose=verbose,
                sample_print_limit=sample_print_limit,
                chat_template=self.runner.is_chat_ft 
            )
            task_results.append(result)

        performance = None
        if collect_performance:
            performance = self.runner.collect_performance()
            if verbose and performance is not None:
                memory = performance.memory
                memory_str = ""
                if memory is not None:
                    if memory.peak_device_memory_mb is not None:
                        memory_str = f" peak_mem={memory.peak_device_memory_mb:.2f}MB"
                    elif memory.current_cpu_memory_mb is not None:
                        memory_str = f" cpu_mem={memory.current_cpu_memory_mb:.2f}MB"
                print(
                    f"[Benchmark] Performance latency={performance.latency_ms:.2f}ms "
                    f"throughput={performance.throughput_tokens_per_second:.2f} tok/s "
                    f"batch_time={performance.batch_inference_time_ms:.2f}ms"
                    f"{memory_str}"
                )

        return BenchmarkReport(task_results=task_results, performance=performance)

    def _evaluate_task(
        self,
        harness_model: Any,
        task: BenchmarkTask,
        verbose: bool = False,
        sample_print_limit: int | None = None,
        chat_template: bool = False,
    ) -> TaskResult:
        generation_kwargs: Dict[str, Any] | None = None

        all_metrics: List[Dict[str, Any]] = []
        captured_samples: Dict[str, Sequence[Dict[str, Any]]] = {}

        for harness_name in task.harness_task_names():
            task_generation_kwargs = generation_kwargs
            evaluation = simple_evaluate(
                harness_model,
                tasks=[harness_name],
                num_fewshot=task.config.num_fewshot,
                batch_size=task.config.batch_size,
                limit=task.config.limit,
                log_samples=True,
                gen_kwargs=task_generation_kwargs,
                apply_chat_template=chat_template,
                fewshot_as_multiturn=True,
            )

            all_metrics.append(evaluation["results"][harness_name])
            samples = evaluation.get("samples", {}).get(harness_name, [])
            if verbose and samples:
                captured_samples[harness_name] = samples

        accuracy = task.aggregate_metrics(all_metrics)
        num_samples = task.config.limit

        if verbose:
            print(f"[Benchmark] Task={task.name} accuracy={accuracy:.4f} samples={num_samples}")
            self._print_samples(task.name, captured_samples, sample_print_limit)

        return TaskResult(
            task_name=task.name,
            accuracy=accuracy,
            num_samples=num_samples,
        )

    def _extract_sample_field(self, sample: Dict[str, Any], keys: Sequence[str]) -> str:
        for key in keys:
            value = sample.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                if not value:
                    continue
                value = value[0]
            return str(value)
        return ""

    def _print_samples(
        self,
        task_name: str,
        samples_by_harness: Dict[str, Sequence[Dict[str, Any]]],
        sample_print_limit: int | None,
    ) -> None:
        if not samples_by_harness or sample_print_limit == 0:
            return

        for harness_name, samples in samples_by_harness.items():
            if not samples:
                continue
            limit = len(samples) if sample_print_limit is None else min(sample_print_limit, len(samples))
            for idx, sample in enumerate(samples[:limit]):
                if not isinstance(sample, dict):
                    print(
                        f"[Benchmark] Task={task_name} Harness={harness_name} "
                        f"Sample={idx} unexpected sample type: {type(sample).__name__}"
                    )
                    continue
                doc = sample.get("doc") if isinstance(sample.get("doc"), dict) else None
                prompt = doc.get("question") if isinstance(doc, dict) else None
                response = sample.get("resps")
                print(f"[Benchmark] Task={task_name} Harness={harness_name} Sample={idx} Prompt: {prompt!r}")
                print(f"[Benchmark] Task={task_name} Harness={harness_name} Sample={idx} Response: {response!r}")


__all__ = [
    "BenchmarkSuite",
    "BenchmarkReport",
    "TaskResult",
]
