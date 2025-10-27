from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function # noqa: F401

from transformers import PreTrainedModel, PreTrainedTokenizerBase, LlamaForCausalLM

from better_prune.utils.consts import MODEL_CACHE

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    psutil = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from lm_eval.models.huggingface import HFLM  # type: ignore

    LM_EVAL_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    HFLM = None  # type: ignore
    LM_EVAL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    latency_ms: float
    throughput_tokens_per_second: float
    batch_inference_time_ms: float
    memory: "MemoryMetrics | None" = None


@dataclass
class MemoryMetrics:
    device_type: str
    peak_device_memory_mb: float | None
    current_device_memory_mb: float | None
    peak_cpu_memory_mb: float | None = None
    current_cpu_memory_mb: float | None = None


@dataclass
class LogLikelihood:
    total_log_prob: float
    token_count: int


class ModelRunner(ABC):
    def __init__(self, model: Any, identifier: str | None = None) -> None:
        """Initialize the model runner."""
        self.identifier = identifier or self.__class__.__name__
        self.model = model
        self.device = torch.device("cpu") 

    @abstractmethod
    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> Sequence[str]:
        """Generate continuations for a batch of prompts."""

    @abstractmethod
    def measure_latency(self, prompt: str, max_new_tokens: int, runs: int = 5) -> float:
        """Measure average latency for a single prompt in milliseconds."""

    @abstractmethod
    def measure_throughput(self, prompts: Sequence[str], max_new_tokens: int, runs: int = 3) -> float:
        """Measure generated token throughput in tokens per second."""

    @abstractmethod
    def measure_batch_inference_time(self, prompts: Sequence[str], max_new_tokens: int, runs: int = 3) -> float:
        """Measure average batch inference time in milliseconds."""

    @abstractmethod
    def get_lm_eval_model(self, batch_size: int) -> Any:
        """Return an lm-eval harness model adapter for this runner."""

    @abstractmethod
    def measure_memory_usage(self, prompts: Sequence[str], max_new_tokens: int) -> MemoryMetrics:
        """Measure device and host memory usage for a representative batch."""

    @abstractmethod
    def get_layerwise_perf(self, inp: torch.Tensor, layer: nn.Module) -> Dict[str, float]:
        """Get layerwise performance metrics given input and output tensors."""

    def default_prompts(self) -> Sequence[str]:
        """Return a small set of representative prompts used for quick diagnostics."""
        return (
            "Explain the difference between supervised and unsupervised learning.",
            "Write a haiku about autumn evenings in the mountains.",
            "Summarize the key steps of the water cycle in two sentences.",
            "Translate the phrase 'knowledge is power' into French.",
        )

    def collect_performance(
        self,
        prompts: Sequence[str] | None = None,
        max_new_tokens: int = 128,
        latency_runs: int = 5,
        throughput_runs: int = 3,
        batch_runs: int = 3,
    ) -> PerformanceMetrics:
        """Collect latency, throughput, batch inference, and memory metrics."""
        prompts = list(prompts or self.default_prompts())
        if not prompts:
            raise ValueError("At least one prompt is required to collect performance metrics.")
        latency_prompt = prompts[0]
        latency_ms = self.measure_latency(latency_prompt, max_new_tokens=max_new_tokens, runs=latency_runs)
        throughput = self.measure_throughput(prompts, max_new_tokens=max_new_tokens, runs=throughput_runs)
        batch_time = self.measure_batch_inference_time(prompts, max_new_tokens=max_new_tokens, runs=batch_runs)
        memory = self.measure_memory_usage(prompts, max_new_tokens=max_new_tokens)
        return PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_tokens_per_second=throughput,
            batch_inference_time_ms=batch_time,
            memory=memory,
        )

    @abstractmethod
    def compute_log_likelihoods(self, inputs: Sequence[str]) -> Sequence[LogLikelihood]:
        """Compute token-level log likelihood statistics for each input sequence."""


class NativeModelRunner(ModelRunner):
    is_chat_ft = False
    generate_kwargs: Dict[str, Any]

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str | torch.device | None = None,
        identifier: str | None = None,
        is_chat_ft: bool = True,
        generate_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize a runner that executes models on the local host."""
        super().__init__(model, identifier=identifier or getattr(model.config, "name_or_path", None))
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.model.eval()
        self.is_chat_ft = is_chat_ft
        self.generate_kwargs = dict(generate_kwargs or {})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if not getattr(self.model.config, "is_encoder_decoder", False) and getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

    def generate(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        """Generate continuations for a batch of prompts, applying chat template automatically."""
        outputs, _ = self._generate_internal(
            prompts,
            max_new_tokens=max_new_tokens,
            use_chat_template=self.is_chat_ft,
            record_timing=False,
        )
        return outputs

    def generate_no_template(self, prompts: Sequence[str], max_new_tokens: int) -> List[str]:
        """Generate continuations for a batch of prompts."""
        outputs, _ = self._generate_internal(
            prompts,
            max_new_tokens=max_new_tokens,
            use_chat_template=False,
            record_timing=False,
        )
        return outputs

    def measure_latency(self, prompt: str, max_new_tokens: int, runs: int = 5) -> float:
        """Measure average latency for a single prompt in milliseconds."""
        #self._run_warmup([prompt], max_new_tokens, warmup)
        durations: List[float] = []
        for _ in range(runs):
            _, duration_ms = self._generate_internal(
                [prompt],
                max_new_tokens=max_new_tokens,
                use_chat_template=self.is_chat_ft,
                record_timing=True,
            )
            if duration_ms is not None:
                durations.append(duration_ms)
        return sum(durations) / len(durations)

    def measure_throughput(self, prompts: Sequence[str], max_new_tokens: int, runs: int = 3) -> float:
        """Measure generated token throughput in tokens per second."""
        #self._run_warmup(prompts, max_new_tokens, 1)
        total_tokens = 0
        total_time = 0.0
        for _ in range(runs):
            outputs, duration_ms = self._generate_internal(
                prompts,
                max_new_tokens=max_new_tokens,
                use_chat_template=self.is_chat_ft,
                record_timing=True,
            )
            total_time += (duration_ms or 0.0) / 1000.0
            total_tokens += self._count_generated_tokens(prompts, outputs)
            del outputs
        return total_tokens / total_time if total_time > 0 else 0.0

    def measure_batch_inference_time(self, prompts: Sequence[str], max_new_tokens: int, runs: int = 3) -> float:
        """Measure average batch inference time in milliseconds."""
        #self._run_warmup(prompts, max_new_tokens, 1)
        durations: List[float] = []
        for _ in range(runs):
            _, duration_ms = self._generate_internal(
                prompts,
                max_new_tokens=max_new_tokens,
                use_chat_template=self.is_chat_ft,
                record_timing=True,
            )
            if duration_ms is not None:
                durations.append(duration_ms)
        return sum(durations) / len(durations)

    def measure_memory_usage(self, prompts: Sequence[str], max_new_tokens: int) -> MemoryMetrics:
        """Capture peak device memory and host RSS for the provided prompts."""
        #self._run_warmup(prompts, max_new_tokens, warmup)

        process = psutil.Process(os.getpid()) if psutil is not None else None
        cpu_before = process.memory_info().rss if process is not None else None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            self._sync_device()

        outputs, _ = self._generate_internal(
            prompts,
            max_new_tokens=max_new_tokens,
            use_chat_template=self.is_chat_ft,
            record_timing=False,
        )
        del outputs
        self._sync_device()

        if self.device.type == "cuda":
            peak_device = float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 ** 2)
            current_device = float(torch.cuda.memory_allocated(self.device)) / (1024.0 ** 2)
        else:
            peak_device = None
            current_device = None

        cpu_after = process.memory_info().rss if process is not None else None
        peak_cpu = None
        current_cpu = None
        if cpu_before is not None or cpu_after is not None:
            before = float(cpu_before) if cpu_before is not None else 0.0
            after = float(cpu_after) if cpu_after is not None else before
            peak_cpu = max(before, after) / (1024.0 ** 2)
            current_cpu = after / (1024.0 ** 2)

        return MemoryMetrics(
            device_type=self.device.type,
            peak_device_memory_mb=peak_device,
            current_device_memory_mb=current_device,
            peak_cpu_memory_mb=peak_cpu,
            current_cpu_memory_mb=current_cpu,
        )
    def get_layerwise_perf(self, inp, layer: nn.Module, layer_kwargs: Dict) -> Dict[str, Any]:
        layer_dev = next(iter(layer.parameters())).device
        layer.to(self.device)
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            _ = layer(inp, **layer_kwargs)
        cuda_time = sum([e.cuda_time_total for e in prof.events()])
        layer.to(layer_dev)
        del prof
        return {"time": cuda_time}


    def get_lm_eval_model(self, batch_size: int) -> Any:
        """Create an lm-eval harness adapter for the local model."""
        if not LM_EVAL_AVAILABLE:
            raise ImportError("lm_eval is required to create an evaluation model. Install eleuther-ai/lm-eval-harness.")
        return HFLM(
            pretrained=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            device=str(self.device),
            trust_remote_code=True,
        )

    def _run_warmup(self, prompts: Sequence[str], max_new_tokens: int, warmup: int) -> None:
        """Execute warmup runs to stabilize performance measurement."""
        if warmup <= 0:
            return
        for _ in range(warmup):
            outputs, _ = self._generate_internal(
                prompts,
                max_new_tokens=max_new_tokens,
                use_chat_template=self.is_chat_ft,
                record_timing=False,
            )
            del outputs

    def _count_generated_tokens(self, prompts: Sequence[str], generations: Sequence[str]) -> int:
        """Count generated tokens for throughput estimation."""
        generated_tokens = self.tokenizer(list(generations), add_special_tokens=False, padding=False)["input_ids"]
        return sum(len(tokens) for tokens in generated_tokens)

    def _sync_device(self) -> None:
        """Synchronize CUDA device when available."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _generate_internal(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        use_chat_template: bool,
        record_timing: bool,
    ) -> Tuple[List[str], float | None]:
        """Internal helper that handles formatting, timing, and decoding."""
        formatted_prompts = self._format_prompts(prompts, use_chat_template=use_chat_template)

        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        eos_token_id = self.tokenizer.eos_token_id

        start_event: torch.cuda.Event | None = None
        end_event: torch.cuda.Event | None = None
        start_time: float | None = None
        duration_ms: float | None = None

        if record_timing and self.device.type == "cuda":
            self._sync_device()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        elif record_timing:
            start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                **self.generate_kwargs,
            )

        if record_timing and self.device.type == "cuda" and start_event is not None and end_event is not None:
            end_event.record()
            self._sync_device()
            duration_ms = float(start_event.elapsed_time(end_event))
        elif record_timing and start_time is not None:
            duration_ms = float((time.perf_counter() - start_time) * 1000.0)

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded, duration_ms

    def _format_prompts(self, prompts: Sequence[str], use_chat_template: bool) -> List[str]:
        """Apply chat templates when requested; otherwise return raw prompts."""
        if not use_chat_template:
            return list(prompts)

        formatted = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted.append(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                formatted.append(f"System: You are a helpful AI assistant.\nUser: {prompt}\nAssistant:")
        return formatted

    def compute_log_likelihoods(self, inputs: Sequence[str]) -> List[LogLikelihood]:
        """Compute log likelihood statistics for each provided sequence."""
        if not inputs:
            return []

        encoded = self.tokenizer(
            list(inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        attention_mask = encoded["attention_mask"]
        input_ids = encoded["input_ids"]

        with torch.no_grad():
            outputs = self.model(**encoded)

        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Align logits with next-token labels and ignore padding tokens.
        shifted_log_probs = log_probs[:, :-1, :]
        shifted_labels = input_ids[:, 1:]
        shifted_mask = attention_mask[:, 1:]

        token_log_probs = shifted_log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs * shifted_mask

        summed = masked_log_probs.sum(dim=-1)
        counts = shifted_mask.sum(dim=-1)

        results: List[LogLikelihood] = []
        for log_prob, count in zip(summed.tolist(), counts.tolist()):
            results.append(
                LogLikelihood(
                    total_log_prob=float(log_prob),
                    token_count=int(count),
                )
            )
        return results

def skip(*args, **kwargs):
    pass


def get_model(model_id: str, is_chat_ft: bool=False, seqlen:int=2048) -> NativeModelRunner:
    """Load a medium-scale causal LM for testing."""
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE,
                                              trust_remote_code=True, model_max_len=seqlen)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE,
        trust_remote_code=True,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.seqlen=2048
    return NativeModelRunner(model=model, tokenizer=tokenizer, device="cuda", is_chat_ft=is_chat_ft)
