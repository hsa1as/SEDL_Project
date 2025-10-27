# stuff for search
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from .obc import DuoGPTConfig

__all__ = [
    "ParameterType",
    "ParameterSpec",
    "PruningSearchSpace",
    "LayerPrunePlan",
    "ToDuoGPTConfig",
    "PerformanceMetric",
    "LayerEvaluation",
    "EvaluationDataset",
    "LayerMetrics",
    "build_layer_evaluation",
    "GaussianProcess",
    "expected_improvement",
    "BayesianTuner",
]


class ParameterType(Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: ParameterType
    bounds: Tuple[float, float]
    choices: Optional[Sequence[float]] = None
    log_scale: bool = False

    def __post_init__(self) -> None:
        lo, hi = self.bounds
        if lo >= hi:
            raise ValueError(f"Invalid bounds {self.bounds} for {self.name}")
        if self.choices is not None and not self.choices:
            raise ValueError(f"{self.name} choices must not be empty.")
        if self.choices is not None:
            sorted_choices = sorted(self.choices)
            object.__setattr__(self, "choices", tuple(sorted_choices))

    def validate(self, value: float) -> None:
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{value} not permitted for parameter {self.name}")
        lo, hi = self.bounds
        if value < lo or value > hi:
            raise ValueError(f"{value} outside bounds {self.bounds} for {self.name}")
        if self.type == ParameterType.INTEGER and not float(value).is_integer():
            raise ValueError(f"{self.name} requires an integer value, got {value}")

    def normalise(self, value: float) -> float:
        self.validate(value)
        if self.choices is not None:
            idx = self.choices.index(value)  # value validated above
            return idx / (len(self.choices) - 1) if len(self.choices) > 1 else 0.0
        lo, hi = self.bounds
        if self.log_scale:
            lo = torch.log(torch.tensor(lo))
            hi = torch.log(torch.tensor(hi))
            value = torch.log(torch.tensor(value))
            return float((value - lo) / (hi - lo))
        return float((value - lo) / (hi - lo))

    def denormalise(self, value: float) -> float:
        value = float(torch.clamp(torch.tensor(value), 0.0, 1.0))
        if self.choices is not None:
            if len(self.choices) == 1:
                return float(self.choices[0])
            idx = min(int(round(value * (len(self.choices) - 1))), len(self.choices) - 1)
            return float(self.choices[idx])
        lo, hi = self.bounds
        raw = lo + value * (hi - lo)
        if self.log_scale:
            return float(torch.exp(torch.tensor(raw)))
        if self.type == ParameterType.INTEGER:
            return float(round(raw))
        return raw

@dataclass
class PruningSearchSpace:
    parameters: Dict[str, ParameterSpec]

    def __post_init__(self) -> None:
        for spec in self.parameters.values():
            if spec.name not in self.parameters:
                raise KeyError(f"Parameter {spec.name} not registered.")

    @classmethod
    def default(cls) -> "PruningSearchSpace":
        params = {
            "weight_sparsity": ParameterSpec(
                name="weight_sparsity",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 0.98),
            ),
            "blocksize": ParameterSpec(
                name="blocksize",
                type=ParameterType.INTEGER,
                bounds=(16, 512),
                choices=(16, 32, 64, 128, 256, 512),
            ),
            "prunen": ParameterSpec(
                name="prunen",
                type=ParameterType.INTEGER,
                bounds=(0, 8),
                choices=(0, 1, 2, 4, 8),
            ),
            "prunem": ParameterSpec(
                name="prunem",
                type=ParameterType.INTEGER,
                bounds=(1, 16),
                choices=(1, 2, 4, 8, 16),
            ),
        }
        return cls(parameters=params)

    def spec(self, name: str) -> ParameterSpec:
        if name not in self.parameters:
            raise KeyError(f"Unknown parameter {name}")
        return self.parameters[name]

    def normalise_plan(self, plan: Mapping[str, float]) -> torch.Tensor:
        values = []
        for name, spec in self.parameters.items():
            if name not in plan:
                raise KeyError(f"{name} missing from plan.")
            values.append(spec.normalise(plan[name]))
        return torch.tensor(values, dtype=torch.float32)

    def denormalise_plan(self, encoded: torch.Tensor) -> Dict[str, float]:
        if encoded.numel() != len(self.parameters):
            raise ValueError("Encoded vector size does not match parameter count.")
        decoded: Dict[str, float] = {}
        for idx, (name, spec) in enumerate(self.parameters.items()):
            decoded[name] = spec.denormalise(float(encoded[idx]))
        return decoded

    def sample_sobol(self, n: int, seed: Optional[int] = None) -> List[Dict[str, float]]:
        if n <= 0:
            raise ValueError("Number of samples must be positive.")
        engine = torch.quasirandom.SobolEngine(dim=len(self.parameters), scramble=True, seed=seed)
        samples = engine.draw(n)
        return [self.denormalise_plan(sample) for sample in samples]


@dataclass
class LayerPrunePlan:
    weight_sparsity: float
    blocksize: int
    prunen: int
    prunem: int
    def __str__(self) -> str:
        return (
            f"LayerPrunePlan("
            f"weight_sparsity={self.weight_sparsity:.4f}, "
            f"blocksize={self.blocksize}, "
            f"prunen={self.prunen}, "
            f"prunem={self.prunem})"
        )
    def as_dict(self) -> Dict[str, float]:
        return {
            "weight_sparsity": float(self.weight_sparsity),
            "blocksize": float(self.blocksize),
            "prunen": float(self.prunen),
            "prunem": float(self.prunem),
        }

    def validate(self, space: PruningSearchSpace) -> None:
        params = self.as_dict()
        if self.prunen and self.prunen > self.prunem:
            raise ValueError("prunen must be <= prunem.")
        for name, value in params.items():
            space.spec(name).validate(value)


@dataclass
class ToDuoGPTConfig:
    base_config: DuoGPTConfig
    space: PruningSearchSpace = field(default_factory=PruningSearchSpace.default)

    def for_layer(self, plan: LayerPrunePlan) -> DuoGPTConfig:
        plan.validate(self.space)
        cfg = replace(self.base_config)
        cfg.sparsity = plan.weight_sparsity
        cfg.blocksize = int(plan.blocksize)
        cfg.prunen = int(plan.prunen)
        cfg.prunem = int(plan.prunem)
        return cfg

def duo_to_layer(cfg: DuoGPTConfig) -> LayerPrunePlan:
    return LayerPrunePlan(
        weight_sparsity=cfg.sparsity,
        blocksize=cfg.blocksize,
        prunen=cfg.prunen,
        prunem=cfg.prunem,
    )


class PerformanceMetric(Enum):
    LATENCY_MS = "latency_ms"
    THROUGHPUT_TOKENS_PER_S = "throughput_tokens_per_second"
    BATCH_TIME_MS = "batch_inference_time_ms"


@dataclass
class LayerEvaluation:
    layer_idx: int
    plan: LayerPrunePlan
    metrics: Dict[str, float]
    notes: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(self.metrics.items()))
        notes_str = ", ".join(f"{k}: {v:.4f}" for k, v in sorted(self.notes.items())) if self.notes else "—"
        return (
            f"LayerEvaluation(\n"
            f"  layer_idx = {self.layer_idx},\n"
            f"  plan = {self.plan},\n"
            f"  metrics = {{{metrics_str}}},\n"
            f"  notes = {{{notes_str}}}\n"
            f")"
        )

    def get_metric(self, metric: PerformanceMetric) -> Optional[float]:
        return self.metrics.get(metric.value)


@dataclass
class EvaluationDataset:
    space: PruningSearchSpace
    records: List[LayerEvaluation] = field(default_factory=list)

    def add(self, record: LayerEvaluation) -> None:
        record.plan.validate(self.space)
        self.records.append(record)

    def as_tensors(
        self,
        objective: PerformanceMetric,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.records:
            raise ValueError("Dataset is empty.")
        features = []
        targets = []
        for record in self.records:
            features.append(self.space.normalise_plan(record.plan.as_dict()))
            metric = record.get_metric(objective)
            if metric is None:
                raise KeyError(f"Metric {objective.value} missing for record.")
            targets.append(metric)
        feature_tensor = torch.stack(features, dim=0)
        target_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        return feature_tensor, target_tensor


@dataclass
class LayerMetrics:

    residual_norm: float = 0.0  
    sparsity_applied: float = 0.0 
    timing: float = 0.0 

def build_layer_evaluation(
    layer_idx: int,
    plan: LayerPrunePlan,
    summary: LayerMetrics,
    *,
    extra_metrics: Optional[Dict[str, float]] = None,
) -> LayerEvaluation:
    metrics: Dict[str, float] = {
        "residual_norm": summary.residual_norm,
        "sparsity_applied": summary.sparsity_applied,
        "timing": summary.timing,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    notes: Dict[str, float] = {}
    return LayerEvaluation(layer_idx=layer_idx, plan=plan, metrics=metrics, notes=notes)

class GaussianProcess(torch.nn.Module):
    def __init__(self, input_dim: int, noise: float = 1e-4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.noise = noise
        self.lengthscale = torch.nn.Parameter(torch.ones(1) * 0.2)
        self.outputscale = torch.nn.Parameter(torch.ones(1))

    def kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        sqdist = torch.cdist(x1, x2) ** 2
        return self.outputscale ** 2 * torch.exp(-0.5 * sqdist)

    def forward(self, train_x: torch.Tensor, train_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_xx = self.kernel(train_x, train_x)
        noise = self.noise * torch.eye(len(train_x), device=train_x.device)
        L = torch.linalg.cholesky(k_xx + noise)
        alpha = torch.cholesky_solve(train_y, L)
        return L, alpha

    def predict(self, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L, alpha = self(train_x, train_y)
        k_xs = self.kernel(train_x, test_x)
        mean = k_xs.transpose(0, 1).matmul(alpha)
        v = torch.cholesky_solve(k_xs, L)
        k_ss = self.kernel(test_x, test_x)
        var = k_ss - k_xs.transpose(0, 1).matmul(v)
        return mean.squeeze(-1), var.diag().clamp_min(1e-9)


def expected_improvement(
    mean: torch.Tensor,
    var: torch.Tensor,
    best: float,
    explore: float,
) -> torch.Tensor:
    std = var.sqrt()
    z = (best - mean - explore) / std.clamp_min(1e-9)
    normal = torch.distributions.Normal(0, 1)
    return (best - mean - explore) * normal.cdf(z) + std * normal.log_prob(z).exp()


@dataclass
class BayesianTuner:
    space: PruningSearchSpace
    objective: PerformanceMetric
    surrogate_noise: float = 1e-4
    explore: float = 0.01
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.surrogate = GaussianProcess(len(self.space.parameters), self.surrogate_noise).to(self.device)
        self.dataset = EvaluationDataset(self.space)

    def seed(self, plans: Sequence[LayerPrunePlan], evaluations: Sequence[LayerEvaluation]) -> None:
        for plan, eval_ in zip(plans, evaluations):
            self.dataset.add(eval_)

    def best(self) -> Optional[LayerEvaluation]:
        if not self.dataset.records:
            return None
        metric = self.objective.value
        return min(self.dataset.records, key=lambda r: r.metrics[metric])

    def next(self, candidates: Sequence[LayerPrunePlan]) -> Optional[LayerPrunePlan]:
        if not self.dataset.records:
            return candidates[0] if candidates else None
        x_train, y_train = self.dataset.as_tensors(self.objective)
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        mean: List[float] = []
        var: List[float] = []
        for plan in candidates:
            x_test = self.space.normalise_plan(plan.as_dict()).unsqueeze(0).to(self.device)
            m, v = self.surrogate.predict(x_train, y_train, x_test)
            mean.append(float(m.item()))
            var.append(float(v.item()))
        best_value = min(r.metrics[self.objective.value] for r in self.dataset.records)
        ei = expected_improvement(torch.tensor(mean), torch.tensor(var), best_value, self.explore)
        idx = int(torch.argmax(ei).item())
        return candidates[idx]

    def update(self, evaluation: LayerEvaluation) -> None:
        self.dataset.add(evaluation)
