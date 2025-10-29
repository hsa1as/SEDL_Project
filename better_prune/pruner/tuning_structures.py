# stuff for search
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

import random

import torch
from torch import Tensor
from .obc import DuoGPTConfig, prealloc_inps_outs, capture_embeddings, CalibrationResult, cleanup_memory

__all__ = [
    "LayerEvaluation",
    "LayerMetrics",
    "build_layer_evaluation",
]


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

from typing import Dict, Any, Tuple, List, Optional
import torch
from torch import Tensor
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

import matplotlib.pyplot as plt

from .obc import DEFAULT_CONFIG
from ..model_runner import ModelRunner


class BayesianForObc:

    def __init__(
        self,
        layer_idx: int,
        model_runner: ModelRunner,
        loader: Any,
        n_init: int = 8,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.double,
        timing_samples: int = 8,
        timing_runs: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.timing_samples = min(128, timing_samples)
        self.timing_runs = timing_runs
        self.path = "pareto_layer_" + str(layer_idx) + ".png"
        self.evals = 0
        self.dim: int = 5
        self.model_runner = model_runner
        model_runner.model.to("cpu") # move to cpu
        self.loader = loader
        self.n_init: int = n_init
        self.device: torch.device = torch.device(device)
        self.dtype: torch.dtype = dtype
        self.layer_idx = layer_idx
        self.layer_acts_init = False
        self.inps = None
        self.outs = None
        self.position_embeddings = None
        self.attention_mask = None
        self.weights_copy = None
        self.blocksize_max = 64
        self.nm_max = 64
        self.sparsity_max = 0.9
        self.act_blocksize_max = 64
        if seed is not None:
            torch.manual_seed(seed)
        self.train_X: Optional[Tensor] = None
        self.train_Y: Optional[Tensor] = None
        self.train_Y_raw: Optional[Tensor] = None
        self.ref_point: Optional[Tensor] = None

    def sample_config(self) -> DuoGPTConfig:
        config: DuoGPTConfig = DEFAULT_CONFIG
        # DuoGPTConfig: Hyperparameters changed are
        # blocksize [1, 256] int
        # prunen, prunem n < m ints, [0, 256]
        # sparsity [0,1] float
        # act_blocksize [0, 256] int
        # 5 dimensions
        config.blocksize = random.randint(1, self.blocksize_max)
        config.prunem = random.randint(0, self.nm_max)
        config.prunen = random.randint(0, max(0,config.prunem-1))
        config.sparsity = min(random.random(), self.sparsity_max)
        config.act_blocksize = random.randint(0, self.act_blocksize_max)

        return config
    # blocksize, prunen, prunem, sparsity, act_blocksize
    def config_to_vector(self, config: DuoGPTConfig) -> Tensor:
        ret = torch.ones(self.dim, device=self.device, dtype=self.dtype)
        ret[0] = config.blocksize / self.blocksize_max
        ret[1] = config.prunem / self.nm_max
        ret[2] = config.prunen / self.nm_max
        ret[3] = config.sparsity
        ret[4] = config.act_blocksize / self.act_blocksize_max
        return ret


    def vector_to_config(self, x: Tensor) -> DuoGPTConfig:
        ret = DEFAULT_CONFIG
        ret.blocksize = max(1,round(x[0].item() * self.blocksize_max))
        ret.prunem = round(x[1].item() * self.nm_max)
        ret.prunen = min(round(x[2].item() * self.nm_max), max(0,ret.prunem-1))
        ret.sparsity = min(x[3].item(), self.sparsity_max)
        ret.act_blocksize = round(x[4].item() * self.act_blocksize_max)
        return ret

    def free(self):
        for name in self.weights_copy:
            self.weights_copy[name] = None
            self.inps = None
            self.outs = None
            self.position_embeddings = None
            self.attention_mask = None
            self.layer_acts_init = False
            cleanup_memory()

    def reset_weights(self):
        if self.weights_copy is None:
            return
        for name in self.weights_copy:
            self.model_runner.model.model.layers[self.layer_idx]\
            .get_submodule(name).weight.data = self.weights_copy[name]

    @torch.no_grad()
    def evaluate(self, config: DuoGPTConfig) -> Tuple[float, float]:
        self.evals += 1
        print("Evaluating config:", config)
        layer = self.model_runner.model.model.layers[self.layer_idx]
        res = (0., 0.)
        if not self.layer_acts_init:
            inps, outs = prealloc_inps_outs(self.model_runner, config)
            self.inps = inps
            self.outs = outs
        calib_result = capture_embeddings(self.model_runner, self.loader, self.layer_idx, self.inps, self.outs,
                                          inps_contain_layer_inps= self.layer_acts_init,
                                          position_embeddings = self.position_embeddings,
                                          attention_mask = self.attention_mask,args= config)
        self.position_embeddings = calib_result.position_embeddings
        self.attention_mask = calib_result.attention_mask
        self.layer_acts_init = True

        extra = {"attention_mask": self.attention_mask, "position_embeddings": self.position_embeddings}
        # save weights
        self.weights_copy = {}
        for name, item in calib_result.pruner_state.items():
            self.weights_copy[name] = item.layer.weight.data.clone().cpu()
        cleanup_memory()
        layer_dev = next(iter(layer.parameters())).device
        layer.to(self.model_runner.device)
        time_unpruned = self.model_runner.get_layerwise_perf(self.inps[:self.timing_samples], layer, extra, runs=self.timing_runs)["time"]
        layer.to(layer_dev)
        cleanup_memory()
        for name in calib_result.pruner_state:
            calib_result.pruner_state[name].fasterprune(config)

        for name in calib_result.pruner_state:
            calib_result.pruner_state[name].free()

        layer.to(self.model_runner.device)
        time_pruned = self.model_runner.get_layerwise_perf(self.inps[:self.timing_samples], layer, extra, runs=self.timing_runs)["time"]
        layer.to(layer_dev)

        layer.to(self.model_runner.device)

        residual = 0. # 0-3
        speedup = time_unpruned/time_pruned if time_pruned != 0 else 1
        with torch.no_grad():
            new = layer(self.inps[:8], attention_mask = self.attention_mask, position_embeddings=self.position_embeddings)
            cleanup_memory()
            residual = (
                (new - self.outs[:8]).norm(dim=-1) / (self.outs[:8].norm(dim=-1) + 1e-8)
            ).mean().item()
            del new

        layer.to(layer_dev)
        print(f"Evaluation complete: speedup={speedup}, residual={residual}")
        res = (speedup/10, residual/3)

        self.reset_weights()
        cleanup_memory()
        return res

    def _update_ref_point(self) -> None:
        if self.train_Y is None:
            raise RuntimeError("train_Y is None")
        y_min: Tensor = self.train_Y.min(dim=0).values
        self.ref_point = (y_min - 0.1).to(device=self.device, dtype=self.dtype)

    def initialize(self) -> None:
        xs: List[Tensor] = []
        ys_raw: List[Tuple[float, float]] = []
        for _ in range(self.n_init):
            cfg: DuoGPTConfig = self.sample_config()
            x: Tensor = self.config_to_vector(cfg).to(self.device, self.dtype)
            r, t = self.evaluate(cfg)
            xs.append(x)
            ys_raw.append((r, t))
        self.train_X = torch.stack(xs)
        self.train_Y_raw = torch.tensor(ys_raw, device=self.device, dtype=self.dtype)
        speedup: Tensor = self.train_Y_raw[:, [0]]
        residual: Tensor = self.train_Y_raw[:, [1]]
        self.train_Y = torch.cat([speedup, -residual], dim=-1)
        self._update_ref_point()

    def _build_model(self) -> ModelListGP:
        if self.train_X is None or self.train_Y is None:
            raise RuntimeError("Call initialize() first")
        y1: Tensor = self.train_Y[:, [0]]
        y2: Tensor = self.train_Y[:, [1]]
        m1: SingleTaskGP = SingleTaskGP(self.train_X, y1)
        m2: SingleTaskGP = SingleTaskGP(self.train_X, y2)
        return ModelListGP(m1, m2)

    def get_saved(self) -> Tuple:
        if(self.layer_acts_init is False):
            print("attempting to get saved results of layer before it is processed")
            exit(-1)
        return (self.outs, self.inps, self.attention_mask, self.position_embeddings)

    def load_saved(self, saved):
        self.layer_acts_init = True
        # invert inps, outs
        self.inps, self.outs, self.attention_mask, self.position_embeddings = saved

    def step(
        self,
        q: int = 1,
        num_restarts: int = 5,
        raw_samples: int = 64,
        num_samples_acq: int = 128,
    ) -> Tuple[Tensor, Tensor]:
        if self.train_X is None or self.train_Y is None:
            self.initialize()
        if self.train_X is None or self.train_Y is None or self.ref_point is None:
            raise RuntimeError("Initialization failed")
        model: ModelListGP = self._build_model().to(self.device, self.dtype)
        mll: SumMarginalLogLikelihood = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        partitioning: NondominatedPartitioning = NondominatedPartitioning(
            ref_point=self.ref_point,
            Y=self.train_Y,
        )
        sampler: SobolQMCNormalSampler = SobolQMCNormalSampler(torch.Size((num_samples_acq,)))
        acq = qLogExpectedHypervolumeImprovement( #qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        bounds: Tensor = torch.stack(
            [
                torch.zeros(self.dim, device=self.device, dtype=self.dtype),
                torch.ones(self.dim, device=self.device, dtype=self.dtype),
            ]
        )
        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
        )
        new_y_raw_list: List[Tuple[float, float]] = []
        for x in candidates:
            cfg: DuoGPTConfig = self.vector_to_config(x.detach())
            r, t = self.evaluate(cfg)
            new_y_raw_list.append((r, t))
        new_Y_raw: Tensor = torch.tensor(new_y_raw_list, device=self.device, dtype=self.dtype)
        new_speedup: Tensor = new_Y_raw[:, [0]]
        new_residual: Tensor = new_Y_raw[:, [1]]
        new_Y: Tensor = torch.cat([new_speedup, -new_residual], dim=-1)
        self.train_X = torch.cat([self.train_X, candidates], dim=0)
        self.train_Y_raw = torch.cat([self.train_Y_raw, new_Y_raw], dim=0)
        self.train_Y = torch.cat([self.train_Y, new_Y], dim=0)

        self._update_ref_point()
        return candidates, new_Y_raw

    def run(self, n_steps: int, **step_kwargs: Any) -> None:
        for _ in range(n_steps):
            self.step(**step_kwargs)

    def pareto_front(self) -> Tuple[Tensor, Tensor]:
        if self.train_X is None or self.train_Y_raw is None:
            raise RuntimeError("No data")
        Y: Tensor = self.train_Y_raw
        n: int = Y.shape[0]
        mask: Tensor = torch.ones(n, dtype=torch.bool, device=Y.device)
        for i in range(n):
            if not mask[i]:
                continue
            s_i: Tensor = Y[i, 0]
            r_i: Tensor = Y[i, 1]
            better_or_equal_speedup: Tensor = Y[:, 0] >= s_i
            better_or_equal_residual: Tensor = Y[:, 1] <= r_i
            strictly_better: Tensor = (Y[:, 0] > s_i) | (Y[:, 1] < r_i)
            dominates: Tensor = better_or_equal_speedup & better_or_equal_residual & strictly_better
            mask[dominates] = False
        return self.train_X[mask], self.train_Y_raw[mask]


    def plot(self) -> None :
        if self.train_Y_raw is None or self.train_X is None:
            raise RuntimeError("No data")
        X_pf, Y_pf = self.pareto_front()
        Y_all: Tensor = self.train_Y_raw
        Y_pf[:,0] *= 10
        Y_pf[:, 1] *=3
        Y_all[:,0] *=10
        Y_all[:, 1] *= 3
        fig, ax = plt.subplots()
        ax.scatter(
            Y_all[:, 0].detach().cpu().numpy(),
            Y_all[:, 1].detach().cpu().numpy(),
            alpha=0.4,
        )
        ax.scatter(
            Y_pf[:, 0].detach().cpu().numpy(),
            Y_pf[:, 1].detach().cpu().numpy(),
        )
        ax.set_xlabel("speedup")
        ax.set_ylabel("residual")
        fig.tight_layout()
        fig.savefig(self.path)
        plt.close(fig)
