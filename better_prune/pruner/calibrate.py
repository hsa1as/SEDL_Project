# does not have any pruning code,only calibration
from typing import Any, Dict

import torch
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: F401

from better_prune.model_runner import ModelRunner
from better_prune.pruner.obc import (  # noqa: F401
    DEFAULT_CONFIG,
    ActPruneWrapper,
    add_actprune,
    cleanup_memory,
    disable_act_sparsity,
    find_layers,
    prealloc_inps_outs,
    capture_embeddings,
)
from better_prune.pruner.tuning_structures import (
    LayerMetrics,
    LayerPrunePlan, # noqa: F401
    build_layer_evaluation,
    duo_to_layer,
)
from better_prune.utils.helpers import tensor_sparsity


@torch.no_grad
def calibrate_model(model: ModelRunner, loader: Any, layer_stats: Dict = {}) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.model.eval()
    model.model.to("cpu")

    pruners = find_layers(model.model, layers=[ActPruneWrapper])
    for n in pruners:
        pruners[n].pruner.configure(sparsity=0.5)

    inps, outs = prealloc_inps_outs(model)
    inps_contain_layer_inps = False
    attn_mask = None
    pos_embed = None

    for i, layer in enumerate(model.model.model.layers):
        x = capture_embeddings(
            model,
            loader,
            i,
            inps,
            outs,
            inps_contain_layer_inps=inps_contain_layer_inps,
            attention_mask=attn_mask,
            position_embeddings=pos_embed,
        )
        if x is None:
            print("Error while capturing embeddings, aborting")
            break

        inps_contain_layer_inps = True
        attn_mask = getattr(x, "attention_mask")
        pos_embed = getattr(x, "position_embeddings")

        total = 0
        zeros = 0
        totalp = 0
        zerosp = 0

        for _, param in layer.named_parameters():
            _, zero_count, total_count = tensor_sparsity(param)
            zeros += zero_count
            total += total_count

        layer_dev = next(iter(layer.parameters())).device
        layer.to(model.device)
        disable_act_sparsity(model.model)

        extra = {"attention_mask": attn_mask, "position_embeddings": pos_embed}
        total_cuda_time = model.get_layerwise_perf(inps[:8], layer, extra)["time"]
        print(f"Total CUDA time unpruned: {total_cuda_time:.3f} us")

        layer.to(layer_dev)
        cleanup_memory()

        state = x.pruner_state
        for k in state:
            state[k].fasterprune()
        for k in state:
            state[k].free()
        del x
        cleanup_memory()

        layer.to(model.device)
        residual = 0.0
        disable_act_sparsity(model.model)
        with torch.no_grad():
            inps[:8] = layer(
                inps[:8], attention_mask=attn_mask, position_embeddings=pos_embed
            )
            residual = (
                (inps[:8] - outs[:8]).norm(dim=-1) / (outs[:8].norm(dim=-1) + 1e-8)
            ).mean().item()

        layer.to(layer_dev)
        total_cuda_time = model.get_layerwise_perf(inps[:8], layer, extra)["time"]
        cleanup_memory()

        for _, param in layer.named_parameters():
            _, zero_count, total_count = tensor_sparsity(param)
            zerosp += zero_count
            totalp += total_count
        actual_sparsity = zerosp / totalp

        layer_eval = build_layer_evaluation(
            i,
            duo_to_layer(DEFAULT_CONFIG),
            LayerMetrics(residual, actual_sparsity, total_cuda_time),
        )
        print(layer_eval)
        inps, outs = outs, inps

    model.model.to(model.device)


def main() -> None:
    pass


if __name__ == "__main__":
    main()
