import torch
from torch import nn
from typing import List 

def find_layers(module: nn.Module, layers: List[nn.Module],
                name: str=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers,
            name=name + '.' + name1 if name != '' else name1
        ))
    return res

@torch.no_grad()
def tensor_sparsity(t: torch.Tensor) -> float:
    total = t.numel()
    zeros = (t == 0).sum().item()
    return zeros / total, zeros, total
