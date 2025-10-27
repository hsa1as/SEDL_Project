import math
from typing import Tuple, List, Dict, Optional
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def module_dtype(module: torch.nn.Module):
    for p in module.parameters():
        return p.dtype
    for b in module.buffers():
        return b.dtype
    return None  # no tensors inside

def next_power_of_two_strict(x: int) -> int:
    if x < 0:
        raise ValueError("x must be non-negative")
    return 1 << (x.bit_length())


def compute_k_for_energy(W: torch.Tensor, energy: float = 0.9, device: Optional[torch.device] = None) -> int:
    """
    Returns k (1-based count). If W is all zeros returns 0.
    """
    if W.ndim != 2:
        raise ValueError("W must be 2D (out_features, in_features)")
    dev = device or torch.device(device="cpu")
    Wf = W.detach().to(device="cuda", dtype=torch.float32)
    s = torch.linalg.svdvals(Wf)  # descending
    sv2 = s.pow(2)
    total = sv2.sum().item()
    if total == 0.0:
        return 0
    cumsum = torch.cumsum(sv2, dim=0).cpu().numpy()
    target = energy * total
    for idx, val in enumerate(cumsum):
        if val >= target:
            return idx + 1
    return s.numel()


def can_replace_linear(linear: nn.Linear, energy_threshold: float = 0.95, max_k90_threshold: int = 1024) -> Tuple[bool, int]:
    """
    Decide whether a given `nn.Linear` layer is a candidate for low-rank replacement.
    Returns:
      (can_replace, k90)
    """
    if not isinstance(linear, nn.Linear):
        return False, 0
    W = linear.weight.detach()
    k90 = compute_k_for_energy(W, energy=energy_threshold)
    gc.collect()
    torch.cuda.empty_cache()
    can = (0 < k90 < max_k90_threshold)
    return bool(can), int(k90)


class LowRankLinear(nn.Module):
    """
    Replace an nn.Linear (out_features x in_features) with a low-rank factorization
    WA @ WB where WA has shape (out_features, k) and WB has shape (k, in_features).
    """

    def __init__(self, in_features: int, out_features: int, k: int, bias: bool = True, init_from: Optional[nn.Linear] = None):
        super().__init__()
        if not is_power_of_two(k):
            raise ValueError(f"k must be a power of two (got {k})")

        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.WB = nn.Linear(in_features, k, bias=False, dtype=module_dtype(init_from) if init_from else torch.float)
        self.WA = nn.Linear(k, out_features, bias=bias, dtype=module_dtype(init_from) if init_from else torch.float)

        if init_from is not None:
            if not isinstance(init_from, nn.Linear):
                raise ValueError("init_from must be an instance of nn.Linear")
            W = init_from.weight.detach()  # shape (out_features, in_features)
            if W.shape != (out_features, in_features):
                raise ValueError("init_from weight shape doesn't match provided in/out features")

            Wf = W.float().cpu()
            U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)  # U:(out,kfull), Vt:(kfull,in)
            r = S.numel()
            k_eff = min(self.k, r)

            Uk = U[:, :k_eff]              # out x k_eff
            Sk = S[:k_eff]                 # k_eff
            Vtk = Vt[:k_eff, :]            # k_eff x in

            sqrtS = torch.sqrt(Sk)         # k_eff
            WA_init = (Uk * sqrtS.unsqueeze(0)).to(dtype=self.WA.weight.dtype, device=self.WA.weight.device)
            WB_init = (sqrtS.unsqueeze(1) * Vtk).to(dtype=self.WB.weight.dtype, device=self.WB.weight.device)

            if k_eff < self.k:
                WA_pad = torch.zeros((out_features, self.k), dtype=WA_init.dtype, device=WA_init.device)
                WA_pad[:, :k_eff] = WA_init
                WA_init = WA_pad
                WB_pad = torch.zeros((self.k, in_features), dtype=WB_init.dtype, device=WB_init.device)
                WB_pad[:k_eff, :] = WB_init
                WB_init = WB_pad

            self.WA.weight.data.copy_(WA_init.to(self.WA.weight.device, dtype=self.WA.weight.dtype))
            self.WB.weight.data.copy_(WB_init.to(self.WB.weight.device, dtype=self.WB.weight.dtype))

            if self.WA.bias is not None and init_from.bias is not None:
                self.WA.bias.data.copy_(init_from.bias.detach().to(self.WA.bias.device, dtype=self.WA.bias.dtype))
            try:
                del WA_init, WA_pad, WB_init, WB_pad, Wf, U, S, Vt, r, k_eff, Uk, Sk, Vtk, sqrtS
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as _e:
                pass
        else:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.WA(self.WB(x))


def _replace_module_in_parent(parent: nn.Module, child_name: str, new_mod: nn.Module):
    if not hasattr(parent, "_modules"):
        raise ValueError("parent module has no attribute _modules")
    parent._modules[child_name] = new_mod


def find_and_replace_linears_with_lowrank(model: nn.Module,
                                          energy_threshold: float = 0.90,
                                          max_k90_threshold: int = 1024) -> List[Dict]:
    replacements = []
    allowed_names=["q_proj", "o_proj", "gate_proj", "up_proj","down_proj"]
    def _rec(parent: nn.Module):
        if isinstance(parent, LowRankLinear):
            return
        # iterate over direct children only so we can replace in parent
        for name, child in list(parent.named_children()):
            if name not in allowed_names:
                _rec(child)
                continue
            if isinstance(child, LowRankLinear):
                continue
            # If child is Linear, evaluate
            if isinstance(child, nn.Linear):
                can, k90 = can_replace_linear(child, energy_threshold, max_k90_threshold)
                gc.collect()
                torch.cuda.empty_cache()
                if can:
                    k_chosen = next_power_of_two_strict(max(1, k90))
                    in_f = child.in_features
                    out_f = child.out_features
                    lowrank = LowRankLinear(in_features=in_f, out_features=out_f, k=k_chosen, bias=(child.bias is not None), init_from=child)
                    _replace_module_in_parent(parent, name, lowrank)
                    replacements.append({
                        'parent': parent,
                        'name': name,
                        'orig_shape': (out_f, in_f),
                        'k90': k90,
                        'k_chosen': k_chosen
                    })
                    try:
                        # delete local reference; in this scope `child` still refers to it
                        del child, in_f, out_f
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                else:
                    continue
            else:
                _rec(child)

    _rec(model)
    print("Replaced: ", len(replacements), " Layers")
    print(replacements)
    gc.collect()
    torch.cuda.empty_cache()
    return replacements
