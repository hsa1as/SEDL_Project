import torch
from torch import nn
from typing import List 

import os
import signal
import platform
import faulthandler

import torch

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


def install_segfault_handler():
    # ---- Static debug info ----
    print("=== Runtime info ===")
    print(f"PID: {os.getpid()}")
    print(f"Python : {platform.python_version()}")
    print(f"OS     : {platform.system()} {platform.release()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA   : {torch.version.cuda}")

    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            print(f"GPU    : {torch.cuda.get_device_name(dev)}")
            print(f"SM CC  : {torch.cuda.get_device_capability(dev)}")
        except Exception as e:
            print(f"GPU info query failed: {e}")
    else:
        print("CUDA   : NOT AVAILABLE")

    print("\nAttach debugger if needed:")
    print(f"  gdb -p {os.getpid()}")
    print("====================\n")

    # ---- Segfault / abort handler ----
    faulthandler.enable(all_threads=True)

    # Register for common crash signals (ignore if not supported on this OS)
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGFPE):
        try:
            faulthandler.register(sig, all_threads=True, chain=True)
            print(f"Registered faulthandler for {sig.name}")
        except (ValueError, OSError):
            # Not available on this platform / already registered
            pass



