from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Tuple
from abc import ABC, abstractmethod
import hashlib

import numpy as np
import torch
from torch import nn

from .obc import cleanup_memory, DuoGPTConfig, ActPruneWrapper, prealloc_inps_outs, capture_embeddings, disable_act_sparsity, enable_act_sparsity, DEFAULT_CONFIG
from ..model_runner import ModelRunner
from ..utils.helpers import tensor_sparsity, find_layers
from ..utils.data_utils import get_loaders
from ..utils.consts import SEQLEN, DEFAULT_DATASET, DEFAULT_MODEL_ID, DEFAULT_CALIBRATION_SAMPLES
from .tuning_structures import (LayerMetrics, build_layer_evaluation, duo_to_layer)

def stable_json_dumps(obj: Any) -> str:
    """JSON dump for canonicalization.
    - sort_keys=True
    - separators to remove whitespace
    - ensure_ascii=False for readability
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def canonicalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow canonicalized dict where nested ops are canonicalized recursively."""
    out = {}
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, Op):
            out[k] = json.loads(v.canonical_json())
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, Op):
                    new_list.append(json.loads(item.canonical_json()))
                else:
                    new_list.append(item)
            out[k] = new_list
        elif isinstance(v, dict):
            out[k] = canonicalize_dict(v)
        else:
            out[k] = v
    return out

def sha256_of_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class Op(ABC):
    """
    base class for pruning ops.
    """

    def __init__(self, op_id: str):
        self.op_id = op_id

    """
    abstract method to apply the op.
    - layer is the path to the Decoder Layer to apply the op to. path is applied on the hf transformers object
    - 
    """
    @abstractmethod
    def apply(self, model: ModelRunner, layer: str):
        pass

class ObcOp(Op):
    """
        Class for obc style pruning
        DuoGPTConfig required to create
    """

    def __init__(self, config: DuoGPTConfig = DEFAULT_CONFIG, calibration_dataset: str = DEFAULT_DATASET,
                 calibration_nsamples: int = DEFAULT_CALIBRATION_SAMPLES, model_id: str = DEFAULT_MODEL_ID):
        super().__init__("obc")
        self.calibration_dataset = calibration_dataset
        self.calibration_nsamples = calibration_nsamples
        self.config = config
        self.model_id = model_id
        self.loader = get_loaders(
            self.calibration_dataset,
            self.calibration_nsamples,
            model=self.model_id,
            seqlen=SEQLEN,
            eval_mode=False,
        )

    """
        Apply Obc style operation on a layer

    """
    def apply(self, model: ModelRunner, layer: str):

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.model.eval()
        model.model.to("cpu")

        # get layer idx from the string
        l = layer.split(".")
        assert(l[0] == "model")
        assert(l[1] == "layers")
        try: 
            layer_idx = int(l[2])
            self._apply_inner(model, layer_idx)
            cleanup_memory()
        except ValueError:
            print("(ObcOp.apply) Path string did not have the layer index at the expected location")
            exit(-1)
    
    def _apply_inner(self, model: ModelRunner, layer_idx: int):
        args = self.config
        loader = self.loader
        pruners = find_layers(model.model, layers=[ActPruneWrapper])
        for n in pruners:
            pruners[n].pruner.configure(sparsity=0.5)

        inps, outs = prealloc_inps_outs(model)
        inps_contain_layer_inps = False
        attn_mask = None
        pos_embed = None

        for i, layer in enumerate(model.model.model.layers):
            print(f"Capturing embeddings for layer {i} until {layer_idx}")
            x = capture_embeddings(
                model,
                loader,
                i,
                inps,
                outs,
                inps_contain_layer_inps=inps_contain_layer_inps,
                attention_mask=attn_mask,
                position_embeddings=pos_embed,
                args=self.config
            )
            if x is None:
                print("Error while capturing embeddings, aborting")
                break

            inps_contain_layer_inps = True
            attn_mask = getattr(x, "attention_mask")
            pos_embed = getattr(x, "position_embeddings")

            extra = {"attention_mask": attn_mask, "position_embeddings": pos_embed}
            layer_dev = next(iter(layer.parameters())).device
            disable_act_sparsity(model.model)
            cleanup_memory()

            if i == layer_idx:
                # reached layer, prune
                state = x.pruner_state
                for k in state:
                    state[k].fasterprune(args)
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

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                layer.to(layer_dev)
                total_cuda_time = model.get_layerwise_perf(inps[:8], layer, extra)["time"]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cleanup_memory()

                zerosp = 0
                totalp = 0
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
                inps, outs = outs, inps
                self.layer_eval = layer_eval
                return

            # otherwise, we have to continue    
            state = x.pruner_state
            for k in state:
                state[k].free()
            del x
            cleanup_memory()


            
class RemoveHeadOp(Op):
    op_type: str = field(init=False, default="remove_head")
    params={}

    def __init__(self, layer_path: str, head_index: int, meta: Optional[Dict[str, Any]] = None):
        super().__init__(op_id="RemoveHeadOp")
        self.params["head_index"] = int(head_index)

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            head_index=int(p["head_index"]),
            meta=data.get("meta", {}),
        )

class LowRankProjOp(Op):
    op_type: str = field(init=False, default="lowrank_proj")
    params={}

    def __init__(self, layer_path: str, proj: str = "W_Q", rank: int = 512,
                 method: str = "svd_trunc", meta: Optional[Dict[str, Any]] = None):
        super().__init__(op_id="LowRankProj")
        self.params.update({
            "proj": proj,
            "rank": int(rank),
            "method": method
        })

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            proj=p.get("proj", "W_Q"),
            rank=int(p.get("rank", 512)),
            method=p.get("method", "svd_trunc"),
            meta=data.get("meta", {}),
        )
