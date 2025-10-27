from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
import hashlib

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

@dataclass
class Op:
    """base class for pruning ops.

    subclasses should:
    - override op_type class var
    - include operation-specific fields in dataclass signature
    - not perform model mutation here applier shall do the mutation using op data.
    """
    op_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    op_type: str = field(init=False, default="op")
    layer_path: Optional[str] = None                 # canonical dotted module path (HF)
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- serialization -----------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        """Serialize op to a JSON-serializable dict (not canonicalized)."""
        return {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "layer_path": self.layer_path,
            "params": self.params,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Op":
        """Factory that uses the registry to construct the correct subclass."""
        if "op_type" not in data:
            raise ValueError("op_type missing in op json")
        op_type = data["op_type"]
        op_cls = OP_REGISTRY.get(op_type)
        if op_cls is None:
            # fallback: create a generic Op
            base = Op(
                op_id=data.get("op_id", str(uuid.uuid4())),
                layer_path=data.get("layer_path"),
                params=data.get("params", {}),
                meta=data.get("meta", {}),
            )
            base.op_type = op_type
            return base
        return op_cls._from_json_inner(data)

    @classmethod
    def _from_json_inner(cls, data: Dict[str, Any]) -> "Op":
        """Default inner constructor used by subclasses if they don't override."""
        return cls(
            op_id=data.get("op_id", str(uuid.uuid4())),
            layer_path=data.get("layer_path"),
            params=data.get("params", {}),
            meta=data.get("meta", {}),
        )

    # -- canonicalization & hashing -------------------------------------
    def canonical_dict(self) -> Dict[str, Any]:
        """Return canonical dict (sorted keys, nested ops canonicalized)."""
        base = {
            "op_id": self.op_id,
            "op_type": self.op_type,
            "layer_path": self.layer_path,
            "params": self.params,
            "meta": self.meta,
        }
        # remove ephemeral fields from meta before canonicalization if present
        canon_meta = dict(self.meta)
        # drop timestamps because they differ across runs, but keep created_by etc.
        if "created_at" in canon_meta:
            # keep created_at as part of op identity? Usually no — remove for canonical hash
            canon_meta.pop("created_at", None)
        base["meta"] = canon_meta
        return canonicalize_dict(base)

    def canonical_json(self) -> str:
        """Deterministic JSON string for hashing/caching."""
        return stable_json_dumps(self.canonical_dict())

    def sha256_hash(self) -> str:
        """Hash over canonical JSON representation."""
        return sha256_of_string(self.canonical_json())

    # -- validation hook -------------------------------------------------
    def validate(self, model=None) -> None:
        """Optional lightweight validation. Applier should run stronger checks."""
        # default: no-op
        return

    # -- apply hook ------------------------------------------------------
    def apply(self, model, applier) -> Any:
        """Apply this op to a model using an 'applier' object that implements mutation logic.

        The contract: the applier provides methods named according to op types, e.g.
        - applier.apply_shrink_ffn(op, model)
        - applier.apply_remove_head(op, model)
        ...
        This pattern keeps model-surgery code separated from the op representation.
        """
        method_name = f"apply_{self.op_type}"
        if not hasattr(applier, method_name):
            raise NotImplementedError(f"Applier has no method '{method_name}' for op {self.op_type}")
        method = getattr(applier, method_name)
        return method(self, model)

@dataclass
class MaskWeightsOp(Op):
    op_type: str = field(init=False, default="mask_weights")

    def __init__(
        self,
        layer_path: str,
        mask_type: str = "unstructured",
        pattern: Optional[str] = None,
        indices: Optional[List[int]] = None,
        threshold: Optional[float] = None,
        method: str = "magnitude",
        meta: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(layer_path=layer_path, params={}, meta=meta or {})
        self.params.update({
            "mask_type": mask_type,
            "pattern": pattern,
            "indices": indices,
            "threshold": threshold,
            "method": method,
        })

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            mask_type=p.get("mask_type", "unstructured"),
            pattern=p.get("pattern"),
            indices=p.get("indices"),
            threshold=p.get("threshold"),
            method=p.get("method", "magnitude"),
            meta=data.get("meta", {}),
        )

@dataclass
class ReplaceLayerOp(Op):
    op_type: str = field(init=False, default="replace_layer")

    def __init__(self, layer_path: str, replace_with: str, args: Optional[Dict[str, Any]] = None,
                 init_method: str = "svd_trunc", meta: Optional[Dict[str, Any]] = None):
        super().__init__(layer_path=layer_path, params={}, meta=meta or {})
        self.params.update({
            "replace_with": replace_with,
            "args": args or {},
            "init_method": init_method
        })

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            replace_with=p.get("replace_with"),
            args=p.get("args", {}),
            init_method=p.get("init_method", "svd_trunc"),
            meta=data.get("meta", {}),
        )

@dataclass
class RemoveHeadOp(Op):
    op_type: str = field(init=False, default="remove_head")

    def __init__(self, layer_path: str, head_index: int, meta: Optional[Dict[str, Any]] = None):
        super().__init__(layer_path=layer_path, params={}, meta=meta or {})
        self.params["head_index"] = int(head_index)

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            head_index=int(p["head_index"]),
            meta=data.get("meta", {}),
        )

@dataclass
class ShrinkFFNOp(Op):
    op_type: str = field(init=False, default="shrink_ffn")

    def __init__(self, layer_path: str, method: str = "pca_projection",
                 target_dim: Optional[int] = None, seed: int = 42, meta: Optional[Dict[str, Any]] = None):
        super().__init__(layer_path=layer_path, params={}, meta=meta or {})
        self.params.update({
            "method": method,
            "target_dim": target_dim,
            "seed": int(seed)
        })

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            method=p.get("method", "pca_projection"),
            target_dim=p.get("target_dim"),
            seed=p.get("seed", 42),
            meta=data.get("meta", {}),
        )

@dataclass
class LowRankProjOp(Op):
    op_type: str = field(init=False, default="lowrank_proj")

    def __init__(self, layer_path: str, proj: str = "W_Q", rank: int = 512,
                 method: str = "svd_trunc", meta: Optional[Dict[str, Any]] = None):
        super().__init__(layer_path=layer_path, params={}, meta=meta or {})
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

@dataclass
class RemoveLayerOp(Op):
    op_type: str = field(init=False, default="remove_layer")

    def __init__(self, layer_path: str, mode: str = "zero",
                 scaled_skip_init: float = 0.0, collapse_target: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None):
        super().__init__(layer_path=layer_path, params={}, meta=meta or {})
        self.params.update({
            "mode": mode,
            "scaled_skip_init": float(scaled_skip_init),
            "collapse_target": collapse_target
        })

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        return cls(
            layer_path=data.get("layer_path"),
            mode=p.get("mode", "zero"),
            scaled_skip_init=float(p.get("scaled_skip_init", 0.0)),
            collapse_target=p.get("collapse_target"),
            meta=data.get("meta", {}),
        )

@dataclass
class MetaGroupOp(Op):
    op_type: str = field(init=False, default="meta_group")

    def __init__(self, ops: List[Op], atomic: bool = True, rollback_on_fail: bool = True,
                 meta: Optional[Dict[str, Any]] = None):
        super().__init__(layer_path=None, params={}, meta=meta or {})
        # keep inner ops as Op objects
        self._ops: List[Op] = ops
        self.params.update({
            "ops": [op.to_json() for op in ops],
            "atomic": bool(atomic),
            "rollback_on_fail": bool(rollback_on_fail)
        })

    @property
    def ops(self) -> List[Op]:
        return self._ops

    def to_json(self) -> Dict[str, Any]:
        base = super().to_json()
        base["params"] = {
            "ops": [op.to_json() for op in self._ops],
            "atomic": self.params.get("atomic", True),
            "rollback_on_fail": self.params.get("rollback_on_fail", True),
        }
        return base

    @classmethod
    def _from_json_inner(cls, data):
        p = data.get("params", {})
        ops_json = p.get("ops", [])
        ops = [Op.from_json(o) for o in ops_json]
        return cls(
            ops=ops,
            atomic=p.get("atomic", True),
            rollback_on_fail=p.get("rollback_on_fail", True),
            meta=data.get("meta", {}),
        )


OP_REGISTRY: Dict[str, Type[Op]] = {
    "mask_weights": MaskWeightsOp,
    "replace_layer": ReplaceLayerOp,
    "remove_head": RemoveHeadOp,
    "shrink_ffn": ShrinkFFNOp,
    "lowrank_proj": LowRankProjOp,
    "remove_layer": RemoveLayerOp,
    "meta_group": MetaGroupOp,
}


def ops_to_canonical_hash(ops: List[Op]) -> str:
    """Compute canonical sha256 hash for a sequence of ops (order matters)."""
    canonical_list = [json.loads(op.canonical_json()) for op in ops]
    s = stable_json_dumps({"ops": canonical_list})
    return sha256_of_string(s)

def load_ops_from_list(json_list: List[Dict[str, Any]]) -> List[Op]:
    """Construct Op objects from a list of op json dicts."""
    out = []
    for j in json_list:
        out.append(Op.from_json(j))
    return out
