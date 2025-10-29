import math
import transformers
import torch
import torch.nn as nn
import tqdm

from typing import Dict, List, NamedTuple, Optional, Any, Tuple
from dataclasses import dataclass
import functools

from ..model_runner import ModelRunner
from ..utils.helpers import find_layers
from ..utils.consts import VERBOSITY

@dataclass
class DuoGPTConfig:
    dev: torch.device = torch.device("cuda")
    enable_ap_calibration: bool = True
    w_bits: int = 32
    prunen: int = 0
    prunem: int = 0
    percdamp: float =  0.05
    blocksize: int = 128
    scale_alpha: float = 0.125
    nsamples: int =128
    a_sparsity: float = 0.5
    sparsity: float = 0.5
    act_order:bool = True
    block_act: bool = False
    dxxt_permutation: bool = False
    act_blocksize: int = 32

DEFAULT_CONFIG = DuoGPTConfig()

class DuoGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []
        self.dXdXT = torch.zeros(self.columns, device=self.dev) #! added for DuoGPT
        self.residual: None | torch.Tensor = None

    @torch.no_grad() 
    def update_residual(self, residual: torch.Tensor) -> None:
        self.residual=residual

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.dXdXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(1 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

        #! DuoGPT's feature
        dX = self.fp_inp[0].float() * math.sqrt(1 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())
        self.dXdXT += torch.sum(dX**2, dim=1)
        
        del self.fp_inp[0]
    @torch.no_grad
    def fasterprune(self, args: DuoGPTConfig = DEFAULT_CONFIG):
        self.layer.to(args.dev)
        sparsity = args.sparsity
        prunen=args.prunen
        prunem=args.prunem
        blocksize=args.blocksize
        percdamp=args.percdamp
        actorder=args.act_order
        alpha=args.scale_alpha
        W = self.layer.weight.data.clone()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dXdXT = self.dXdXT
        del self.dXdXT

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        # self.dXXT[:, dead] = 0 #! Does not require this, guarantee to be 0 for dead entry.

        if actorder:
            if args.block_act: 
                global_sorted_indices = torch.argsort(torch.diag(H), descending=True)
                perm = torch.zeros(self.columns, dtype=torch.int, device=self.dev)
                num_blocks = self.columns // blocksize
                for i, global_idx in enumerate(global_sorted_indices):
                    block_num = (i // args.act_blocksize) % num_blocks
                    offset_within_block = (i // (args.act_blocksize * num_blocks)) * args.act_blocksize
                    pos_within_block = i % args.act_blocksize
                    new_pos = block_num * blocksize + offset_within_block + pos_within_block
                    perm[new_pos] = global_idx
            else:
                perm = torch.argsort(torch.diag(H), descending=True) if not args.dxxt_permutation else torch.argsort(torch.diag(H) + torch.diag(torch.abs(self.dXXT)), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            dXdXT = dXdXT[perm] #! New for DuoGPT
            invperm = torch.argsort(perm)
            

        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        #! V2 implementation
        mask_triangle = torch.ones_like(Hinv).triu_(diagonal=1)
        dXXTL = self.dXXT @ Hinv.T
        P = alpha * (dXXTL * mask_triangle) @ Hinv

        dXXTL2 = torch.sum((dXXTL*mask_triangle)**2, dim=1)
        lP_COE4 = 2*torch.diag(dXXTL)/torch.diag(Hinv)
        del self.dXXT
        

        mask = None #! Add the masks for pruning weights

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            #! focus on the current block
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2] #! V2

            Lp_coe2 = dXdXT[i1:i2].unsqueeze(0)
            Lp_coe3 = dXXTL2[i1:i2].unsqueeze(0)
            Lp_coe4 = lP_COE4[i1:i2].unsqueeze(0)

            #! Mask selection
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    #! Fold the division into a multiplication of scale will result into small deviation.
                    #! Scale by alpha to align with the updates.
                    tmp = W1 ** 2 / torch.diag(Hinv1).reshape((1, -1)) ** 2 + W1**2 * (Lp_coe2-Lp_coe3+Lp_coe4) * alpha

                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)] #! Keep using this method to align with SparseGPT.
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1


            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0 and prunen < W1[:, i:(i+prunem)].shape[1]:
                    tmp = W1[:, i:(i + prunem)] ** 2 / torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1)) ** 2 + W1[:, i:(i + prunem)]**2 * (Lp_coe2-Lp_coe3+Lp_coe4)[:,i:(i + prunem)] * alpha
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)


                q = w.clone()
                q[mask1[:, i]] = 0
                
                if hasattr(self, 'quantizer'):
                    q = self.quantizer.quantize(q.unsqueeze(1)).flatten()

                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))#! V2
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])#! lazy batch update

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            raise ValueError('NaN in weights')
        return mask

    def free(self):
        del self.fp_inp
        self.layer.to(torch.device("cpu"))
        self.H = None
        # self.Losses = None
        self.dXXT = None
        self.dXdXT = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        cleanup_memory()


class FPInputsCache:
    """
    class for saving the full-precision output in each layer.
    """
    def __init__(self, sequential):
        self.fp_cache = {}
        self.names = sequential[0]+sequential[1]+sequential[2]+sequential[3]
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        self.fp_cache[name] += [inp.t()]
        

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []

class CalibrationResult(NamedTuple):
    layer_idx: int
    layer_inp: torch.Tensor
    layer_out: torch.Tensor
    pruner_state: Dict[str,DuoGPT]
    position_embeddings: Tuple[torch.Tensor, torch.Tensor] | None
    attention_mask: torch.Tensor | None

def prealloc_inps_outs(model_runner: ModelRunner, args: DuoGPTConfig = DEFAULT_CONFIG) ->Tuple[torch.Tensor, torch.Tensor]:
    dtype = next(iter(model_runner.model.parameters())).dtype
    dev = model_runner.device
    inps = torch.zeros((args.nsamples, model_runner.model.seqlen, model_runner.model.config.hidden_size),
                        dtype=dtype, device=dev)
    outs = torch.zeros_like(inps)
    return (inps, outs)

# process single layer
@torch.no_grad()
def process_layer(layer, fp_inputs_cache:FPInputsCache, fp_inps:torch.Tensor, attention_mask,
                  position_embeddings, args: DuoGPTConfig, full, sequential, inps: torch.Tensor,
                  outs: torch.Tensor) -> Dict[str, DuoGPT]:
    dev = inps.device
    layer_dev = next(iter(layer.parameters())).device
    layer.to(dev)
    fp_inputs_cache.add_hook(full)#! V2
    if args.enable_ap_calibration:
        disable_act_sparsity(layer)
    #! getting FP X
    for j in range(args.nsamples):
        fp_inps[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
    fp_inputs_cache.clear_hook()

    if args.enable_ap_calibration:
        enable_act_sparsity(layer, args.a_sparsity)#! turn on activation sparsity

    gpts = {}
    for names in sequential:
        subset = {n: full[n] for n in names}

        for name in subset:
            gpts[name] = DuoGPT(subset[name])
            gpts[name].fp_inp = fp_inputs_cache.fp_cache[name] #! FP X
            if args.w_bits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.w_bits, perchannel=True, sym=False, mse=False, grouprows=128
                )
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        #! only calculate the H and dXXT for one of the parallel block, then duplicate.
        first_module_name = list(subset.keys())[0]
        handle = subset[first_module_name].register_forward_hook(add_batch(first_module_name))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,
                        position_embeddings=position_embeddings)[0]
        handle.remove()

        # copy H and dXXT, and dXdXT
        for name in subset:
            if name != first_module_name:
                gpts[name].H = gpts[first_module_name].H
                gpts[name].dXXT = gpts[first_module_name].dXXT
                gpts[name].dXdXT = gpts[first_module_name].dXdXT #! New for DuoGPT
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    layer.to(layer_dev) # move back
    cleanup_memory()
    return gpts


# do this layer by layer, i.e need to put model on cpu, move individual layers to gpu
# to avoid oom
# warn: possibly invalidates any layer_inp contained in stored LayerCalibrationResult
# this will give you the DuoGPT struct for one decoder stack, all linear layers inside 
# entire model needs to be in cpu() before this is called. it will move required layers to gpu
# inps contains the tensor to be used to generate the values at the input of the layer being processed
# if inps_contain_layer_inps is True, there is no need to generate it from previous layers procedurally
# and the function assumes that inps is already valid
# remember to swap inps and outs after each layer processing, 
@torch.no_grad()
def capture_embeddings(model_runner: ModelRunner, dataloader: Any, 
                       layer_idx: int, 
                       inps: torch.Tensor, outs: torch.Tensor, inps_contain_layer_inps:bool = False,
                       position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
                       attention_mask: Optional[torch.Tensor] = None,
                       args: DuoGPTConfig = DEFAULT_CONFIG) -> CalibrationResult | None:
    print(f"Capturing embeddings for layer {layer_idx}")
    model = model_runner.model
    nsamples = args.nsamples
    if(layer_idx >= len(model.model.layers)):
        print("WARN: Attempting to calibrate non-existent layer")
        print("      Embeddings not captured")
        return

    dev = model_runner.device
    layers = model.model.layers
    model.model.norm = model.model.norm.to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)

    
    # start by moving first to gpu
    layers[0] = layers[0].to(dev)

    # inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size),
    #                    dtype=dtype, device=dev)
    # use preallocated inps and outs

    expected_inps_shape: tuple[int,int,int] = (nsamples, model.seqlen, model.config.hidden_size)
    if(inps.shape != expected_inps_shape):
        print("WRONG PREALLOCATED INPS SHAPES ENCOUNTERED")
        print("Encountered ", inps.shape)
        print("Expected", expected_inps_shape)
        print("Calibration stopped with error")
        return
    if(outs.shape != expected_inps_shape):
        print("WRONG PREALLOCATED outs SHAPES ENCOUNTERED")
        print("Encountered ", outs.shape)
        print("Expected", expected_inps_shape)
        print("Calibration stopped with error")
        return
    if not inps_contain_layer_inps:
        cache = {'i': 0, 'attention_mask': None, 'position_embeddings': None}
        class Cacher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                if hasattr(module, "attention_type"):
                    self.attention_type = module.attention_type

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_embeddings'] = kwargs['position_embeddings']
                raise ValueError # lol

        layers[0] = Cacher(layers[0]) #! Add catcher at the first transformer layer.
        for batch in dataloader:
            try:
                model(batch[0].to(dev)) #! This will catch the output states from the embedding.
            except ValueError:
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        layers[0] = layers[0].module #! Remove the catcher module.
        layers[0] = layers[0].cpu() #! Put the layer back to cpu.

        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        model.model.rotary_emb = model.model.rotary_emb.cpu()

        torch.cuda.empty_cache()
        position_embeddings = cache['position_embeddings']
        attention_mask = cache['attention_mask']

    sequential = [
        ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
        ['self_attn.o_proj.module'],
        ['mlp.up_proj.module', 'mlp.gate_proj.module'],
        ['mlp.down_proj.module']
    ]

    fp_inputs_cache = FPInputsCache(sequential)
    fp_inps = inps.clone()
    i = 0
    if inps_contain_layer_inps:
        i = layer_idx
    while i < layer_idx + 1:
        full = find_layers(layers[i], layers=[torch.nn.Linear])
        if VERBOSITY.info:
            print("Processing layer ", i)
        gpts = process_layer(layers[i], fp_inputs_cache, 
                            fp_inps, attention_mask,
                            position_embeddings, args, full,
                            sequential, inps, outs)
        for names in sequential:
            subset = {n: full[n] for n in names}
            if i != layer_idx:
                for name in subset:
                    #gpts[name].fasterprune(
                    #    args=args
                    #)
                    gpts[name].free()

        #! For generating the outputs for the next layer.
        #  no need if this is the target layer_idx
        if i != layer_idx:
            fp_inputs_cache.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            layers[i] = layers[i].cpu()
            del gpts
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inps, outs = outs, inps
        else:
            return CalibrationResult(layer_idx, inps, outs, gpts, position_embeddings, attention_mask)
        i += 1

######### activation stuff ##########
def prune(x: torch.Tensor, sparsity: float =0.5):

    #! there will be case for opt models, there is a reshape before the fc1,
    if len(x.shape)==2:
        x = x.unsqueeze(0)
    #* x shape: [batch_size, seq_len, hidden_dim]
    
    #! For each token or batch, find the threshold value (sort along hidden_dim axis)
    #! In reality, for decoding, batch and token will assumed be 1.
    thresh = torch.sort(torch.abs(x), dim=-1)[0][:, :, int(x.shape[-1] * sparsity)].unsqueeze(-1)

    # thresh, _ = torch.kthvalue(torch.abs(x),int(x.shape[-1] * sparsity))
    mask = torch.abs(x) >= thresh
    return x * mask

class ActPruner(torch.nn.Module):
    '''
        A class for pruning the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActPruner, self).__init__()
        self.sparsity = 0.
        self.enable = True
        self.mask = None
        self.annealr = 0.
        self.annealing = False
        self.annealer_cnter = 0


    def free(self):
        self.mask = None
        self.annealer_cnter = 0

    def forward(self, x):
        if self.enable:
            x = prune(x, self.sparsity)
        return x

    def configure(self, sparsity, annealing=False, annealer = 0.0):
        self.sparsity=sparsity
        self.annealing=annealing
        self.annealer=annealer
        assert self.sparsity < 1 and self.sparsity >= 0, 'sparsity should be in [0, 1)'

class ActPruneWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation pruning.
    '''

    def __init__(self, module:torch.nn.Linear, name=None):
        super(ActPruneWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        if name:
            self.name = name
        else:
            self.name = 'nobody'
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.pruner = ActPruner()

    def extra_repr(self) -> str:
        str_ = f'Input Pruner Sparsity: {self.pruner.sparsity}'
        return str_

    def forward(self, x):
        x = self.pruner(x)
        x = self.module(x)
        return x

def add_actprune(module: nn.Module, name: str='',
                 layers: List = 
                    [torch.nn.Linear,
                    ActPruneWrapper,
                    transformers.models.falcon.modeling_falcon.FalconLinear
                    ]):
    if isinstance(module, ActPruneWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActPruneWrapper(tmp,name=attr))
        if type(tmp) is torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActPruneWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) is torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActPruneWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actprune(child, name + '.' + name1 if name != '' else name1, layers)

def disable_act_sparsity(module: nn.Module):
    for name, m in module.named_modules():
        if isinstance(m, ActPruneWrapper):
            m.pruner.enable = False

def enable_act_sparsity(module: nn.Module, sparsity: float | None):
    for name, m in module.named_modules():
        if isinstance(m, ActPruneWrapper):
            m.pruner.enable = True
            if sparsity is not None:
                m.pruner.sparsity = sparsity
            


##### quantizer #####

def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
            self,
            bits, perchannel=False, sym=True, 
            mse=False, norm=2.4, grid=100, maxshrink=.8,
            grouprows=1
        ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.grouprows = grouprows

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if self.grouprows > 1: 
                    x = x.reshape((x.shape[0] // self.grouprows, -1))
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            if self.grouprows > 1:
                self.scale = self.scale.unsqueeze(1).repeat(1, self.grouprows)
                self.zero = self.zero.unsqueeze(1).repeat(1, self.grouprows)
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


#### defeat oom ####
def cleanup_memory() -> None:
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if VERBOSITY.memory:
            print(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
