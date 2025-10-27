import random
from typing import Any, List, Tuple

import datasets
import transformers

from better_prune.utils.consts import MODEL_CACHE


# Reuse the shared HF cache path from project constants
HF_CACHE_PATH = MODEL_CACHE


def _build_tokenizer(model: str, hf_token):
    kwargs = {"use_fast": False, "cache_dir": HF_CACHE_PATH}
    if hf_token is not None:
        kwargs["use_auth_token"] = hf_token
    return transformers.AutoTokenizer.from_pretrained(model, **kwargs)


def _sample_windows_from_encoding(enc, nsamples: int, seed: int, seqlen: int) -> List[Tuple[Any, Any]]:
    random.seed(seed)
    windows: List[Tuple[Any, Any]] = []
    total = enc.input_ids.shape[1]
    for _ in range(nsamples):
        i = random.randint(0, total - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        windows.append((inp, tar))
    return windows


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    tokenizer = _build_tokenizer(model, hf_token)

    if eval_mode:
        testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    traindata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    return _sample_windows_from_encoding(trainenc, nsamples, seed, seqlen)


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _build_tokenizer(model, hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        return TokenizerWrapper(valenc)

    traindata = datasets.load_dataset(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )

    random.seed(seed)
    trainloader: List[Tuple[Any, Any]] = []
    for _ in range(nsamples):
        # Find a document long enough, then slice a random contiguous window
        while True:
            idx = random.randint(0, len(traindata) - 1)
            enc = tokenizer(traindata[idx]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        trainloader.extend(_sample_windows_from_encoding(enc, 1, seed, seqlen))
    return trainloader


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    tokenizer = _build_tokenizer(model, hf_token)

    if eval_mode:
        testdata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="test")
        return tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    traindata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="train")
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    return _sample_windows_from_encoding(trainenc, nsamples, seed, seqlen)


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "ptb" in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "c4" in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
