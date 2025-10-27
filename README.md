better_prune
============

CS6886 Course Project submission - CS25S025 R Sai Ashwin

Install
-------

- Regular install from source:
  - `pip install .`
- Editable (development) install:
  - `pip install -e .`

Requires Python 3.9+ and a working PyTorch/CUDA stack for GPU usage.

Requirements
------------

- Auto-install via helper:
  - `python -m better_prune.requirements`
  - Options: `--skip-torch` to manage torch yourself; `--pip-extra-args "..."` to forward args to pip.

- Or install manually (example):
  - `pip install accelerate>=0.27.0 datasets>=2.16.0 huggingface_hub>=0.19.0 lm_eval==0.4.0 numpy>=1.23.0 scikit-learn>=1.3.0 tokenizers>=0.15.0 transformers>=4.37.0`
  - `pip install torch>=2.1.0`  (pick the wheel that matches your CUDA/CPU setup)

Run
---

- Show CLI help:
  - `python -m better_prune --help`

- Run with defaults (calibrate on C4, then measure pruned perf and run benchmarks):
  - `python -m better_prune`

- Example with explicit options:
  - `python -m better_prune --model-id meta-llama/Meta-Llama-3-8B --tasks gsm8k boolq --measure-base-perf --run-base-benchmarks`

The entrypoint `python -m better_prune` also works 
