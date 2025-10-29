from pathlib import Path
from dataclasses import dataclass

MODEL_CACHE = Path.home() / ".cache" / "better_prune" / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

@dataclass
class Verbosity:
    info: bool = True
    memory: bool = True

VERBOSITY = Verbosity()

SEQLEN = 2048
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
#DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATASET = "c4"
DEFAULT_CALIBRATION_SAMPLES = 128
DEFAULT_BATCH_SIZES = (64, 128, 256)
DEFAULT_MAX_TOKENS_PERF = 32
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TASKS = ("boolq",)
LATENCY_RUNS = 5
THROUGHPUT_RUNS = 10
BATCH_RUNS = 5
