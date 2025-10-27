from pathlib import Path
from dataclasses import dataclass

MODEL_CACHE = Path.home() / ".cache" / "better_prune" / "models"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

@dataclass
class Verbosity:
    info: bool = True
    memory: bool = False

VERBOSITY = Verbosity()
