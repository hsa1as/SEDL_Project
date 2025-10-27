from __future__ import annotations

import argparse
import subprocess
import sys

CORE_PACKAGES = [
    "accelerate>=0.27.0",
    "datasets>=2.16.0",
    "huggingface_hub>=0.19.0",
    "lm_eval==0.4.0",
    "numpy>=1.23.0",
    "scikit-learn>=1.3.0",
    "tokenizers>=0.15.0",
    "transformers>=4.37.0",
]

TORCH_PACKAGE = "torch>=2.1.0"


def install_packages(packages: list[str], extra_args: list[str]) -> None:
    """Install a list of packages using pip."""
    if not packages:
        return
    command = [sys.executable, "-m", "pip", "install", "--upgrade", *packages, *extra_args]
    print(" ".join(command))
    subprocess.check_call(command)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Install BetterPrune dependencies.")
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip installing torch. Useful if you manage torch separately for GPU support.",
    )
    parser.add_argument(
        "--pip-extra-args",
        default="",
        help="Additional arguments forwarded to pip (quoted string).",
    )
    return parser.parse_args()


def main() -> None:
    """Install core dependencies required for benchmarking and pruning."""
    args = parse_args()
    extra_args = args.pip_extra_args.split() if args.pip_extra_args else []
    install_packages(CORE_PACKAGES, extra_args)
    if not args.skip_torch:
        install_packages([TORCH_PACKAGE], extra_args)


if __name__ == "__main__":
    main()
