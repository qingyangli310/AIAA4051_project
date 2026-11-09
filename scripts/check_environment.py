"""Check the shared Project 6 software and GPU environment."""

from __future__ import annotations

import platform
import sys

import torch
import transformers


EXPECTED_PYTHON = (3, 11)
EXPECTED_TORCH = "2.9.1"
EXPECTED_TRANSFORMERS = "4.57.6"
EXPECTED_CUDA = "12.6"


def main() -> int:
    python_version = platform.python_version()
    torch_version = torch.__version__.split("+")[0]
    transformers_version = transformers.__version__
    cuda_runtime = torch.version.cuda or "not available"
    cuda_available = torch.cuda.is_available()

    print(f"Python: {python_version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers_version}")
    print(f"CUDA runtime: {cuda_runtime}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        memory_gb = properties.total_memory / 1024**3
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {memory_gb:.2f} GB")
    else:
        print("GPU: none")

    problems: list[str] = []

    if sys.version_info[:2] != EXPECTED_PYTHON:
        problems.append(
            f"Expected Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.x, "
            f"found {python_version}."
        )
    if torch_version != EXPECTED_TORCH:
        problems.append(
            f"Expected PyTorch {EXPECTED_TORCH}, found {torch.__version__}."
        )
    if transformers_version != EXPECTED_TRANSFORMERS:
        problems.append(
            "Expected Transformers "
            f"{EXPECTED_TRANSFORMERS}, found {transformers_version}."
        )
    if cuda_available and cuda_runtime != EXPECTED_CUDA:
        problems.append(
            f"Expected CUDA runtime {EXPECTED_CUDA}, found {cuda_runtime}."
        )

    if problems:
        print("\nEnvironment check: FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1

    if not cuda_available:
        print("\nEnvironment check: CPU DEVELOPMENT MODE")
        print("The software versions match, but final model experiments require the RTX 4090.")
        return 0

    print("\nEnvironment check: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
