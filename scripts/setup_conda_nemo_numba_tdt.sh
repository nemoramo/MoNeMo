#!/usr/bin/env bash

# Create a Conda env, install NeMo with ASR+audio (numba-enabled), and run a TDT numba smoke test.

set -euo pipefail

ENV_NAME="${ENV_NAME:-ramosnemo}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10.12}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

echo "Using repo: ${REPO_ROOT}"
echo "Target conda env: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -Fx "${ENV_NAME}" >/dev/null 2>&1; then
  echo "Conda env ${ENV_NAME} already exists; skipping creation."
else
  echo "Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION}..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION} toolchain into ${ENV_NAME}..."
conda install -y -n "${ENV_NAME}" "pytorch=${PYTORCH_VERSION}" torchvision torchaudio "pytorch-cuda=${CUDA_VERSION}" -c pytorch -c nvidia

echo "Upgrading pip/setuptools/wheel..."
conda run -n "${ENV_NAME}" pip install --upgrade pip setuptools wheel

echo "Installing NeMo (editable) with ASR+audio extras..."
cd "${REPO_ROOT}"
conda run -n "${ENV_NAME}" pip install -v -e ".[asr,audio]"

echo "Running TDT numba smoke test..."
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
from nemo.collections.asr.losses.rnnt import TDTLossNumba
from nemo.core.utils import numba_utils

assert torch.cuda.is_available(), "CUDA is required for TDT numba test."

durations = [0, 1, 2, 3]
print("Numba CUDA support:", numba_utils.numba_cuda_is_supported(numba_utils.__NUMBA_MINIMUM_VERSION__))

device = "cuda"
B, T, U, V = 2, 4, 3, 5  # U = max target len + 1
acts = torch.randn(B, T, U, V + len(durations), device=device, requires_grad=True)
labels = torch.tensor([[1, 2], [2, 3]], device=device, dtype=torch.int64)
act_lens = torch.tensor([T, T], device=device, dtype=torch.int64)
label_lens = torch.tensor([2, 2], device=device, dtype=torch.int64)

loss_fn = TDTLossNumba(blank=0, durations=durations, reduction="mean", sigma=0.05, fastemit_lambda=0.001, omega=0.3)
loss = loss_fn(acts, labels, act_lens, label_lens)
loss.backward()

print("TDT numba loss:", float(loss.item()))
print("Grad finite:", torch.isfinite(acts.grad).all().item())
PY

echo "Done. To work in this env later: conda activate ${ENV_NAME}"
