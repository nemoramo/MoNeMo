#!/usr/bin/env bash
# Install NVIDIA driver 580 (server-open) and CUDA Toolkit 12.8 on Ubuntu 22.04.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

log() { echo "[cuda-setup] $*"; }

export DEBIAN_FRONTEND=noninteractive

CUDA_KEYRING="/etc/apt/keyrings/cuda-archive-keyring.gpg"
CUDA_LIST="/etc/apt/sources.list.d/cuda-ubuntu2204.list"
CUDA_REPO_LINE="deb [signed-by=${CUDA_KEYRING}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

log "Ensuring CUDA apt repo key and list are present"
mkdir -p /etc/apt/keyrings
if [[ ! -f "${CUDA_KEYRING}" ]]; then
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-archive-keyring.gpg \
    | tee "${CUDA_KEYRING}" >/dev/null
fi
echo "${CUDA_REPO_LINE}" | tee "${CUDA_LIST}" >/dev/null

log "Updating apt cache"
apt-get update

log "Installing NVIDIA driver 580 (server-open) with Fabric Manager and CUDA Toolkit 12.8"
apt-get install -y \
  nvidia-driver-580-server-open \
  nvidia-fabricmanager-580 \
  cuda-toolkit-12-8

log "Pointing /usr/local/cuda to CUDA 12.8"
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

log "Writing /etc/profile.d/cuda.sh for PATH/LD_LIBRARY_PATH"
cat >/etc/profile.d/cuda.sh <<'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
EOF

log "Refreshing linker cache"
ldconfig

if command -v /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
  log "nvcc version:"
  /usr/local/cuda/bin/nvcc --version | sed 's/^/[cuda-setup] /'
else
  log "nvcc not found; check installation output above."
fi

log "Done. Reboot recommended to load the new driver and start fabric manager."
