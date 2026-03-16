#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${RAMOSNEMO_SOURCE_DIR:-/opt/ramosnemo_source}"
FALLBACK_DIR="/workspace/RamosNeMo"
EXTRAS="${RAMOSNEMO_EXTRAS:-asr,audio}"
DEFAULT_PIP_PACKAGES="${RAMOSNEMO_DEFAULT_PIP_PACKAGES:-boto3[crt] s3fs==0.4.2 tenacity tokenizers>=0.22.0,<=0.23.0 sentencepiece<1.0.0 polars>=1.6.0}"
EXTRA_PIP_PACKAGES="${RAMOSNEMO_PIP_PACKAGES:-}"

install_runtime_packages() {
  local package_string="$1"
  local label="$2"
  local -a packages=()

  if [[ -z "${package_string//[[:space:]]/}" ]]; then
    return
  fi

  # Split by shell words so requirement specs stay intact (e.g., tokenizers>=0.22.0,<=0.23.0).
  read -r -a packages <<< "${package_string}"
  echo "${label}: ${package_string}"
  python -m pip install --no-cache-dir "${packages[@]}"
}

if [[ -z "${AWS_ACCESS_KEY_ID:-}" && -n "${TOS_ACCESS_KEY_ID:-}" ]]; then
  export AWS_ACCESS_KEY_ID="${TOS_ACCESS_KEY_ID}"
fi
if [[ -z "${AWS_SECRET_ACCESS_KEY:-}" && -n "${TOS_SECRET_ACCESS_KEY:-}" ]]; then
  export AWS_SECRET_ACCESS_KEY="${TOS_SECRET_ACCESS_KEY}"
fi
if [[ -z "${AWS_DEFAULT_REGION:-}" && -n "${TOS_REGION:-}" ]]; then
  export AWS_DEFAULT_REGION="${TOS_REGION}"
fi
if [[ -z "${AWS_ENDPOINT_URL:-}" && -n "${TOS_ENDPOINT:-}" ]]; then
  export AWS_ENDPOINT_URL="${TOS_ENDPOINT}"
fi
if [[ -z "${AWS_SESSION_TOKEN:-}" && -n "${TOS_SESSION_TOKEN:-}" ]]; then
  export AWS_SESSION_TOKEN="${TOS_SESSION_TOKEN}"
fi
if [[ -z "${AWS_S3_ADDRESSING_STYLE:-}" && -n "${TOS_ADDRESSING_STYLE:-}" ]]; then
  export AWS_S3_ADDRESSING_STYLE="${TOS_ADDRESSING_STYLE}"
fi

if [[ -d "${SOURCE_DIR}" ]]; then
  install_dir="${SOURCE_DIR}"
elif [[ -d "${FALLBACK_DIR}" ]]; then
  install_dir="${FALLBACK_DIR}"
else
  echo "ERROR: RamosNeMo source not found." >&2
  echo "Checked: ${SOURCE_DIR} and ${FALLBACK_DIR}" >&2
  exit 1
fi

if [[ -n "${EXTRAS}" && "${EXTRAS}" != "none" ]]; then
  echo "Installing RamosNeMo (editable) from: ${install_dir} (extras: ${EXTRAS})"
  python -m pip install -e "${install_dir}[${EXTRAS}]"
else
  echo "Installing RamosNeMo (editable) from: ${install_dir}"
  python -m pip install -e "${install_dir}"
fi

install_runtime_packages "${DEFAULT_PIP_PACKAGES}" "Ensuring default runtime packages"
install_runtime_packages "${EXTRA_PIP_PACKAGES}" "Installing extra runtime packages"
