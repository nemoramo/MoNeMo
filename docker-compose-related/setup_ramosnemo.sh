#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${RAMOSNEMO_SOURCE_DIR:-/opt/ramosnemo_source}"
FALLBACK_DIR="/workspace/RamosNeMo"
EXTRAS="${RAMOSNEMO_EXTRAS:-asr,audio}"
DEFAULT_PIP_PACKAGES="${RAMOSNEMO_DEFAULT_PIP_PACKAGES:-boto3[crt] s3fs==0.4.2 tenacity tokenizers>=0.22.0,<=0.23.0 sentencepiece<1.0.0 polars>=1.6.0}"
EXTRA_PIP_PACKAGES="${RAMOSNEMO_PIP_PACKAGES:-}"
INSTALL_RUNTIME_PACKAGES="${RAMOSNEMO_INSTALL_RUNTIME_PACKAGES:-0}"
RUNTIME_INSTALL_MODE="${RAMOSNEMO_RUNTIME_PACKAGE_INSTALL_MODE:-missing}"

is_truthy() {
  local value="${1:-}"
  value="${value,,}"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

extract_dist_name() {
  local spec="$1"
  local dist_name="${spec%%[<>=!~;[:space:]]*}"
  dist_name="${dist_name%%\[*}"
  printf '%s' "${dist_name}"
}

is_dist_installed() {
  local dist_name="$1"
  python - "$dist_name" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

name = sys.argv[1]
try:
    version(name)
except PackageNotFoundError:
    raise SystemExit(1)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

install_runtime_packages() {
  local package_string="$1"
  local label="$2"
  local mode="$3"
  local dist_name=""
  local -a packages=()
  local -a install_list=()

  if [[ -z "${package_string//[[:space:]]/}" ]]; then
    return
  fi

  # Split by shell words so requirement specs stay intact (e.g., tokenizers>=0.22.0,<=0.23.0).
  read -r -a packages <<< "${package_string}"

  if [[ "${mode}" == "all" ]]; then
    install_list=("${packages[@]}")
  else
    for package_spec in "${packages[@]}"; do
      dist_name="$(extract_dist_name "${package_spec}")"
      if [[ -z "${dist_name}" ]]; then
        install_list+=("${package_spec}")
        continue
      fi
      if is_dist_installed "${dist_name}"; then
        echo "Runtime package already installed, skip: ${package_spec}"
      else
        install_list+=("${package_spec}")
      fi
    done
  fi

  if [[ "${#install_list[@]}" -eq 0 ]]; then
    echo "${label}: all packages already available, nothing to install"
    return
  fi

  echo "${label}: installing ${install_list[*]}"
  python -m pip install --no-cache-dir "${install_list[@]}"
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

runtime_mode="${RUNTIME_INSTALL_MODE,,}"
if [[ "${runtime_mode}" != "all" && "${runtime_mode}" != "missing" ]]; then
  echo "WARN: invalid RAMOSNEMO_RUNTIME_PACKAGE_INSTALL_MODE='${RUNTIME_INSTALL_MODE}', fallback to 'missing'" >&2
  runtime_mode="missing"
fi

if is_truthy "${INSTALL_RUNTIME_PACKAGES}"; then
  install_runtime_packages "${DEFAULT_PIP_PACKAGES}" "Ensuring default runtime packages" "${runtime_mode}"
  install_runtime_packages "${EXTRA_PIP_PACKAGES}" "Installing extra runtime packages" "${runtime_mode}"
else
  echo "Skipping runtime package installation (set RAMOSNEMO_INSTALL_RUNTIME_PACKAGES=1 to enable)"
fi
