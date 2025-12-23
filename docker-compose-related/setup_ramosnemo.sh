#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${RAMOSNEMO_SOURCE_DIR:-/opt/ramosnemo_source}"
FALLBACK_DIR="/workspace/RamosNeMo"
EXTRAS="${RAMOSNEMO_EXTRAS:-asr,audio}"

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
