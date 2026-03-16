#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/fix_run_permissions.sh /path/to/exp_dir
  bash scripts/fix_run_permissions.sh --exp-dir /path/to/exp_dir
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

EXP_DIR=""

if [[ $# -eq 1 && "${1}" != "--exp-dir" && "${1}" != "-h" && "${1}" != "--help" ]]; then
  EXP_DIR="$1"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --exp-dir)
        EXP_DIR="${2:-}"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
fi

[[ -n "${EXP_DIR}" ]] || die "exp_dir is required"
[[ -e "${EXP_DIR}" ]] || die "path does not exist: ${EXP_DIR}"

echo "[INFO] chown -R mayufeng:mayufeng ${EXP_DIR}"
chown -R mayufeng:mayufeng "${EXP_DIR}"
echo "[OK] ownership updated: ${EXP_DIR}"
