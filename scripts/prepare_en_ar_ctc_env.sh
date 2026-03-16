#!/usr/bin/env bash
set -euo pipefail

GPUS=1
TRAIN_MANIFEST=""
VAL_MANIFEST=""
TOKENIZER_DIR=""
PRETRAINED_NEMO=""
RUN_NAME="en-ar-ctc-110m-$(date +%Y%m%d-%H%M%S)"
EXP_DIR="/data2/mayufeng/nemo_exps"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_en_ar_ctc_env.sh \
    --gpus 1|2|4 \
    --train-manifest /abs/path/train.manifest \
    --val-manifest /abs/path/val.manifest \
    --tokenizer-dir /abs/path/tokenizer_dir \
    --pretrained-nemo /abs/path/parakeet_110m.nemo \
    [--exp-dir /data2/mayufeng/nemo_exps] \
    [--run-name en-ar-ctc-110m-xxx]
EOF
}

check_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "[OK] command found: ${cmd}"
    return 0
  fi
  echo "[FAIL] command missing: ${cmd}"
  return 1
}

check_dir() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    echo "[OK] dir exists: ${p}"
    return 0
  fi
  echo "[FAIL] dir missing: ${p}"
  return 1
}

check_file() {
  local p="$1"
  if [[ -f "${p}" ]]; then
    echo "[OK] file exists: ${p}"
    return 0
  fi
  echo "[FAIL] file missing: ${p}"
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="${2:-}"
      shift 2
      ;;
    --train-manifest)
      TRAIN_MANIFEST="${2:-}"
      shift 2
      ;;
    --val-manifest)
      VAL_MANIFEST="${2:-}"
      shift 2
      ;;
    --tokenizer-dir)
      TOKENIZER_DIR="${2:-}"
      shift 2
      ;;
    --pretrained-nemo)
      PRETRAINED_NEMO="${2:-}"
      shift 2
      ;;
    --exp-dir)
      EXP_DIR="${2:-}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[FAIL] unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

fail=0

case "${GPUS}" in
  1|2|4)
    echo "[OK] gpus setting: ${GPUS}"
    ;;
  *)
    echo "[FAIL] --gpus must be one of: 1, 2, 4"
    fail=1
    ;;
esac

check_cmd docker || fail=1
check_cmd nvidia-smi || fail=1

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "[OK] docker daemon reachable"
  else
    echo "[FAIL] docker daemon not reachable"
    fail=1
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    echo "[OK] nvidia-smi works"
  else
    echo "[FAIL] nvidia-smi failed"
    fail=1
  fi
fi

for d in /data1 /data2 /data3 /mnt/asr-audio-data; do
  check_dir "${d}" || fail=1
done

if [[ -z "${TRAIN_MANIFEST}" ]]; then
  echo "[FAIL] missing --train-manifest"
  fail=1
else
  check_file "${TRAIN_MANIFEST}" || fail=1
fi

if [[ -z "${VAL_MANIFEST}" ]]; then
  echo "[FAIL] missing --val-manifest"
  fail=1
else
  check_file "${VAL_MANIFEST}" || fail=1
fi

if [[ -z "${TOKENIZER_DIR}" ]]; then
  echo "[FAIL] missing --tokenizer-dir"
  fail=1
else
  check_dir "${TOKENIZER_DIR}" || fail=1
fi

if [[ -z "${PRETRAINED_NEMO}" ]]; then
  echo "[FAIL] missing --pretrained-nemo"
  fail=1
else
  check_file "${PRETRAINED_NEMO}" || fail=1
fi

if [[ -z "${EXP_DIR}" ]]; then
  echo "[FAIL] --exp-dir cannot be empty"
  fail=1
else
  mkdir -p "${EXP_DIR}" || fail=1
  if touch "${EXP_DIR}/.write_test.$$" 2>/dev/null; then
    rm -f "${EXP_DIR}/.write_test.$$"
    echo "[OK] exp_dir writable: ${EXP_DIR}"
  else
    echo "[FAIL] exp_dir not writable: ${EXP_DIR}"
    fail=1
  fi
fi

echo
if [[ "${fail}" -eq 0 ]]; then
  echo "[PASS] environment check passed."
else
  echo "[FAIL] environment check failed."
fi

echo
echo "Next step command:"
echo "bash scripts/train_en_ar_ctc_docker.sh \\"
echo "  --gpus ${GPUS} \\"
echo "  --train-manifest ${TRAIN_MANIFEST} \\"
echo "  --val-manifest ${VAL_MANIFEST} \\"
echo "  --tokenizer-dir ${TOKENIZER_DIR} \\"
echo "  --pretrained-nemo ${PRETRAINED_NEMO} \\"
echo "  --exp-dir ${EXP_DIR} \\"
echo "  --run-name ${RUN_NAME}"

exit "${fail}"
