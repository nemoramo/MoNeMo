#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${NEMO_IMAGE:-nemo-25.11-custom:latest}"
MAX_EPOCHS=15
TRAIN_BSZ=16
VAL_BSZ=16
VAL_CHECK_INTERVAL=2000
SAVE_TOP_K=10
NUM_WORKERS=""
TRAIN_PREFETCH_FACTOR=""
VAL_PREFETCH_FACTOR=""
PERSISTENT_WORKERS=""

GPUS=1
TRAIN_MANIFEST=""
VAL_MANIFEST=""
TOKENIZER_DIR=""
PRETRAINED_NEMO=""
RUN_NAME="en-ar-ctc-110m-$(date +%Y%m%d-%H%M%S)"
EXP_DIR="/data2/mayufeng/nemo_exps"
TOS_ENV_FILE=""
CONTAINER_NAME=""
KEEP_CONTAINER=0
DETACH=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_en_ar_ctc_docker.sh \
    --gpus 1|2|4 \
    --train-manifest /abs/path/train.manifest \
    --val-manifest /abs/path/val.manifest \
    --tokenizer-dir /abs/path/tokenizer_dir \
    --pretrained-nemo /abs/path/parakeet_110m.nemo \
    [--exp-dir /data2/mayufeng/nemo_exps] \
    [--run-name en-ar-ctc-110m-xxx] \
    [--max-epochs 15] \
    [--train-bsz 16] \
    [--val-bsz 16] \
    [--val-check-interval 2000] \
    [--save-top-k 10] \
    [--num-workers 8] \
    [--train-prefetch-factor 4] \
    [--val-prefetch-factor 4] \
    [--persistent-workers true|false] \
    [--keep-container] \
    [--detach] \
    [--tos-env-file /home/mayufeng/projects/speech_related_tools/.env] \
    [--container-name en_ar_ctc_110m_4g_tos]

Notes:
  - Uses examples/asr/asr_ctc/speech_to_text_ctc_bpe.py
  - Uses fast-conformer_ctc_bpe_110m
  - Supports 1/2/4 GPUs
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

read_env_value() {
  local env_file="$1"
  local env_key="$2"
  awk -F= -v k="$env_key" '
    $0 !~ /^[[:space:]]*#/ && index($0, "=") > 0 {
      key=$1
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      if (key == k) {
        val=substr($0, index($0, "=") + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
        print val
        exit
      }
    }
  ' "$env_file"
}

strip_wrapping_quotes() {
  local value="$1"
  value="${value#\"}"
  value="${value%\"}"
  value="${value#\'}"
  value="${value%\'}"
  printf '%s' "$value"
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
    --max-epochs)
      MAX_EPOCHS="${2:-}"
      shift 2
      ;;
    --train-bsz)
      TRAIN_BSZ="${2:-}"
      shift 2
      ;;
    --val-bsz)
      VAL_BSZ="${2:-}"
      shift 2
      ;;
    --val-check-interval)
      VAL_CHECK_INTERVAL="${2:-}"
      shift 2
      ;;
    --save-top-k)
      SAVE_TOP_K="${2:-}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:-}"
      shift 2
      ;;
    --train-prefetch-factor)
      TRAIN_PREFETCH_FACTOR="${2:-}"
      shift 2
      ;;
    --val-prefetch-factor)
      VAL_PREFETCH_FACTOR="${2:-}"
      shift 2
      ;;
    --persistent-workers)
      PERSISTENT_WORKERS="${2:-}"
      shift 2
      ;;
    --tos-env-file)
      TOS_ENV_FILE="${2:-}"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="${2:-}"
      shift 2
      ;;
    --keep-container)
      KEEP_CONTAINER=1
      shift
      ;;
    --detach)
      DETACH=1
      shift
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

case "${GPUS}" in
  1)
    GPU_DEVICES="${GPU_DEVICES:-0}"
    TRAINER_STRATEGY="auto"
    ;;
  2)
    GPU_DEVICES="${GPU_DEVICES:-0,1}"
    TRAINER_STRATEGY="ddp"
    ;;
  4)
    GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
    TRAINER_STRATEGY="ddp"
    ;;
  *)
    die "--gpus must be one of: 1, 2, 4"
    ;;
esac

[[ -n "${TRAIN_MANIFEST}" ]] || die "--train-manifest is required"
[[ -n "${VAL_MANIFEST}" ]] || die "--val-manifest is required"
[[ -n "${TOKENIZER_DIR}" ]] || die "--tokenizer-dir is required"
[[ -n "${PRETRAINED_NEMO}" ]] || die "--pretrained-nemo is required"
[[ -n "${EXP_DIR}" ]] || die "--exp-dir cannot be empty"
[[ -n "${RUN_NAME}" ]] || die "--run-name cannot be empty"
[[ "${MAX_EPOCHS}" =~ ^[0-9]+$ ]] || die "--max-epochs must be a positive integer"
[[ "${MAX_EPOCHS}" -gt 0 ]] || die "--max-epochs must be > 0"
[[ "${TRAIN_BSZ}" =~ ^[0-9]+$ ]] || die "--train-bsz must be a positive integer"
[[ "${TRAIN_BSZ}" -gt 0 ]] || die "--train-bsz must be > 0"
[[ "${VAL_BSZ}" =~ ^[0-9]+$ ]] || die "--val-bsz must be a positive integer"
[[ "${VAL_BSZ}" -gt 0 ]] || die "--val-bsz must be > 0"
[[ "${VAL_CHECK_INTERVAL}" =~ ^[0-9]+$ ]] || die "--val-check-interval must be a positive integer (steps)"
[[ "${VAL_CHECK_INTERVAL}" -gt 0 ]] || die "--val-check-interval must be > 0"
[[ "${SAVE_TOP_K}" =~ ^[0-9]+$ ]] || die "--save-top-k must be a non-negative integer"
if [[ -n "${NUM_WORKERS}" ]]; then
  [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "--num-workers must be a non-negative integer"
fi
if [[ -n "${TRAIN_PREFETCH_FACTOR}" ]]; then
  [[ "${TRAIN_PREFETCH_FACTOR}" =~ ^[0-9]+$ ]] || die "--train-prefetch-factor must be a non-negative integer"
fi
if [[ -n "${VAL_PREFETCH_FACTOR}" ]]; then
  [[ "${VAL_PREFETCH_FACTOR}" =~ ^[0-9]+$ ]] || die "--val-prefetch-factor must be a non-negative integer"
fi
if [[ -n "${PERSISTENT_WORKERS}" ]]; then
  case "${PERSISTENT_WORKERS}" in
    true|false)
      ;;
    *)
      die "--persistent-workers must be true or false"
      ;;
  esac
fi

[[ -f "${TRAIN_MANIFEST}" ]] || die "train manifest not found: ${TRAIN_MANIFEST}"
[[ -f "${VAL_MANIFEST}" ]] || die "val manifest not found: ${VAL_MANIFEST}"
[[ -d "${TOKENIZER_DIR}" ]] || die "tokenizer dir not found: ${TOKENIZER_DIR}"
[[ -f "${PRETRAINED_NEMO}" ]] || die "pretrained nemo not found: ${PRETRAINED_NEMO}"
[[ -z "${TOS_ENV_FILE}" || -f "${TOS_ENV_FILE}" ]] || die "tos env file not found: ${TOS_ENV_FILE}"

mkdir -p "${EXP_DIR}"
touch "${EXP_DIR}/.write_test.$$" && rm -f "${EXP_DIR}/.write_test.$$"

train_args=(
  "python" "examples/asr/asr_ctc/speech_to_text_ctc_bpe.py"
  "--config-path=/opt/ramosnemo_source/examples/asr/conf/fastconformer"
  "--config-name=fast-conformer_ctc_bpe_110m"
  "model.train_ds.manifest_filepath=${TRAIN_MANIFEST}"
  "model.validation_ds.manifest_filepath=${VAL_MANIFEST}"
  "model.tokenizer.dir=${TOKENIZER_DIR}"
  "model.tokenizer.type=bpe"
  "init_from_nemo_model.model0.path=${PRETRAINED_NEMO}"
  "trainer.devices=${GPUS}"
  "trainer.num_nodes=1"
  "trainer.accelerator=gpu"
  "trainer.strategy=${TRAINER_STRATEGY}"
  "trainer.precision=bf16"
  "trainer.max_epochs=${MAX_EPOCHS}"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
  "model.train_ds.batch_size=${TRAIN_BSZ}"
  "model.validation_ds.batch_size=${VAL_BSZ}"
  "exp_manager.exp_dir=${EXP_DIR}"
  "exp_manager.name=${RUN_NAME}"
  "exp_manager.create_wandb_logger=false"
  "exp_manager.checkpoint_callback_params.save_top_k=${SAVE_TOP_K}"
)

if [[ -n "${NUM_WORKERS}" ]]; then
  train_args+=("model.train_ds.num_workers=${NUM_WORKERS}")
  train_args+=("model.validation_ds.num_workers=${NUM_WORKERS}")
  train_args+=("model.test_ds.num_workers=${NUM_WORKERS}")
fi
if [[ -n "${TRAIN_PREFETCH_FACTOR}" ]]; then
  train_args+=("+model.train_ds.prefetch_factor=${TRAIN_PREFETCH_FACTOR}")
fi
if [[ -n "${VAL_PREFETCH_FACTOR}" ]]; then
  train_args+=("+model.validation_ds.prefetch_factor=${VAL_PREFETCH_FACTOR}")
fi
if [[ -n "${PERSISTENT_WORKERS}" ]]; then
  train_args+=("+model.train_ds.persistent_workers=${PERSISTENT_WORKERS}")
  train_args+=("+model.validation_ds.persistent_workers=${PERSISTENT_WORKERS}")
  train_args+=("+model.test_ds.persistent_workers=${PERSISTENT_WORKERS}")
fi

printf -v train_cmd '%q ' "${train_args[@]}"

docker_env_args=(
  -e CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
  -e RAMOSNEMO_SOURCE_DIR=/opt/ramosnemo_source
  -e NEMO_S3_CACHE_DIR="${NEMO_S3_CACHE_DIR:-/data2/mayufeng/.cache/nemo_s3}"
  -e NEMO_S3_CACHE_SIZE_GB="${NEMO_S3_CACHE_SIZE_GB:-500}"
)

if [[ -n "${NEMO_S3_CACHE_DISABLE:-}" ]]; then
  docker_env_args+=(-e NEMO_S3_CACHE_DISABLE="${NEMO_S3_CACHE_DISABLE}")
fi

if [[ -n "${TOS_ENV_FILE}" ]]; then
  tos_ak="$(strip_wrapping_quotes "$(read_env_value "${TOS_ENV_FILE}" "TOS_ACCESS_KEY_ID")")"
  tos_sk="$(strip_wrapping_quotes "$(read_env_value "${TOS_ENV_FILE}" "TOS_SECRET_ACCESS_KEY")")"
  tos_region="$(strip_wrapping_quotes "$(read_env_value "${TOS_ENV_FILE}" "TOS_REGION")")"
  tos_endpoint="$(strip_wrapping_quotes "$(read_env_value "${TOS_ENV_FILE}" "TOS_ENDPOINT")")"
  tos_session_token="$(strip_wrapping_quotes "$(read_env_value "${TOS_ENV_FILE}" "TOS_SESSION_TOKEN")")"
  tos_addressing_style="$(strip_wrapping_quotes "$(read_env_value "${TOS_ENV_FILE}" "TOS_ADDRESSING_STYLE")")"

  [[ -n "${tos_ak}" ]] || die "TOS_ACCESS_KEY_ID missing in ${TOS_ENV_FILE}"
  [[ -n "${tos_sk}" ]] || die "TOS_SECRET_ACCESS_KEY missing in ${TOS_ENV_FILE}"
  [[ -n "${tos_region}" ]] || die "TOS_REGION missing in ${TOS_ENV_FILE}"
  [[ -n "${tos_endpoint}" ]] || die "TOS_ENDPOINT missing in ${TOS_ENV_FILE}"

  docker_env_args+=(-e AWS_ACCESS_KEY_ID="${tos_ak}")
  docker_env_args+=(-e AWS_SECRET_ACCESS_KEY="${tos_sk}")
  docker_env_args+=(-e AWS_DEFAULT_REGION="${tos_region}")
  docker_env_args+=(-e AWS_ENDPOINT_URL="${tos_endpoint}")
  if [[ -n "${tos_session_token}" ]]; then
    docker_env_args+=(-e AWS_SESSION_TOKEN="${tos_session_token}")
  fi
  if [[ -n "${tos_addressing_style}" ]]; then
    docker_env_args+=(-e AWS_S3_ADDRESSING_STYLE="${tos_addressing_style}")
  fi
fi

echo "[INFO] Docker image: ${IMAGE}"
echo "[INFO] GPU devices: ${GPU_DEVICES} (count=${GPUS})"
echo "[INFO] exp_dir: ${EXP_DIR}"
echo "[INFO] run_name: ${RUN_NAME}"
echo "[INFO] tos_env_file: ${TOS_ENV_FILE:-<none>}"
echo "[INFO] container_name: ${CONTAINER_NAME:-<auto>}"
echo "[INFO] keep_container: ${KEEP_CONTAINER}"
echo "[INFO] detach: ${DETACH}"

DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  DOCKER_TTY_ARGS+=(-i)
  DOCKER_TTY_ARGS+=(-t)
fi

docker_name_args=()
if [[ -n "${CONTAINER_NAME}" ]]; then
  docker_name_args+=(--name "${CONTAINER_NAME}")
fi

docker_rm_args=(--rm)
if [[ "${KEEP_CONTAINER}" == "1" ]]; then
  docker_rm_args=()
fi

docker_run_args=(
  "${docker_rm_args[@]}"
  "${DOCKER_TTY_ARGS[@]}"
  "${docker_name_args[@]}"
  --gpus all
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  -v /data1:/data1
  -v /data2:/data2
  -v /data3:/data3
  -v /mnt/asr-audio-data:/mnt/asr-audio-data
  -v "${REPO_ROOT}:/opt/ramosnemo_source"
  -w /opt/ramosnemo_source
  "${docker_env_args[@]}"
  "${IMAGE}"
  /bin/bash -lc "set -euo pipefail; /opt/setup_ramosnemo.sh; ${train_cmd}"
)

if [[ "${DETACH}" == "1" ]]; then
  container_id="$(docker run -d "${docker_run_args[@]}")"
  echo "[INFO] detached_container_id: ${container_id}"
else
  docker run "${docker_run_args[@]}"
fi
