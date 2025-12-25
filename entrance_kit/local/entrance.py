#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Local launcher that mirrors the SageMaker entrance.")
    p.add_argument("--train-manifest", required=True, help="Absolute path to train manifest (jsonl).")
    p.add_argument("--val-manifest", required=True, help="Absolute path to val manifest (jsonl).")
    p.add_argument("--test-manifest", default="", help="Optional path to test manifest (jsonl).")
    p.add_argument("--tokenizer-dir", required=True, help="Tokenizer directory.")
    p.add_argument("--pretrained", required=True, help="Path to pretrained .nemo file.")
    p.add_argument("--config-name", default="fastconformer_hybrid_tdt_ctc_bpe_110m")
    p.add_argument("--out", default="runs/local_launch", help="Output root (checkpoints/logs/TB).")
    p.add_argument("--run-name", default="nemo-local")
    p.add_argument("--language", default="")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--train-bsz", type=int, default=32)
    p.add_argument("--val-bsz", type=int, default=32)
    p.add_argument("--test-bsz", type=int, default=32)
    p.add_argument("--test-num-workers", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--max-steps", default="")
    p.add_argument("--val-check-interval", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--ema-enable", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--auto-batch-per-gpu", type=int, default=1)
    p.add_argument("--train-audio-s3", action="store_true", help="Set TRAIN_AUDIO_S3=1 for S3 direct read.")
    p.add_argument("--train-manifest-s3-uri", default="", help="S3 uri of train manifest when TRAIN_AUDIO_S3=1.")
    p.add_argument("--val-audio-s3", action="store_true", help="Optional S3 direct read for val.")
    p.add_argument("--val-manifest-s3-uri", default="", help="S3 uri of val manifest when VAL_AUDIO_S3=1.")
    p.add_argument("--train-is-tarred", action="store_true")
    p.add_argument("--train-tar-pattern", default="")
    p.add_argument("--val-is-tarred", action="store_true")
    p.add_argument("--val-tar-pattern", default="")
    p.add_argument("--max-duration-train", default="")
    p.add_argument("--min-duration-train", default="")
    p.add_argument("--max-duration-val", default="")
    p.add_argument("--min-duration-val", default="")
    p.add_argument("--sample-k", type=int, default=64)
    p.add_argument("--ckpt-every-steps", type=int, default=0)
    p.add_argument("--ckpt-every-epochs", type=int, default=0)
    p.add_argument("--use-lhotse", action="store_true", help="Force use_lhotse=true for train/val/test.")
    p.add_argument("--train-batch-duration", default="", help="Lhotse max batch duration for train set.")
    p.add_argument("--val-batch-duration", default="", help="Lhotse max batch duration for val set.")
    p.add_argument("--test-batch-duration", default="", help="Lhotse max batch duration for test set.")
    p.add_argument("--use-polars", action="store_true", help="Enable Polars-based manifest reading (non-Lhotse).")
    p.add_argument(
        "--pretokenize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to pretokenize manifests at load time (default: true).",
    )
    return p.parse_args()


def _configure_numba_cuda_linker_env(env: dict) -> None:
    """
    Keep NeMo/Numba RNNT(TDT) kernels working on CUDA>=12.

    numba-cuda 在 driver>=12 时已不支持 MVCLinker（NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1 会直接报错）。
    本地在 H20 + CUDA forward-compat 场景下，开启 pynvjitlink 可能触发 PTX 版本不匹配；默认关闭更稳。
    如需启用，可自行设置 NUMBA_CUDA_ENABLE_PYNVJITLINK=1。
    """

    env["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "0"

    if "NUMBA_CUDA_ENABLE_PYNVJITLINK" not in env and "CUDA_ENABLE_PYNVJITLINK" not in env:
        env["NUMBA_CUDA_ENABLE_PYNVJITLINK"] = "0"


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sm_entry = repo_root / "entrance_kit" / "sagemaker" / "train_entry.py"
    if not sm_entry.is_file():
        raise SystemExit(f"train_entry not found at {sm_entry}")

    train_manifest = Path(args.train_manifest).expanduser().resolve()
    val_manifest = Path(args.val_manifest).expanduser().resolve()
    test_manifest = Path(args.test_manifest).expanduser().resolve() if args.test_manifest else None
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    pretrained = Path(args.pretrained).expanduser().resolve()

    missing_paths = []
    reqs = [train_manifest, val_manifest, tokenizer_dir, pretrained]
    if test_manifest:
        reqs.append(test_manifest)
    for required_path in reqs:
        if not required_path.exists():
            missing_paths.append(str(required_path))
    if missing_paths:
        raise SystemExit(f"The following required paths were not found: {', '.join(missing_paths)}")

    out_root = Path(args.out).resolve()
    model_dir = out_root / "model"
    output_dir = out_root / "output"
    tb_base = output_dir
    ckpt_base = out_root / "checkpoints"
    train_s3_dir = out_root / "train_s3_manifest"

    for d in [model_dir, output_dir, tb_base, ckpt_base, train_s3_dir]:
        os.makedirs(d, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH','')}"
    env.update(
        {
            "SM_CHANNEL_TRAIN": str(train_manifest.parent),
            "SM_CHANNEL_VAL": str(val_manifest.parent),
            "SM_CHANNEL_TEST": str(test_manifest.parent) if test_manifest else "",
            "SM_CHANNEL_TOKENIZER": str(tokenizer_dir),
            "SM_CHANNEL_PRETRAINED": str(pretrained.parent),
            "SM_MODEL_DIR": str(model_dir),
            "SM_OUTPUT_DATA_DIR": str(output_dir),
            "OUTPUT_BASE_DIR": str(tb_base),
            "CHECKPOINT_BASE_DIR": str(ckpt_base),
            "TRAIN_MANIFEST_LOCAL_DIR": str(train_s3_dir),
            "TRAIN_MANIFEST": train_manifest.name,
            "VAL_MANIFEST": val_manifest.name,
            "TEST_MANIFEST": test_manifest.name if test_manifest else "",
            "PRETRAINED_FILENAME": pretrained.name,
            "CONFIG_NAME": args.config_name,
            "RUN_NAME": args.run_name,
            "LANGUAGE": args.language,
            "DEVICES": str(args.devices),
            "PRECISION": args.precision,
            "GRAD_ACCUM": str(args.grad_accum),
            "NUM_WORKERS": str(args.num_workers),
            "TEST_NUM_WORKERS": str(args.test_num_workers),
            "AUTO_BATCH_PER_GPU": str(int(bool(args.auto_batch_per_gpu))),
            "TRAIN_BATCH": str(args.train_bsz),
            "VAL_BATCH": str(args.val_bsz),
            "TEST_BATCH": str(args.test_bsz),
            "MAX_EPOCHS": str(args.max_epochs),
            "MAX_STEPS": args.max_steps,
            "VAL_CHECK_INTERVAL": str(args.val_check_interval),
            "LR": str(args.lr),
            "EMA_ENABLE": "1" if args.ema_enable else "0",
            "EMA_DECAY": str(args.ema_decay),
            "RESUME": "1" if args.resume else "0",
            "TRAIN_AUDIO_S3": "1" if args.train_audio_s3 else "0",
            "VAL_AUDIO_S3": "1" if args.val_audio_s3 else "0",
            "TRAIN_MANIFEST_S3_URI": args.train_manifest_s3_uri,
            "VAL_MANIFEST_S3_URI": args.val_manifest_s3_uri,
            "TRAIN_IS_TARRED": "1" if args.train_is_tarred else "0",
            "VAL_IS_TARRED": "1" if args.val_is_tarred else "0",
            "TRAIN_TAR_PATTERN": args.train_tar_pattern,
            "VAL_TAR_PATTERN": args.val_tar_pattern,
            "MAX_DURATION_TRAIN": args.max_duration_train,
            "MIN_DURATION_TRAIN": args.min_duration_train,
            "MAX_DURATION_VAL": args.max_duration_val,
            "MIN_DURATION_VAL": args.min_duration_val,
            "SAMPLE_K": str(args.sample_k),
            "CKPT_EVERY_STEPS": str(args.ckpt_every_steps),
            "CKPT_EVERY_EPOCHS": str(args.ckpt_every_epochs),
            "USE_LHOTSE": "1" if args.use_lhotse else "0",
            "BATCH_DURATION_TRAIN": args.train_batch_duration,
            "BATCH_DURATION_VAL": args.val_batch_duration,
            "BATCH_DURATION_TEST": args.test_batch_duration,
            "USE_POLARS": "1" if args.use_polars else "0",
            "PRETOKENIZE": "1" if args.pretokenize else "0",
            "S3_OUT": "",
            "S3_TB": "",
            "S3_CKPT": "",
            "REGION": "",
        }
    )

    _configure_numba_cuda_linker_env(env)

    cmd = [sys.executable, str(sm_entry)]
    print("Launching:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


if __name__ == "__main__":
    main()
