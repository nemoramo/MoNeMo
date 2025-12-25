#!/usr/bin/env python3
import os, sys, json, subprocess, random, time
from pathlib import Path
from urllib.parse import urlparse

from omegaconf import OmegaConf

from nemo.core.optim.lr_scheduler import compute_max_steps

# --- 兜底：确保优先导入 /opt/nemo 和本仓库根目录 ---
_p = "/opt/nemo"
if os.path.isdir(_p) and _p not in sys.path:
    sys.path.insert(0, _p)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def env(name, default=None, required=False, cast=str):
    v = os.environ.get(name, default)
    if required and (v is None or v == ""):
        print(f"[X] missing env: {name}")
        sys.exit(2)
    if v is not None and cast is not str:
        try:
            v = cast(v)
        except Exception:
            pass
    return v

def resolve_nemo_entry():
    """Resolve training script and config paths (prefer repo copy, fallback to /opt/nemo)."""
    repo_script = REPO_ROOT / "examples" / "asr" / "asr_hybrid_transducer_ctc" / "speech_to_text_hybrid_rnnt_ctc_bpe.py"
    repo_cfg_dir = REPO_ROOT / "examples" / "asr" / "conf" / "fastconformer" / "hybrid_transducer_ctc"
    if repo_script.is_file():
        return str(repo_script), str(repo_cfg_dir)

    fallback_script = Path("/opt/nemo/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py")
    fallback_cfg = Path("/opt/nemo/examples/asr/conf/fastconformer/hybrid_transducer_ctc")
    if fallback_script.is_file():
        return str(fallback_script), str(fallback_cfg)

    raise SystemExit("Cannot locate NeMo training entry script (looked under repo and /opt/nemo)")

# === 数据模式相关 env ===
TRAIN_AUDIO_S3        = env("TRAIN_AUDIO_S3", 0, cast=int)
VAL_AUDIO_S3          = env("VAL_AUDIO_S3",   0, cast=int)
TRAIN_MANIFEST_S3_URI = os.environ.get("TRAIN_MANIFEST_S3_URI", "")
VAL_MANIFEST_S3_URI   = os.environ.get("VAL_MANIFEST_S3_URI", "")
TEST_MANIFEST_ENV     = os.environ.get("TEST_MANIFEST", "")
CH_TEST               = os.environ.get("SM_CHANNEL_TEST", "")

TRAIN_IS_TARRED    = env("TRAIN_IS_TARRED", 0, cast=int)
TRAIN_TAR_PATTERN  = os.environ.get("TRAIN_TAR_PATTERN", "")
VAL_IS_TARRED      = env("VAL_IS_TARRED", 0, cast=int)
VAL_TAR_PATTERN    = os.environ.get("VAL_TAR_PATTERN", "")
USE_LHOTSE         = env("USE_LHOTSE", 0, cast=int)
BATCH_DURATION_TRAIN = os.environ.get("BATCH_DURATION_TRAIN", "")
BATCH_DURATION_VAL   = os.environ.get("BATCH_DURATION_VAL", "")
BATCH_DURATION_TEST  = os.environ.get("BATCH_DURATION_TEST", "")
USE_POLARS         = env("USE_POLARS", 0, cast=int)
PRETOKENIZE        = env("PRETOKENIZE", 1, cast=int)

if TRAIN_IS_TARRED and TRAIN_AUDIO_S3:
    print("[X] TRAIN_IS_TARRED=1 和 TRAIN_AUDIO_S3=1 不可同时开启")
    sys.exit(2)

# === SageMaker Script Mode 环境（容器内） ===
# train 通道：只有在非 S3 模式下才是 required
CH_TRAIN      = os.environ.get("SM_CHANNEL_TRAIN")
CH_VAL        = env("SM_CHANNEL_VAL",        required=True)
CH_TOKENIZER  = env("SM_CHANNEL_TOKENIZER",  required=True)
CH_PRETRAINED = env("SM_CHANNEL_PRETRAINED", required=True)
MODEL_DIR     = env("SM_MODEL_DIR",          required=True)
OUT_DIR       = env("SM_OUTPUT_DATA_DIR",    required=True)
OUTPUT_BASE   = os.environ.get("OUTPUT_BASE_DIR", "/opt/ml/output")
TB_DIR        = os.path.join(OUTPUT_BASE, "tensorboard")

# 训练/评估文件名
TRAIN_MANIFEST       = env("TRAIN_MANIFEST", required=True)
VAL_MANIFEST         = env("VAL_MANIFEST",   required=True)
TEST_MANIFEST        = TEST_MANIFEST_ENV
PRETRAINED_FILENAME  = env("PRETRAINED_FILENAME", "parakeet-tdt_ctc-110m.nemo")
CONFIG_NAME          = env("CONFIG_NAME", "fastconformer_hybrid_tdt_ctc_bpe_110m")
RUN_NAME             = env("RUN_NAME", "nemo-run")
LANGUAGE_TAG         = os.environ.get("LANGUAGE", "")

# 训练控制
DEVICES              = env("DEVICES",            8,  cast=int)
PRECISION            = env("PRECISION",          "bf16")
VAL_CHECK_INTERVAL   = env("VAL_CHECK_INTERVAL", 2000, cast=int)
RESUME               = env("RESUME",             0,   cast=int)
MAX_EPOCHS           = env("MAX_EPOCHS",         20,  cast=int)
MAX_STEPS_ENV        = os.environ.get("MAX_STEPS", "")
LR                   = env("LR",                 1.0e-3, cast=float)
EMA_ENABLE           = env("EMA_ENABLE",         1, cast=int)
EMA_DECAY            = env("EMA_DECAY",          0.999, cast=float)

# 批量与 Loader
TRAIN_BATCH          = env("TRAIN_BATCH",        "", cast=str)
VAL_BATCH            = env("VAL_BATCH",          "", cast=str)
TEST_BATCH           = env("TEST_BATCH",         "", cast=str)
GRAD_ACCUM           = env("GRAD_ACCUM",         1,   cast=int)
NUM_WORKERS          = env("NUM_WORKERS",        8,   cast=int)
TEST_NUM_WORKERS     = env("TEST_NUM_WORKERS",   NUM_WORKERS, cast=int)
AUTO_BATCH_PER_GPU   = env("AUTO_BATCH_PER_GPU", 1,   cast=int)

# 时长裁剪
MAX_DURATION_TRAIN   = os.environ.get("MAX_DURATION_TRAIN", "")
MIN_DURATION_TRAIN   = os.environ.get("MIN_DURATION_TRAIN", "")
MAX_DURATION_VAL     = os.environ.get("MAX_DURATION_VAL", "")
MIN_DURATION_VAL     = os.environ.get("MIN_DURATION_VAL", "")

# 抽样检查
SAMPLE_K             = env("SAMPLE_K", 64, cast=int)
SAMPLE_SEED_ENV      = os.environ.get("SAMPLE_SEED")

# Checkpoint 频率
CKPT_EVERY_STEPS     = env("CKPT_EVERY_STEPS",  0, cast=int)
CKPT_EVERY_EPOCHS    = env("CKPT_EVERY_EPOCHS", 0, cast=int)

# 信息提示
S3_OUT = os.environ.get("S3_OUT", "")
S3_TB  = os.environ.get("S3_TB", "")
S3_CKPT= os.environ.get("S3_CKPT", "")
REGION = os.environ.get("REGION", "")

# === AUGMENTOR 环境变量 ===
AUG_ON_VAL          = env("AUG_ON_VAL",           0,    cast=int)

# noise
AUG_NOISE_PROB      = env("AUG_NOISE_PROB",       0.0,  cast=float)
AUG_NOISE_MANIFEST  = env("AUG_NOISE_MANIFEST",   "",   cast=str)
AUG_NOISE_MIN_SNR   = env("AUG_NOISE_MIN_SNR_DB", 0.0,  cast=float)
AUG_NOISE_MAX_SNR   = env("AUG_NOISE_MAX_SNR_DB", 15.0, cast=float)

# speed
AUG_SPEED_PROB      = env("AUG_SPEED_PROB",       0.0,  cast=float)
AUG_SPEED_SR        = env("AUG_SPEED_SR",         16000, cast=int)
AUG_SPEED_RESAMPLE  = env("AUG_SPEED_RESAMPLE",   "kaiser_fast", cast=str)
AUG_SPEED_MIN_RATE  = env("AUG_SPEED_MIN_RATE",   0.95, cast=float)
AUG_SPEED_MAX_RATE  = env("AUG_SPEED_MAX_RATE",   1.05, cast=float)

# gain
AUG_GAIN_PROB       = env("AUG_GAIN_PROB",        0.0,  cast=float)
AUG_GAIN_MIN_DB     = env("AUG_GAIN_MIN_DB",   -10.0, cast=float)
AUG_GAIN_MAX_DB     = env("AUG_GAIN_MAX_DB",    10.0, cast=float)

def resolve_path(base_dir: str, p: str) -> str:
    if not p:
        return ""
    if p.startswith("/") or p.startswith("s3://"):
        return p
    return os.path.join(base_dir, p)

def parse_s3_uri(u: str):
    p = urlparse(u)
    assert p.scheme == "s3", f"not an s3 uri: {u}"
    return p.netloc, p.path.lstrip("/")


def load_base_cfg(cfg_dir: str, config_name: str):
    cfg_path = Path(cfg_dir) / f"{config_name}.yaml"
    if not cfg_path.is_file():
        return None
    try:
        return OmegaConf.load(cfg_path)
    except Exception as e:
        print(f"[WARN] 无法加载配置 {cfg_path}: {e}")
        return None


def cfg_select(cfg, key: str):
    if cfg is None:
        return None
    try:
        return OmegaConf.select(cfg, key)
    except Exception:
        return None


def count_manifest_entries(manifest_path: str):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"[WARN] 无法统计 manifest 行数 {manifest_path}: {e}")
        return None


def resolve_device_count(devices_hint: int) -> int:
    if devices_hint and devices_hint > 0:
        return devices_hint
    try:
        import torch

        cnt = torch.cuda.device_count()
        return max(1, cnt)
    except Exception:
        return 1

print("== PRE-FLIGHT (Script Mode) ==")
if LANGUAGE_TAG:
    print(f"Language: {LANGUAGE_TAG}")
print(f"Run name: {RUN_NAME}")
print("SM_CHANNELS:", {
    "train": CH_TRAIN if CH_TRAIN else "(none - S3 mode?)",
    "val": CH_VAL,
    "test": CH_TEST if CH_TEST else "(none)",
    "tokenizer": CH_TOKENIZER,
    "pretrained": CH_PRETRAINED,
})

print("\n--- WHERE TO WATCH (printed by container) ---")
print(f"[TensorBoard S3]  {S3_TB}")
print(f"[Checkpoints S3]  {S3_CKPT}")
print(f"[Output S3]       {S3_OUT}")
print("------------------------------------------------\n")

print("DATA MODES:")
print(f"  train: is_tarred={bool(TRAIN_IS_TARRED)}  audio_s3={bool(TRAIN_AUDIO_S3)}")
print(f"  val  : is_tarred={bool(VAL_IS_TARRED)}    audio_s3={bool(VAL_AUDIO_S3)}\n")
print("OPTIM:")
print(f"  lr={LR}  ema={'on' if EMA_ENABLE else 'off'}{'' if not EMA_ENABLE else f' decay={EMA_DECAY}'}\n")

# 1) GPU / Torch 检查
subprocess.run(["nvidia-smi"], check=False)
py = (
    "import sys, torch; "
    "print('[Torch]', torch.__version__, 'cuda?', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count()); "
    "assert torch.cuda.is_available(), 'CUDA not available'"
)
subprocess.check_call([sys.executable, "-c", py])

# 2) 处理 manifest 路径（含 S3 直读模式）
def obtain_manifest(path_local: str, s3_uri: str) -> str:
    """若给了 s3_uri，则从 S3 下载到 path_local；否则认为 path_local 已存在。"""
    if not s3_uri:
        return path_local
    import boto3
    s3 = boto3.client("s3", region_name=REGION or None)
    bucket, key = parse_s3_uri(s3_uri)
    os.makedirs(os.path.dirname(path_local), exist_ok=True)
    print(f"[S3] downloading manifest from {s3_uri} -> {path_local}")
    s3.download_file(bucket, key, path_local)
    return path_local

# train manifest
if TRAIN_AUDIO_S3:
    if not TRAIN_MANIFEST_S3_URI:
        print("[X] TRAIN_AUDIO_S3=1 但 TRAIN_MANIFEST_S3_URI 为空")
        sys.exit(2)
    train_manifest_dir = os.environ.get("TRAIN_MANIFEST_LOCAL_DIR", "/opt/ml/input/data/train_s3")
    train_manifest_path = os.path.join(train_manifest_dir, TRAIN_MANIFEST)
    train_manifest_path = obtain_manifest(train_manifest_path, TRAIN_MANIFEST_S3_URI)
    train_base_dir = train_manifest_dir
else:
    if not CH_TRAIN:
        print("[X] TRAIN_AUDIO_S3=0 且缺少 SM_CHANNEL_TRAIN")
        sys.exit(2)
    train_manifest_path = os.path.join(CH_TRAIN, TRAIN_MANIFEST)
    train_base_dir = CH_TRAIN

# val manifest（这里暂时不做 S3 直读，VAL_AUDIO_S3 将来可对称支持）
val_manifest_path = os.path.join(CH_VAL, VAL_MANIFEST)
val_base_dir = CH_VAL

# test manifest（可选，本地路径）
test_manifest_path = ""
if TEST_MANIFEST:
    if CH_TEST:
        test_manifest_path = os.path.join(CH_TEST, TEST_MANIFEST)
    else:
        test_manifest_path = TEST_MANIFEST

# 3) 关键文件存在性（train manifest 在 S3 模式下已经下载到本地）
tok_dir   = CH_TOKENIZER
nemo_path = os.path.join(CH_PRETRAINED, PRETRAINED_FILENAME)

for p in [train_manifest_path, val_manifest_path, tok_dir, nemo_path] + ([test_manifest_path] if test_manifest_path else []):
    if not (os.path.isdir(p) or os.path.isfile(p)):
        print("[X] missing:", p); sys.exit(2)

if TRAIN_IS_TARRED:
    print(f"[TAR] train dataset is tarred, pattern={TRAIN_TAR_PATTERN!r}")
    if not TRAIN_TAR_PATTERN:
        print("[X] TRAIN_IS_TARRED=1 但 TRAIN_TAR_PATTERN 为空")
        sys.exit(2)
if VAL_IS_TARRED:
    print(f"[TAR] val dataset is tarred, pattern={VAL_TAR_PATTERN!r}")
    if not VAL_TAR_PATTERN:
        print("[X] VAL_IS_TARRED=1 但 VAL_TAR_PATTERN 为空")
        sys.exit(2)

# 4) 随机抽样检查 manifest
def sample_check_random(manifest_path, base_dir, k=64, seed=None, show_n=10, is_remote=False):
    if seed is None:
        if SAMPLE_SEED_ENV:
            try: seed = int(SAMPLE_SEED_ENV)
            except Exception: seed = None
    if seed is None:
        seed = (int(time.time()) & 0xFFFFFFFF)
    rng = random.Random(seed)

    reservoir = []
    total = 0
    bad_json = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                ex = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            if len(reservoir) < k:
                reservoir.append((total, ex))
            else:
                j = rng.randint(1, total)
                if j <= k:
                    reservoir[j - 1] = (total, ex)

    misses, abs_paths, missing_field = [], [], []
    for idx, ex in reservoir:
        rel = ex.get("audio_filepath")
        if not isinstance(rel, str) or not rel:
            missing_field.append((idx, "audio_filepath missing/invalid"))
            continue

        if not is_remote:
            # 本地模式支持绝对路径
            if rel.startswith("/"):
                fp = rel
            else:
                fp = os.path.join(base_dir, rel)
            if not os.path.exists(fp):
                misses.append((idx, fp))
        else:
            # 远端模式仍要求相对路径
            if rel.startswith("/"):
                abs_paths.append((idx, rel))
                continue
            fp = os.path.join(base_dir, rel)
            # 远端不检查存在性

    print(f"[RANDSAMPLE] manifest={manifest_path} seed={seed} picked={len(reservoir)} "
          f"/ seen={total} bad_json={bad_json} is_remote={is_remote}")

    if missing_field:
        print(f"[RANDSAMPLE] entries missing field: {len(missing_field)} (show {min(show_n,len(missing_field))})")
        for t in missing_field[:show_n]:
            print("  - line#", t[0], t[1])
        raise SystemExit("manifest sample has entries missing audio_filepath")

    if abs_paths:
        print(f"[RANDSAMPLE] absolute-path entries: {len(abs_paths)} (show {min(show_n,len(abs_paths))})")
        for t in abs_paths[:show_n]:
            print("  - line#", t[0], t[1])
        raise SystemExit("absolute paths found in manifest sample")

    if misses:
        print(f"[RANDSAMPLE] missing files: {len(misses)} (show {min(show_n,len(misses))})")
        for t in misses[:show_n]:
            print("  - line#", t[0], t[1])
        raise SystemExit(f"missing {len(misses)} files in random sample")

    print("[OK] random sample existence check passed")

sample_check_random(
    train_manifest_path,
    train_base_dir,
    k=SAMPLE_K,
    is_remote=bool(TRAIN_IS_TARRED or TRAIN_AUDIO_S3),
)
sample_check_random(
    val_manifest_path,
    val_base_dir,
    k=SAMPLE_K,
    is_remote=bool(VAL_IS_TARRED or VAL_AUDIO_S3),
)

# 5) TB & model 可写测试
py = (
    f"from torch.utils.tensorboard import SummaryWriter; "
    f"open('{MODEL_DIR}/_write_ok.txt','w').write('ok'); "
    f"w=SummaryWriter('{TB_DIR}'); "
    f"w.add_text('hello', 'tensorboard is ready'); "
    f"w.add_scalar('sanity/ok',1.0,0); "
    f"w.flush(); w.close(); "
    f"print('[OK] TB & model dir writable (wrote hello + sanity/ok)')"
)
subprocess.check_call([sys.executable, "-c", py])

# 6) Checkpoint 目录
TS = time.strftime("%Y%m%d-%H%M%S", time.localtime())
CKPT_BASE = os.environ.get("CHECKPOINT_BASE_DIR", "/opt/ml/checkpoints")
CKPT_DIR_TS = os.path.join(CKPT_BASE, f"ts-{TS}")
os.makedirs(CKPT_DIR_TS, exist_ok=True)

job_name = ""
try:
    sm_env = os.environ.get("SM_TRAINING_ENV")
    if sm_env:
        job_name = json.loads(sm_env).get("job_name", "")
except Exception:
    pass

print(f"[CKPT] local dir : {CKPT_DIR_TS}")
if S3_CKPT:
    if job_name:
        print(f"[CKPT] S3 dir    : {S3_CKPT.rstrip('/')}/{job_name}/ts-{TS}/")
    else:
        print(f"[CKPT] S3 parent : {S3_CKPT} (actual under <JobName>/ts-{TS}/)")

# 7) batch 相关推断（保持你原来的逻辑）
def detect_gpu_name():
    try:
        py = "import torch; print(torch.cuda.get_device_name(0))"
        out = subprocess.check_output([sys.executable, "-c", py], text=True).strip()
        return out
    except Exception:
        return ""

gpu_name = detect_gpu_name()

def default_batch_per_gpu_by_gpu(gpu_name: str) -> int:
    g = gpu_name.lower()
    if "h100" in g:                  return 112
    if "a100" in g and "80" in g:    return 96
    if "a100" in g:                  return 64
    if "a10"  in g:                  return 32
    return -1

train_bs_override = None
val_bs_override   = None
if TRAIN_BATCH.strip():
    train_bs_override = int(TRAIN_BATCH)
elif AUTO_BATCH_PER_GPU:
    guess = default_batch_per_gpu_by_gpu(gpu_name)
    if guess > 0:
        train_bs_override = guess

if VAL_BATCH.strip():
    val_bs_override = int(VAL_BATCH)
elif train_bs_override is not None:
    val_bs_override = max(8, min(train_bs_override, 64))

try:
    global_bs_val = (train_bs_override if train_bs_override is not None else None)
    if isinstance(global_bs_val, int):
        global_bs_val = global_bs_val * DEVICES * GRAD_ACCUM
except Exception:
    global_bs_val = None

print("== BATCH CONFIG SUMMARY ==")
print(f"[GPU]        {gpu_name}")
print(f"[devices]    {DEVICES}")
print(f"[train_bs]   {train_bs_override if train_bs_override is not None else '(use YAML)'}  (per GPU)")
print(f"[val_bs]     {val_bs_override   if val_bs_override   is not None else '(use YAML)'}  (per GPU)")
print(f"[grad_accum] {GRAD_ACCUM}")
print(f"[global_bs]  {global_bs_val if global_bs_val is not None else '(YAML or unknown)'}\n")

# === AUGMENTOR 覆盖组装（略，保留你原来的逻辑，这里就不再重复解释）===
def build_augmentor_overrides(for_val: bool):
    ds_key = "model.validation_ds" if for_val else "model.train_ds"
    overrides = []
    # noise
    if AUG_NOISE_PROB and AUG_NOISE_PROB > 0:
        manifest = resolve_path(train_base_dir, AUG_NOISE_MANIFEST)
        if manifest and os.path.isfile(manifest):
            overrides += [
                f"++{ds_key}.augmentor.noise.prob={AUG_NOISE_PROB}",
                f"++{ds_key}.augmentor.noise.manifest_path={manifest}",
                f"++{ds_key}.augmentor.noise.min_snr_db={AUG_NOISE_MIN_SNR}",
                f"++{ds_key}.augmentor.noise.max_snr_db={AUG_NOISE_MAX_SNR}",
            ]
        else:
            print(f"[WARN] noise.prob={AUG_NOISE_PROB} 但未找到 manifest 文件：{manifest!r}，将跳过 noise 覆盖。")
    # speed
    if AUG_SPEED_PROB and AUG_SPEED_PROB > 0:
        overrides += [
            f"++{ds_key}.augmentor.speed.prob={AUG_SPEED_PROB}",
            f"++{ds_key}.augmentor.speed.sr={AUG_SPEED_SR}",
            f"++{ds_key}.augmentor.speed.resample_type={AUG_SPEED_RESAMPLE}",
            f"++{ds_key}.augmentor.speed.min_speed_rate={AUG_SPEED_MIN_RATE}",
            f"++{ds_key}.augmentor.speed.max_speed_rate={AUG_SPEED_MAX_RATE}",
        ]
    # gain
    if AUG_GAIN_PROB and AUG_GAIN_PROB > 0:
        overrides += [
            f"++{ds_key}.augmentor.gain.prob={AUG_GAIN_PROB}",
            f"++{ds_key}.augmentor.gain.min_gain_dbfs={AUG_GAIN_MIN_DB}",
            f"++{ds_key}.augmentor.gain.max_gain_dbfs={AUG_GAIN_MAX_DB}",
        ]
    return overrides

aug_train_overrides = build_augmentor_overrides(for_val=False)
aug_val_overrides   = build_augmentor_overrides(for_val=True) if AUG_ON_VAL else []

print("== AUGMENTOR SUMMARY ==")
def _yn(x): return "ON" if x else "OFF"
print(f"[train] noise={_yn(any('augmentor.noise' in s for s in aug_train_overrides))} "
      f"speed={_yn(any('augmentor.speed' in s for s in aug_train_overrides))} "
      f"gain={_yn(any('augmentor.gain'  in s for s in aug_train_overrides))}")
print(f"[valid] enabled={bool(AUG_ON_VAL)} | "
      f"noise={_yn(any('augmentor.noise' in s for s in aug_val_overrides))} "
      f"speed={_yn(any('augmentor.speed' in s for s in aug_val_overrides))} "
      f"gain={_yn(any('augmentor.gain'  in s for s in aug_val_overrides))}\n")

# 额外：推断 Cosine LR 所需的 max_steps（若未显式指定）
nemo_script, nemo_cfg_dir = resolve_nemo_entry()
base_cfg = load_base_cfg(nemo_cfg_dir, CONFIG_NAME)
sched_name_in_cfg = cfg_select(base_cfg, "model.optim.sched.name")
cfg_sched_max_steps = cfg_select(base_cfg, "model.optim.sched.max_steps")
cfg_train_bs = cfg_select(base_cfg, "model.train_ds.batch_size")
cfg_drop_last = bool(cfg_select(base_cfg, "model.train_ds.drop_last") or False)
target_is_cosine = (sched_name_in_cfg is None) or ("cosine" in str(sched_name_in_cfg).lower())

per_gpu_bs_for_sched = train_bs_override or (int(cfg_train_bs) if cfg_train_bs else None)
sched_max_steps = None
sched_max_steps_info = ""

if MAX_STEPS_ENV and MAX_STEPS_ENV.strip():
    try:
        sched_max_steps = int(MAX_STEPS_ENV.strip())
        sched_max_steps_info = "来自 MAX_STEPS 环境变量"
    except ValueError:
        print(f"[WARN] MAX_STEPS={MAX_STEPS_ENV!r} 不是整数，忽略该值用于调度器")
elif target_is_cosine:
    samples = count_manifest_entries(train_manifest_path)
    device_count_for_sched = resolve_device_count(DEVICES)
    if samples is not None and per_gpu_bs_for_sched and MAX_EPOCHS > 0:
        try:
            sched_max_steps = compute_max_steps(
                max_epochs=MAX_EPOCHS,
                accumulate_grad_batches=GRAD_ACCUM,
                limit_train_batches=1.0,
                num_workers=device_count_for_sched,
                num_samples=samples,
                batch_size=per_gpu_bs_for_sched,
                drop_last=cfg_drop_last,
            )
            sched_max_steps_info = (
                f"估算值：samples={samples} per_gpu_bs={per_gpu_bs_for_sched} "
                f"devices={device_count_for_sched} grad_accum={GRAD_ACCUM} epochs={MAX_EPOCHS}"
            )
        except Exception as e:
            print(f"[WARN] 估算调度器 max_steps 失败：{e}")
    else:
        print("[WARN] 缺少 batch_size 或样本数，无法推断调度器 max_steps")

if sched_max_steps is not None and target_is_cosine:
    if cfg_sched_max_steps not in (None, -1):
        print(f"[SCHED] 覆盖原有调度器 max_steps={cfg_sched_max_steps} -> {sched_max_steps}")
    else:
        print(f"[SCHED] 设置调度器 max_steps={sched_max_steps}")
    if sched_max_steps_info:
        print(f"[SCHED] 来源：{sched_max_steps_info}")
else:
    if target_is_cosine:
        print("[SCHED] Cosine 调度器 max_steps 未设置，将使用配置文件默认值")

print("== PRE-FLIGHT PASSED ==\n")

# 8) 组装 Hydra 覆写并启动 NeMo 训练
args = [
    sys.executable, nemo_script,
    f"--config-path={nemo_cfg_dir}",
    f"--config-name={CONFIG_NAME}",

    # 数据路径（train/val manifest 使用我们刚刚准备好的本地路径）
    f"model.train_ds.manifest_filepath={train_manifest_path}",
    f"model.validation_ds.manifest_filepath={val_manifest_path}",
    f"model.tokenizer.dir={CH_TOKENIZER}",

    f"model.optim.lr={LR}",

    f"init_from_nemo_model.model0.path={nemo_path}",
    "init_from_nemo_model.model0.exclude=[decoder,joint,aux_ctc]",

    "trainer.num_nodes=1",
    "trainer.accelerator=gpu",
    "trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy",
    "trainer.strategy.gradient_as_bucket_view=true",
    f"trainer.devices={DEVICES}",
    f"trainer.precision={PRECISION}",
    f"trainer.val_check_interval={VAL_CHECK_INTERVAL}",
    f"trainer.max_epochs={MAX_EPOCHS}",
    f"trainer.accumulate_grad_batches={GRAD_ACCUM}",

    "trainer.log_every_n_steps=50",

    "++model.decoding.greedy.loop_labels=false",
    "++model.decoding.greedy.use_cuda_graph_decoder=false", 
]

if MAX_STEPS_ENV and MAX_STEPS_ENV.strip():
    args += [f"trainer.max_steps={MAX_STEPS_ENV.strip()}", "trainer.max_epochs=null"]

if sched_max_steps is not None and target_is_cosine:
    args += [f"++model.optim.sched.max_steps={sched_max_steps}"]

# EMA 开关与衰减
if EMA_ENABLE:
    args += [
        "++exp_manager.ema.enable=true",
        f"++exp_manager.ema.decay={EMA_DECAY}",
    ]
else:
    args += ["++exp_manager.ema.enable=false"]

# Tarred dataset 覆写
if TRAIN_IS_TARRED:
    train_tar_abs = resolve_path(train_base_dir, TRAIN_TAR_PATTERN)
    args += [
        "model.train_ds.is_tarred=true",
        f"model.train_ds.tarred_audio_filepaths={train_tar_abs}",
    ]
if VAL_IS_TARRED:
    val_tar_abs = resolve_path(val_base_dir, VAL_TAR_PATTERN)
    args += [
        "model.validation_ds.is_tarred=true",
        f"model.validation_ds.tarred_audio_filepaths={val_tar_abs}",
    ]

# Lhotse 开关
if USE_LHOTSE:
    args += [
        "model.train_ds.use_lhotse=true",
        "model.validation_ds.use_lhotse=true",
    ]

# Polars manifest 开关（仅在非 Lhotse 时生效）
if USE_POLARS:
    if USE_LHOTSE:
        print("[WARN] USE_POLARS=1 ignored because USE_LHOTSE=1")
    else:
        args += [
            "++model.train_ds.use_polars=true",
            "++model.validation_ds.use_polars=true",
        ]
        if test_manifest_path:
            args += ["++model.test_ds.use_polars=true"]

# Pretokenize 开关（仅在非 Lhotse 时有意义）
if not PRETOKENIZE:
    if USE_LHOTSE:
        print("[WARN] PRETOKENIZE=0 ignored because USE_LHOTSE=1")
    else:
        args += [
            "++model.train_ds.pretokenize=false",
            "++model.validation_ds.pretokenize=false",
        ]
        if test_manifest_path:
            args += ["++model.test_ds.pretokenize=false"]

# batch 大小
if train_bs_override is not None:
    args += [f"model.train_ds.batch_size={train_bs_override}"]
if val_bs_override is not None:
    args += [f"model.validation_ds.batch_size={val_bs_override}"]

# DataLoader worker
args += [
    f"model.train_ds.num_workers={NUM_WORKERS}",
    f"model.validation_ds.num_workers={NUM_WORKERS}",
]

# test ds 覆写（可选）
if test_manifest_path:
    tb = TEST_BATCH.strip()
    test_bs = int(tb) if tb else 16
    args += [
        f"++model.test_ds.manifest_filepath={test_manifest_path}",
        "++model.test_ds.shuffle=false",
        f"++model.test_ds.batch_size={test_bs}",
        f"++model.test_ds.num_workers={TEST_NUM_WORKERS}",
        "++model.test_ds.pin_memory=true",
    ]
    if USE_LHOTSE:
        args += ["++model.test_ds.use_lhotse=true"]
    elif USE_POLARS:
        args += ["++model.test_ds.use_lhotse=false"]

# Lhotse 动态 batch duration 覆写
if BATCH_DURATION_TRAIN.strip():
    args += [f"model.train_ds.batch_duration={BATCH_DURATION_TRAIN.strip()}"]
if BATCH_DURATION_VAL.strip():
    args += [f"model.validation_ds.batch_duration={BATCH_DURATION_VAL.strip()}"]
if test_manifest_path and BATCH_DURATION_TEST.strip():
    args += [f"++model.test_ds.batch_duration={BATCH_DURATION_TEST.strip()}"]

# 时长裁剪
if MAX_DURATION_TRAIN.strip():
    args += [f"model.train_ds.max_duration={MAX_DURATION_TRAIN.strip()}"]
if MIN_DURATION_TRAIN.strip():
    args += [f"model.train_ds.min_duration={MIN_DURATION_TRAIN.strip()}"]
if MAX_DURATION_VAL.strip():
    args += [f"model.validation_ds.max_duration={MAX_DURATION_VAL.strip()}"]
if MIN_DURATION_VAL.strip():
    args += [f"++model.validation_ds.min_duration={MIN_DURATION_VAL.strip()}"]

# Checkpoint
args += [f"+exp_manager.checkpoint_callback_params.dirpath={CKPT_DIR_TS}"]
if CKPT_EVERY_STEPS > 0:
    args += [f"+exp_manager.checkpoint_callback_params.every_n_train_steps={CKPT_EVERY_STEPS}"]
elif CKPT_EVERY_EPOCHS > 0:
    args += [f"+exp_manager.checkpoint_callback_params.every_n_epochs={CKPT_EVERY_EPOCHS}"]
if CKPT_EVERY_EPOCHS == 0:
    args += ["++exp_manager.checkpoint_callback_params.every_n_epochs=0"]
args += ["+exp_manager.checkpoint_callback_params.save_last=true"]

# TensorBoard
args += [
    "exp_manager.create_tensorboard_logger=true",
    f"exp_manager.exp_dir={TB_DIR}",
    f"exp_manager.name={RUN_NAME}",
]

# 续训策略
if RESUME:
    args += [
        "exp_manager.resume_if_exists=true",
        "exp_manager.resume_ignore_no_checkpoint=true",
    ]
else:
    args += ["exp_manager.resume_if_exists=false"]

# Augmentor 覆写
if aug_train_overrides:
    args += aug_train_overrides
if aug_val_overrides:
    args += aug_val_overrides

print("Launch:", " ".join(args))
subprocess.check_call(args)
