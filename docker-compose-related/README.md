# RamosNeMo docker-compose quickstart

## 启动容器
```bash
cd RamosNeMo/docker-compose-related
docker compose -f docker-compose.yml build nemo-training
docker compose -f docker-compose.yml up -d nemo-training
```

说明：`docker-compose.yml` 默认使用仓库根目录的 `Dockerfile.nemo25` 构建镜像。

进入容器：
```bash
docker compose -f docker-compose.yml exec -it nemo-training bash
# 或者直接用容器名（默认 projects-nemo-training-1）
docker exec -it projects-nemo-training-1 bash
```

## 训练示例命令
容器内执行（使用挂载的代码和数据）：
```bash
python /opt/ramosnemo_source/entrance_kit/local/entrance.py \
  --config-name fastconformer_ctc_tdt_hybrid_0.6b \
  --train-manifest /data2/<user>/swahili/train.manifest \
  --val-manifest /data2/<user>/swahili/val.manifest \
  --tokenizer-dir /data1/<user>/tokenizer_2048_swa/tokenizer_spe_bpe_v2048 \
  --pretrained /data1/<user>/models/parakeet-tdt-0.6b-v3.nemo \
  --language swahili \
  --out /data2/<user>/nemo_exps/swahili_0.6b_v6plus \
  --run-name swahili-0.6b-v6plus \
  --devices 8 --precision bf16 --train-bsz 32 --val-bsz 32 \
  --max-epochs 30 --ckpt-every-steps 5000 --val-check-interval 2000
```

要按步数存 checkpoint：`--ckpt-every-steps 5000`。  
要调整验证频率：`--val-check-interval <steps>`（默认为 2000）。  

## S3/TOS 训练示例（Docker + env-file）
宿主机执行（推荐把凭证放在 `speech_related_tools/.env`）：
```bash
cd /path/to/RamosNeMo/docker-compose-related
ENV_FILE=/path/to/speech_related_tools/.env
docker compose --env-file "${ENV_FILE}" -f docker-compose.yml up -d nemo-training
```

容器内启动 EN+AR CTC 110M（manifest 可用 `s3://` / TOS S3 兼容路径）：
```bash
docker compose --env-file "${ENV_FILE}" -f docker-compose.yml exec -it nemo-training bash -lc "
python /opt/ramosnemo_source/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
  --config-path=/opt/ramosnemo_source/examples/asr/conf/fastconformer \
  --config-name=fast-conformer_ctc_bpe_110m_en_ar_tos \
  model.train_ds.manifest_filepath=s3://<bucket>/<prefix>/train_normalized.jsonl \
  model.validation_ds.manifest_filepath=s3://<bucket>/<prefix>/val_normalized.jsonl \
  model.tokenizer.dir=/data2/<user>/tokenizers/en_ar_bpe \
  init_from_nemo_model.model0.path=/data2/<user>/models/parakeet-tdt_ctc-110m.nemo \
  trainer.devices=4 \
  trainer.accelerator=gpu \
  trainer.strategy=ddp \
  trainer.max_epochs=15 \
  model.train_ds.batch_size=96 \
  model.validation_ds.batch_size=96 \
  exp_manager.exp_dir=/data2/<user>/nemo_exps \
  exp_manager.name=en_ar_ctc_110m_tos
"
```

说明：
- `docker-compose.yml` 会把 `TOS_*` 和 `AWS_*` 环境注入容器，`setup_ramosnemo.sh` 会自动做 `TOS_* -> AWS_*` 兼容映射。
- 默认关闭音频 S3/TOS 本地缓存（`NEMO_S3_CACHE_DISABLE=1`）。
- 若要开启缓存：设置 `NEMO_S3_CACHE_DISABLE=0`，并按需设置 `NEMO_S3_CACHE_DIR`、`NEMO_S3_CACHE_SIZE_GB`。
- 新增示例配置：`examples/asr/conf/fastconformer/fast-conformer_ctc_bpe_110m_en_ar_tos.yaml`。

## entrance.py 运行示例（Hybrid RNNT+CTC）
如果你希望复用 `entrance_kit/local/entrance.py`（对齐 SageMaker 启动方式），可在容器内这样执行：
```bash
docker compose --env-file "${ENV_FILE}" -f docker-compose.yml exec -it nemo-training bash -lc '
python /opt/ramosnemo_source/entrance_kit/local/entrance.py \
  --config-name fastconformer_hybrid_tdt_ctc_bpe_110m \
  --train-manifest /data2/<user>/manifests/train.jsonl \
  --val-manifest /data2/<user>/manifests/val.jsonl \
  --tokenizer-dir /data2/<user>/tokenizers/en_ar_bpe \
  --pretrained /data2/<user>/models/parakeet-tdt_ctc-110m.nemo \
  --out /data2/<user>/nemo_exps/en_ar_hybrid \
  --run-name en_ar_hybrid \
  --devices 4 --precision bf16 \
  --train-bsz 96 --val-bsz 96 \
  --num-workers 8 \
  --max-epochs 15 \
  --val-check-interval 2000 \
  --ckpt-every-steps 2000
'
```

说明：
- `entrance.py` 当前默认走 hybrid 训练入口（RNNT+CTC）。
- 如果要纯 CTC 训练，使用上面 `speech_to_text_ctc_bpe.py` + `fast-conformer_ctc_bpe_110m_en_ar_tos.yaml` 的示例。

## 挂载与依赖
- `/data1`, `/data2` 挂载到容器内同路径；`../RamosNeMo` 挂载到 `/opt/ramosnemo_source`。  
- `Dockerfile.nemo25` 已安装 `ffmpeg`；启动时 `/opt/setup_ramosnemo.sh` 默认会执行 `pip install -e "/opt/ramosnemo_source[asr,audio]"`（可用 `RAMOSNEMO_EXTRAS=none` 关闭 extras）。 

## numba / pynvjitlink（RNNT/TDT 必需）
在 CUDA>=12 环境下，`numba-cuda` 的 MVCLinker（`NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1`）会直接报错：`Use CUDA_ENABLE_PYNVJITLINK for CUDA >= 12.0 MVC`。  
本仓库的 `Dockerfile.nemo25` 已默认安装 `pynvjitlink-cu12` 并设置：
- `NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=0`
- `NUMBA_CUDA_ENABLE_PYNVJITLINK=0`（默认更稳）

若你修改过镜像或环境变量导致再次报错，重新构建镜像即可：
```bash
cd RamosNeMo/docker-compose-related
docker compose -f docker-compose.yml build --no-cache nemo-training
```

如需尝试开启 `pynvjitlink`（可能在某些 forward-compat / CUDA 版本组合下触发 PTX 版本不匹配报错），可在启动容器时显式设置：
```bash
export NUMBA_CUDA_ENABLE_PYNVJITLINK=1
docker compose -f docker-compose.yml up -d nemo-training
```
