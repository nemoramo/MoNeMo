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
  --train-manifest /data2/mayufeng/swahili/swahili_v6_plus.filtered.filtered.abs.manifest \
  --val-manifest /data2/mayufeng/nemo_val/nemo_val_test/swahili_returndata.abs.manifest.exists \
  --tokenizer-dir /data1/mayufeng/tokenizer_2048_swa/tokenizer_spe_bpe_v2048 \
  --pretrained /data1/mayufeng/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/6d590f77001d318fb17a0b5bf7ee329a91b52598/parakeet-tdt-0.6b-v3.nemo \
  --language swahili \
  --out /data2/mayufeng/nemo_exps/swahili_0.6b_v6plus \
  --run-name swahili-0.6b-v6plus \
  --devices 8 --precision bf16 --train-bsz 32 --val-bsz 32 \
  --max-epochs 30 --ckpt-every-steps 5000 --val-check-interval 2000
```

要按步数存 checkpoint：`--ckpt-every-steps 5000`。  
要调整验证频率：`--val-check-interval <steps>`（默认为 2000）。  

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
