# ramos nemo docker-compose quickstart

这套 compose 主要给两类工作准备环境：

- 在容器里直接跑 `ramos nemo` 训练/推理。
- 给 `speech_related_tools` 的 Gemini -> NFA 流水线提供一个可复用的 NeMo/NFA 运行时。

容器内仓库会挂载到 `/opt/ramosnemo_source`。目录名可能沿用旧名，但项目名称仍按当前 `ramos nemo` 称呼。

## 启动容器

```bash
cd /path/to/your/ramos-nemo-repo/docker-compose-related
docker compose -f docker-compose.yml build nemo-training
docker compose -f docker-compose.yml up -d nemo-training
docker compose -f docker-compose.yml exec -it nemo-training bash
```

说明：

- `docker-compose.yml` 默认使用仓库根目录的 `Dockerfile.nemo25` 构建镜像。
- 如果你的环境只有 `docker-compose`，下面所有 `docker compose` 命令都可以等价替换为 `docker-compose`。
- 容器启动时会执行 `/opt/setup_ramosnemo.sh`，默认安装 `pip install -e "/opt/ramosnemo_source[asr,audio]"`。
- 如需关闭 extras，可在启动前设置 `RAMOSNEMO_EXTRAS=none`。

## 挂载目录

当前 compose 会把以下宿主机路径映射到容器内同路径：

- `/data1`
- `/data2`
- `/data3`
- `/mnt/asr-audio-data`
- `/mnt/asr-audio-roufo`

另外：

- 当前仓库 `..` 挂载到 `/opt/ramosnemo_source`
- `../logs` 挂载到 `/workspace/logs`

### 为什么要加 `/mnt/asr-audio-roufo`

这是为了给 `speech_related_tools` 的 NFA 流水线做准备。

`speech_related_tools/scripts/gemini_nfa_pipeline.py` 最终会调用：

```bash
python /opt/ramosnemo_source/tools/nemo_forced_aligner/align.py ...
```

NFA 读取的 manifest 里，`audio_filepath` 通常保留原始绝对路径，例如：

```json
{"audio_filepath": "/mnt/asr-audio-roufo/project_a/demo.wav", "text": "example transcript"}
```

如果容器里没有把 `/mnt/asr-audio-roufo` 挂载成同路径，`align.py` 会直接找不到音频文件。  
因此现在同时保留：

- `/mnt/asr-audio-data:/mnt/asr-audio-data`
- `/mnt/asr-audio-roufo:/mnt/asr-audio-roufo`

如果你的 manifest 使用别的绝对路径前缀，也建议按同样方式补一条“宿主机路径:容器路径一致”的映射。

## NFA 环境准备

### 与 `speech_related_tools` 的衔接关系

从 `speech_related_tools` 侧接入时，重点关注以下约定：

- `--nemo-home` 应指向包含 `tools/nemo_forced_aligner/align.py` 的 `ramos nemo` 仓库。
- 在这个容器里，`--nemo-home` 应写成 `/opt/ramosnemo_source`。
- NFA 目前只支持 CTC 类 ASR 模型，不要传 RNNT/TDT 模型给 `align.py`。
- manifest 至少要有 `audio_filepath` 和 `text`；如果走 `align_using_pred_text=True`，则允许不提供 `text`，但这不是 `speech_related_tools` 默认路径。
- 默认的 `RAMOSNEMO_EXTRAS=asr,audio` 通常够用；如果启动 NFA 时遇到 `Missing required dependency for NFA`，可在容器内补执行 `pip install -e "/opt/ramosnemo_source[all]"`。

### 容器内直接跑 NFA

```bash
python /opt/ramosnemo_source/tools/nemo_forced_aligner/align.py \
  pretrained_name=stt_en_conformer_ctc_large \
  manifest_filepath=/data2/mayufeng/nfa_demo/manifest.jsonl \
  output_dir=/data2/mayufeng/nfa_demo_out \
  transcribe_device=cuda \
  viterbi_device=cuda \
  batch_size=4
```

### 给 `speech_related_tools` 用的最小参数参考

如果你是在这个容器里额外挂载了 `speech_related_tools`，常用组合大致如下：

```bash
python scripts/gemini_nfa_pipeline.py \
  --audio-dir /mnt/asr-audio-roufo/project_a/raw_audio \
  --nemo-home /opt/ramosnemo_source \
  --pretrained-name stt_en_conformer_ctc_large \
  --output-root /data2/mayufeng/nfa_results \
  --device cuda \
  --batch-size 4
```

## 常用参数说明

### compose / 容器层

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | 控制容器内可见 GPU | 当前 compose 默认为 `0,1,2,3,4,5,6,7` |
| `RAMOSNEMO_EXTRAS` | 控制启动时 `pip install -e` 的 extras | 默认 `asr,audio`；若只想最小安装可设 `none` |
| `RAMOSNEMO_SOURCE_DIR` | 指定源码目录 | 默认 `/opt/ramosnemo_source` |

### NFA / `speech_related_tools` 层

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `--nemo-home` | `ramos nemo` 仓库根目录 | 容器里写 `/opt/ramosnemo_source`；宿主机执行时写实际仓库路径 |
| `--pretrained-name` | 使用预训练 CTC 模型 | 和 `--model-path` 二选一；快速验证时优先它 |
| `--model-path` | 使用本地 `.nemo` 模型 | 和 `--pretrained-name` 二选一 |
| `--device` | NFA 运行设备 | 默认用 `cuda`；显存不足时可退到 `cpu` |
| `--batch-size` | NFA 批大小 | 短音频可适当放大；OOM 时先减小 |
| `--output-root` | 输出根目录 | 建议放在 `/data2` 这类大盘路径 |
| `--segment-sep` | 传给 NFA 的段分隔符 | `speech_related_tools` 默认是 `\\|` |
| `--skip-gemini` | 复用已有 transcript，不重新转写 | 本地复跑 NFA 时很实用 |
| `--use-vtt` | 从同名 `.vtt` 生成 transcript | 适合已有字幕的音频 |
| `--soft-manifest` | 仅输出带偏移的 manifest，不实际切音频 | 适合先验收对齐质量 |
| `--nfa-auto-group` | 按音频时长自动分组跑 NFA | 混合长短音频时建议开启 |
| `--nfa-group-policy` | 动态分组策略 | 默认 `300:6,500:4,inf:1`，长音频可更保守 |

### 常见坑

- `audio_filepath` 在宿主机能访问，不等于容器里也能访问。只要 manifest 用的是绝对路径，就要保证 compose 里有同路径挂载。
- `align.py` 只支持 CTC 模型。`fastconformer_ctc_*`、`stt_*_ctc_*` 这类一般是安全选择。
- 如果显存紧张，优先调小 `--batch-size`，再考虑把 `transcribe_device` / `viterbi_device` 切到 CPU。
- 如果只是为了复跑对齐，不想重复调用 Gemini，优先使用 `--skip-gemini`。

## 训练示例命令

容器内执行：

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

- 要按步数存 checkpoint：`--ckpt-every-steps 5000`
- 要调整验证频率：`--val-check-interval <steps>`，默认 `2000`

## numba / pynvjitlink（RNNT/TDT 必需）

在 CUDA >= 12 环境下，`numba-cuda` 的 MVCLinker（`NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1`）会直接报错：

```text
Use CUDA_ENABLE_PYNVJITLINK for CUDA >= 12.0 MVC
```

本仓库的 `Dockerfile.nemo25` 已默认安装 `pynvjitlink-cu12` 并设置：

- `NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=0`
- `NUMBA_CUDA_ENABLE_PYNVJITLINK=0`

如果你修改过镜像或环境变量导致再次报错，重新构建镜像即可：

```bash
cd /path/to/your/ramos-nemo-repo/docker-compose-related
docker compose -f docker-compose.yml build --no-cache nemo-training
```

如需尝试显式开启 `pynvjitlink`，可在启动前设置：

```bash
export NUMBA_CUDA_ENABLE_PYNVJITLINK=1
docker compose -f docker-compose.yml up -d nemo-training
```
