# 可复制命令

在 **M-WAM仓库根目录**、已激活的Python环境中执行。变量指向你自己的本地资产；`${VAR:?…}` 会在遗漏时清楚报错，不下载或猜测文件。

交互包装在成功和失败时都返回原终端，并显示任务真实退出码；其外层为了保留终端返回0，不要作为自动调度入口。自动化直接调用包装内的实际任务程序并读取真实退出码。所有代码块已通过 `/bin/sh` 外层和 Bash 内层语法检查；GPU命令只是待执行命令，不是已完成记录。

数据目录与checkpoint会较大；源码包不包含它们。配置解释见 [training_guide_zh.md](training_guide_zh.md)。

## 安装

本輪已构建wheel；新环境安装及依赖下载未执行。SAM2须另按固定commit安装。

```sh
/bin/bash -lc 'python3 -m pip install -e ".[test,distributed,perception]"; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 帮助

已执行；只显示帮助，不生成结果文件。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py --help; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 准备数据：复用现有分割

清单示例见 examples/episodes.json。已用真实2 episode执行通过；各环境路径由变量提供。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py prepare-data --manifest "${MWAM_EPISODES:?set MWAM_EPISODES}" --template configs/data_raw_v1.yaml --text-cache "${MWAM_TEXT_CACHE:?set MWAM_TEXT_CACHE}" --seg-manifest "${MWAM_SEG_MANIFEST:?set MWAM_SEG_MANIFEST}" --output outputs/prepared; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 准备数据：需要新生成分割

与上一条二选一，输出目录必须尚不存在；清单生成已验证。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py prepare-data --manifest "${MWAM_EPISODES:?set MWAM_EPISODES}" --template configs/data_raw_v1.yaml --text-cache "${MWAM_TEXT_CACHE:?set MWAM_TEXT_CACHE}" --output outputs/prepared; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 从RGB生成分割缓存

CUDA与本地GroundingDINO/SAM2资产就绪后执行。本轮仅通过原脚本self-test，没有执行GPU分割。必须使用兼容SAM2 commit；脚本内不加载DA3。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/perception/run_robotwin_sam2_role_refresh_seg_only_cache.py --manifest outputs/prepared/perception_manifest.json --output-dir outputs/seg_cache --grounding-model "${MWAM_GROUNDING_MODEL:?set MWAM_GROUNDING_MODEL}" --sam2-repo "${MWAM_SAM2_REPO:?set MWAM_SAM2_REPO}" --sam2-checkpoint "${MWAM_SAM2_CHECKPOINT:?set MWAM_SAM2_CHECKPOINT}" --rank 0 --world-size 1 --device cuda:0 --fail-fast; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 索引新分割

只接受完整预测缓存；没有生成缓存时此命令会失败。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py index-segmentation --manifest outputs/prepared/perception_manifest.json --cache-dir outputs/seg_cache --output outputs/prepared/seg_manifest.json; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 生成或复用文本编码

已验证复用真实T5缓存。新生成路径需本地权重/tokenizer，本轮未做新编码；CPU可用 --device cpu，但T5需要较多内存。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py encode-text --prompts outputs/prepared/prompts.json --text-cache "${MWAM_TEXT_CACHE:?set MWAM_TEXT_CACHE}" --text-weights "${MWAM_TEXT_WEIGHTS:?set MWAM_TEXT_WEIGHTS}" --tokenizer "${MWAM_TOKENIZER:?set MWAM_TOKENIZER}" --device cuda --batch-size 1 --output outputs/text_summary.json; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 检查真实训练batch

已在CPU检查2个真实batch；这不等于整库质量检查。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py check-data --data-config outputs/prepared/data.yaml --batches 2 --workers 0 --output outputs/data_check.json; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 检查完整训练配置

已验证；只组装和校验配置，不分配完整模型，不启动训练。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py check-training-config --config configs/train.yaml --model-config configs/robotwin_3d_ac.yaml --data-config outputs/prepared/data.yaml --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --output-dir outputs/train --output outputs/train_config_check.json; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 单GPU训练

待CUDA实机验收，未执行。完整1.61B模型显存需求需要实测；不要因batch=1就假定任意GPU都能运行。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py train --config configs/train.yaml --model-config configs/robotwin_3d_ac.yaml --data-config outputs/prepared/data.yaml --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --output-dir outputs/train; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 8GPU ZeRO-2训练

待8卡资源/DeepSpeed环境就绪，未执行。与单卡训练二选一。公开batch=1；根据显存调整，不等价于原16卡×24训练。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- accelerate launch --config_file configs/accelerate_zero2.yaml scripts/mwam.py train --config configs/train.yaml --model-config configs/robotwin_3d_ac.yaml --data-config outputs/prepared/data.yaml --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --output-dir outputs/train --set deepspeed_zero2_graph_task_cache_enabled=true; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 8GPU DDP训练

待硬件验收，未执行。DDP不分片optimizer，显存需求比ZeRO-2更高；与其他训练命令二选一。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- torchrun --standalone --nproc_per_node=8 scripts/mwam.py train --config configs/train.yaml --model-config configs/robotwin_3d_ac.yaml --data-config outputs/prepared/data.yaml --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --output-dir outputs/train; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 短程GPU训练检查

待CUDA验收，未执行。另建 train_probe 目录，保留50000总预算，在第2步保存停止。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py train --config configs/train.yaml --model-config configs/robotwin_3d_ac.yaml --data-config outputs/prepared/data.yaml --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --output-dir outputs/train_probe --set stop_after_step=2; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 从第2步恢复并完成

待上一条产生完整state后执行，保持相同卡数、batch、数据、配置；本条对应单GPU短程检查。恢复后继续至max_steps=50000。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py train --config configs/train.yaml --model-config configs/robotwin_3d_ac.yaml --data-config outputs/prepared/data.yaml --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --output-dir outputs/train_probe --resume outputs/train_probe/checkpoints/state/step_000002 --set stop_after_step=null; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 方法与训练器测试

已通过。训练器测试使用小型CPU网络，包含真实优化、梯度累积和保存恢复，不代表完整M-WAM反向或GPU通过。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 -m pytest -q -p no:cacheprovider tests; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

## 严格加载与动作推理

需真实全3D权重和符合输入契约的NPZ；GPU版本未执行。step190075旧混合版本不能用。

```sh
/bin/bash -lc 'python3 scripts/run_with_result.py -- python3 scripts/mwam.py infer --config configs/robotwin_3d_ac.yaml --checkpoint "${MWAM_CHECKPOINT:?set MWAM_CHECKPOINT}" --pixeldit-pretrained "${MWAM_PIXELDIT_PRETRAINED:?set MWAM_PIXELDIT_PRETRAINED}" --observation "${MWAM_OBSERVATION:?set MWAM_OBSERVATION}" --device cuda --steps 5 --verify-direct --output outputs/action.npz; mwam_status=$?; printf "任务退出码=%s\n" "$mwam_status"; exit 0'
```

