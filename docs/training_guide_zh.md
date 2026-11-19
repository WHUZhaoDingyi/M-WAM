# 在 RoboTwin 2.0 上训练 M-WAM

从仓库根目录执行下列命令，示例中的 `/path/to/...` 替换为本机路径。

## 示范数据

每条 RoboTwin 2.0 示范对应一个 HDF5 文件和一句任务指令。训练读取 M-WAM 使用的 HDF5 导出格式，每个文件包含以下数据。

| 数据 | 用途 |
| --- | --- |
| 头部、左腕和右腕相机的 RGB、深度、相机内外参 | 构造当前观测，将深度反投影为 XYZ |
| 物体和机器人部件的像素标签、逐帧位姿及时间戳 | 将源帧上的点随部件位姿变换，得到未来的 3D 运动标签 |
| 双臂的控制指令及时间戳 | 读取当前控制状态，生成未来 32 步、每步 14 维的动作标签 |

文件字段和存储布局见 [HDF5 读写代码](../src/mwam/robotwin_momap/raw_episode_hdf5.py)。将每条示范的 HDF5 路径和任务指令写入 [episodes.json](../examples/episodes.json) 格式的清单。相对路径以清单所在目录为起点。

## 准备训练输入

安装训练依赖，生成数据配置和文本、分割清单。

```bash
python -m pip install -e .
mwam prepare-data \
  --manifest /path/to/episodes.json \
  --template configs/robotwin_data.yaml \
  --text-cache data/text \
  --output data/prepared
```

**文本。** 每条指令用 Wan T5 编码一次，训练时读取缓存。已有对应的缓存时，将上面的 `--text-cache` 指向该目录即可。生成缓存的命令如下。

```bash
mwam encode-text \
  --prompts data/prepared/prompts.json \
  --text-cache data/text \
  --text-weights /path/to/wan_t5_weights.pth \
  --tokenizer /path/to/tokenizer \
  --output data/prepared/text_summary.json
```

**分割。** 训练使用头部相机的预测实例分割，两个腕部视角的分割通道填零。已有分割缓存时，在上面的 `prepare-data` 命令中添加 `--seg-manifest /path/to/seg_manifest.json`。首次生成分割缓存时，安装感知依赖和 SAM 2，准备 Grounding DINO 权重目录和 SAM 2.1 Hiera Base+ 权重，然后运行以下命令。脚本使用 [SAM 2 对应版本](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4)。

```bash
python -m pip install -e '.[perception]'
python -m pip install -e /path/to/sam2
python scripts/perception/run_robotwin_sam2_role_refresh_seg_only_cache.py \
  --manifest data/prepared/perception_manifest.json \
  --output-dir data/segmentation \
  --grounding-model /path/to/grounding_dino \
  --sam2-repo /path/to/sam2 \
  --sam2-checkpoint /path/to/sam2.1_hiera_base_plus.pt
mwam index-segmentation \
  --manifest data/prepared/perception_manifest.json \
  --cache-dir data/segmentation \
  --output data/prepared/seg_manifest.json
```

## 联合训练

模型用 PixelDiT XL 的 ImageNet 权重初始化。每个样本以当前观测、任务文本和机器人状态为条件，同时学习 15 Hz 的动作序列和第 8、16、24、32 步的三维位移。三个视角的位移都在源帧头部相机坐标系中表示。

```bash
mwam train \
  --config configs/train.yaml \
  --model-config configs/robotwin_3d_ac.yaml \
  --data-config data/prepared/data.yaml \
  --pixeldit-pretrained /path/to/pixeldit_imagenet.pth \
  --output-dir runs/robotwin
```

每卡批大小、学习率和更新步数由 [train.yaml](../configs/train.yaml) 设置。训练结束时，模型权重写入 `runs/robotwin/model.pt`。
