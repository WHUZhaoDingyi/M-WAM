# 从数据准备到训练结束

## 哪些功能是必需的

| 功能 | 是否必要 | 当前交付 |
|---|---|---|
| 原始数据读取、合法窗口、完整 DataLoader | 训练必需 | 已加入原生实现，支持 raw-v1 HDF5 和固定 runtime index |
| 优化器、学习率调度、训练循环、最终保存 | 训练必需 | 原生 AdamW、warmup/cosine、梯度累积、日志和 checkpoint |
| 断点续训 | 长训练必需 | 权重、optimizer、scheduler、RNG、采样游标和恢复契约 |
| 分布式训练 | 算法非必需，大模型训练实用必需 | Accelerate；提供单机8卡 ZeRO-2配置；也支持 torchrun DDP |
| 深度→XYZ | 必需 | 原生 depth-ray 重建、相机坐标变换、有效掩码与统一 mosaic |
| 文本编码 | 必需，但可离线做一次 | Wan T5 编码和按完整 prompt 哈希寻址的缓存读取 |
| RGB→分割 | 主训练配方的输入依赖 | 原生 GroundingDINO + SAM2 因果跟踪缓存生成脚本；可复用现有缓存 |
| 模拟器/真机适配、动作执行器、闭环评测 | 离线训练不需要，部署和测 SR 需要 | 尚未迁入。现有 policy 输出动作，不自动控制真实机械臂 |
| egg-info、Python cache | 不属于方法 | 不打进源码包；安装或 import 后可自动生成 |

## 输入数据的边界

本训练器读取的是本项目 **RoboTwin raw-v1 HDF5**，不是任意 RGB 视频，也不能把官方不同结构的 HDF5 改个文件名直接使用。必须包含当前三相机 RGB、深度/ray、相机标定、动作控制流、时间对齐、物体及机器人几何/位姿和 episode 元数据。读取/写出格式见 `src/mwam/robotwin_momap/raw_episode_hdf5.py` 的 `RawEpisodeReader` / `RawEpisodeWriter`。

未来几何用来构造 3D 运动监督，未来动作是 action 标签。它们不作为推理条件。当前 XYZ 来自当前深度和标定，不能直接用未来点坐标替代。

```text
当前观测：RGB、预测头分割、当前 depth→XYZ、文本、proprio
                          ↓
                       条件 C
                          ├── 世界分支 ← 未来3D运动加噪（训练随机变量）
                          └── 动作分支 ← 未来动作加噪（训练随机变量）
                                 ↓
                        运动 loss + action loss

私有几何 / 未来动作 → 监督标签；不进入当前条件 C。
闭环成功判定 → 事后评测；不进入训练 loss。
```

### 新机器需要提供的资产

1. raw-v1 数据及每个 episode 的正确任务指令，参照 `examples/episodes.json`。相对 episode 路径按清单文件所在目录解析。
2. 原 PixelDiT XL ImageNet 初始化权重。当前 factory 要求该权重；`null` 是公开模板占位，CLI 必须传入实际路径。
3. 文本缓存，或本地 Wan T5 权重及 tokenizer。只用真实编码，不用随机/零向量代替。
4. 原生 v3 预测分割缓存及 manifest；没有时用附带的 GroundingDINO/SAM2 脚本生成。第三方代码、权重和许可证需要另行准备，不内嵌下载。
5. CUDA 训练机器。本轮所在容器无 GPU；CPU 能做数据检查和小型训练器测试，不能据此宣称完整模型 GPU 训练已验证。

固定运动/action/proprio 归一化统计已保留在模型配置中。换成其他机器人、动作维度或差异较大的新数据集时，需要重新标定相应统计并定义新契约，不应沿用现有 digest 冒充旧统计。

## 数据准备

`prepare-data` 读取每条 HDF5 的真实元数据，绑定指令，输出 `data.yaml`、`prompt_index.json`、`prompts.json`、`perception_manifest.json` 和摘要。它不修改原始文件。

- 已有分割：传 `--seg-manifest`；manifest 内每个 `cache_path` 也必须能在新机器访问。迁移数据时要显式更新这些路径。
- 没有分割：不传此参数。训练数据配置先指向待生成的 `prepared/seg_manifest.json`，随后运行分割缓存脚本，再用 `index-segmentation` 建索引。预处理清单覆盖帧1到最后一帧；模型每个时刻只看当前及过去 RGB，没有倒放 SAM2。
- `index-segmentation` 只索引 PASS 的预测缓存，不把缺失预测替换成 GT。新索引使用 `p_gt=0`、`predicted_only_unscored`，不声称做过质量评分。
- 本主方法 `strict_pred_head_only=true`，使用头部预测分割，腕部分割条件置零；XYZ 与运动监督仍覆盖三个视角。该模式下原质量混合表的 p_gt 不用于回退 GT。
- SAM2 固定兼容 commit 为 `2b90b9f5ceec907a1c18123530e92e794ad901a4`；需要安装该仓库依赖。分割入口只加载 GroundingDINO 与 SAM2，不加载 Depth Anything，尽管历史依赖脚本文件名包含 da3。
- `encode-text` 会复用并验证已有编码，仅生成缺失项。缓存对象为 context `[128,4096]` 和 bool mask `[128]`，prompt 使用原生固定模板。
- `check-data` 构造真实 Dataset 和 PyTorch DataLoader，计算实际 XYZ 与未来运动/动作监督，检查选定 batch。它不是所有样本质量通过的证明。

公开原始入口每次建立窗口索引，适合新数据和调试；大规模已有数据可用保留的 `create_fixed_v1_momap_dataset` 和对应固定 index 配置避免重复建索引。不得把旧索引与不同 episode 清单混用。

## 启动、结束与恢复

`train` 合并模型、训练和已准备的数据三份配置，执行原生训练器。每步记录 loss；`max_steps` 是优化器更新次数，不是 microbatch 数。

```text
effective_batch = batch_size_per_rank * world_size * gradient_accumulation_steps
```

公开模板 batch_size=1 是可调整的起点，不承诺任何显存大小都能运行，也不等价于原16卡×24的训练预算。实际显存取决于精度、ZeRO、激活和 batch。多卡命令中的进程数必须与使用的卡数相符。

训练输出：

- `config.yaml`：本次解析后的配置，供运行追溯，不应把机器路径文件提交到公开仓库。
- `training_metrics.jsonl`：训练 loss、LR 和进度；不是闭环成功率。
- `checkpoints/weights/step_XXXXXX.pt`：供推理的模型权重。
- `checkpoints/state/step_XXXXXX/`：完整训练恢复目录。
- `summary.json`：CLI 正常返回时写入真实完成 step；异常中断不生成新的完成结论。

`max_steps` 到达后训练器保存最终权重和完整状态。短程检查用 `stop_after_step`，保持 `max_steps` 的最终预算不变；恢复时传完整 state 目录，再取消或提高 `stop_after_step`。不能把权重 `.pt` 当成完整续训状态。

精确恢复要求模型、训练/数据契约、拓扑、batch 与采样模式兼容，不能随意改卡数或学习率。当前新增的 DataLoader 随机流隔离也纳入契约：旧集群的 full-state 不自动承诺兼容；本包产生的 state 可按测试覆盖的流程恢复。模型权重加载兼容与完整训练状态恢复是两件事。

MoMap 主训练 `eval_every=0`：原 generic trainer 的内置 evaluator 是视频任务协议，不应打开后冒充 MoMap/SR 评测。训练后离线验证和模拟器 SR 是独立步骤。

## 本轮必要的基础设施修复

- Python 包从 `fastwam` 改为 `mwam`；保留旧缓存和 checkpoint 的序列化标识与参数键。
- 不再在 import 时要求原集群 `configs/paths.json`；主流程的资产都通过配置传入。
- DataLoader 使用独立 generator，避免恢复时重建 iterator 额外消耗模型 CPU 随机流。原生数据增强按 sample/epoch 确定，不依赖 worker 全局随机流。此修复不改变网络结构或 loss，但不声称与旧训练的整个随机序列逐步一致。
- 主配置直接验证 KFlowContract，修正此前 architecture_version 的拼写。

代码来源和改动哈希见 `training_source_inventory.json`，分割脚本来源见 `perception_source_inventory.json`。

## train.yaml 词条说明

| 词条 | 含义 |
|---|---|
| output_dir | 运行输出目录，由 CLI 覆盖；公开模板不写机器路径 |
| batch_size | 每个 rank、每次前向的样本数，不是有效全局 batch |
| num_workers | 每个 rank 的 DataLoader 子进程数 |
| prefetch_factor | 每 worker 预取 batch 数；workers=0时须为null |
| dataloader_non_blocking | Accelerate 递归搬运到GPU时允许非阻塞拷贝 |
| dataloader_rng_isolated | DataLoader base seed 独立于模型CPU随机流；已加入恢复契约 |
| persistent_workers | epoch之间保留worker；workers=0时须为false |
| lr_scheduler_type | 学习率调度类型，主配置cosine |
| learning_rate | AdamW基础学习率；具名参数组另有显式值 |
| cosine_min_learning_rate | 余弦调度末端学习率 |
| warmup_ratio | warmup步数占max_steps的比例 |
| num_epochs | 未指定max_steps时用于推导预算；当前显式max_steps优先 |
| max_steps | 总优化器更新次数 |
| log_every | 每多少次更新写训练日志 |
| save_every | 定期保存间隔；0关闭定期保存，不取消正常终点保存 |
| eval_every | 本主方法必须为0，避免误用视频任务 evaluator |
| eval_num_inference_steps | 原通用evaluator的推理步数；eval_every=0时不生效 |
| gradient_accumulation_steps | 每次optimizer更新累积的microbatch数 |
| training_stage | 主配置joint，联合训练世界与动作；本公开入口不接收其他stage |
| strict_resumable_sampler | 使用能精确恢复顺序和游标的原生采样器 |
| mixed_precision | bf16/fp16/no；同时决定模型构造dtype及Accelerate精度模式 |
| seed | 模型、采样和训练随机种子；不是闭环评测初态seed |
| max_grad_norm | 梯度范数裁剪阈值 |
| weight_decay | AdamW权重衰减 |
| resume | 完整状态目录；CLI --resume覆盖，null代表新训练 |
| weights_only_warmstart | 历史权重迁移入口，当前null；不等于精确恢复 |
| wandb | 可选实验日志配置分组 |
| wandb.enabled | 是否使用W&B，默认false，不联网记录 |
| wandb.workspace | W&B entity/工作区 |
| wandb.project | W&B项目名 |
| wandb.name | 运行显示名 |
| wandb.group | 可选运行分组 |
| wandb.mode | W&B online/offline等模式，仅enabled=true时生效 |
| train_sampling_mode | all_windows_per_epoch：遍历本epoch全部合法窗口并按原生采样规则shuffle/pad |
| failure_replay_manifest | 历史失败场景replay资产，当前null |
| adaptive_replay | 历史动态replay配置，当前null |
| training_timing_enabled | 记录数据等待、前向、反向、optimizer时间 |
| deepspeed_zero2_graph_task_cache_enabled | 原实现ZeRO-2重入checkpoint兼容补丁；只在对应DeepSpeed启动时开启 |
| optimizer_parameter_groups | 为action_backbone、action_io、world分别配置优化器学习率 |
| optimizer_parameter_groups.action_backbone | 动作主干参数组 |
| optimizer_parameter_groups.action_io | 动作输入/输出及相关接口参数组，以原生参数归组逻辑为准 |
| optimizer_parameter_groups.world | 世界分支参数组 |
| 各参数组.learning_rate | 该组峰值学习率 |
| 各参数组.cosine_min_learning_rate | 该组余弦末端学习率 |
| 各参数组.warmup_start_learning_rate | 该组warmup初始学习率 |
| stop_after_step | 临时停止的绝对optimizer step；null表示到max_steps |
| checkpoint_on_stop_after_step | 临时停止时是否保存完整状态，默认true |
| controlled_model_math_migration | 历史受控数学契约迁移扩展，当前null |
| fresh_stage_weights_transition | 历史新阶段权重迁移扩展，当前null |
| runtime_checkpoint_control_enabled | 通过控制文件请求checkpoint/stop的原生扩展；公开默认false，不接入旧任务控制器 |

`configs/accelerate_zero2.yaml` 只负责启动拓扑与ZeRO配置，不替代上述优化器/DataLoader配置。DDP/ZeRO需要相同代码、数据和外部资产在各rank可见。
