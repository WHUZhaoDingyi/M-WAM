# 主方法实现清单与模块边界

## 固定的方法选择

| 项目 | 配置 / 实现 |
|---|---|
| 全视角 3D | `all_views_dxyz_matched_uv_valid_v1`，头部与双腕均以 `head_camera@source_t0` 坐标系中的米表示 |
| AC | `kv_source=action_condition`；C 只读取 C，W 读取 C+W，A 读取 C+自身 |
| 轨迹 token | `joint14`：输入 `[B,32,14]` 转置为 14 个长度为 32 的控制轨迹，再映射到隐藏维 |
| 高噪声采样 | World 和 Action 均使用 `WanHighNoise405505UniformWeightFlowMatchScheduler`，训练时刻独立 |
| 主干 | PixelDiT 26 个 global blocks + 4 个 pixel blocks；Action 30 层，hidden 1152 |
| 时域 | 15 Hz，32 步动作；运动偏移 `[8,16,24,32]` |

高噪声分布：40% 概率在 sigma `[0,0.95)`，55% 在 `[0.95,1)`，5% 在 sigma=1。
区间内部使用原 shifted-uniform 分布的条件采样，不能误写为区间内均匀采样。
每个样本的损失权重为 1；这不等于推理初始噪声强度的调整。

## 原实现位置

以下路径均相对 `src/mwam/`：

| 功能 | 文件 / 接口 |
|---|---|
| 原生模型组装 | `models/pixeldit/factory.py::create_hybrid_k4_fastwam_pixeldit` |
| 当前条件编码和独立 proprio 处理 | `models/pixeldit/fastwam_pixeldit.py::_prepare_observation/_prepare_context` |
| 世界 C/W 分支 | `models/pixeldit/hybrid_cw_pixeldit.py::HybridCWPixelDiT` |
| 动作分支与轨迹 token | `models/pixeldit/pixeldit_aligned_action.py::PixelDiTAlignedActionDiT` |
| AC 可见性及分支调度 | `models/pixeldit/hybrid_cw_action_mot.py::HybridCWActionMoT` |
| 全视角 3D 损失 | `models/pixeldit/hybrid_fastwam_pixeldit.py::_compute_momap_velocity_loss` |
| 3D 归一化 | `models/pixeldit/hybrid_normalization.py` |
| 加噪与采样分布 | `models/wan22/schedulers/scheduler_continuous.py` |
| 联合 Euler 推理、严格加载 | `models/pixeldit/fastwam_pixeldit.py::infer_joint/load_checkpoint` |

原生模型、数据与训练依赖已迁入 mwam；来源和变化见 `training_source_inventory.json`。参数键及序列化格式标签保持不变。训练基础设施修复见 training_guide_zh.md。`source_inventory.json` 是第一阶段的历史记录。

## NPZ 输入约定

模型接收拼接完成的当前观测，不接收未来 motion、未来 RGB 或 GT action：

| 字段 | 形状 | 含义 |
|---|---|---|
| source_rgb | `[B,3,384,320]` | RGB，范围 [-1,1] |
| source_seg_condition | 同上 | 原生分割条件编码，范围 [-1,1]；不能直接放任意 actor IDs |
| source_xyz_ref | 同上 | 当前各视角像素对应的 XYZ，参考头相机当前坐标系，原始米制 |
| source_condition_valid | `[B,1,384,320]` | bool 几何有效掩码 |
| context | `[B,L,4096]` | 预计算的原生文本 embedding |
| context_mask | `[B,L]` | bool 文本有效掩码 |
| proprio | `[B,14]` 或 `[B,1,14]` | 当前 last-command drive target，未归一化 |

布局：上部 256×320 为头相机，下部两块 128×160 分别为左右腕相机。XYZ 和 RGB 必须经过相同的像素对应预处理。原评测协议腕部 segmentation 条件为零，应复用该协议；本模块不从 RGB 重新计算这些输入。

NPZ 不包含字符串或 pickle。接口补充 `proprio_mode=last_command_drive_target` 和 `proprio_normalization=none`；proprio 仅进入动作侧。输出 `action_raw` 为 `[B,32,14]`，已反归一化的双臂关节/夹爪 drive targets，不是末端位姿。

## 权重与环境

当前核对的配置来源是全 3D 训练快照，检查权重是该 run 的 step 8000；这不是权重优选或新成功率结果。模型配置与路径分离，动作/状态归一化参数直接保留其原数值与摘要。

构造模型仍复用原 factory，因此加载完整策略权重前需要提供原 PixelDiT ImageNet 初始化文件。本阶段不替换加载协议、不静默跳过权重校验。未来可单独实现免重复初始化的加载器，再验证等价性。

训练与评测资产路径没有写入公开配置。CLI 不调用现有训练或 GPU 控制器，也不依赖原目录才能导入模型。

## 新增训练模块

数据准备与训练入口位于 `data_tools.py`、`train_cli.py`、`runtime.py`。`datasets/robotwin_momap` 区分当前 ModelInputs 与用于构造监督的私有几何。Dataset 输出原始三视角张量 `[B,3,C,240,320]`，模型适配器统一拼成384×320；不能把它与已经拼好的推理NPZ形状混淆。

完整训练流程、资产格式、预处理和恢复限制见 [training_guide_zh.md](training_guide_zh.md)。
