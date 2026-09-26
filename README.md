# M-WAM

主方法：**全视角 3D 运动表示 / AC 注意力 / 轨迹 action token / 高噪声训练采样**。

本仓库包含从 RoboTwin raw-v1 数据准备、窗口采样、DataLoader，到模型训练、保存、断点恢复和结束的代码；另提供严格权重加载及动作推理接口。训练依赖外部数据、PixelDiT 初始化权重、文本编码与预测分割资产。它不是包含所有权重和模拟器的免配置安装包。

先读 [训练流程与必需资产](docs/training_guide_zh.md)，执行命令见 [commands.md](docs/commands.md)。模型配置每个词条见 [config_guide_zh.md](docs/config_guide_zh.md)，实际验证范围见 [validation.md](docs/validation.md)。

```text
raw-v1 HDF5 + 每条示范的任务指令
    ├─ RGB → GroundingDINO + SAM2 → 预测头分割缓存
    ├─ 指令 → Wan T5 → 文本缓存
    ├─ 当前深度 + 相机标定 → 当前 XYZ
    └─ 未来几何 / 动作流 → 3D motion / action 监督
                         ↓
              合法窗口索引 → Dataset → DataLoader
                         ↓
          全 3D / AC / 轨迹 token / 高噪声训练
                         ↓
       AdamW + LR 调度 → 定期保存 → max_steps → 最终保存
                         ↑
            full-state checkpoint 恢复继续
```

未来几何与动作是训练标签；推理接口只接收当前观测。完整数据边界见 [implementation.md](docs/implementation.md)。

| 位置 | 内容 |
|---|---|
| `configs/robotwin_3d_ac.yaml` | 模型、归一化和推理配置 |
| `configs/train.yaml` | 优化器、预算、DataLoader、保存恢复设置 |
| `configs/data_raw_v1.yaml` | 原始数据与监督生成配置模板 |
| `src/mwam/data_tools.py` | 清单、文本编码、分割索引和真实 batch 检查 |
| `src/mwam/datasets/robotwin_momap/` | 原始读取、合法窗口、XYZ、运动目标、collator |
| `src/mwam/models/` | 方法计算，保留原 checkpoint 参数名 |
| `src/mwam/training/`、`trainer.py` | 训练、采样、日志、保存恢复 |
| `scripts/perception/` | 原生 RGB→预测分割预处理及依赖 |
| `src/mwam/policy.py` | 权重加载与动作预测 |
| `tests/` | 方法约束及真实训练器 CPU 保存恢复测试 |

只有一个 Python 包 `mwam`，不再提供 `src/fastwam`。内部部分函数、格式标签仍带 FastWAM/Wan 的历史名字，以保持权重和数据兼容；它们不是额外启动的 Fast-WAM 基线。通用训练器保留了未启用的历史 replay 扩展，当前公开主配置不启用这些扩展。

`*.egg-info` 是 pip 安装时生成的包元数据；`__pycache__` 是 Python 自动生成的字节码缓存。都不是算法、数据或权重，源码包排除它们，安装/运行后再次出现是正常的。

来源为全 3D 快照 `robotwin_all3d_imagenet50k_20260924`。历史 head-dXYZ/wrist-dUV 的 step190075 不兼容本全 3D 配置。当前 `infer_joint` 仍计算 W，虽然动作不读取未来 W tokens；尚未做省略 W 的部署优化。

本轮没有启动现有 GPU 任务，没有上传 GitHub。源码不包含数据、第三方权重或模拟器资产；保留原来源许可证。
