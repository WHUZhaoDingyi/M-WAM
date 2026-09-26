# 验证记录

当前第二阶段结果见下方；第一阶段数据作为历史记录保留，不能代替当前所有改动的验证。

## 第一阶段（历史）

本次验证基于 `robotwin_all3d_imagenet50k_20260924` 全3D模型的 step 8000，不是此前鲁棒性评测使用的旧 hybrid step 190075。

| 检查 | 结果 | 证明范围 |
|---|---|---|
| 40个模型依赖文件与来源SHA256比对 | 全部一致 | 复制未改数学实现 |
| 模型配置比对 | 通过 | 除路径、归一化内联及显式joint14默认值，其余字段完全一致 |
| 动作/状态归一化 | 原数值与digest一致 | 去除了原集群统计文件路径依赖 |
| 针对性pytest | 10通过 | 全3D腕Z、AC键值选择、高噪声概率/权重、消融误用和GT拒绝、接口透传 |
| 完整checkpoint严格加载 | 通过 | CPU BF16，1,610,024,088个参数，原生模型/数学契约与state校验 |
| 完整5步推理 | 通过 | CPU，单份合成观测，输出[1,32,14]且数值有限 |
| 新接口与直接原生infer_joint对照 | 逐元素完全相同 | 同一已加载模型、相同观测与seed=20260923；零容差比较 |
| Python wheel构建 | 通过 | 无依赖下载构建 |
| wheel脱离源码目录导入与配置检查 | 通过 | 从/tmp运行，使用解包wheel |
| 交互命令语法 | 7组通过 | 外层/bin/sh与内层Bash；不是GPU执行证明 |

运行环境：Python 3.10，torch 2.6.0+cu124，NumPy 1.26.4，CUDA不可用。

## 尚未覆盖

- 真实观测与模拟器闭环成功率、GPU数值/速度、训练反向和优化器更新等价性。
- 数据集、训练入口、RoboTwin/LIBERO/真机适配的发布版迁移。
- 省略世界分支的动作部署优化；当前仍执行原生joint solver。

同模型包装接口对照不是独立训练复现。原计算源码逐字保留是本阶段不改变模型行为的另一个依据。

机器可读摘要：[validation_receipt.json](validation_receipt.json)。完整本地日志位于工作区 `outputs/mwam_modular_step1/`。

## 构建过程说明

首次wheel构建因默认pip缓存目录只读失败；使用 `--no-cache-dir --no-index --no-deps --no-build-isolation` 后成功，未修改环境或外部目录。

工作区没有Git元数据，无法产生普通git diff；已记录源文件哈希、配置差异与新增接口unified diff。未初始化Git仓库、提交或推送。


## 第二阶段：数据准备与训练代码迁入（2026-09-26）

| 检查 | 当前结果 | 边界 |
|---|---|---|
| 主方法与训练器测试 | 12通过 | 小型CPU网络调用真实AdamW/Accelerate/保存恢复；不是完整M-WAM反向 |
| raw-v1真实数据 | 2 episode、197合法窗口、2 batch通过 | 原生预测头分割＋深度噪声，不用GT分割替代 |
| 新旧DataLoader对照 | 首末两个窗口48个张量逐元素相同 | 对照全3D训练快照 |
| 现有文本缓存 | 真实prompt编码读取通过 | 没有执行新的T5前向 |
| 原分割脚本self-test | PASS | 不含GPU模型推理 |
| 分割缓存索引 | 2个已有真实cache索引通过 | 不代表新生成分割完成 |
| 主训练配置组装 | PASS | 没有启动完整训练 |
| 命令语法 | 16组通过 | `/bin/sh`外层与Bash内层 |
| 当前完整模型CPU推理复测 | 未完成，进程exit137 | 16GB容器内存限制；未产生新输出。没有停止其他进程 |

测试中发现并修复的两个迁移问题：保持旧缓存schema字符串不改名；移除原集群paths.json导入依赖。恢复测试还发现DataLoader重建会消耗模型CPU RNG，已用独立generator修复并纳入恢复契约。

CPU训练器测试暂时屏蔽Accelerate对**可选DeepSpeed**的探测，使用普通CPU后端。原因是当前sandbox中的DeepSpeed CPU探测依赖/proc，不能正常初始化。测试没有替换AdamW、真实训练循环或checkpoint保存恢复，不证明DeepSpeed已通过。

**仍待实机验收：**完整1.61B M-WAM GPU前向/反向、两步保存恢复、DDP/ZeRO-2、从RGB生成新分割、新T5编码。当前工程提供对应代码和命令，不能写成已完成GPU训练或论文成功率复现。

本轮源码独立打包，未启动/停止现有训练、评测或服务。证据目录：工作区 `outputs/mwam_training_package/`。最终构建和覆盖摘要见 `training_validation_receipt.json`。

双worker检查已在允许本地IPC的环境复测通过。wheel构建及脱离源码目录导入也通过，未导入旧fastwam包。首次构建误将目录当作包名，改为显式 `./M-WAM` 后成功；未下载依赖、未修改已安装环境。
