# robotwin_3d_ac.yaml 逐项中文说明

对应文件：[robotwin_3d_ac.yaml](../configs/robotwin_3d_ac.yaml)。本说明按当前文件的 **141 个键（包括分组键）** 逐一解释；同名键在不同分组下分别说明。

## 先回答：这是不是完整的方法？

**核心方法和完整离线训练链路已包含。** 原始数据读取、窗口采样、DataLoader、AdamW、分布式训练入口和断点恢复已经迁入；深度到XYZ与文本编码也已接入，RGB分割通过附带的原生离线预处理脚本生成。

模拟器/真机执行器与闭环成功率评测尚未迁入：它们不参与离线训练，属于训练后的部署/评测适配。

权重和数据需外部提供。当前这份YAML仍是**模型与推理配置**；学习率、batch size等位于 `configs/train.yaml`，数据配置位于 `configs/data_raw_v1.yaml`。完整流程见 [training_guide_zh.md](training_guide_zh.md)。

另外，AC是注意力依赖关系，不等于已经省掉W的计算：现有 `infer_joint` 仍计算世界分支，动作分支不读其未来tokens。省略W的部署优化尚未整理。

## 你问到的四个地方

1. **`pixeldit_pretrained_path: null`**：没有把本机绝对路径写死在共享配置里。运行时必须用 `--pixeldit-pretrained` 传入，入口会覆盖这个字段。不是“不用预训练”，也不是最终策略权重；最终M-WAM权重另由 `--checkpoint` 提供。当前加载路径确实仍需要这两个文件。
2. **`pixeldit_weight_source: auto`**：在所给文件内，先找 `ema_denoiser.` 参数，再找 `denoiser.`，最后尝试无前缀的纯参数字典。EMA是参数的指数滑动平均版本；这里没有联网自动找模型的行为。
3. **`fastwam_clip_frame@15hz`**：帧偏移以15Hz重采样序列为基准，所以偏移8对应8/15秒。它是时间语义标签，不是文件或自动控制相机频率的命令。
4. **`view_mode: robotwin_fastwam_mosaic_384x320_v1`**：冒号是YAML的“键: 值”分隔符，右侧只是预先约定的字符串。代码明确检查它并选择对应处理函数，不会读懂英文后自己执行。这里是**384×320**，不是380×320。

```text
view_mode 字符串
      ↓ 代码检查是否为支持的模式
MoMapBatchAdapter 选择对应拼接函数
      ↓
头部256×320 + 下方左腕128×160、右腕128×160
      ↓
得到384×320的拼接输入
```

如果已经通过当前最小接口传入拼好的mosaic，则不再做这次原始多视角拼接。

`_target_`与这些普通字符串不同：Hydra会将它作为Python模块/函数或类的入口路径导入。`schema_version`、`architecture_version`、`method_contract`中的长名字主要是版本/兼容性标识，不能靠修改名字来实现另一种算法。

## 版本标识

`architecture_version` 已修正为代码支持的 `momap_wam_cw_hybrid_allxyz_384x320_k4s8_g26p4_a30_v15`。配置检查现在直接构造原生 `KFlowContract`，拼写错误会在分配模型前报错。版本标识是精确匹配的程序字符串。

## 阅读约定

- `null`：YAML空值，对应Python的None。`none`：这里是普通字符串，表示某一种“未预先归一化”的协议；两者不同。
- `true/false`：布尔开关。`[]`：空列表，不等于字段不存在。
- C=当前条件，W=未来世界/运动，A=动作；Q/K/V=注意力中的query/key/value。
- XYZ=三维位置，dXYZ=相对源时刻的三维位移；UV=二维像素坐标/位移，不能把标识中的UV误读为这里全部输出都是2D。
- K=4个未来时刻；B=batch大小；32=动作时间步数；14=每步动作维度；1152=token隐藏宽度。这些数字不是同一维度。
- `v1/v2/...`：代码约定的格式版本，不是训练step；`schema`=格式，`contract`=必须一致的数据/模型约定，`artifact`=保存在文件里的统计或资产。
- 三项XYZ列表依次对应X/Y/Z；mean/std是数据统计值，不是待挑选的超参数。14项动作/状态统计按输入向量维度一一对应，双臂各6个关节加1个夹爪，维度顺序必须沿用原数据/执行协议。
- 数值公式：`normalized=(raw-mean)/std`，反变换为 `raw=normalized*std+mean`。设置输入的`source_normalization: none`并不会取消这一步。

## 全部词条

下面标题给出分组的完整路径，表格左列是该分组内的键。14维统计长列表不重复抄录所有小数，原值以YAML为准；每项对应的含义和单位均在说明中给出。

### `method`

主方法的简明身份说明。新入口会检查这组值；真正的网络和采样行为由下面 model 中的具体配置实现。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `name` | `M-WAM` | 方法名称 M-WAM，用于标识这套配置。 |
| `motion_representation` | `all_views_3d` | all_views_3d 表示头部和两只腕相机的运动目标都是三维位移 dX、dY、dZ；不是旧版头部3D、腕部2D。 |
| `attention` | `AC` | AC 表示动作分支 A 读取当前条件 C 和自身的动作 tokens，不读取未来世界 W 的 tokens。 |
| `action_tokens` | `trajectory` | trajectory 表示沿动作维度划分 token：每个 token 包含一个控制维度在整个时间窗口内的轨迹。 |
| `noise_sampling` | `high_noise_405505` | high_noise_405505 是本项目给高噪声训练分布起的名称：40%低于0.95，55%在[0.95,1)，5%恰为1；两个连续区间内保留 shifted-uniform 条件密度。 |

### `model`

模型构造参数：网络结构、输入/目标约定、归一化、噪声调度与损失。它不是整个训练任务的配置。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `_target_` | `mwam.models.pixeldit.factory.create_hybrid_k4_fastwam_pixeldit` | Hydra 要导入并调用的 Python 工厂函数。点号分隔模块和函数名，确实是代码入口路径。该工厂组装世界与动作分支；名字里的 hybrid 是沿用的历史命名。 |
| `proprio_dim` | `14` | 当前机器人状态向量的维度，14维；本协议为双臂关节与夹爪 drive target。它只进入动作侧，不加入共享世界条件。 |
| `pixeldit_pretrained_path` | `null` | PixelDiT 的 ImageNet 初始化权重文件路径。当前 null 是留给运行时填写的占位，不是关闭预训练；MWAMPolicy.from_checkpoint 总是用传入的 pixeldit_pretrained 参数覆盖它，CLI对应 --pixeldit-pretrained。直接用这个 null 调原 factory 会报错。它也不是最终 M-WAM checkpoint 的路径。 |
| `pixeldit_weight_source` | `auto` | auto 按固定优先级从上述权重文件内选择参数：先 ema_denoiser.，再 denoiser.，最后尝试纯 tensor state_dict。不是自动下载、自动选训练step或比较优劣；还可显式用 ema 或 denoiser。 |
| `action_initialization_seed` | `43` | 动作网络初始随机参数的种子43。之后可从世界主干复制兼容参数，最终完整策略checkpoint再覆盖模型状态。它与推理采样用的 inference.seed 不同。 |
| `action_normalization_artifact_path` | `null` | 动作统计JSON文件的可选路径。当前为null，因为下方 action_normalization_contract 已内联同一套统计；二者只能选一个来源。 |
| `proprio_normalization_artifact_path` | `null` | 状态统计JSON文件的可选路径。当前为null，因为下方 proprio_normalization_contract 已给出统计；不表示状态不做归一化。 |
| `timestep_coupling` | `independent` | independent 表示训练中世界分支和动作分支各自采样噪声时刻，不强制二者使用同一个噪声强度。 |
| `kv_source` | `action_condition` | action_condition 是实现 AC 的实际开关：动作注意力拼接C的K/V和动作自身K/V，不拼接W。不是读取名为action_condition的文件。 |
| `normalization_artifact_path` | `null` | 几何/运动统计文件的可选路径；当前null，因为 normalization_contract 已内联。与 model.pixeldit_pretrained_path 的必填占位含义不同。 |
| `action_tokenization` | `joint14` | joint14 是轨迹维token的原生代码选项：32步×14维动作转置为14个token，每个token覆盖32步。它是实际开关，需与上面的 trajectory 和 method_contract 一致。 |

#### `model.kflow_contract`

运动目标和相机输入布局的约定。K代表预测的未来时刻数，这里K=4；contract是数据/模型一致性约定，不是另一个网络。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `future_offsets` | `[8, 16, 24, 32]` | 四个预测时刻相对当前帧的偏移：[8,16,24,32]，单位由 future_offset_basis 指定。不是去噪步数，也不是四个物体。 |
| `future_horizons_seconds` | `[0.5333333333333333, 1.0666666666666667, 1.6, 2.1333333333333333]` | 上述偏移对应的物理时间，分别为8/15、16/15、24/15、32/15秒，即约0.533、1.067、1.600、2.133秒。 |
| `future_offset_basis` | `fastwam_clip_frame@15hz` | fastwam_clip_frame@15hz 是项目定义的时间基准标签：按15帧/秒采样后的clip帧编号计算偏移。@15hz只是字符串的一部分，不是Python语法，也不会自动把摄像头或模拟器设置为15Hz。 |
| `view_mode` | `robotwin_fastwam_mosaic_384x320_v1` | robotwin_fastwam_mosaic_384x320_v1 是代码支持的布局模式名称。KFlowContract先校验字符串，MoMapBatchAdapter再据此选拼接函数。384为高、320为宽：上256×320为头部，下方左右各128×160为双腕。不是路径、网址或自动识别英文；已有mosaic输入不会因这个词再次变换视角。 |
| `target_representation` | `all_views_dxyz_matched_uv_valid_v1` | all_views_dxyz_matched_uv_valid_v1 指全视角三维位移目标；matched_uv_valid表示沿用与2D UV对照匹配的有效监督点/掩码约定，不表示腕部预测UV。实际标签构造在原数据管线，发布第一阶段未包含该管线。 |
| `source_coordinate_frame` | `head_camera@source_t0` | 输入当前XYZ所使用的坐标系：head_camera@source_t0，即当前观测时刻头相机的参考坐标系。其他相机的XYZ要先转换到同一参考系。 |
| `source_normalization` | `none` | none 是字符串，表示数据送到模型入口时XYZ尚未做这套模型标准化，仍为原始米制值。模型内部仍会使用 normalization_contract；不是关闭所有归一化。 |
| `coordinate_frame` | `head_camera@source_t0` | 未来位移目标的参考坐标系，同样固定在源时刻t0的头相机坐标系；不是每个未来时刻都重新换参考系。 |
| `unit` | `all_views_canonical_meter` | all_views_canonical_meter 是所有视角统一参考系、统一米制单位的标识；模型不会仅凭这个字符串把毫米自动转换为米。 |
| `channel_order` | `k_major_xyz` | k_major_xyz 表示按未来时刻优先排列12个通道：[dx8,dy8,dz8, dx16,dy16,dz16, dx24,dy24,dz24, dx32,dy32,dz32]。 |
| `camera_order` | `["head_camera", "left_camera", "right_camera"]` | 三个视角的固定顺序：head_camera为头部，left_camera为左腕，right_camera为右腕。列表顺序决定拼接位置。 |
| `architecture_version` | `momap_wam_cw_hybrid_allxyz_384x320_k4s8_g26p4_a30_v15` | 这套张量布局/模型结构的版本标识，代码要求与支持的常量完全匹配；不会解析名字自动生成结构。当前值与代码支持的版本一致。 |

#### `model.normalization_contract`

当前XYZ和未来3D位移的固定归一化参数，数值直接写在本配置内。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `statistics_id` | `robotwin_allxyz_matched2d_stratified4096_20260924` | 该套几何/运动统计的来源标识。名字中的matched2d说明统计与2D对照的有效点口径相关，不把本模型改成2D；它不是要读取的路径。 |
| `future_offsets` | `[8, 16, 24, 32]` | 统计适用的四个未来偏移，必须与运动目标[8,16,24,32]一致。 |
| `source_mean_xyz_meters` | `[0.010645708305307268, 0.04895817558894431, -0.7550955002897234]` | 当前源点XYZ的固定均值，列表三项分别对应X、Y、Z，单位米。 |
| `source_std_xyz_meters` | `[0.2734047798851918, 0.17070957653298427, 0.15400744531857438]` | 当前源点XYZ的固定标准差，三项对应X、Y、Z，单位米；用于逐坐标标准化。 |
| `head_scale_channels_meters` | `[]` | 历史逐通道缩放方案的兼容字段。当前使用pooled z-score，因此必须为空列表[]；不是忘填参数。 |
| `wrist_scale_channels_meters` | `[]` | 历史腕部逐通道缩放参数。当前模式不用，必须为空列表[]。 |
| `head_flow_mean_xyz_meters` | `[0.0004454643716099158, -0.0005298441303830609, -0.0006565998981648588]` | 头部像素对应的3D位移均值，三项为dX/dY/dZ，四个未来偏移共用这组统计。 |
| `head_flow_std_xyz_meters` | `[0.020128777503447156, 0.014664635871552903, 0.019597151696224013]` | 头部3D位移标准差，三项为dX/dY/dZ，四个偏移共用。 |
| `wrist_flow_mean_xyz_meters` | `[-0.0011799793721345363, 0.00043401040904521656, 0.0026781860166423584]` | 双腕区域3D位移的合并均值，三项为dX/dY/dZ，左右腕与四个偏移共用。 |
| `wrist_flow_std_xyz_meters` | `[0.034183018564904515, 0.02497882126418233, 0.027535178196940777]` | 双腕区域3D位移的合并标准差，三项为dX/dY/dZ；包含真实Z分量，不是旧版2D的补零通道。 |
| `head_height` | `256` | 拼接图中头部区域高度256像素，即前256行。 |
| `full_height` | `384` | 完整拼接图高度384像素，包含头部256行和腕部128行。 |
| `full_width` | `320` | 完整拼接图宽度320像素；下方左右腕各占160列。 |
| `reference_frame` | `head_camera@source_t0` | 归一化统计所在的参考坐标系，必须与输入/目标的head_camera@source_t0一致。 |
| `source_unit` | `all_views_canonical_meter` | 统计所针对的原始数据单位，all_views_canonical_meter表示所有视角在统一坐标系中以米表示。 |
| `channel_order` | `k_major_xyz` | 统计应用到张量时采用k_major_xyz，与运动目标12通道顺序一致。 |
| `schema_version` | `robotwin_momap_k4_all_xyz_normalization_v2` | 归一化参数的数据结构版本标识，用来选择并校验可兼容的统计格式。 |
| `mode` | `fixed_source_xyz_affine_head_dxyz_pooled_zscore_wrist_dxyz_pooled_meter_zscore_v1` | 固定源XYZ仿射标准化，加上头部/腕部分开的3D位移pooled z-score。pooled表示跨四个未来时刻共用统计，不表示对每个样本现场重新估计均值方差。 |
| `digest` | `sha256:0caf16f3daca38fa0c2e26936317362df133c14f14943a682fc13f8c192aa6f3` | 这些几何归一化参数的SHA256摘要，作用类似内容指纹。构造时会重算并检查匹配；它不是文件路径或模型权重。 |

#### `model.action_normalization_contract`

14维动作目标的固定均值/标准差。用于训练标准化及输出动作反标准化。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `digest` | `sha256:fda8d7aeae30f380ca6201e62c59b7192ab39ec06fdfae5f22064b36ed8c6bfc` | 未来动作目标统计的内容摘要。当前factory读取内联feature统计时会去除外层digest并按实际参数重算，随后完整checkpoint按数值契约校验；不靠这段字符串指定文件。 |
| `feature_dim` | `14` | 向量维数14的说明/校验信息，源实现从mean/std长度得到对应维度；不能只改此处数字来改变机器人自由度。 |
| `feature_kind` | `action` | action 表示这组统计用于未来动作目标，使动作与当前状态统计不会混用。 |
| `mean` | 列表，共14项；精确数值见YAML | 未来动作目标14个分量各自的固定均值；第i个数对应第i个控制维度，不是14个未来时刻。 |
| `mode` | `fixed_feature_zscore` | fixed_feature_zscore 表示固定逐维z-score：normalized[i]=(raw[i]-mean[i])/std[i]。 |
| `schema_version` | `fastwam_fixed_feature_zscore_v1` | 这套固定feature归一化的数据结构版本，模型加载时校验兼容性。 |
| `semantic_kind` | `low_level_joint_drive_target` | low_level_joint_drive_target 表示低层关节/夹爪控制目标，不是末端位置、物体轨迹或高层技能标签。 |
| `source_normalization` | `none` | none表示传入的未来动作目标未预先做模型z-score，模型内部再按本统计归一化。 |
| `statistics_id` | `robotwin-momap-formal-train-v1-action14` | 未来动作目标统计的来源名称，用于追溯和区分动作/状态统计；不是目录。 |
| `std` | 列表，共14项；精确数值见YAML | 未来动作目标14个分量各自的固定标准差，与mean及输入向量采用相同维度顺序；不是评测结果的Mean±STD。 |

#### `model.proprio_normalization_contract`

14维当前状态的固定均值/标准差，与未来动作目标的统计分开。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `digest` | `sha256:3534ea2a6d243bdc8b1c0826c5a987c57821d602b13cbfda4bfbfe2671388873` | 当前机器人状态统计的内容摘要。当前factory读取内联feature统计时会去除外层digest并按实际参数重算，随后完整checkpoint按数值契约校验；不靠这段字符串指定文件。 |
| `feature_dim` | `14` | 向量维数14的说明/校验信息，源实现从mean/std长度得到对应维度；不能只改此处数字来改变机器人自由度。 |
| `feature_kind` | `proprio` | proprio 表示这组统计用于当前机器人状态，使动作与当前状态统计不会混用。 |
| `mean` | 列表，共14项；精确数值见YAML | 当前机器人状态14个分量各自的固定均值；第i个数对应第i个控制维度，不是14个未来时刻。 |
| `mode` | `fixed_feature_zscore` | fixed_feature_zscore 表示固定逐维z-score：normalized[i]=(raw[i]-mean[i])/std[i]。 |
| `schema_version` | `fastwam_fixed_feature_zscore_v1` | 这套固定feature归一化的数据结构版本，模型加载时校验兼容性。 |
| `semantic_kind` | `last_command_drive_target` | last_command_drive_target 表示最近一次发送的关节/夹爪控制目标，不等同于传感器测到的实际关节位置；输入时必须保持该语义。 |
| `source_normalization` | `none` | none表示传入的当前机器人状态未预先做模型z-score，模型内部再按本统计归一化。 |
| `statistics_id` | `robotwin-momap-formal-train-v1-proprio14` | 当前机器人状态统计的来源名称，用于追溯和区分动作/状态统计；不是目录。 |
| `std` | 列表，共14项；精确数值见YAML | 当前机器人状态14个分量各自的固定标准差，与mean及输入向量采用相同维度顺序；不是评测结果的Mean±STD。 |

#### `model.pixeldit_config`

世界分支 PixelDiT 的网络结构与执行选项。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `num_groups` | `16` | 世界注意力的分组/头数16。hidden_size=1152，因此每头宽度72；不是16张图或16台GPU。 |
| `hidden_size` | `1152` | 全局patch token隐藏维度1152。 |
| `pixel_hidden_size` | `16` | 局部pixel token隐藏维度16；与上面的全局token宽度不是一个层级。 |
| `patch_depth` | `26` | 全局patch层数量26，不是图像深度或深度相机量程。 |
| `pixel_depth` | `4` | 局部PiT/pixel层数量4，不是第四个相机。 |
| `patch_size` | `16` | 每个patch为16×16像素。384×320因此有24×20=480个空间patch。 |
| `num_classes` | `1000` | 保留ImageNet预训练的1000类embedding兼容结构。默认使用额外的空类别索引1000，不是1000个机器人任务，也不是语言类别数。 |
| `use_pixel_abs_pos` | `true` | true表示局部pixel tokens加入二维绝对位置编码，帮助辨别像素位置；不表示使用绝对机器人关节角。 |
| `pit_adaln_post_modulation` | `false` | false使用原PiT的调制方式：在归一化后的注意力/MLP输入做shift/scale，残差输出乘gate。true是另一种输出后调制结构，会改变参数形状；不是无关紧要的性能开关。 |
| `momap_output_hidden_size` | `128` | 运动输出头融合全局和局部特征时的中间隐藏宽度128。最终输出仍是12个运动通道，不是128个未来帧。 |
| `use_gradient_checkpointing` | `true` | true表示训练反向时重算部分前向以省显存，代价是额外计算；不指保存checkpoint文件，普通无梯度推理不因此保存/重算训练激活。 |
| `validate_finite` | `false` | false关闭部分网络层内额外NaN/Inf检查以减少开销；入口、输出和checkpoint仍有其他检查，不表示允许任意无效数值。 |

#### `model.aligned_action_dit_config`

动作分支 ActionDiT 的结构；aligned指隐藏宽度和注意力头等与世界分支对齐，便于直接使用其K/V。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `action_horizon` | `32` | 每个动作chunk包含32个时间步。它与世界预测的4个稀疏时刻、推理的5个去噪步是三种不同数量。 |
| `action_dim` | `14` | 每一步14维控制值；当前双臂各6个关节加1个夹爪。轨迹维分词后对应14个action tokens。 |
| `hidden_size` | `1152` | 动作token隐藏宽度1152，与世界K/V宽度对齐。 |
| `num_heads` | `16` | 动作注意力头数16，每头宽度72。 |
| `num_layers` | `30` | 动作Transformer层数30，与世界26个global层和4个pixel层对应交互。 |
| `text_dim` | `4096` | 输入语言embedding每个token的特征宽度4096；不是4096个单词或文本最大长度。 |
| `mlp_ratio` | `4.0` | 动作块前馈层的扩展比例4.0，内部前馈宽度按hidden_size×4设置。 |
| `use_gradient_checkpointing` | `true` | 动作分支训练时启用激活重算以节省显存；不是文件存盘策略。 |
| `validate_finite` | `false` | 动作网络层内的额外有限数检查开关。false不取消整个模型的全部数值检查。 |

#### `model.train_world_scheduler`

世界分支训练时的加噪、训练时刻采样与时间权重规则。scheduler在这里不是学习率调度器。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `_target_` | `mwam.models.wan22.schedulers.scheduler_continuous.WanHighNoise405505UniformWeightFlowMatchScheduler` | Hydra构造高噪声405505连续Flow-Matching scheduler的Python类路径。它实现该训练分布及恒为1的时间权重。 |
| `num_train_timesteps` | `1000` | 连续噪声时间坐标的标度1000：内部t=1000×sigma。不是训练总步数，也不要求推理跑1000次；实际推理步数在inference.num_inference_steps。 |
| `shift` | `5.0` | 噪声时间轴变换参数5.0，使用sigma=5*u/(1+4*u)。训练连续采样与推理时间网格都按各自scheduler的此参数生成，不是相机位移。 |
| `eps` | `1e-10` | 数值稳定小量1e-10。继承的基础scheduler用于权重归一化分母；本高噪声类的训练权重固定为1，因此它不是调节高噪声概率的参数。 |

#### `model.infer_world_scheduler`

世界分支推理时的噪声时间网格和Euler更新规则。没有 _target_ 时，原factory显式使用 WanContinuousFlowMatchScheduler。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `num_train_timesteps` | `1000` | 连续噪声时间坐标的标度1000：内部t=1000×sigma。不是训练总步数，也不要求推理跑1000次；实际推理步数在inference.num_inference_steps。 |
| `shift` | `5.0` | 噪声时间轴变换参数5.0，使用sigma=5*u/(1+4*u)。训练连续采样与推理时间网格都按各自scheduler的此参数生成，不是相机位移。 |
| `eps` | `1e-10` | 沿用基础scheduler的数值稳定小量1e-10；在本实现推理时间网格和Euler更新中不直接使用它，不是推理精度阈值。 |

#### `model.train_action_scheduler`

动作分支训练时的加噪、时刻采样与权重规则。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `_target_` | `mwam.models.wan22.schedulers.scheduler_continuous.WanHighNoise405505UniformWeightFlowMatchScheduler` | Hydra构造高噪声405505连续Flow-Matching scheduler的Python类路径。它实现该训练分布及恒为1的时间权重。 |
| `num_train_timesteps` | `1000` | 连续噪声时间坐标的标度1000：内部t=1000×sigma。不是训练总步数，也不要求推理跑1000次；实际推理步数在inference.num_inference_steps。 |
| `shift` | `5.0` | 噪声时间轴变换参数5.0，使用sigma=5*u/(1+4*u)。训练连续采样与推理时间网格都按各自scheduler的此参数生成，不是相机位移。 |
| `eps` | `1e-10` | 数值稳定小量1e-10。继承的基础scheduler用于权重归一化分母；本高噪声类的训练权重固定为1，因此它不是调节高噪声概率的参数。 |

#### `model.infer_action_scheduler`

动作分支推理时的噪声时间网格和Euler更新规则；原factory同样显式构造基础连续Flow-Matching scheduler。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `num_train_timesteps` | `1000` | 连续噪声时间坐标的标度1000：内部t=1000×sigma。不是训练总步数，也不要求推理跑1000次；实际推理步数在inference.num_inference_steps。 |
| `shift` | `5.0` | 噪声时间轴变换参数5.0，使用sigma=5*u/(1+4*u)。训练连续采样与推理时间网格都按各自scheduler的此参数生成，不是相机位移。 |
| `eps` | `1e-10` | 沿用基础scheduler的数值稳定小量1e-10；在本实现推理时间网格和Euler更新中不直接使用它，不是推理精度阈值。 |

#### `model.loss`

两个训练目标的组合权重。总损失为 lambda_momap * motion_loss + lambda_action * action_loss。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `lambda_momap` | `1.0` | 运动/世界分支损失系数1.0；不直接表示世界分支成功率的重要性。 |
| `lambda_action` | `1.0` | 动作分支损失系数1.0；两项系数相同，不意味着两项实际损失数值必然相同。 |

#### `model.method_contract`

对具体模型行为的完整版本描述，保存到checkpoint并参与一致性校验。下面多数长字符串是固定标识，不能靠改英文就改变算法。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `schema_version` | `fastwam_momap_k4_allxyz_method_contract_ac_v1` | 整套方法约定的版本名称：这里为全视角3D、AC版本。必须与代码支持的契约精确一致。 |
| `source_conditioning` | `source_rgb3_pred_instance_seg3_noisy_canonical_xyz3_head_only_seg_v1` | 声明源条件由RGB三通道、分割条件三通道、XYZ三通道组成；noisy XYZ/pred segmentation描述训练数据协议，head_only_seg表示分割条件只在头部提供。这个标识本身不执行分割预测或深度加噪，推理入口也不会偷偷修改图像。 |
| `world_topology` | `shared_weight_c_only_and_w_reads_cw_g26_p4_local16_k4_v1` | 声明C/W共享世界网络参数，C只读C、W读C+W，结构为26个global层、4个pixel层，局部特征宽度16，4个未来时刻。 |
| `world_state` | `all_dxyz12_head_wrist_carrier12_v1` | 声明头部和双腕都使用12通道dXYZ载体：4时刻×3坐标。头/腕在空间位置上分块，而不是混成2D。 |
| `world_attention` | `c_queries_c_only_w_queries_concat_cw_v1` | 世界分支注意力规则：C的query只读C；W的query读取拼接后的C和W。 |
| `world_timestep` | `c_clean_fraction_one_w_one_minus_sigma_v1` | C的clean-fraction时刻固定为1（不对当前条件做扩散加噪）；W使用1-sigma_world。训练数据本身的感知增强是另一件事。 |
| `world_position` | `shared_mosaic_spatial_coordinates_no_cw_type_embed_v1` | C/W使用同一mosaic空间坐标的位置编码，不额外加专门区分C与W的type embedding。 |
| `world_loss` | `normalized_velocity_half_head_xyz_half_mean_nonempty_wrist_xyz_v1` | 在有效监督元素上计算标准化速度误差：每个时刻头部占1/2，非空左右腕平均后占1/2；该时刻腕部都无有效点时只用头部。再在有效未来时刻及样本上聚合。 |
| `world_normalization` | `source_xyz_affine_head_dxyz_pooled_zscore_wrist_dxyz_pooled_meter_zscore_v1` | 声明当前源XYZ使用固定仿射标准化；头部/腕部dXYZ分别使用跨未来时刻合并的米制均值/标准差，与normalization_contract一致。 |
| `action_tokenization` | `transpose_b32x14_to_b14x32_then_linear32_to1152_v1` | 声明[B,32,14]先转置为[B,14,32]，再通过Linear(32,1152)得到14个轨迹tokens。不是把时间与动作维度随意展平。 |
| `action_position` | `fixed_sincos_actuator_index_0_13_no_temporal_rope_v1` | 按控制维度索引0到13使用固定sin/cos位置编码；动作token不使用时间轴RoPE。 |
| `kv_adaptation` | `none_equal_hidden1152_heads16_head_dim72_v1` | 世界与动作均为hidden1152、16头、每头72维，因此直接使用原生K/V，不加宽度适配投影。 |
| `mixed_attention` | `action_no_rope_q_queries_concat_selected_world_native_then_action_self_kv_direct_no_count_bias_v1` | 动作query不使用RoPE，每层一次注意力读取选定世界K/V（AC只选C）和动作自身K/V；直接拼接，不按token数量加入历史count bias。 |
| `train_timestep_coupling` | `independent` | 训练两分支噪声时刻独立，与model.timestep_coupling必须一致。 |
| `inference_timestep_coupling` | `lockstep_equal_step_count_distinct_schedulers_v1` | 原生联合推理使用两个各自的scheduler，以相同的步数同步迭代；这不把训练时的独立采样改成同一时刻。 |
| `world_model_timestep_coordinate` | `pixeldit_clean_fraction_from_one_minus_wan_noise_fraction_v1` | 世界网络接收clean fraction=1-sigma_world，1代表干净端。 |
| `action_model_timestep_coordinate` | `wan_noise_fraction_0_clean_1_noise_v1` | 动作网络接收noise fraction=sigma_action，0代表干净、1代表纯噪声；与世界网络的时间坐标方向相反。 |
| `topology` | `hybrid_cw_g26_p4_action30_joint14_action_condition_v1` | 整机模型结构的固定版本名，声明C/W世界26+4层、动作30层、14轨迹tokens和AC连接；不是根据这段文字动态创建网络。 |
| `bridge` | `all_30_layers_concat_c_then_action_self_kv_v1` | 全部30个动作层都读取拼接的C K/V和动作自身K/V，不读取W K/V。 |
| `action_world_visibility` | `c_only_480_world_keys_v1` | c_only_480_world_keys_v1声明动作仅能看到C的480个空间keys：384/16 × 320/16=480。动作自身14个keys另算，语言cross-attention也不包含在这480个里。 |

### `inference`

最小推理入口使用的默认去噪步数和随机种子。CLI 的 --steps / --seed 可覆盖。

| 词条 | 当前值 | 含义与作用 |
|---|---|---|
| `num_inference_steps` | `5` | 默认5次去噪/Euler更新后得到一个动作chunk。不是只输出5个动作，也不是5个物理控制步；完整chunk仍为32×14。 |
| `seed` | `20260923` | 推理随机种子20260923，用于生成初始噪声。原生实现世界噪声用seed、动作噪声用seed+1；它不是环境初态seed，也不是训练初始化seed43。 |

## 对照的实现位置

- 配置填入与权重加载：[policy.py](../src/mwam/policy.py)。
- 网络工厂及参数校验：[factory.py](../src/mwam/models/pixeldit/factory.py)。
- auto选权重前缀：[model.py](../src/mwam/models/pixeldit/model.py) 中 `_select_denoiser_state`。
- 布局、偏移与版本：[kflow.py](../src/mwam/models/pixeldit/kflow.py)。
- 几何标准化：[hybrid_normalization.py](../src/mwam/models/pixeldit/hybrid_normalization.py)。
- C/W世界分支：[hybrid_cw_pixeldit.py](../src/mwam/models/pixeldit/hybrid_cw_pixeldit.py)。
- AC连接：[hybrid_cw_action_mot.py](../src/mwam/models/pixeldit/hybrid_cw_action_mot.py)。
- 轨迹token：[pixeldit_aligned_action.py](../src/mwam/models/pixeldit/pixeldit_aligned_action.py)。
- 世界损失：[hybrid_fastwam_pixeldit.py](../src/mwam/models/pixeldit/hybrid_fastwam_pixeldit.py)。
- 训练噪声和推理时间网格：[scheduler_continuous.py](../src/mwam/models/wan22/schedulers/scheduler_continuous.py)。

本轮只新增说明文档和README入口，不修改配置、模型或运行任务。
