# 在 LIBERO 上训练 M-WAM

使用 LIBERO 的 Spatial、Object、Goal 和 Long 四组示范，联合训练动作和三维运动预测。每个样本包含主相机和腕部相机的观测、任务指令，以及当前 8 维机器人状态（末端位置、旋转向量和两个夹爪关节的位置）。

## 准备数据

在独立的 Python 3.10 环境中，从仓库根目录进入 `libero/`。四卡参考配置使用 PyTorch 2.9.1 和 CUDA 12.8。

```bash
cd libero
python -m pip install -e .
python prepare_data.py --index data/index.json --out data/ready
```

`data/index.json` 列出每条示范的几何数据、机器人状态、主相机实例分割和指令。准备脚本按任务划分示范，并计算训练集的归一化统计量。[数据格式与清单示例](DATA.md)

## 启动训练

将 PixelDiT 的初始权重保存为 `data/pixeldit_t2i_v1.pth`。[NVIDIA PixelDiT 权重](https://huggingface.co/nvidia/PixelDiT-1300M-1024px/tree/7c63b99a7a399918a1d6478b095698a65f664847)

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  --data data/ready --pretrained data/pixeldit_t2i_v1.pth --output runs/libero
```

默认每卡每步读取 8 个片段。每次抽取片段时重新分配分割颜色。模型预测 32 步七维动作，以及源帧中各点在第 8、16、24、32 步的三维位移。两项损失等权相加，各自独立采样噪声时间。

AdamW 先用 1,000 步将学习率升至 `1e-4`，随后按余弦曲线降至 `1e-6`。完成 100,000 次更新后，权重写入 `runs/libero/model.pt`。[训练配置](configs/libero_joint.yaml)
