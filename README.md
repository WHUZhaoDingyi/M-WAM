# M-WAM

M-WAM 用机器人示范联合学习动作和三维运动。模型根据当前观测、机器人状态和任务指令预测接下来 32 步的动作。训练时还预测当前画面中各点在第 8、16、24、32 步的三维位移。

| 数据集 | 训练入口 | 数据准备与启动命令 |
| --- | --- | --- |
| RoboTwin 2.0 | [scripts/mwam.py](scripts/mwam.py) | [RoboTwin 训练指南](docs/training_guide_zh.md) |
| LIBERO | [libero/train.py](libero/train.py) | [LIBERO 训练指南](libero/README.md) |
