<h1 align="center">
  UFO：面向人形机器人控制的无监督强化学习框架
</h1>

<p align="center">
  <a href="https://roboparty.github.io/UFO/"><img alt="Website" src="https://img.shields.io/badge/Website-roboparty.github.io%2FUFO-2563eb?style=for-the-badge" /></a>
  <a href="https://youtu.be/uJPcLdn9sNA"><img alt="Video" src="https://img.shields.io/badge/Video-Demo-7c3aed?style=for-the-badge" /></a>
  <a href="https://roboparty.github.io/UFO/assets/UFO.pdf"><img alt="PDF" src="https://img.shields.io/badge/PDF-Available-0f766e?style=for-the-badge" /></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="./assets/UFO.png" alt="UFO" height="72" style="vertical-align: middle;" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/rplab_logo.png" alt="ROBO PARTY LAB Logo" height="72" style="vertical-align: middle;" />
</p>

## UFO 是什么？

UFO 是一个开源的无监督强化学习人形机器人控制框架。`main` 分支主要用于 MJLab 训练、RobotState 数据导入、tracking/goal/reward inference，以及 ONNX 导出。`deploy` 分支用于 Unitree G1 实机部署和遥操作运行时。

当前最完整、测试最充分的路线是 Unitree G1。新机器人适配已经有实验性接口，但需要用户准备目标机器人的 MuJoCo XML、可选 URDF，以及已经 retarget 到该机器人的 RobotState motion data。UFO 不会自动把人类动作或其他机器人的动作 retarget 到新机器人；不同机器人之间也不能直接复用同一个 checkpoint。

## 当前支持范围

| 功能 | 状态 |
| --- | --- |
| G1 训练 | 支持，测试最充分 |
| RobotState CSV / NPZ / `ufo_pkl` | 支持 |
| 多数据源 manifest | 支持 |
| Tracking inference | Robot-config aware |
| Goal inference | 支持 robot config；非 G1 需要机器人专属 goal JSON |
| Reward inference | G1 支持完整默认任务；非 G1 当前主要支持 root/locomotion 任务 |
| 实机部署 / 遥操作 | 使用 [`deploy` 分支](https://github.com/Roboparty/UFO/tree/deploy) |
| 自动 motion retargeting | 不支持 |
| 跨机器人复用同一个 checkpoint | 不支持 |

> [!NOTE]
> `main` 分支：训练、数据导入、推理、ONNX 导出。
> `deploy` 分支：G1 实机部署和遥操作运行时。

## 路线 A：Unitree G1 快速开始

### 1. 安装环境

```bash
git clone https://github.com/Roboparty/UFO.git
cd UFO
```

安装 [`uv`](https://docs.astral.sh/uv/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

或者：

```bash
python -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```

安装项目环境：

```bash
uv sync
```

如需使用 W&B：

```bash
uv run wandb login
# 或者: export WANDB_API_KEY=your_wandb_api_key
```

### 2. 下载 G1 LaFAN 数据

大数据不放在 Git 仓库中。使用下面的命令下载默认的 G1 LaFAN 数据：

```bash
bash scripts/download_data.sh g1_lafan
ls -lh humanoidverse/data/lafan_29dof_10s-clipped.pkl
```

### 3. Smoke test

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_g1
```

### 4. FB 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_fb_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 100 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_fb_g1
```

### 5. TeCH 训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent tech \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_tech_g1 \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --update-z-every-step 10 \
  --buffer-size 5120000 \
  --use-wandb \
  --wandb-run-name ufo_tech_g1
```

TeCH 在早期 UFO 版本中曾经叫 TLDR。`--agent tldr` 仍然保留为 `--agent tech` 的兼容 alias，但已经不推荐继续使用。

### 6. Tracking inference

推理时建议使用 full motion sequences，不要使用裁剪后的 training clips：

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run python -m humanoidverse.tracking_inference \
  --model-folder runs/ufo_fb_g1 \
  --data-path /path/to/full_motions.pkl \
  --device cuda:0 \
  --headless \
  --save-mp4 \
  --motion-list 0
```

输出会写到 `<model-folder>/tracking_inference/`。

### 7. ONNX 导出说明

在 tracking inference 命令中加入 `--export-onnx true` 可以导出 robot-config-aware ONNX policy 和 metadata JSON。导出的 ONNX 和当前 checkpoint 的机器人、动作维度、观测维度绑定，不能直接用于其他机器人。

## 路线 B：适配新机器人

这条路线是 experimental。你需要先准备：

1. 目标机器人的 MuJoCo XML；
2. 可选的匹配 URDF；
3. 已经适配或 retarget 到目标机器人的 RobotState 数据。

UFO 不负责自动把人类动作或其他机器人动作 retarget 到新机器人。用户需要先通过 `hhtools`、GMR 或自定义 retargeting pipeline 得到目标机器人的 RobotState 数据，再导入 UFO。

### 1. 生成 robot config 草稿

```bash
uv run python -m humanoidverse.tools.robot_inspect \
  --xml /path/to/robot.xml \
  --urdf /path/to/robot.urdf \
  --name my_robot \
  --out configs/robots/my_robot.yaml \
  --hydra-out humanoidverse/config/robot/my_robot/my_robot_auto.yaml
```

如果没有 URDF，可以省略 `--urdf`。URDF 只是辅助信息；MuJoCo XML 仍然是 qpos/qvel、action layout 和 actuator order 的 source of truth。

### 2. 人工检查 robot config

自动生成的配置只是草稿。大规模训练前必须人工检查 base body、control-joint order、feet、hands、key bodies、initial state、PD gains、actuator limits、contact bodies，以及和 reward/termination 相关的语义。

### 3. 构建 RobotState data manifest

```bash
uv run python -m humanoidverse.tools.data_build \
  --robot configs/robots/my_robot.yaml \
  --source "/path/to/motions/*.csv" \
  --format robot_state_csv \
  --name my_motion \
  --fps 50 \
  --clip-seconds 10 \
  --out configs/data/my_motion_auto_build.yaml \
  --rebuild-cache
```

无表头 CSV 支持两种格式：`root_pos` xyz、`root_quat` xyzw、随后是 XML/control-joint order 的 DOF position；也可以在最前面增加可选的 `time` 列。

如只想检查 CSV schema，可以先运行 `humanoidverse.tools.data_inspect`。

### 4. Smoke training

```bash
./run_train.sh \
  --agent fb \
  --robot-config configs/robots/my_robot.yaml \
  --data-manifest configs/data/my_motion_auto_build.yaml \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_my_robot
```

## 常见注意事项

- G1 是当前最完整、测试最充分的路径。
- 新机器人适配是 experimental，通常还需要调 controller、reward、contact 和 termination 语义。
- `main` 分支用于 training、data import、inference、ONNX export。
- `deploy` 分支当前主要面向 G1 实机部署和遥操作。
- 非 G1 的 goal inference 需要机器人专属 goal JSON。
- 非 G1 的 reward inference 当前主要支持 root/locomotion 任务，除非额外补充机器人语义。
- TeCH 曾经叫 TLDR，`--agent tldr` 仍是兼容 alias。
- 不同机器人之间不能直接复用同一个 checkpoint。

## 多数据源技能注入

UFO 支持基于 manifest 的多数据源混合。每个数据源之间的采样比例保持固定，prioritized sampling 在每个数据源内部进行。这适合在保持基础动作分布的同时注入少量稀有高敏捷技能，例如 cartwheel。可以参考 `configs/data/example_mix.yaml`。

## 文档链接

- [Import Wizard](docs/import_wizard.md)：RobotState schema、数据检查和数据构建。
- [Robot-Config Training](docs/robot_config_training.md)：实验性的 robot-aware training 初始化说明。
- [Training and Inference](docs/TRAIN_INFERENCE.md)：更多训练和推理命令。
- [Deploy branch](https://github.com/Roboparty/UFO/tree/deploy)：G1 实机部署和遥操作运行时。

## 引用 / 许可证

Citation coming soon.

License: see [LICENSE](LICENSE).
