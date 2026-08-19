# Action-to-position-target mappings / 动作到关节位置目标的映射

## Overview / 概述

UFO interprets policy actions in a normalized action space. The environment can
convert them to joint-position targets with either of the mappings below.

UFO 在归一化动作空间中解释策略输出。环境可以使用下列两种映射之一，将策略动作转换为关节位置目标。

## Mapping modes / 映射模式

- `effort_kp` (default) preserves the existing behavior. Its per-joint target
  scale is derived from the configured action scale and, when action rescaling
  is enabled, the effort-limit-to-stiffness ratio.
- `soft_limit_bias` clips each policy action to `[-1, 1]` and maps it affinely
  onto that joint's configured soft position-limit interval.

- `effort_kp`（默认）保留原有行为。每个关节的位置目标缩放系数来自配置的 action scale；启用动作重缩放时，还会乘以该关节的力矩上限与刚度之比。
- `soft_limit_bias` 先将每个策略动作裁剪到 `[-1, 1]`，再通过仿射变换将其映射到对应关节配置的软位置限位区间。

The second mapping is useful for robots with strongly asymmetric joint ranges.
Unlike a zero-centered scale, its bias lets the normalized action cover both
ends of an asymmetric interval.

第二种映射适用于关节范围明显不对称的机器人。与以零为中心的缩放不同，它通过偏置项使归一化动作能够覆盖不对称区间的两个端点。

## Mapping formula / 映射公式

For a soft interval `[q_min, q_max]`, default position `q_default`, and MJLab
target scale `s`, the mapping computes:

对于软限位区间 `[q_min, q_max]`、默认关节位置 `q_default` 和 MJLab 位置目标缩放系数 `s`，映射计算如下：

```text
lower = (q_min - q_default) / s
upper = (q_max - q_default) / s
bias = (lower + upper) / 2
half_range = (upper - lower) / 2
mapped_action = bias + half_range * clamp(policy_action, -1, 1)
```

Consequently, policy outputs `-1` and `1` reconstruct `q_min` and `q_max`
after MJLab applies its default-position offset and target scale. The default
remains `effort_kp`, so existing launch commands and checkpoints retain their
current behavior.

因此，在 MJLab 应用默认关节位置偏置和位置目标缩放后，策略输出 `-1` 和 `1` 会分别还原为 `q_min` 和 `q_max`。默认模式仍为 `effort_kp`，所以已有启动命令和检查点的行为不会改变。

## Training usage / 训练用法

Select the alternative mapping explicitly when training:

训练时需要显式选择新的映射模式：

```bash
./run_train.sh \
  --agent fb \
  --data-manifest configs/data/example_mix.yaml \
  --rebuild-motion-cache \
  --gpu-ids single \
  --action-mapping soft_limit_bias \
  --work-dir runs/ufo_fb_soft_limit_bias
```

Models trained with different mappings should be evaluated with the same
mapping used during training.

使用不同映射训练的模型，在评估时必须继续使用各自训练时的映射。

## ONNX metadata / ONNX 元数据

Tracking ONNX export records the mapping name in the companion `.meta.json`
file. For `soft_limit_bias`, it also records the affine bias and range, action
input endpoints, MJLab target scales, default joint positions, and resolved
soft position limits.

Tracking ONNX 导出会在配套的 `.meta.json` 文件中记录映射名称。对于 `soft_limit_bias`，还会记录仿射偏置与范围、动作输入端点、MJLab 位置目标缩放系数、默认关节位置以及最终解析出的软位置限位。
