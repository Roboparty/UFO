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

## Concrete G1 example / G1 具体例子

The repository's G1 configuration provides a concrete asymmetric case. For
`left_hip_roll_joint`, it specifies:

仓库中的 G1 配置提供了一个具体的不对称关节案例。对于 `left_hip_roll_joint`，配置为：

```text
hard limits        = [-0.52360, 2.96710] rad = [-30.00°, 170.00°]
soft-limit factor  = 0.95
soft limits        = [-0.43633, 2.87983] rad = [-25.00°, 165.00°]
default position   = 0 rad
effort limit       = 139 N·m
Kp                 = 99.09843 N·m/rad
base action scale  = 0.25
action normalize   = 5.0
```

With the existing mapping, the MJLab target scale is
`s = 0.25 * 139 / 99.09843 = 0.35066`. A normalized policy action
`a in [-1, 1]` is first multiplied by `5`, producing this target interval:

使用现有映射时，MJLab 的位置目标缩放系数为
`s = 0.25 * 139 / 99.09843 = 0.35066`。归一化策略动作
`a in [-1, 1]` 会先乘以 `5`，因此得到的位置目标区间为：

```text
q_target = q_default + s * (5 * a)
         = [-1.75331, 1.75331] rad
         = [-100.46°, 100.46°]
```

The negative side requests positions far beyond the joint's `-25°` soft
limit, while the positive side cannot reach the `165°` soft upper limit; it is
short by `1.12653 rad` (`64.55°`). The mirrored right-hip-roll joint has the
same problem in the opposite direction.

负方向的位置目标远远超过关节 `-25°` 的软下限，而正方向最多只能到 `100.46°`，距离 `165°` 的软上限还差 `1.12653 rad`（`64.55°`）。镜像的右髋 roll 关节则在相反方向存在同样的问题。

For the same joint, `soft_limit_bias` computes the action-space endpoints and
affine parameters as follows:

对于同一个关节，`soft_limit_bias` 计算出的动作空间端点和仿射参数为：

```text
lower      = -1.24431
upper      =  8.21257
bias       =  3.48413
half_range =  4.72844
```

Therefore policy outputs `-1`, `0`, and `1` produce position targets `-25°`,
`70°`, and `165°`, respectively. This proves the coverage property of the
mapping: both soft-limit endpoints are reachable and no normalized-action
range is allocated to targets outside that interval. It is a geometric
correctness argument, not by itself a claim that every training task must
improve.

因此，策略输出 `-1`、`0` 和 `1` 时，位置目标分别为 `-25°`、`70°` 和 `165°`。这证明了该映射的空间覆盖性质：两个软限位端点均可达，且不会把归一化动作范围浪费在软限位区间之外。这是对映射几何正确性的证明，但它本身不等价于“所有训练任务都必然获得提升”。

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
