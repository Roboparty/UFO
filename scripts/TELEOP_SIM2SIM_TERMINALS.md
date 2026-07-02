# Teleop Sim2Sim terminals

在本地工作站开 4 个终端，按顺序启动。

```bash
cd "$UFO_ROOT"
conda activate bfm0real
```

终端 A：仿真
```bash
python -m sim_env.base_sim \
  --robot_config=./config/robot/g1.yaml \
  --scene_config=./config/scene/g1_29dof.yaml
```

终端 B：PICO teleop 服务
```bash
scripts/teleop/teleop_pose_50hz.sh
```

终端 C：realtime z（FK + backward_map，跑在 4090 GPU）
```bash
python scripts/realtime/realtime_z_server.py \
  --teleop_req tcp://127.0.0.1:28701 \
  --teleop_rep tcp://127.0.0.1:28702 \
  --teleop_ctrl tcp://127.0.0.1:28703 \
  --enable-pico-control \
  --z_bind tcp://*:28711 \
  --hz 50 \
  --mujoco_xml data/robots/g1/scene_29dof_freebase.xml \
  --backward_onnx model/g1_policy/exported/backward_encoder.onnx \
  --device cuda \
  --root_height_obs \
  --wall-clock-dt \
  --fix-quat-continuity \
  --angvel-delta-frame world \
  --max-retarget-age-ms 200 \
  --max-z-delta 0.75
```

终端 D：策略执行
```bash
python rl_policy/bfm_zero.py \
  --robot_config config/robot/g1.yaml \
  --policy_config config/policy/g1_policy.yaml \
  --model_path ./model/g1_policy/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/teleop.yaml
```
