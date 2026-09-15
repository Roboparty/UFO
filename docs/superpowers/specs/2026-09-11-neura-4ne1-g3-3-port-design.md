# Porting UFO to the NEURA 4NE-1 G3.3 — design notes

Date: 2026-09-11. Status: implemented; first FB training run launched from this design.

## 1. Goal

Train a UFO Forward-Backward (FB) policy for the NEURA 4NE-1 G3.3 humanoid on the
LaFAN1 motions that were retargeted from Unitree G1 onto the 4NE-1, using the existing
MJLab training path, with W&B logging and periodic preview videos. Only files inside this
repository are changed; the robot assets and motion CSVs are consumed as delivered.

## 2. Inputs (verified)

| Item | Fact |
| --- | --- |
| MJCF | `humanoidverse/data/robots/neura_4ne1_g3_3/neura_4ne1_g3_3_without_hands.xml`: nq 39, nu 32, 70 bodies (fingers and sensor mounts are welded, jointless bodies), freejoint on `link_pelvis`, no floor. |
| Actuated joints (MJCF order) | left leg (hip_y, hip_x, hip_z, knee, ankle_y, ankle_x), right leg, torso (z, x, y), left arm (shoulder_y, shoulder_x, shoulder_z, elbow, wrist_z, wrist_x, wrist_y), right arm, head (z, x, y). Actuator order equals joint order. |
| Foot geometry | The only foot collision hulls are `left_foot_collision` / `right_foot_collision` on the near-massless welded bodies `sensor_{left,right}_foot_fts` (identity transform w.r.t. `link_{side}_ankle_x`). |
| Motion data | 40 headerless CSVs, 39 columns = root xyz, root quaternion xyzw, 32 joint angles in MJCF order, 30 fps (the retargeter ran at `FPS=30` on the 30 fps G1 LaFAN data; frame counts match the G1 pickle exactly), 264,705 frames. Every value is inside the MJCF joint limits. |
| Gains and limits | `neura_4ne1_g3_3_constants.py` next to the MJCF (0810 system identification, 2026-08-28 ankle spec) and the sim2sim config used by the NEURA tracking stack. |
| Compute | One RTX PRO 6000 Blackwell (96 GB) shared with a running AMP job (~16 GB, ~80 % util); 24 CPU cores; 125 GB RAM. |

## 3. Reference-data quality and the ground-alignment decision

The retargeter matches keypoints without a floor constraint. Forward kinematics of every
frame with the 4NE-1 collision hulls shows:

| Dataset | frames with > 2 cm penetration | frames with > 10 cm | notes |
| --- | --- | --- | --- |
| G1 LaFAN (UFO reference) | 0.9 % | 0.0 % | after removing the constant 9.3 cm offset of the G1 foot capsules |
| 4NE-1 retargets | 33.4 % | 8.2 % | lying poses are 0.2 to 0.6 m below the floor; running clips dig the toe in by 10 to 20 cm |

Options considered:

1. Train on the data as is. Rejected: underground root heights enter the expert state
   distribution (`root_height_obs=True`) and every fall/get-up clip is physically impossible.
2. Drop every clip with penetration above a threshold. Rejected as the only measure: a 10 cm
   threshold removes 55 % of clips including almost all running.
3. Per-frame root lift plus a residual filter. **Chosen.** Lift the root by
   `max(0, penetration - 2 cm)` smoothed with a centered 0.3 s moving average, then drop 10 s
   training clips whose residual penetration still exceeds 15 cm. Sustained segments (lying
   on the floor) are lifted fully; brief toe digs get a small smooth correction instead of a
   vertical bounce; clearly broken retargets are removed.

Result: 852 of 902 ten-second clips are kept (94 %), including all running and sprint clips.
Fall/get-up clips are mostly dropped (13 of 90 survive). The full-length inference motions
are lifted the same way and never filtered.

Implementation: `humanoidverse/utils/motion_data/ground_align.py`, wired through
`robot_state_dict_to_ufo_motion_dict`, the manifest `ground_alignment` block, and
`data_build --ground-align`. Alignment is opt-in and only exists for `robot_state_*` sources;
`ufo_pkl` sources are untouched. Tests: `tests/test_ground_alignment.py`.

## 4. Robot configuration decisions

`configs/robots/neura_4ne1_g3_3.yaml` (consumed through `RobotTrainingSpec.to_env_dict()`) and
the Hydra group `neura_4ne1_g3_3/neura_4ne1_g3_3` (only its non-patched fields matter).

| Field | Value | Reason |
| --- | --- | --- |
| `feet` / `semantics.contact_bodies` | `sensor_left_foot_fts`, `sensor_right_foot_fts` | The MJLab contact sensor runs in body mode; only these bodies own foot collision geoms. Left first: the aux rewards index `feet_indices[0]` as left. |
| `torso_name` | `link_torso_y` | Upper torso link that carries shoulders, head and the IMU; used for base-COM randomization and torso metrics. |
| `undesired_contact_bodies` | `link_pelvis`, `shoulder`, `hip` | Substring patterns, same semantics as the G1 set. |
| ankle dof names | `[*_ankle_y, *_ankle_x]` | Pitch first, roll second; the ankle-roll penalty reads index 1. |
| `default_joint_angles` | legs -0.1 / 0 / 0 / 0.3 / -0.2 / 0, torso 0, arms 0.2 / ±0.05 / 0 / -0.4 / 0 / 0 / 0, head 0 | Standing pose shared with the NEURA sim2sim and tracking stacks; actions are offsets from it. |
| `init_state.pos.z` | 0.942 m | Puts the lowest foot-hull vertex on the floor for that pose (verified with FK). |
| PD gains | hip_y 763.7/64.8, hip_x 387.7/32.9, hip_z 208.3/17.7, knee 387.7/32.9, ankle_y 350/30, ankle_x 180/10, torso 208.3/17.7, shoulder_y/x 208.3/17.7, shoulder_z and elbow 92.8/7.9, wrist 10/1, head 15/0.75 | Identified gains validated on hardware by the NEURA stack. Keys are joint-name substrings; the first match wins. |
| `effort_limit` | MJCF `<motor>` forcerange (rated torque) | Also fixes the action scale: `0.25 * effort / kp`, reproducing the NEURA action scales exactly (e.g. hip_y 0.036 rad). |
| `velocity_limit` | 20 rad/s, ankles 10.97 / 16.18 | The URDF has no velocity limits; ankle values come from the 2026-08-28 spec table. Only shapes the DC-motor torque-speed envelope. |
| `actuator.source: yaml` | armature and Coulomb friction per joint copied from the MJCF | The G1 mode-15 table raises on 4NE-1 joint names; MJLab replaces the XML actuators with DC-motor PD actuators and keeps the XML viscous damping. |
| `hydra_overrides` | empty (G1 uses `lie_down_init=True`, `lie_down_init_prob=0.3`) | The ground-alignment filter removes almost all fall/get-up references, so 30 % side-lying resets would spend samples on states without expert data. Re-enable once floor-aware get-up retargets exist. |
| bodies | all 70 MJCF bodies | RobotSpec has no subset mechanism and MotionLib FK needs one `pose_aa` slot per MJCF body. Cost: `privileged_state` is 1047 floats (G1: 448). Trimming sensor and finger bodies is a documented follow-up. |
| `extend_config` | empty | The 4NE-1 has real head bodies; G1's virtual `head_link` is dropped. |

Alternative MJCF: `neura_4ne1_g3_3_sysid_without_hands.xml` (a different passive-dynamics
identification) can be swapped in `xml_path` without other changes; the NEURA default task
uses the non-sysid file, so this port does too. The full-hand `neura_4ne1_g3_3.xml` must not
be used: MotionLib asserts that the number of joints equals the number of actuators.

## 5. Code changes

| Change | Files | Why |
| --- | --- | --- |
| Ground alignment | `humanoidverse/utils/motion_data/ground_align.py`, `robot_state_convert.py`, `adapters.py`, `manifest.py`, `tools/data_build.py`, `tests/test_ground_alignment.py` | Section 3. |
| Comment-tolerant MJCF parsing | `humanoidverse/utils/mjcf_xml.py`, `utils/motion_lib/skeleton.py`, `utils/motion_lib/torch_humanoid_batch.py`, `mjlab_inference_utils.py` | The exported MJCFs contain `--` inside an XML comment. MuJoCo accepts it; `xml.etree` and `lxml` reject the file, which crashed MotionLib initialisation. Stripping comments before parsing keeps the generated assets untouched. |
| Training-time video previews | `training/workspace.py` (`video_*` fields, `build_video_preview_command`, `_launch/_poll/_finish_video_job`), `train.py` (`--video-every-checkpoints`, `--video-motion-ids`, `--video-max-steps`, `--video-render-size`), `tests/test_video_preview.py` | After every N-th checkpoint, rank 0 runs `tracking_inference --save-mp4` in a subprocess on the saved checkpoint (own 1-env MJLab instance, EGL renderer) and logs the side-by-side expert/policy MP4s to W&B under `video/`. The training loop is never blocked. |
| torch CUDA 12.8 index | `pyproject.toml`, `uv.lock` | Default cu126 wheels have no sm_120 kernels for Blackwell GPUs. cu128 wheels also run on older GPUs. |
| 4NE-1 configs | `configs/robots/neura_4ne1_g3_3.yaml`, `humanoidverse/config/robot/neura_4ne1_g3_3/neura_4ne1_g3_3.yaml`, `configs/data/lafan_neura_4ne1_g3_3.yaml` | Section 4. |

`tags.robot` is now recorded in the train config so W&B runs of different robots can be
told apart.

## 6. Training plan

Single shared GPU, so per-rank defaults are scaled down where memory matters:

- `--num-envs 1024` (default), `--buffer-size 2560000` (half of the default; the trajectory
  buffer lives on the GPU and each stored 4NE-1 transition is about 7.7 kB, so 2.56M
  transitions take about 20 GB).
- `--num-env-steps 192000000` (default budget), `--checkpoint-every-steps 9600000` to keep the
  replay-buffer dump cadence reasonable, `--video-every-checkpoints 1` with three motions
  (walk1_subject1 = 28, run1_subject2 = 22, dance1_subject1 = 0), 400 steps at 50 Hz.
- `--use-wandb --wandb-run-name ufo_fb_4ne1_g3_3` on the NEURA W&B instance
  (`~/.config/wandb/settings`), project `ufo-humanoid`, group `ufo_fb`.
- Launched in a dedicated tmux session with `scripts/train_4ne1_g3_3.sh`; README "Path C" has
  the exact command.

Stability criteria used while monitoring: no NaN/inf (the workspace asserts), training FPS
steady, `train/` losses finite and decreasing, tracking-eval success/MPJPE improving across
the first evaluations, preview videos showing the robot standing and following the motion.

## 7. Known limitations and follow-ups

- Toe/heel penetration of up to 15 cm remains in some running frames; the true fix is a
  floor-aware retargeter (`robot_retargeter` is outside this repo).
- 70-body observation: a training MJCF without sensor/finger bodies (keeping the two foot
  bodies) would halve `privileged_state` and replay memory.
- Goal and reward inference remain G1-centric (goal JSON, relabel body names); tracking
  inference and ONNX export work with `--robot-config` and `--data-manifest`.
- Inference defaults (`configs/robots/g1_29dof.yaml`, G1 pickle) still silently select G1 when
  flags are omitted; always pass the 4NE-1 robot config and data.

## 8. Run log

| Item | Value |
| --- | --- |
| Launched | 2026-09-11 20:48 local (attempt 1, stopped after 4 min to disable `lie_down_init`; W&B run 6ey14guh is stale), relaunched 20:53, tmux session `ufo_4ne1_train`, log `runs/ufo_fb_4ne1_g3_3.log` |
| Command | `scripts/train_4ne1_g3_3.sh` (FB, 1024 envs, buffer 2.56M, 192M env steps, checkpoint every 9.6M, videos every checkpoint) |
| W&B | https://neura.wandb.io/lei-li/ufo-humanoid/runs/jzu3by66 (project `ufo-humanoid`, group `ufo_fb`, run `ufo_fb_4ne1_g3_3`); the aborted attempt 1 is run 6ey14guh |
| Observation dims | actor 1557 (state 70, last_action 32, history 408, privileged 1047), critic 1047, actions 32 |
| Data | 852 training clips of 10 s at 30 fps from 40 motions; 40 full motions for eval/videos |
| Smoke tests | `runs/smoke_4ne1_g3_3` (16 envs, 2048 steps): env build, MotionLib, checkpoint, and preview video verified before launch |

Early observations (first 25 minutes):

| Metric | Value | Reading |
| --- | --- | --- |
| Step-0 tracking eval | 852 clips evaluated, proximity 0.23, MPJPE 3.7 m | untrained policy baseline; eval + prioritization path works for the 4NE-1 |
| Throughput | ~556 env-steps/s between 384k and 768k steps (GPU shared with the AMP job at ~99 % total utilization) | 192M steps take about 4 days at this rate; expect roughly 2x once the neighbour job ends |
| Process memory | ~42 GB GPU, 10 GB host RSS, flat | within budget next to the 16 GB neighbour |
| Losses at 768k steps | fb_loss -9447, actor_loss 3419, critic_loss 2.7, disc_loss 0.046, mean aux reward -1.24 (from -1.72) | all finite; typical early FB magnitudes |
| Posture at 768k steps | torso height 0.19 m, torso contact force ~440 N | robot still falls; standing is expected to emerge after tens of millions of steps |

Note: training-loop prints are block-buffered when piped through `tee`, so the tmux console and
`runs/ufo_fb_4ne1_g3_3.log` lag behind `train_log.txt` and W&B for this first run; the launch
script now sets `PYTHONUNBUFFERED=1` for future runs.

Update after 2.3 h (4.6M steps, 12 logging intervals): throughput constant at ~553 env-steps/s,
no errors, GPU memory flat at ~44 GB, all losses finite; the action-rate penalty fell from 310
to 13 and the undesired-contact fraction started to decline (0.63 to 0.49). The robot does not
stand yet, as expected this early. First checkpoint and preview videos are due at 9.6M steps
(about 4.8 h after launch).

Tracking-eval trend (mean over all 852 training clips):

| Env steps | proximity | MPJPE [mm] | EMD | vel dist | accel dist |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.227 | 3718 | 3.63 | 276 | 27.0 |
| 3.2M | 0.504 | 3042 | 2.93 | 159 | 6.5 |
| 6.4M | 0.553 | 2950 | 2.84 | 157 | 6.4 |

At 9.2M steps (4.7 h) the undesired-contact fraction had fallen from 0.63 to 0.26 and throughput
was still ~555 env-steps/s with no errors.

First checkpoint (9.6M steps, 4.9 h): checkpoint written (26 GB with the replay buffer), the
preview job rendered walk1_subject1, run1_subject2 and dance1_subject1 in 12.6 min on the shared
GPU, and all three MP4s were logged to W&B under `video/` with the checkpoint step as caption.
The frames show the policy standing and following the reference posture for all three motions.
Monitoring was reduced to failure signals only; the run is left to train.

## 9. Performance review at 125M steps (2026-09-14)

Method: eval CSV statistics, replay/loss trends, renders of the worst clips with the 122.9M
checkpoint, and a four-lens analysis (data, env config, algorithm, evaluation) whose proposals
were each vetted by two adversarial verifiers.

Where the policy stands: median clip proximity 0.96, 59 % of clips >= 0.9 (walk, dance1,
jumps, sprint), run/fight 0.66-0.77, fall/get-up 0.45-0.67; mean 0.79-0.85 on a plateau since
~50M steps. The mean oscillation is a common-mode fall-rate swing on ~365 mid-difficulty clips
under DR-on evaluation, not eval sampling noise. Failing clips split into falls (fast or
floor motions; the policy cannot get up) and ~30 clips whose reference arms sit on the far
Euler branch (shoulder_y/z, wrist_z near +-pi).

Root cause found for the plateau (rank 1, confirmed): the commandable PD set-point range is
far too narrow. With the tanh actor (+-1, x5, clip +-5) and `action_scale = 0.25 * effort / kp`,
the NEURA gains give max |target - default| of 0.18 rad (hip_y), 0.32 (knee), 0.36 (shoulder,
torso), 0.40 (elbow) versus 1.1-4.3 rad on G1. 75-80 % of reference hip/knee frames are
unreachable and 46-58 % of replay knee actions are pinned at the bound. The NEURA action-scale
parity assumed in section 4 is therefore wrong for UFO's bounded actor.

Ranked plan (details in the workflow journal `wf_5a3979e9-ab3`):

| # | Change | When | Confidence |
| --- | --- | --- | --- |
| 1 | Per-joint action scales (~joint range / 5) with reduced actor noise | v2 | 0.80 |
| 2 | Snapshot checkpoints; render failing clips DR-on/off; per-joint error dumps | now | 0.75 |
| 3 | Tail and per-type eval statistics (median, p10, frac<0.5, per type) | now / v2 | 0.85 |
| 4 | Shoulder Euler-branch fix at import (3-joint flip, Viterbi selection) | v2 | 0.75 |
| 5 | Identified DC-motor envelope (saturation torque, no-load speed) | v2 | 0.50 |
| 6-8 | Data hygiene: soft-limit clamp before alignment, wrist zeroing, root-teleport repair in ground alignment | v2 | 0.40-0.60 |
| 9 | Truncate long-fallen episodes (with per-env expert-rollout fix) | v3, after measuring v2 | 0.40 |
| 10-14 | Diagnostics keys, fb_pessimism_penalty 0.5, buffer 5.12M, priority mask for never-tracked clips, DR-free second eval | v2 | 0.30-0.70 |
| 15-17 | Root-relative body-position metrics, retargeter feedback package, body trim | later | 0.35-0.70 |

Sequence: keep v1 running as a baseline (snapshots under `runs/ufo_fb_4ne1_g3_3/snapshots/`),
bundle all v2 items, rebuild caches, smoke test, launch v2 on the now-unshared GPU (~4 days),
compare against v1 snapshots at equal steps on non-near-pi p10, frac<0.5, per-type run/fight
means, per-joint errors and replay action saturation. Correction to section 3: 52 of 96
fall/get-up windows are in the training set (not 13 of 90).
