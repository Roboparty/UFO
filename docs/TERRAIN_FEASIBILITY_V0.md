# Terrain-Conditioned FB V0

This opt-in experiment tests whether the original LaFAN-only UFO behavior prior
can adapt one task latent across physical terrain through simulation interaction.
It does not add terrain demonstrations, terrain-conformal motion processing,
vision, depth, or a learned terrain encoder.

## Architecture

The standard `fb` preset is unchanged. `fb_terrain` routes observations as follows:

```text
B:             state + privileged_state
Discriminator: state + privileged_state + z
Actor:         state + last_action + history_actor + terrain_actor + z
F:             state + privileged_state + last_action + history_actor + terrain_priv + action + z
Critic:        state + privileged_state + last_action + history_actor + terrain_priv + action + z
Aux critic:    state + privileged_state + last_action + history_actor + terrain_priv + action + z
```

One 21 x 13 (273D) collision-ray grid produces two observation channels. Local
+x is robot forward and local +y is robot left. The grid covers x `[-0.4,
1.6]` m and y `[-0.6, 0.6]` m at 0.1 m resolution. `terrain_actor` contains
raw pelvis-to-terrain clearances. `terrain_priv` contains metric terrain
heights relative to the ground directly below the pelvis and is clipped to
`[-0.5, 0.5]` m. The same center ray defines the ground-relative root height
in `privileged_state`; the original `fb` world-z observation is unchanged.
MJLab's yaw-aligned GPU ray sensor queries the same MuJoCo geoms used by
physics.

The run used an easy-to-medium fixed distribution through global step
99,295,232: 40% flat, 25% slope, 10% stairs, and 25% rough. Phase 2 deliberately
shifts this to 20% flat, 20% slope, 30% stairs, and 30% rough. This is a fixed
distribution shift at resume, not a time-varying curriculum manager. Slope
difficulty remains 5-8 degrees. Stairs now use 20 concentric levels, 0.30 m
depth, and a per-step height range of 0.08-0.15 m before reaching a bounded
outer plateau. The 0.30 m depth lets the Actor's 1.6 m forward scan observe
multiple step edges. Rough height amplitude remains 0.03-0.05 m with smooth
two-octave Perlin structure. Patches are 30 x 30 m. Startup validates every
motion clip against the patch size, sensor footprint, and policy margin, while
runtime boundary checks fail the run before a robot can enter another patch.
All current values and mix proportions are in
`humanoidverse/config/terrain/terrain_ufo_v0.yaml`.

## Smoke

```bash
./run_train.sh \
  --agent fb_terrain \
  --terrain-mode mixed \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --gpu-ids single \
  --smoke \
  --work-dir /tmp/ufo_smoke_terrain
```

Smoke mode uses a bounded replay buffer and completes one agent update. It is
not a formal training run.

During formal `fb_terrain` training, the original 3.2M-step EMD prioritization
cadence and priority formula are unchanged. Only that evaluator uses a lazy,
persistent terrain-aware flat environment, preventing random mixed-terrain
assignment from changing LaFAN motion sampling weights. Training rollouts remain
on the configured mixed terrain. The fast flat path returns constant pelvis
clearance in `terrain_actor`, zero geometry in `terrain_priv`, and does not
create a terrain height sensor.

## Future Full Training

Do not use this command as a smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./run_train.sh \
  --agent fb_terrain \
  --terrain-mode mixed \
  --data-path humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  --gpu-ids all \
  --num-envs 1024 \
  --num-env-steps 192000000 \
  --work-dir runs/ufo_terrain_v0
```

## Same-Z Evaluation

The evaluator computes a reward, goal, or tracking latent once, saves a SHA256
over the entire tensor (including a complete tracking sequence), clones it, and
asserts the checksum before and after all terrain rollouts. Results are written
as JSON and CSV.

Reward:

```bash
uv run python -m humanoidverse.terrain_transfer_inference \
  --model-folder runs/ufo_terrain_v0 \
  --prompt-type reward \
  --reward-task move-ego-0-0.7 \
  --terrains flat,slope,stairs,rough \
  --output runs/ufo_terrain_v0/terrain_transfer_reward.json
```

Goal:

```bash
uv run python -m humanoidverse.terrain_transfer_inference \
  --model-folder runs/ufo_terrain_v0 \
  --prompt-type goal \
  --goal-index 0 \
  --terrains flat,slope,stairs,rough \
  --output runs/ufo_terrain_v0/terrain_transfer_goal.json
```

Tracking:

```bash
uv run python -m humanoidverse.terrain_transfer_inference \
  --model-folder runs/ufo_terrain_v0 \
  --prompt-type tracking \
  --data-path humanoidverse/data/lafan_29dof.pkl \
  --motion-id 0 \
  --terrains flat,slope,stairs,rough \
  --output runs/ufo_terrain_v0/terrain_transfer_tracking.json
```
