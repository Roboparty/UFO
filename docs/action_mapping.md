# Action-to-position-target mappings

UFO policies emit normalized actions. The environment can convert them to
joint-position targets with either of these mappings:

- `effort_kp` (default) preserves the existing behavior. Its per-joint target
  scale is derived from the configured action scale and, when action rescaling
  is enabled, the effort-limit-to-stiffness ratio.
- `soft_limit_bias` maps each clipped policy action in `[-1, 1]` affinely onto
  that joint's configured soft position-limit interval.

The second mapping is useful for robots with strongly asymmetric joint ranges.
For a soft interval `[q_min, q_max]`, default position `q_default`, and MJLab
target scale `s`, it computes

```text
lower = (q_min - q_default) / s
upper = (q_max - q_default) / s
bias = (lower + upper) / 2
half_range = (upper - lower) / 2
mapped_action = bias + half_range * clamp(policy_action, -1, 1)
```

Consequently, policy outputs `-1` and `1` correspond to `q_min` and `q_max`.
The default remains `effort_kp`, so existing launch commands and checkpoints
retain their current behavior.

Select the alternative mapping when training:

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
mapping used during training. Tracking ONNX export records the mapping name and,
for `soft_limit_bias`, the affine parameters and resolved soft limits in the
companion `.meta.json` file.
