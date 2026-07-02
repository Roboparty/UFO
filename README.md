# UFO Deploy

This branch is reserved for deployment and teleoperation code for UFO humanoid policies.

For training, MJLab inference, and policy export, use the `main` branch.

## Branches

- `main`: UFO training, MJLab inference, and export tooling.
- `deploy`: real-world deployment, teleoperation, realtime latent control, and policy runners.

## Expected Artifacts

The deploy stack should consume exported artifacts produced from the `main` branch, for example:

```text
artifacts/
  policy.onnx
  backward_encoder.onnx
  normalizer.json
  policy_config.yaml
  robot_config.yaml
```

Large model artifacts are intentionally ignored by git. Keep them under `artifacts/` locally or download them from a release/checkpoint store.

## Planned Layout

```text
deploy/
  scripts/      # realtime policy runner, z server, teleop clients
  configs/      # robot and policy runtime configs
  artifacts/    # local exported ONNX/checkpoint artifacts; ignored by git

docs/
  DEPLOYMENT_PLAN.md
```

The branch currently contains only the deployment scaffold. Add the actual teleop and robot runtime code here once the deployment stack is finalized.
