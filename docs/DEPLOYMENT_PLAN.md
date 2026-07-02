# UFO Deployment Plan

This branch is intentionally separate from `main` so deployment dependencies do not affect training users.

## Artifact Boundary

`main` should export artifacts. `deploy` should load artifacts.

Expected inputs:

- policy ONNX
- backward encoder ONNX
- observation normalizer metadata
- robot runtime configuration
- action scaling and joint ordering metadata

## Runtime Components

- policy runner: reads robot state, builds observations, runs policy ONNX, sends actions
- latent server: computes or streams `z` for goal/reward/teleop control
- teleop client: sends user commands or goal selections to the latent server
- safety layer: clamps actions and checks state validity before sending commands

## Open Items

- finalize robot communication backend
- finalize teleop input device protocol
- finalize exported artifact schema
- add smoke tests for ONNX loading and observation assembly
