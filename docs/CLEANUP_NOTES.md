# Cleanup Notes

- Public-facing training docs and CLI now expose the historical TLDR preset as TeCH.
- `--agent tldr` is retained as a deprecated compatibility alias for `--agent tech`.
- Internal `tldr` module names, config classes, checkpoint names, config fields, and metric keys are retained for checkpoint/config compatibility.
- A future migration can rename internal modules and checkpoint config classes only with explicit backward-compatibility shims.
- Cleanup audit found no remaining public `ufo_npz` or `csv_ufo` format docs to delete. Existing deploy notes, G1 reward relabel paths, G1 data download paths, RobotState import, XML/Hydra draft generation, and goal/reward inference docs are retained.
