#!/usr/bin/env python3
"""Download the released G1 ONNX policy and matching task files; no login required."""

import argparse

from huggingface_hub import snapshot_download


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="model", help="Parent directory for g1_policy/")
    parser.add_argument("--revision", default="da3727034a058eae588a16af142764ee34bdada4", help="Hugging Face release revision")
    args = parser.parse_args(argv)
    path = snapshot_download(
        repo_id="xuewang/ufo-g1-policy",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=[
            "g1_policy/exported/*", "g1_policy/tasks/*",
            "g1_policy/tracking_inference/*.pkl", "g1_policy/goal_inference/*.pkl",
            "g1_policy/reward_inference/*.pkl", "g1_policy/release_manifest.json",
        ],
    )
    print(f"Downloaded G1 deployment files to {path}/g1_policy")


if __name__ == "__main__":
    main()
